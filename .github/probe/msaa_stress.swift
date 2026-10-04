// Diagnostic for #1016 and #707: does the hosted image's Metal device lose or
// hang multisampled work without mtld3d or Wine in the process?
//
// Plain Metal, shaped like the e2e tests that fail on the Intel image:
//   A  4x colour cleared to white and resolved by the store action.
//   E  4x colour, a triangle edge drawn over a black clear, resolved.
//   B  4x colour + 4x Depth32Float_Stencil8: clear, near draw, store; a
//      clear-only pass on another target; reload, far draw that must fail the
//      depth test, resolve (one_off_passes::color_fill_leaves_a_multisampled_
//      depth_attachment_bound). Even iterations keep the stencil plane, odd
//      ones discard it.
//   C  the RESZ transfer: blit the 4x depth-stencil into a second 4x texture,
//      read sample zero of depth and stencil in compute.
// Every iteration builds fresh textures, encodes A, E, B and C into one
// command buffer, waits, and checks every value read back.
//
// Phases: msaa1 (time-boxed loops), storm (threads creating and releasing
// textures and sRGB views, which on the Intel image provokes the kernel's
// `addObject: Object already exists`), msaa2 (loops again).
//
// Output: `PROBE ...` lines. One `PROBE-SUMMARY` line per phase. The exit
// status is 0 whatever the device did; a non-zero status means the probe
// itself broke.

import Foundation
import Metal

setvbuf(stdout, nil, _IOLBF, 0)

let args = CommandLine.arguments
let msaaSeconds = args.count > 1 ? Double(args[1]) ?? 210 : 210
let stormSeconds = args.count > 2 ? Double(args[2]) ?? 30 : 30
let started = Date()

func now() -> String { String(format: "%.1f", Date().timeIntervalSince(started)) }
func say(_ s: String) { print("PROBE t=\(now()) \(s)") }

guard let device = MTLCreateSystemDefaultDevice() else {
    print("PROBE-ERROR no Metal device")
    exit(2)
}
guard let queue = device.makeCommandQueue() else {
    print("PROBE-ERROR no command queue")
    exit(2)
}

say("device name=\(device.name) registryID=\(device.registryID) unified=\(device.hasUnifiedMemory) lowPower=\(device.isLowPower) headless=\(device.isHeadless) removable=\(device.isRemovable) location=\(device.location.rawValue) locationNumber=\(device.locationNumber) peerGroupID=\(device.peerGroupID) peerIndex=\(device.peerIndex) peerCount=\(device.peerCount)")
say("device maxTransferRate=\(device.maxTransferRate) workingSet=\(device.recommendedMaxWorkingSetSize) maxBuffer=\(device.maxBufferLength) maxThreadgroupMemory=\(device.maxThreadgroupMemoryLength) maxThreadsPerThreadgroup=\(device.maxThreadsPerThreadgroup.width)x\(device.maxThreadsPerThreadgroup.height)x\(device.maxThreadsPerThreadgroup.depth)")
say("device d24s8=\(device.isDepth24Stencil8PixelFormatSupported) rog=\(device.areRasterOrderGroupsSupported) bc=\(device.supportsBCTextureCompression) rwTier=\(device.readWriteTextureSupport.rawValue) argTier=\(device.argumentBuffersSupport.rawValue) float32Filter=\(device.supports32BitFloatFiltering) msaa32=\(device.supports32BitMSAA) raytracing=\(device.supportsRaytracing) sparseTile=\(device.sparseTileSizeInBytes) mac2=\(device.supportsFamily(.mac2)) metal3=\(device.supportsFamily(.metal3)) apple7=\(device.supportsFamily(.apple7))")
say("device sampleCounts=\([1, 2, 4, 8].filter { device.supportsTextureSampleCount($0) }) linearAlign=\(device.minimumLinearTextureAlignment(for: .bgra8Unorm)) bufferAlign=\(device.minimumTextureBufferAlignment(for: .bgra8Unorm))")

let source = """
#include <metal_stdlib>
using namespace metal;
struct VOut { float4 pos [[position]]; float4 color; };
struct Params { float4 color; float depth; uint shape; };
vertex VOut vs(uint vid [[vertex_id]], constant Params &p [[buffer(0)]]) {
    float2 full[3] = { float2(-1, -1), float2(3, -1), float2(-1, 3) };
    // A right triangle whose diagonal crosses every row through pixel
    // centres, so each row has a pixel with two of its four samples covered.
    float2 edge[3] = { float2(-1, -1), float2(1, -1), float2(-1, 1) };
    float2 xy = p.shape == 0 ? full[vid] : edge[vid];
    VOut o;
    o.pos = float4(xy, p.depth, 1);
    o.color = p.color;
    return o;
}
fragment float4 fs(VOut in [[stage_in]]) { return in.color; }
kernel void read_sample_zero(depth2d_ms<float, access::read> d [[texture(0)]],
                             texture2d_ms<uint, access::read> s [[texture(1)]],
                             device float *out_depth [[buffer(0)]],
                             device uint *out_stencil [[buffer(1)]],
                             uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= d.get_width() || gid.y >= d.get_height()) return;
    uint i = gid.y * d.get_width() + gid.x;
    out_depth[i] = d.read(gid, 0);
    out_stencil[i] = s.read(gid, 0).r;
}
"""

let library: MTLLibrary
do {
    library = try device.makeLibrary(source: source, options: nil)
} catch {
    print("PROBE-ERROR library: \(error)")
    exit(2)
}

func renderPipeline(depth: Bool) -> MTLRenderPipelineState {
    let d = MTLRenderPipelineDescriptor()
    d.vertexFunction = library.makeFunction(name: "vs")
    d.fragmentFunction = library.makeFunction(name: "fs")
    d.colorAttachments[0].pixelFormat = .bgra8Unorm
    d.rasterSampleCount = 4
    if depth {
        d.depthAttachmentPixelFormat = .depth32Float_stencil8
        d.stencilAttachmentPixelFormat = .depth32Float_stencil8
    }
    do {
        return try device.makeRenderPipelineState(descriptor: d)
    } catch {
        print("PROBE-ERROR pipeline depth=\(depth): \(error)")
        exit(2)
    }
}

let colourPipeline = renderPipeline(depth: false)
let depthPipeline = renderPipeline(depth: true)
let computePipeline: MTLComputePipelineState
do {
    computePipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "read_sample_zero")!)
} catch {
    print("PROBE-ERROR compute: \(error)")
    exit(2)
}
let depthState: MTLDepthStencilState = {
    let d = MTLDepthStencilDescriptor()
    d.depthCompareFunction = .lessEqual
    d.isDepthWriteEnabled = true
    return device.makeDepthStencilState(descriptor: d)!
}()

let size = 128
let rowBytes = size * 4
let stencilClear: UInt32 = 0x4b

struct Params {
    var color: SIMD4<Float>
    var depth: Float
    var shape: UInt32
    var pad: SIMD2<UInt32> = .zero
}

func texture(_ format: MTLPixelFormat, samples: Int, usage: MTLTextureUsage) -> MTLTexture? {
    let d = MTLTextureDescriptor()
    d.textureType = samples > 1 ? .type2DMultisample : .type2D
    d.pixelFormat = format
    d.width = size
    d.height = size
    d.sampleCount = samples
    d.storageMode = .private
    d.usage = usage
    return device.makeTexture(descriptor: d)
}

func draw(_ enc: MTLRenderCommandEncoder, color: SIMD4<Float>, depth: Float, shape: UInt32) {
    var p = Params(color: color, depth: depth, shape: shape)
    enc.setVertexBytes(&p, length: MemoryLayout<Params>.stride, index: 0)
    enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
}

func readback(_ cb: MTLCommandBuffer, _ tex: MTLTexture) -> MTLBuffer {
    let buf = device.makeBuffer(length: rowBytes * size, options: .storageModeManaged)!
    let blit = cb.makeBlitCommandEncoder()!
    blit.copy(from: tex, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
              sourceSize: MTLSize(width: size, height: size, depth: 1), to: buf,
              destinationOffset: 0, destinationBytesPerRow: rowBytes, destinationBytesPerImage: rowBytes * size)
    blit.synchronize(resource: buf)
    blit.endEncoding()
    return buf
}

struct Iteration {
    var a: MTLBuffer
    var e: MTLBuffer
    var b: MTLBuffer
    var cDepth: MTLBuffer
    var cStencil: MTLBuffer
    var keepStencil: Bool
}

final class Counters {
    var iterations = 0
    var cbErrors = 0
    var consecutiveErrors = 0
    var bad: [String: Int] = ["A": 0, "E": 0, "B": 0, "C": 0]
    var firstBad: [String: String] = [:]
    var firstError = ""
    var createNil = 0
    var maxCbSeconds = 0.0
}

func encodeIteration(_ cb: MTLCommandBuffer, index: Int) -> Iteration? {
    let rt: MTLTextureUsage = [.renderTarget]
    let readable: MTLTextureUsage = [.renderTarget, .shaderRead, .pixelFormatView]
    guard let aMs = texture(.bgra8Unorm, samples: 4, usage: rt),
          let aOut = texture(.bgra8Unorm, samples: 1, usage: rt),
          let eMs = texture(.bgra8Unorm, samples: 4, usage: rt),
          let eOut = texture(.bgra8Unorm, samples: 1, usage: rt),
          let bMs = texture(.bgra8Unorm, samples: 4, usage: rt),
          let bOut = texture(.bgra8Unorm, samples: 1, usage: rt),
          let bDepth = texture(.depth32Float_stencil8, samples: 4, usage: readable),
          let other = texture(.bgra8Unorm, samples: 1, usage: rt),
          let cCopy = texture(.depth32Float_stencil8, samples: 4, usage: readable)
    else { return nil }
    let keepStencil = index % 2 == 0

    // A: clear to white, resolve by the store action.
    var rp = MTLRenderPassDescriptor()
    rp.colorAttachments[0].texture = aMs
    rp.colorAttachments[0].resolveTexture = aOut
    rp.colorAttachments[0].loadAction = .clear
    rp.colorAttachments[0].clearColor = MTLClearColor(red: 1, green: 1, blue: 1, alpha: 1)
    rp.colorAttachments[0].storeAction = .multisampleResolve
    cb.makeRenderCommandEncoder(descriptor: rp)!.endEncoding()

    // E: black clear, opaque red edge triangle, resolve.
    rp = MTLRenderPassDescriptor()
    rp.colorAttachments[0].texture = eMs
    rp.colorAttachments[0].resolveTexture = eOut
    rp.colorAttachments[0].loadAction = .clear
    rp.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
    rp.colorAttachments[0].storeAction = .multisampleResolve
    var enc = cb.makeRenderCommandEncoder(descriptor: rp)!
    enc.setRenderPipelineState(colourPipeline)
    draw(enc, color: SIMD4(1, 0, 0, 1), depth: 0.5, shape: 1)
    enc.endEncoding()

    // B pass 1: clear, near white draw, store colour and depth.
    rp = MTLRenderPassDescriptor()
    rp.colorAttachments[0].texture = bMs
    rp.colorAttachments[0].loadAction = .clear
    rp.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
    rp.colorAttachments[0].storeAction = .store
    rp.depthAttachment.texture = bDepth
    rp.depthAttachment.loadAction = .clear
    rp.depthAttachment.clearDepth = 1.0
    rp.depthAttachment.storeAction = .store
    rp.stencilAttachment.texture = bDepth
    rp.stencilAttachment.loadAction = .clear
    rp.stencilAttachment.clearStencil = stencilClear
    rp.stencilAttachment.storeAction = keepStencil ? .store : .dontCare
    enc = cb.makeRenderCommandEncoder(descriptor: rp)!
    enc.setRenderPipelineState(depthPipeline)
    enc.setDepthStencilState(depthState)
    draw(enc, color: SIMD4(1, 1, 1, 1), depth: 0.2, shape: 0)
    enc.endEncoding()

    // C: the RESZ transfer, from the attachment pass 1 stored.
    let blit = cb.makeBlitCommandEncoder()!
    blit.copy(from: bDepth, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
              sourceSize: MTLSize(width: size, height: size, depth: 1), to: cCopy,
              destinationSlice: 0, destinationLevel: 0, destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
    blit.endEncoding()
    guard let stencilView = cCopy.makeTextureView(pixelFormat: .x32_stencil8) else { return nil }
    let cDepth = device.makeBuffer(length: size * size * 4, options: .storageModeManaged)!
    let cStencil = device.makeBuffer(length: size * size * 4, options: .storageModeManaged)!
    let compute = cb.makeComputeCommandEncoder()!
    compute.setComputePipelineState(computePipeline)
    compute.setTexture(cCopy, index: 0)
    compute.setTexture(stencilView, index: 1)
    compute.setBuffer(cDepth, offset: 0, index: 0)
    compute.setBuffer(cStencil, offset: 0, index: 1)
    compute.dispatchThreadgroups(MTLSize(width: size / 8, height: size / 8, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
    compute.endEncoding()
    let sync = cb.makeBlitCommandEncoder()!
    sync.synchronize(resource: cDepth)
    sync.synchronize(resource: cStencil)
    sync.endEncoding()

    // B pass 2: a clear-only pass on another target, as a ColorFill makes.
    rp = MTLRenderPassDescriptor()
    rp.colorAttachments[0].texture = other
    rp.colorAttachments[0].loadAction = .clear
    rp.colorAttachments[0].clearColor = MTLClearColor(red: 1, green: 0, blue: 0, alpha: 1)
    rp.colorAttachments[0].storeAction = .store
    cb.makeRenderCommandEncoder(descriptor: rp)!.endEncoding()

    // B pass 3: reload, far blue draw that fails the depth test, resolve.
    rp = MTLRenderPassDescriptor()
    rp.colorAttachments[0].texture = bMs
    rp.colorAttachments[0].resolveTexture = bOut
    rp.colorAttachments[0].loadAction = .load
    rp.colorAttachments[0].storeAction = .storeAndMultisampleResolve
    rp.depthAttachment.texture = bDepth
    rp.depthAttachment.loadAction = .load
    rp.depthAttachment.storeAction = .dontCare
    rp.stencilAttachment.texture = bDepth
    rp.stencilAttachment.loadAction = keepStencil ? .load : .dontCare
    rp.stencilAttachment.storeAction = .dontCare
    enc = cb.makeRenderCommandEncoder(descriptor: rp)!
    enc.setRenderPipelineState(depthPipeline)
    enc.setDepthStencilState(depthState)
    draw(enc, color: SIMD4(0, 0, 1, 1), depth: 0.8, shape: 0)
    enc.endEncoding()

    return Iteration(a: readback(cb, aOut), e: readback(cb, eOut), b: readback(cb, bOut),
                     cDepth: cDepth, cStencil: cStencil, keepStencil: keepStencil)
}

func describe(_ words: UnsafeBufferPointer<UInt32>) -> String {
    var counts: [UInt32: Int] = [:]
    for w in words { counts[w, default: 0] += 1 }
    let top = counts.sorted { $0.value > $1.value }.prefix(4)
    let row = (0..<8).map { String(format: "%08x", words[size * (size / 2) + $0]) }.joined(separator: " ")
    return "values=" + top.map { String(format: "%08x x%d", $0.key, $0.value) }.joined(separator: ",") + " midrow[0..8]=" + row
}

/// Checks one iteration; returns the scenario names that read wrong with a note each.
func check(_ it: Iteration) -> [(String, String)] {
    var wrong: [(String, String)] = []
    let n = size * size
    let a = UnsafeBufferPointer(start: it.a.contents().bindMemory(to: UInt32.self, capacity: n), count: n)
    if a.contains(where: { $0 != 0xffff_ffff }) { wrong.append(("A", describe(a))) }

    // E: left column inside the triangle is red, right column outside is
    // black, and each row has at least one pixel strictly between them.
    let e = UnsafeBufferPointer(start: it.e.contents().bindMemory(to: UInt32.self, capacity: n), count: n)
    var eBad = false
    for y in stride(from: 4, to: size - 4, by: 8) {
        let row = Array(e[(y * size)..<(y * size + size)])
        if row[0] != 0xffff_0000 || row[size - 1] != 0xff00_0000 { eBad = true; break }
        if !row.contains(where: { $0 != 0xffff_0000 && $0 != 0xff00_0000 && ($0 >> 24) == 0xff }) { eBad = true; break }
    }
    if eBad { wrong.append(("E", describe(e))) }

    let b = UnsafeBufferPointer(start: it.b.contents().bindMemory(to: UInt32.self, capacity: n), count: n)
    if b.contains(where: { $0 != 0xffff_ffff }) { wrong.append(("B", "stencilKept=\(it.keepStencil) " + describe(b))) }

    let depth = UnsafeBufferPointer(start: it.cDepth.contents().bindMemory(to: Float.self, capacity: n), count: n)
    let stencil = UnsafeBufferPointer(start: it.cStencil.contents().bindMemory(to: UInt32.self, capacity: n), count: n)
    let depthBad = depth.filter { abs($0 - 0.2) > 0.001 }.count
    let stencilBad = it.keepStencil ? stencil.filter { $0 != stencilClear }.count : 0
    if depthBad > 0 || stencilBad > 0 {
        wrong.append(("C", "stencilKept=\(it.keepStencil) depthWrong=\(depthBad) stencilWrong=\(stencilBad) depth[mid]=\(depth[n / 2 + size / 2]) stencil[mid]=\(stencil[n / 2 + size / 2])"))
    }
    return wrong
}

/// The last moment a command buffer finished; a watchdog ends the probe
/// with a summary if the device stops completing work altogether.
let progressLock = NSLock()
var lastProgress = Date()
var currentPhase = "init"
var summaries: [String] = []

func touch() {
    progressLock.lock()
    lastProgress = Date()
    progressLock.unlock()
}

Thread.detachNewThread {
    while true {
        Thread.sleep(forTimeInterval: 5)
        progressLock.lock()
        let idle = Date().timeIntervalSince(lastProgress)
        let phase = currentPhase
        progressLock.unlock()
        if idle > 90 {
            print("PROBE-SUMMARY phase=\(phase) result=stalled idleSeconds=\(Int(idle)) note=no command buffer completed for 90 s; exiting")
            exit(0)
        }
    }
}

func msaaPhase(_ name: String, seconds: Double) {
    progressLock.lock(); currentPhase = name; progressLock.unlock()
    let c = Counters()
    let phaseStart = Date()
    var stoppedForHang = false
    var lastReport = phaseStart
    while Date().timeIntervalSince(phaseStart) < seconds {
        autoreleasepool {
            guard let cb = queue.makeCommandBuffer() else {
                c.cbErrors += 1
                return
            }
            cb.label = "probe-\(name)-\(c.iterations)"
            guard let it = encodeIteration(cb, index: c.iterations) else {
                c.createNil += 1
                if c.createNil <= 5 { say("phase=\(name) iteration=\(c.iterations) a texture, view or buffer create returned nil") }
                return
            }
            let t0 = Date()
            cb.commit()
            cb.waitUntilCompleted()
            touch()
            let dt = Date().timeIntervalSince(t0)
            c.maxCbSeconds = max(c.maxCbSeconds, dt)
            let elapsed = String(format: "%.1f", Date().timeIntervalSince(phaseStart))
            if cb.status != .completed || cb.error != nil {
                c.cbErrors += 1
                c.consecutiveErrors += 1
                let text = "phase=\(name) iteration=\(c.iterations) phaseSeconds=\(elapsed) cbSeconds=\(String(format: "%.2f", dt)) status=\(cb.status.rawValue) error=\(String(describing: cb.error))"
                if c.firstError.isEmpty { c.firstError = "iteration=\(c.iterations) phaseSeconds=\(elapsed)" }
                if c.cbErrors <= 5 { say("CB-ERROR " + text) }
                if c.consecutiveErrors >= 3 { stoppedForHang = true }
            } else {
                c.consecutiveErrors = 0
                for (scenario, note) in check(it) {
                    c.bad[scenario, default: 0] += 1
                    if c.firstBad[scenario] == nil {
                        c.firstBad[scenario] = "iteration=\(c.iterations) phaseSeconds=\(elapsed)"
                        say("BAD phase=\(name) scenario=\(scenario) iteration=\(c.iterations) phaseSeconds=\(elapsed) \(note)")
                    } else if c.bad[scenario]! <= 3 {
                        say("BAD phase=\(name) scenario=\(scenario) iteration=\(c.iterations) phaseSeconds=\(elapsed) \(note)")
                    }
                }
            }
            c.iterations += 1
            if Date().timeIntervalSince(lastReport) >= 30 {
                lastReport = Date()
                say("progress phase=\(name) iterations=\(c.iterations) bad=\(c.bad) cbErrors=\(c.cbErrors) maxCbSeconds=\(String(format: "%.2f", c.maxCbSeconds))")
            }
        }
        if stoppedForHang { break }
    }
    let wall = String(format: "%.1f", Date().timeIntervalSince(phaseStart))
    let first = ["A", "E", "B", "C"].map { "first\($0)=\(c.firstBad[$0].map { "[\($0)]" } ?? "none")" }.joined(separator: " ")
    let line = "PROBE-SUMMARY phase=\(name) result=\(stoppedForHang ? "hung" : (c.cbErrors > 0 || c.bad.values.contains { $0 > 0 } ? "faults" : "clean")) iterations=\(c.iterations) wallSeconds=\(wall) badA=\(c.bad["A"]!) badE=\(c.bad["E"]!) badB=\(c.bad["B"]!) badC=\(c.bad["C"]!) cbErrors=\(c.cbErrors) createNil=\(c.createNil) maxCbSeconds=\(String(format: "%.2f", c.maxCbSeconds)) firstError=[\(c.firstError.isEmpty ? "none" : c.firstError)] \(first)"
    print(line)
    summaries.append(line)
}

func stormPhase(seconds: Double) {
    progressLock.lock(); currentPhase = "storm"; progressLock.unlock()
    touch()
    let threads = 8
    let lock = NSLock()
    var creates = 0
    var textureNil = 0
    var viewNil = 0
    let group = DispatchGroup()
    let phaseStart = Date()
    for _ in 0..<threads {
        group.enter()
        Thread.detachNewThread {
            while Date().timeIntervalSince(phaseStart) < seconds {
                autoreleasepool {
                    let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: 640, height: 480, mipmapped: false)
                    d.storageMode = .private
                    d.usage = [.renderTarget, .shaderRead, .pixelFormatView]
                    let t = device.makeTexture(descriptor: d)
                    let v = t?.makeTextureView(pixelFormat: .bgra8Unorm_srgb)
                    lock.lock()
                    creates += 1
                    if t == nil { textureNil += 1 }
                    if t != nil && v == nil { viewNil += 1 }
                    lock.unlock()
                }
            }
            group.leave()
        }
    }
    group.wait()
    touch()
    let line = "PROBE-SUMMARY phase=storm result=\(textureNil + viewNil > 0 ? "refusals" : "clean") threads=\(threads) creates=\(creates) textureNil=\(textureNil) viewNil=\(viewNil) wallSeconds=\(String(format: "%.1f", Date().timeIntervalSince(phaseStart)))"
    print(line)
    summaries.append(line)
}

msaaPhase("msaa1", seconds: msaaSeconds)
if !summaries.last!.contains("result=hung") {
    stormPhase(seconds: stormSeconds)
    msaaPhase("msaa2", seconds: msaaSeconds)
}
print("PROBE-DONE totalSeconds=\(now())")
for s in summaries { print(s) }
exit(0)
