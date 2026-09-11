#!/usr/bin/env swift
// Verify cursor visibility when input changes twice before the transaction commits.
// Usage: swift scripts/cursor_transaction_probe.swift <existing-output-directory>
import AppKit
import Metal
import QuartzCore

precondition(CommandLine.arguments.count == 2, "pass an existing output directory")
let app = NSApplication.shared
app.setActivationPolicy(.prohibited)
app.finishLaunching()
let device = MTLCreateSystemDefaultDevice()!
let window = NSWindow(
  contentRect: NSRect(x: 100, y: 100, width: 300, height: 300), styleMask: .borderless,
  backing: .buffered, defer: false)
window.isReleasedWhenClosed = false
window.isOpaque = false
window.backgroundColor = .clear
window.ignoresMouseEvents = true
window.level = NSWindow.Level(rawValue: 28)
window.collectionBehavior = [.canJoinAllSpaces, .transient, .ignoresCycle, .fullScreenAuxiliary]
let view = NSView(frame: NSRect(x: 0, y: 0, width: 300, height: 300))
let host = CALayer()
let layer = CAMetalLayer()
layer.device = device
layer.pixelFormat = .rgba16Float
layer.colorspace = CGColorSpace(name: CGColorSpace.extendedLinearDisplayP3)
layer.wantsExtendedDynamicRangeContent = true
layer.isOpaque = false
layer.developerHUDProperties = ["mode": "disabled"]
layer.actions = [
  "position": NSNull(), "bounds": NSNull(), "contents": NSNull(), "contentsScale": NSNull(),
]
layer.anchorPoint = CGPoint(x: 0, y: 0)
layer.frame = CGRect(x: 50, y: 50, width: 32, height: 32)
layer.contentsScale = 2
host.addSublayer(layer)
view.layer = host
view.wantsLayer = true
window.contentView = view
window.orderFrontRegardless()
func submit(_ alpha: Double) {
  let value = Float16(alpha).bitPattern
  let pixels: [UInt16] = [value, 0, 0, value]
  let data = pixels.withUnsafeBytes { Data($0) } as CFData
  let image = CGImage(
    width: 1, height: 1, bitsPerComponent: 16, bitsPerPixel: 64, bytesPerRow: 8,
    space: layer.colorspace!,
    bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
      .union(.byteOrder16Little).union(.floatComponents),
    provider: CGDataProvider(data: data)!, decode: nil, shouldInterpolate: false,
    intent: .defaultIntent)!
  layer.contents = image
}
var pending: Double?
func draw(_ alpha: Double) { pending = alpha }
let observer = CFRunLoopObserverCreateWithHandler(
  nil, CFRunLoopActivity.beforeWaiting.rawValue | CFRunLoopActivity.exit.rawValue, true, 0
) { _, _ in
  if let alpha = pending {
    pending = nil
    submit(alpha)
  }
}
CFRunLoopAddObserver(CFRunLoopGetMain(), observer, .commonModes)
func drain() { RunLoop.current.run(until: Date().addingTimeInterval(1)) }
func capture(_ name: String) {
  let p = Process()
  p.executableURL = URL(fileURLWithPath: "/usr/sbin/screencapture")
  p.arguments = [
    "-x", "-l", String(window.windowNumber), CommandLine.arguments[1] + "/" + name + ".png",
  ]
  try! p.run()
  p.waitUntilExit()
  precondition(p.terminationStatus == 0, "window capture failed")
  let bitmap = NSBitmapImageRep(
    data: try! Data(
      contentsOf:
        URL(fileURLWithPath: CommandLine.arguments[1] + "/" + name + ".png")))!
  let alpha = bitmap.colorAt(x: bitmap.pixelsWide / 2, y: bitmap.pixelsHigh / 2)!.alphaComponent
  let expected = name == "present-show-clear" ? 0.0 : 1.0
  precondition(abs(alpha - expected) < 0.01, "\(name): alpha=\(alpha), expected=\(expected)")
  print("PASS: \(name) alpha=\(alpha)")
}
drain()
draw(1)
drain()
capture("present-seed")
draw(0)
draw(1)
drain()
capture("present-clear-show")
draw(1)
draw(0)
drain()
capture("present-show-clear")
draw(1)
draw(0)
draw(1)
drain()
capture("present-show-clear-show")
CFRunLoopRemoveObserver(CFRunLoopGetMain(), observer, .commonModes)
window.orderOut(nil)
