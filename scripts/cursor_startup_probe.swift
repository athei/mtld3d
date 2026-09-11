#!/usr/bin/env swift

// Capture native and software cursors during a stationary application startup.
// Usage: swift scripts/cursor_startup_probe.swift <app> <output-directory> <x> <y>
// Close the application first. Coordinates are global screen points.
// Screen recording and input posting access must already be available.
import AppKit
import CoreGraphics

let application = NSApplication.shared
application.setActivationPolicy(.prohibited)
application.finishLaunching()
precondition(CommandLine.arguments.count == 5, "pass app, output directory, x and y")
let appURL = URL(fileURLWithPath: CommandLine.arguments[1]).standardizedFileURL
let output = CommandLine.arguments[2]
let parked = CGPoint(x: Double(CommandLine.arguments[3])!, y: Double(CommandLine.arguments[4])!)
precondition(parked.x >= 50 && parked.y >= 50, "capture region must be on screen")
precondition(FileManager.default.fileExists(atPath: output), "output directory must exist")
precondition(
  !NSWorkspace.shared.runningApplications.contains {
    $0.bundleURL?.standardizedFileURL == appURL
  }, "close the application before running the probe")
let region = "\(Int(parked.x) - 50),\(Int(parked.y) - 50),180,180"
func command(_ executable: String, _ args: [String]) {
  let p = Process()
  p.executableURL = URL(fileURLWithPath: executable)
  p.arguments = args
  try! p.run()
  p.waitUntilExit()
  precondition(p.terminationStatus == 0)
}
func capture(_ name: String, full: Bool = false) {
  var args = ["-x", "-C"]
  if !full { args += ["-R", region] }
  args.append(output + "/" + name + ".png")
  command("/usr/sbin/screencapture", args)
  print(
    "CAPTURE \(name) frontmost=\(NSWorkspace.shared.frontmostApplication?.bundleIdentifier ?? "none") pointer=\(CGEvent(source: nil)!.location) time=\(Date().timeIntervalSince1970)"
  )
}
func move(_ point: CGPoint) {
  CGEvent(
    mouseEventSource: nil, mouseType: .mouseMoved, mouseCursorPosition: point, mouseButton: .left)!
    .post(tap: .cghidEventTap)
  Thread.sleep(forTimeInterval: 0.5)
}
func tab() {
  for (key, down, flags): (CGKeyCode, Bool, CGEventFlags) in [
    (55, true, .maskCommand), (48, true, .maskCommand),
    (48, false, .maskCommand), (55, false, [])
  ] {
    let event = CGEvent(keyboardEventSource: nil, virtualKey: key, keyDown: down)!
    event.flags = flags
    event.post(tap: .cghidEventTap)
    Thread.sleep(forTimeInterval: 0.05)
  }
  Thread.sleep(forTimeInterval: 1)
}
func tabRoundTrip(_ name: String) {
  tab()
  capture("away-" + name)
  tab()
  capture("return-" + name)
}
precondition(CGPreflightPostEventAccess())
let finder = NSRunningApplication.runningApplications(withBundleIdentifier: "com.apple.finder")
  .first!
precondition(finder.activate(options: []))
Thread.sleep(forTimeInterval: 1)
move(parked)
capture("before-launch")
command("/usr/bin/open", ["-n", appURL.path])
let start = Date()
for second in [1.0, 3.0, 5.0, 10.0, 20.0] {
  let delay = second - Date().timeIntervalSince(start)
  if delay > 0 { Thread.sleep(forTimeInterval: delay) }
  precondition(
    CGEvent(source: nil)!.location == parked, "external movement invalidated the stationary probe")
  capture("stationary-\(Int(second))")
}
capture("stationary-full", full: true)
command(
  "/usr/sbin/screencapture", ["-x", "-R", region, output + "/stationary-no-system-cursor.png"])
for cycle in 1...2 { tabRoundTrip("before-move-\(cycle)") }
move(CGPoint(x: parked.x + 10, y: parked.y))
capture("after-first-move")
Thread.sleep(forTimeInterval: 2)
capture("after-move-settled")

for cycle in 1...3 { tabRoundTrip("after-move-\(cycle)") }
