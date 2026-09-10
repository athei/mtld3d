#!/usr/bin/env swift
// Reproduce Wine's sendEvent bypass and verify currentEvent at the existing observer phase.
import AppKit
import CoreFoundation

@objc(CursorRunLoopObservationProbe)
final class ProbeApplication: NSApplication {
    var consumed = 0
    override func sendEvent(_ event: NSEvent) { consumed += 1 }
}

let application = ProbeApplication.shared as! ProbeApplication
application.setActivationPolicy(.prohibited)
var localCount = 0
var observed = [Int]()
var previous: NSEvent?
let monitor = NSEvent.addLocalMonitorForEvents(matching: .mouseMoved) { event in
    localCount += 1
    return event
}!
let observer = CFRunLoopObserverCreateWithHandler(nil,
    CFRunLoopActivity.beforeWaiting.rawValue | CFRunLoopActivity.exit.rawValue, true, 0
) { _, _ in
    if let event = application.currentEvent, event.type == .mouseMoved, event !== previous {
        observed.append(event.eventNumber)
        previous = event
    }
}!
CFRunLoopAddObserver(CFRunLoopGetMain(), observer, .commonModes)
for number in 201...203 {
    let event = NSEvent.mouseEvent(with: .mouseMoved, location: NSPoint(x: number, y: 100),
        modifierFlags: [], timestamp: ProcessInfo.processInfo.systemUptime,
        windowNumber: 0, context: nil, eventNumber: number, clickCount: 0, pressure: 0)!
    application.postEvent(event, atStart: true)
    let received = application.nextEvent(matching: .mouseMoved,
        until: Date(timeIntervalSinceNow: 0.1), inMode: .default, dequeue: true)!
    application.sendEvent(received)
    CFRunLoopRunInMode(.defaultMode, 0.001, false)
}
for _ in 0..<10 { CFRunLoopRunInMode(.defaultMode, 0.001, false) }
precondition(application.consumed == 3 && localCount == 0)
precondition(observed == [201, 202, 203], "observer misses input or repeats idle history")
print("PASS: Wine-style consumed events=3, local monitor=0, run-loop observed=\(observed); idle deduplicated")
NSEvent.removeMonitor(monitor)
CFRunLoopRemoveObserver(CFRunLoopGetMain(), observer, .commonModes)
