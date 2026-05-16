import AppKit
import Foundation
import Vision

if CommandLine.arguments.count < 2 {
    FileHandle.standardOutput.write(Data("[]".utf8))
    exit(2)
}

let imagePath = CommandLine.arguments[1]
guard let image = NSImage(contentsOfFile: imagePath) else {
    FileHandle.standardOutput.write(Data("[]".utf8))
    exit(0)
}

var rect = CGRect(origin: .zero, size: image.size)
guard let cgImage = image.cgImage(forProposedRect: &rect, context: nil, hints: nil) else {
    FileHandle.standardOutput.write(Data("[]".utf8))
    exit(0)
}

let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.usesLanguageCorrection = true
request.recognitionLanguages = ["en-US"]

let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
do {
    try handler.perform([request])
} catch {
    FileHandle.standardOutput.write(Data("[]".utf8))
    exit(0)
}

let rows: [[String: Any]] = (request.results ?? []).compactMap { observation in
    guard let top = observation.topCandidates(1).first else {
        return nil
    }
    let box = observation.boundingBox
    return [
        "text": top.string,
        "confidence": top.confidence,
        "box": [box.origin.x, box.origin.y, box.size.width, box.size.height],
    ]
}

if let data = try? JSONSerialization.data(withJSONObject: rows, options: []) {
    FileHandle.standardOutput.write(data)
} else {
    FileHandle.standardOutput.write(Data("[]".utf8))
}
