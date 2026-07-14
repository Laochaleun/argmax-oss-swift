//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Foundation

public protocol ResultWriting {
    var outputDir: String { get }
    func write(result: TranscriptionResult, to file: String, options: [String: Any]?) -> Result<String, Error>
    func formatTime(seconds: Float, alwaysIncludeHours: Bool, decimalMarker: String) -> String
}

public enum TranscriptionEmissionBoundError: Error, Equatable, LocalizedError {
    case invalidAllowedCoordinateInterval(lowerBound: Float, upperBound: Float)
    case segmentCoordinateOutsideAllowedInterval(
        segmentIndex: Int,
        start: Float,
        end: Float,
        lowerBound: Float,
        upperBound: Float
    )
    case wordCoordinateOutsideAllowedInterval(
        segmentIndex: Int,
        wordIndex: Int,
        start: Float,
        end: Float,
        lowerBound: Float,
        upperBound: Float
    )

    public var errorDescription: String? {
        switch self {
            case let .invalidAllowedCoordinateInterval(lowerBound, upperBound):
                return "Invalid allowed coordinate interval [\(lowerBound), \(upperBound)]."
            case let .segmentCoordinateOutsideAllowedInterval(
                segmentIndex,
                start,
                end,
                lowerBound,
                upperBound
            ):
                return "Segment \(segmentIndex) coordinates [\(start), \(end)] are outside allowed interval [\(lowerBound), \(upperBound)]."
            case let .wordCoordinateOutsideAllowedInterval(
                segmentIndex,
                wordIndex,
                start,
                end,
                lowerBound,
                upperBound
            ):
                return "Word \(wordIndex) in segment \(segmentIndex) has coordinates [\(start), \(end)] outside allowed interval [\(lowerBound), \(upperBound)]."
        }
    }
}

public struct AllowedCoordinateInterval: Equatable, Sendable {
    public let lowerBound: Float
    public let upperBound: Float

    public init(lowerBound: Float, upperBound: Float) throws {
        guard lowerBound.isFinite,
              upperBound.isFinite,
              lowerBound <= upperBound
        else {
            throw TranscriptionEmissionBoundError.invalidAllowedCoordinateInterval(
                lowerBound: lowerBound,
                upperBound: upperBound
            )
        }
        self.lowerBound = lowerBound
        self.upperBound = upperBound
    }

    public func validate(result: TranscriptionResult) throws {
        for (segmentIndex, segment) in result.segments.enumerated() {
            guard contains(segment.start), contains(segment.end) else {
                throw TranscriptionEmissionBoundError.segmentCoordinateOutsideAllowedInterval(
                    segmentIndex: segmentIndex,
                    start: segment.start,
                    end: segment.end,
                    lowerBound: lowerBound,
                    upperBound: upperBound
                )
            }
            for (wordIndex, word) in (segment.words ?? []).enumerated() {
                guard contains(word.start), contains(word.end) else {
                    throw TranscriptionEmissionBoundError.wordCoordinateOutsideAllowedInterval(
                        segmentIndex: segmentIndex,
                        wordIndex: wordIndex,
                        start: word.start,
                        end: word.end,
                        lowerBound: lowerBound,
                        upperBound: upperBound
                    )
                }
            }
        }
    }

    private func contains(_ value: Float) -> Bool {
        value.isFinite && value >= lowerBound && value <= upperBound
    }
}

public struct TranscriptionReportPaths: Equatable, Sendable {
    public let srt: String
    public let json: String
}

public struct BoundedTranscriptionReportWriter {
    public let outputDir: String

    public init(outputDir: String) {
        self.outputDir = outputDir
    }

    public func write(
        result: TranscriptionResult,
        to file: String,
        allowedCoordinateInterval: AllowedCoordinateInterval
    ) -> Result<TranscriptionReportPaths, Error> {
        do {
            try allowedCoordinateInterval.validate(result: result)
        } catch {
            return .failure(error)
        }

        let savedSrtReport = WriteSRT(outputDir: outputDir).write(result: result, to: file)
        let srtPath: String
        switch savedSrtReport {
            case let .success(path):
                srtPath = path
            case let .failure(error):
                return .failure(error)
        }

        let savedJsonReport = WriteJSON(outputDir: outputDir).write(result: result, to: file)
        switch savedJsonReport {
            case let .success(path):
                return .success(TranscriptionReportPaths(srt: srtPath, json: path))
            case let .failure(error):
                return .failure(error)
        }
    }
}

public extension ResultWriting {
    /// Format a time value as a string
    func formatTime(seconds: Float, alwaysIncludeHours: Bool, decimalMarker: String) -> String {
        let hrs = Int(seconds / 3600)
        let mins = Int((seconds.truncatingRemainder(dividingBy: 3600)) / 60)
        let secs = Int(seconds.truncatingRemainder(dividingBy: 60))
        let msec = Int((seconds - floor(seconds)) * 1000)

        if alwaysIncludeHours || hrs > 0 {
            return String(format: "%02d:%02d:%02d\(decimalMarker)%03d", hrs, mins, secs, msec)
        } else {
            return String(format: "%02d:%02d\(decimalMarker)%03d", mins, secs, msec)
        }
    }

    func formatSegment(index: Int, start: Float, end: Float, text: String) -> String {
        let startFormatted = formatTime(seconds: Float(start), alwaysIncludeHours: true, decimalMarker: ",")
        let endFormatted = formatTime(seconds: Float(end), alwaysIncludeHours: true, decimalMarker: ",")
        return "\(index)\n\(startFormatted) --> \(endFormatted)\n\(text)\n\n"
    }

    func formatTiming(start: Float, end: Float, text: String) -> String {
        let startFormatted = formatTime(seconds: Float(start), alwaysIncludeHours: false, decimalMarker: ".")
        let endFormatted = formatTime(seconds: Float(end), alwaysIncludeHours: false, decimalMarker: ".")
        return "\(startFormatted) --> \(endFormatted)\n\(text)\n\n"
    }
}

open class WriteJSON: ResultWriting {
    public let outputDir: String

    public init(outputDir: String) {
        self.outputDir = outputDir
    }

    /// Write a transcription result to a JSON file
    /// - Parameters:
    ///   - result: Completed transcription result
    ///   - file: Name of the file to write, without the extension
    ///   - options: Not used
    /// - Returns: The URL of the written file, or a error if the write failed
    public func write(result: TranscriptionResult, to file: String, options: [String: Any]? = nil) -> Result<String, Error> {
        let reportPathURL = URL(fileURLWithPath: outputDir)
        let reportURL = reportPathURL.appendingPathComponent("\(file).json")
        let jsonEncoder = JSONEncoder()
        jsonEncoder.outputFormatting = .prettyPrinted
        do {
            let reportJson = try jsonEncoder.encode(result)
            try reportJson.write(to: reportURL, options: .atomic)
        } catch {
            return .failure(error)
        }

        return .success(reportURL.absoluteString)
    }
}

open class WriteSRT: ResultWriting {
    public let outputDir: String

    public init(outputDir: String) {
        self.outputDir = outputDir
    }

    public func write(result: TranscriptionResult, to file: String, options: [String: Any]? = nil) -> Result<String, Error> {
        let outputPathURL = URL(fileURLWithPath: outputDir)
        let outputFileURL = outputPathURL.appendingPathComponent("\(file).srt")

        do {
            var srtContent = ""
            var index = 1
            for segment in result.segments {
                if let wordTimings = segment.words, !wordTimings.isEmpty {
                    for wordTiming in wordTimings {
                        srtContent += formatSegment(index: index, start: wordTiming.start, end: wordTiming.end, text: wordTiming.word)
                        index += 1
                    }
                } else {
                    // Use segment timing if word timings are not available
                    srtContent += formatSegment(index: index, start: segment.start, end: segment.end, text: segment.text)
                    index += 1
                }
            }

            try srtContent.write(to: outputFileURL, atomically: true, encoding: .utf8)
            return .success(outputFileURL.absoluteString)
        } catch {
            return .failure(error)
        }
    }
}

open class WriteVTT: ResultWriting {
    public let outputDir: String

    public init(outputDir: String) {
        self.outputDir = outputDir
    }

    public func write(result: TranscriptionResult, to file: String, options: [String: Any]? = nil) -> Result<String, Error> {
        let outputPathURL = URL(fileURLWithPath: outputDir)
        let outputFileURL = outputPathURL.appendingPathComponent("\(file).vtt")

        do {
            var vttContent = "WEBVTT\n\n"
            for segment in result.segments {
                if let wordTimings = segment.words, !wordTimings.isEmpty {
                    for wordTiming in wordTimings {
                        vttContent += formatTiming(start: wordTiming.start, end: wordTiming.end, text: wordTiming.word)
                    }
                } else {
                    // Use segment timing if word timings are not available
                    vttContent += formatTiming(start: segment.start, end: segment.end, text: segment.text)
                }
            }

            try vttContent.write(to: outputFileURL, atomically: true, encoding: .utf8)
            return .success(outputFileURL.absoluteString)
        } catch {
            return .failure(error)
        }
    }
}
