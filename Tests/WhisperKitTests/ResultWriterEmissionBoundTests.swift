//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Foundation
@testable import WhisperKit
import XCTest

final class ResultWriterEmissionBoundTests: XCTestCase {
    func testEmissionBoundAcceptsClosedBoundariesAndPreservesCoordinates() throws {
        let outputDirectory = try makeOutputDirectory()
        defer { try? FileManager.default.removeItem(at: outputDirectory) }

        let interval = try AllowedCoordinateInterval(lowerBound: 0, upperBound: 10)
        let result = makeResult(
            segments: [
                TranscriptionSegment(
                    id: 1,
                    start: 0,
                    end: 10,
                    text: "full interval",
                    words: [
                        WordTiming(
                            word: "full",
                            tokens: [1],
                            start: 0,
                            end: 10,
                            probability: 1
                        ),
                    ]
                ),
                TranscriptionSegment(
                    id: 2,
                    start: 0,
                    end: 0,
                    text: "zero duration",
                    words: [
                        WordTiming(
                            word: "boundary",
                            tokens: [2],
                            start: 10,
                            end: 10,
                            probability: 1
                        ),
                    ]
                ),
            ]
        )

        let writeResult = BoundedTranscriptionReportWriter(
            outputDir: outputDirectory.path
        ).write(
            result: result,
            to: "accepted",
            allowedCoordinateInterval: interval
        )

        guard case let .success(paths) = writeResult else {
            return XCTFail("Expected bounded report publication to succeed: \(writeResult)")
        }
        XCTAssertTrue(FileManager.default.fileExists(atPath: try filePath(paths.srt)))
        let jsonPath = try filePath(paths.json)
        XCTAssertTrue(FileManager.default.fileExists(atPath: jsonPath))

        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: jsonPath)))
                as? [String: Any]
        )
        let segments = try XCTUnwrap(object["segments"] as? [[String: Any]])
        XCTAssertEqual(segments[0]["start"] as? Double, 0)
        XCTAssertEqual(segments[0]["end"] as? Double, 10)
        let words = try XCTUnwrap(segments[1]["words"] as? [[String: Any]])
        XCTAssertEqual(words[0]["start"] as? Double, 10)
        XCTAssertEqual(words[0]["end"] as? Double, 10)
    }

    func testEmissionBoundRejectsEverySegmentAndWordBoundaryViolationBeforeAnyReport() throws {
        let interval = try AllowedCoordinateInterval(lowerBound: 0, upperBound: 10)
        let cases: [(name: String, segment: TranscriptionSegment)] = [
            (
                "segment-below",
                TranscriptionSegment(start: -0.001, end: 1, text: "segment below")
            ),
            (
                "segment-above",
                TranscriptionSegment(start: 1, end: 10.001, text: "segment above")
            ),
            (
                "word-below",
                TranscriptionSegment(
                    start: 0,
                    end: 10,
                    text: "word below",
                    words: [
                        WordTiming(
                            word: "below",
                            tokens: [1],
                            start: -0.001,
                            end: 1,
                            probability: 1
                        ),
                    ]
                )
            ),
            (
                "word-above",
                TranscriptionSegment(
                    start: 0,
                    end: 10,
                    text: "word above",
                    words: [
                        WordTiming(
                            word: "above",
                            tokens: [1],
                            start: 1,
                            end: 10.001,
                            probability: 1
                        ),
                    ]
                )
            ),
        ]

        for testCase in cases {
            let outputDirectory = try makeOutputDirectory()
            defer { try? FileManager.default.removeItem(at: outputDirectory) }
            let writeResult = BoundedTranscriptionReportWriter(
                outputDir: outputDirectory.path
            ).write(
                result: makeResult(segments: [testCase.segment]),
                to: testCase.name,
                allowedCoordinateInterval: interval
            )

            guard case .failure = writeResult else {
                XCTFail("Expected \(testCase.name) to fail")
                continue
            }
            XCTAssertFalse(
                FileManager.default.fileExists(
                    atPath: outputDirectory.appendingPathComponent("\(testCase.name).srt").path
                )
            )
            XCTAssertFalse(
                FileManager.default.fileExists(
                    atPath: outputDirectory.appendingPathComponent("\(testCase.name).json").path
                )
            )
        }
    }

    func testEmissionBoundRejectsNonFiniteCoordinatesAndInvalidIntervals() throws {
        XCTAssertThrowsError(
            try AllowedCoordinateInterval(lowerBound: .nan, upperBound: 10)
        )
        XCTAssertThrowsError(
            try AllowedCoordinateInterval(lowerBound: 10, upperBound: 0)
        )

        let interval = try AllowedCoordinateInterval(lowerBound: 0, upperBound: 10)
        let outputDirectory = try makeOutputDirectory()
        defer { try? FileManager.default.removeItem(at: outputDirectory) }
        let result = makeResult(
            segments: [
                TranscriptionSegment(start: 0, end: .infinity, text: "non finite"),
            ]
        )

        guard case .failure = BoundedTranscriptionReportWriter(
            outputDir: outputDirectory.path
        ).write(
            result: result,
            to: "non-finite",
            allowedCoordinateInterval: interval
        ) else {
            return XCTFail("Expected non-finite coordinate rejection")
        }
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: outputDirectory.appendingPathComponent("non-finite.srt").path
            )
        )
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: outputDirectory.appendingPathComponent("non-finite.json").path
            )
        )
    }

    private func makeResult(segments: [TranscriptionSegment]) -> TranscriptionResult {
        TranscriptionResult(
            text: segments.map(\.text).joined(separator: " "),
            segments: segments,
            language: "en",
            timings: TranscriptionTimings()
        )
    }

    private func makeOutputDirectory() throws -> URL {
        let outputDirectory = FileManager.default.temporaryDirectory
            .appendingPathComponent("whisperkit-emission-bound-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: outputDirectory,
            withIntermediateDirectories: true
        )
        return outputDirectory
    }

    private func filePath(_ absoluteString: String) throws -> String {
        try XCTUnwrap(URL(string: absoluteString)?.path)
    }
}
