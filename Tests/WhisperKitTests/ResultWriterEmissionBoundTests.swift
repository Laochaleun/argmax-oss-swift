//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Foundation
@testable import WhisperKit
import XCTest

final class ResultWriterEmissionBoundTests: XCTestCase {
    func testPreEmissionBoundaryAcceptsClosedCoordinatesWithoutChangingThem() throws {
        let interval = try AllowedCoordinateInterval(lowerBound: 0, upperBound: 10)
        let segments = [
            TranscriptionSegment(
                start: 0,
                end: 10,
                text: "closed boundary",
                words: [
                    WordTiming(
                        word: "boundary",
                        tokens: [1],
                        start: 10,
                        end: 10,
                        probability: 1
                    ),
                ]
            ),
        ]
        let originalSegments = segments

        try interval.validateBeforeEmission(
            segments: segments,
            segmentIndexOffset: 0,
            windowSeek: 0,
            windowSegmentSize: 160_000,
            sampleRate: 16_000
        )

        XCTAssertEqual(segments, originalSegments)
    }

    func testPreEmissionBoundaryUsesProducerArithmeticAtNonzeroSeek() throws {
        let windowSeek = 68_160
        let windowSegmentSize = 480_000
        let sampleRate = 16_000
        let start = Float(windowSeek) / Float(sampleRate)
        let end = start + Float(windowSegmentSize) / Float(sampleRate)
        let interval = try AllowedCoordinateInterval(lowerBound: start, upperBound: end)
        let segments = [
            TranscriptionSegment(start: start, end: end, text: "exact producer boundary"),
        ]

        try interval.validateBeforeEmission(
            segments: segments,
            segmentIndexOffset: 0,
            windowSeek: windowSeek,
            windowSegmentSize: windowSegmentSize,
            sampleRate: sampleRate
        )
    }

    func testAllowedCoordinateIntervalDecodingRetainsValidation() throws {
        let reversed = Data(#"{"lowerBound":10,"upperBound":0}"#.utf8)
        XCTAssertThrowsError(
            try JSONDecoder().decode(AllowedCoordinateInterval.self, from: reversed)
        )
    }

    func testPreEmissionBoundaryRejectsSegmentAndWordViolations() throws {
        let interval = try AllowedCoordinateInterval(lowerBound: 2, upperBound: 8)
        let cases = [
            TranscriptionSegment(start: 1.99, end: 3, text: "segment below"),
            TranscriptionSegment(start: 3, end: 8.01, text: "segment above"),
            TranscriptionSegment(
                start: 2,
                end: 8,
                text: "word below",
                words: [
                    WordTiming(
                        word: "below",
                        tokens: [1],
                        start: 1.99,
                        end: 3,
                        probability: 1
                    ),
                ]
            ),
            TranscriptionSegment(
                start: 2,
                end: 8,
                text: "word above",
                words: [
                    WordTiming(
                        word: "above",
                        tokens: [1],
                        start: 3,
                        end: 8.01,
                        probability: 1
                    ),
                ]
            ),
        ]

        for segment in cases {
            XCTAssertThrowsError(
                try interval.validateBeforeEmission(
                    segments: [segment],
                    segmentIndexOffset: 7,
                    windowSeek: 0,
                    windowSegmentSize: 160_000,
                    sampleRate: 16_000
                )
            )
        }
    }

    func testPreEmissionBoundaryRejectsPaddedDecodeAxisOutsideRealWindow() throws {
        let interval = try AllowedCoordinateInterval(lowerBound: 0, upperBound: 30)
        let segment = TranscriptionSegment(
            start: 9,
            end: 10.02,
            text: "padded decode axis",
            words: [
                WordTiming(
                    word: "padded",
                    tokens: [1],
                    start: 9,
                    end: 10,
                    probability: 1
                ),
            ]
        )

        XCTAssertThrowsError(
            try interval.validateBeforeEmission(
                segments: [segment],
                segmentIndexOffset: 0,
                windowSeek: 0,
                windowSegmentSize: 160_000,
                sampleRate: 16_000
            )
        ) { error in
            XCTAssertEqual(
                error as? TranscriptionEmissionBoundError,
                .segmentCoordinateOutsideAllowedInterval(
                    segmentIndex: 0,
                    start: 9,
                    end: 10.02,
                    lowerBound: 0,
                    upperBound: 10
                )
            )
        }
    }

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

    func testDiagnosticNonFiniteFloatsEncodeDeterministicallyWithoutChangingSRT() throws {
        let outputDirectory = try makeOutputDirectory()
        defer { try? FileManager.default.removeItem(at: outputDirectory) }
        let result = makeResult(
            segments: [
                TranscriptionSegment(
                    id: 7,
                    start: 1.25,
                    end: 2.5,
                    text: "diagnostic values",
                    tokenLogProbs: [[42: .infinity]],
                    temperature: .infinity,
                    avgLogprob: -.infinity,
                    compressionRatio: 1.5,
                    noSpeechProb: .nan,
                    words: [
                        WordTiming(
                            word: "diagnostic",
                            tokens: [42],
                            start: 1.25,
                            end: 2.5,
                            probability: .nan
                        ),
                    ]
                ),
            ]
        )

        let writeResult = BoundedTranscriptionReportWriter(
            outputDir: outputDirectory.path
        ).write(
            result: result,
            to: "diagnostics",
            allowedCoordinateInterval: try AllowedCoordinateInterval(
                lowerBound: 0,
                upperBound: 3
            )
        )
        guard case let .success(paths) = writeResult else {
            return XCTFail("Expected diagnostic report publication to succeed: \(writeResult)")
        }

        let srt = try String(contentsOfFile: try filePath(paths.srt), encoding: .utf8)
        XCTAssertEqual(
            srt,
            "1\n00:00:01,250 --> 00:00:02,500\ndiagnostic\n\n"
        )
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(
                with: Data(contentsOf: URL(fileURLWithPath: try filePath(paths.json)))
            ) as? [String: Any]
        )
        let segment = try XCTUnwrap((object["segments"] as? [[String: Any]])?.first)
        XCTAssertEqual(segment["text"] as? String, "diagnostic values")
        XCTAssertEqual(segment["start"] as? Double, 1.25)
        XCTAssertEqual(segment["end"] as? Double, 2.5)
        XCTAssertEqual(segment["compressionRatio"] as? Double, 1.5)
        XCTAssertEqual(segment["temperature"] as? String, "Infinity")
        XCTAssertEqual(segment["avgLogprob"] as? String, "-Infinity")
        XCTAssertEqual(segment["noSpeechProb"] as? String, "NaN")
        let tokenLogProbs = try XCTUnwrap(segment["tokenLogProbs"] as? [[String: Any]])
        XCTAssertEqual(tokenLogProbs[0]["42"] as? String, "Infinity")
        let word = try XCTUnwrap((segment["words"] as? [[String: Any]])?.first)
        XCTAssertEqual(word["word"] as? String, "diagnostic")
        XCTAssertEqual(word["start"] as? Double, 1.25)
        XCTAssertEqual(word["end"] as? Double, 2.5)
        XCTAssertEqual(word["probability"] as? String, "NaN")
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
