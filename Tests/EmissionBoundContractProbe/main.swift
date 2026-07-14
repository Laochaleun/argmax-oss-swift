//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Foundation
import WhisperKit

@main
struct EmissionBoundContractProbe {
    static func main() throws {
        try provePreEmissionBoundary()
        try proveAtomicMultiChunkBoundary()
        try proveClosedBoundaryAcceptance()
        try proveAtomicBoundaryRejection()
        try proveInvalidCoordinateRejection()
        print("EMISSION_BOUND_CONTRACT_OK")
    }

    private static func provePreEmissionBoundary() throws {
        let interval = try AllowedCoordinateInterval(lowerBound: 0, upperBound: 10)
        let validSegment = TranscriptionSegment(
            start: 0,
            end: 10,
            text: "pre-emission boundary",
            words: [
                WordTiming(
                    word: "boundary",
                    tokens: [1],
                    start: 10,
                    end: 10,
                    probability: 1
                ),
            ]
        )
        try interval.validateBeforeEmission(
            segments: [validSegment],
            segmentIndexOffset: 0,
            windowSeek: 0,
            windowSegmentSize: 160_000,
            sampleRate: 16_000
        )

        let invalidSegments = [
            TranscriptionSegment(start: -0.001, end: 1, text: "segment below"),
            TranscriptionSegment(start: 1, end: 10.001, text: "segment above"),
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
            ),
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
            ),
        ]
        for segment in invalidSegments {
            do {
                try interval.validateBeforeEmission(
                    segments: [segment],
                    segmentIndexOffset: 0,
                    windowSeek: 0,
                    windowSegmentSize: 160_000,
                    sampleRate: 16_000
                )
                throw ProbeError("out-of-bound coordinates crossed pre-emission boundary")
            } catch is TranscriptionEmissionBoundError {
                // Expected.
            }
        }

        do {
            try AllowedCoordinateInterval(lowerBound: 0, upperBound: 30)
                .validateBeforeEmission(
                    segments: [
                        TranscriptionSegment(
                            start: 9,
                            end: 10.02,
                            text: "padded decode axis"
                        ),
                    ],
                    segmentIndexOffset: 0,
                    windowSeek: 0,
                    windowSegmentSize: 160_000,
                    sampleRate: 16_000
                )
            throw ProbeError("padded decode-axis coordinate crossed real window")
        } catch is TranscriptionEmissionBoundError {
            // Expected.
        }

        let nonzeroSeek = 68_160
        let nonzeroWindowSize = 480_000
        let nonzeroStart = Float(nonzeroSeek) / Float(16_000)
        let nonzeroEnd = nonzeroStart + Float(nonzeroWindowSize) / Float(16_000)
        try AllowedCoordinateInterval(lowerBound: nonzeroStart, upperBound: nonzeroEnd)
            .validateBeforeEmission(
                segments: [
                    TranscriptionSegment(
                        start: nonzeroStart,
                        end: nonzeroEnd,
                        text: "exact nonzero boundary"
                    ),
                ],
                segmentIndexOffset: 0,
                windowSeek: nonzeroSeek,
                windowSegmentSize: nonzeroWindowSize,
                sampleRate: 16_000
            )

        do {
            _ = try JSONDecoder().decode(
                AllowedCoordinateInterval.self,
                from: Data(#"{"lowerBound":10,"upperBound":0}"#.utf8)
            )
            throw ProbeError("invalid encoded interval bypassed validation")
        } catch is TranscriptionEmissionBoundError {
            // Expected.
        }
    }

    private static func proveAtomicMultiChunkBoundary() throws {
        let chunker = ProbeChunker()
        let chunks = [AudioChunk(seekOffsetIndex: 128_000, audioSamples: [0])]
        let allowedInterval = try AllowedCoordinateInterval(lowerBound: 0, upperBound: 10)

        do {
            _ = try chunker.updateSeekOffsetsForResults(
                chunkedResults: [
                    .failure(
                        TranscriptionEmissionBoundError.segmentCoordinateOutsideAllowedInterval(
                            segmentIndex: 0,
                            start: 0,
                            end: 10.001,
                            lowerBound: 0,
                            upperBound: 10
                        )
                    ),
                ],
                audioChunks: chunks,
                allowedCoordinateInterval: allowedInterval
            )
            throw ProbeError("failed chunk was partially retained")
        } catch is TranscriptionEmissionBoundError {
            // Expected.
        }

        do {
            _ = try chunker.updateSeekOffsetsForResults(
                chunkedResults: [
                    .success([
                        makeResult(
                            segment: TranscriptionSegment(
                                start: 1,
                                end: 3,
                                text: "offset outside call"
                            )
                        ),
                    ]),
                ],
                audioChunks: chunks,
                allowedCoordinateInterval: allowedInterval
            )
            throw ProbeError("offset chunk escaped call-local boundary")
        } catch is TranscriptionEmissionBoundError {
            // Expected.
        }

        let valid = try chunker.updateSeekOffsetsForResults(
            chunkedResults: [
                .success([
                    makeResult(
                        segment: TranscriptionSegment(
                            start: 1,
                            end: 2,
                            text: "offset valid"
                        )
                    ),
                ]),
            ],
            audioChunks: chunks,
            allowedCoordinateInterval: allowedInterval
        )
        guard valid.first?.segments.first?.start == 9,
              valid.first?.segments.first?.end == 10
        else {
            throw ProbeError("valid multi-chunk coordinates changed unexpectedly")
        }

        let batchedChunks = (0..<17).map { index in
            AudioChunk(
                seekOffsetIndex: index * 16_000,
                audioSamples: Array(repeating: 0, count: 16_000 - index)
            )
        }
        let batchedResults: [Result<[TranscriptionResult], Swift.Error>] = (0..<17).map { _ in
            .success([
                makeResult(
                    segment: TranscriptionSegment(
                        start: 0,
                        end: 0.5,
                        text: "valid batched chunk"
                    )
                ),
            ])
        }
        let batchedValid = try chunker.updateSeekOffsetsForResults(
            chunkedResults: batchedResults,
            audioChunks: batchedChunks,
            allowedCoordinateInterval: try AllowedCoordinateInterval(
                lowerBound: 0,
                upperBound: 17
            )
        )
        guard batchedValid.count == 17,
              batchedValid.last?.segments.first?.start == 16,
              batchedValid.last?.segments.first?.end == 16.5
        else {
            throw ProbeError("valid coordinates changed across worker-sized batches")
        }
    }

    private static func proveClosedBoundaryAcceptance() throws {
        let directory = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let interval = try AllowedCoordinateInterval(lowerBound: 0, upperBound: 10)
        let result = makeResult(
            segment: TranscriptionSegment(
                start: 0,
                end: 10,
                text: "boundary",
                words: [
                    WordTiming(
                        word: "boundary",
                        tokens: [1],
                        start: 10,
                        end: 10,
                        probability: 1
                    ),
                ]
            )
        )
        let writeResult = BoundedTranscriptionReportWriter(
            outputDir: directory.path
        ).write(
            result: result,
            to: "accepted",
            allowedCoordinateInterval: interval
        )
        guard case let .success(paths) = writeResult,
              let srtPath = URL(string: paths.srt)?.path,
              let jsonPath = URL(string: paths.json)?.path,
              FileManager.default.fileExists(atPath: srtPath),
              FileManager.default.fileExists(atPath: jsonPath)
        else {
            throw ProbeError("closed boundary report was not published")
        }
        let report = try JSONSerialization.jsonObject(
            with: Data(contentsOf: URL(fileURLWithPath: jsonPath))
        ) as? [String: Any]
        let segments = report?["segments"] as? [[String: Any]]
        let words = segments?.first?["words"] as? [[String: Any]]
        guard segments?.first?["start"] as? Double == 0,
              segments?.first?["end"] as? Double == 10,
              words?.first?["start"] as? Double == 10,
              words?.first?["end"] as? Double == 10
        else {
            throw ProbeError("accepted coordinates changed during publication")
        }
    }

    private static func proveAtomicBoundaryRejection() throws {
        let interval = try AllowedCoordinateInterval(lowerBound: 0, upperBound: 10)
        let cases = [
            TranscriptionSegment(start: -0.001, end: 1, text: "segment below"),
            TranscriptionSegment(start: 1, end: 10.001, text: "segment above"),
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
            ),
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
            ),
        ]
        for (index, segment) in cases.enumerated() {
            let directory = try makeDirectory()
            defer { try? FileManager.default.removeItem(at: directory) }
            let basename = "rejected-\(index)"
            let result = BoundedTranscriptionReportWriter(
                outputDir: directory.path
            ).write(
                result: makeResult(segment: segment),
                to: basename,
                allowedCoordinateInterval: interval
            )
            guard case .failure = result,
                  !FileManager.default.fileExists(
                      atPath: directory.appendingPathComponent("\(basename).srt").path
                  ),
                  !FileManager.default.fileExists(
                      atPath: directory.appendingPathComponent("\(basename).json").path
                  )
            else {
                throw ProbeError("out-of-bound report was partially published")
            }
        }
    }

    private static func proveInvalidCoordinateRejection() throws {
        do {
            _ = try AllowedCoordinateInterval(lowerBound: .nan, upperBound: 10)
            throw ProbeError("non-finite interval was accepted")
        } catch is TranscriptionEmissionBoundError {
            // Expected.
        }
        do {
            _ = try AllowedCoordinateInterval(lowerBound: 10, upperBound: 0)
            throw ProbeError("reversed interval was accepted")
        } catch is TranscriptionEmissionBoundError {
            // Expected.
        }
        let interval = try AllowedCoordinateInterval(lowerBound: 0, upperBound: 10)
        let result = makeResult(
            segment: TranscriptionSegment(start: 0, end: .infinity, text: "non-finite")
        )
        do {
            try interval.validate(result: result)
            throw ProbeError("non-finite coordinate was accepted")
        } catch is TranscriptionEmissionBoundError {
            // Expected.
        }
    }

    private static func makeResult(segment: TranscriptionSegment) -> TranscriptionResult {
        TranscriptionResult(
            text: segment.text,
            segments: [segment],
            language: "en",
            timings: TranscriptionTimings()
        )
    }

    private static func makeDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("whisperkit-emission-probe-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        return directory
    }
}

private struct ProbeError: Error, CustomStringConvertible {
    let description: String

    init(_ description: String) {
        self.description = description
    }
}

private struct ProbeChunker: AudioChunking {
    func chunkAll(
        audioArray: [Float],
        maxChunkLength: Int,
        decodeOptions: DecodingOptions?
    ) async throws -> [AudioChunk] {
        []
    }
}
