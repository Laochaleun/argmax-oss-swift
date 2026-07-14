//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import CoreML
import Foundation
import WhisperKit

@main
struct EmissionBoundContractProbe {
    static func main() async throws {
        try proveRealAudioSelectionDomain()
        try proveTimestampSelectionBeforeSampling()
        try proveAlignmentRestrictionBeforeDTW()
        try proveEmptyAlignmentRowsFailClosed()
        try await proveFallbackAndWindowDomains()
        try await proveSeventeenUnequalWorkers()
        try provePreEmissionBoundary()
        try proveAtomicMultiChunkBoundary()
        try proveClosedBoundaryAcceptance()
        try proveAtomicBoundaryRejection()
        try proveInvalidCoordinateRejection()
        print("EMISSION_BOUND_CONTRACT_OK")
    }

    private static func proveRealAudioSelectionDomain() throws {
        let cases: [(samples: Int, maxOffset: Int, columns: Int)] = [
            (1, 0, 1),
            (319, 0, 1),
            (320, 1, 2),
            (321, 1, 2),
            (160_000, 500, 501),
            (379_904, 1_187, 1_188),
            (446_896, 1_396, 1_397),
            (479_984, 1_499, 1_500),
            (480_000, 1_500, 1_500),
        ]
        for testCase in cases {
            let domain = try makeDomain(samples: testCase.samples)
            guard domain.samplesPerTimestampToken == 320,
                  domain.maximumTimestampTokenOffset == testCase.maxOffset,
                  try domain.validAlignmentColumnCount(matrixColumnCount: 1_500) == testCase.columns
            else {
                throw ProbeError("real-audio integer domain arithmetic changed")
            }
        }

        let nonzeroSeek = try makeDomain(seek: 68_160, samples: 379_904)
        try nonzeroSeek.validateWindow(seek: 68_160, segmentSize: 379_904)

        let unequalChildren = try (1...17).map { childIndex in
            try makeDomain(samples: childIndex * 320)
        }
        guard unequalChildren.map(\.maximumTimestampTokenOffset) == Array(1...17),
              Set(unequalChildren.map(\.realSampleCount)).count == 17
        else {
            throw ProbeError("unequal child domains leaked across workers")
        }

        for invalidSamples in [0, -1] {
            do {
                _ = try makeDomain(samples: invalidSamples)
                throw ProbeError("invalid real-audio domain was accepted")
            } catch is RealAudioSelectionDomainError {
                // Expected.
            }
        }
        do {
            _ = try makeDomain(seek: -1, samples: 320)
            throw ProbeError("negative real-audio seek was accepted")
        } catch is RealAudioSelectionDomainError {
            // Expected.
        }
        do {
            _ = try RealAudioSelectionDomain(
                seekSampleOffset: Int.max,
                realSampleCount: 1,
                sampleRate: WhisperKit.sampleRate,
                secondsPerTimeToken: WhisperKit.secondsPerTimeToken
            )
            throw ProbeError("overflowing real-audio domain was accepted")
        } catch is RealAudioSelectionDomainError {
            // Expected.
        }
        do {
            _ = try RealAudioSelectionDomain(
                seekSampleOffset: 0,
                realSampleCount: 320,
                sampleRate: 16_001,
                secondsPerTimeToken: WhisperKit.secondsPerTimeToken
            )
            throw ProbeError("unsupported timestamp grid was accepted")
        } catch is RealAudioSelectionDomainError {
            // Expected.
        }
    }

    private static func proveTimestampSelectionBeforeSampling() throws {
        let specialTokens = probeSpecialTokens()
        let domainFilter = ValidTimestampDomainFilter(
            specialTokens: specialTokens,
            realAudioSelectionDomain: try makeDomain(samples: 321)
        )
        let timestampRules = TimestampRulesFilter(
            specialTokens: specialTokens,
            sampleBegin: 0,
            maxInitialTimestampIndex: nil,
            isModelMultilingual: false
        )
        let logits = try makeLogits([5, 0, 0, 0, 0, 0, 0.25, 0.5, 100, 90])
        let masked = domainFilter.filterLogits(logits, withTokens: [4])
        guard masked[6].floatValue == 0.25,
              masked[7].floatValue == 0.5,
              masked[8].floatValue == -Float.infinity,
              masked[9].floatValue == -Float.infinity
        else {
            throw ProbeError("timestamp selection domain changed legal logits or retained padded logits")
        }
        let aggregated = timestampRules.filterLogits(masked, withTokens: [4])
        guard aggregated[0].floatValue == 5 else {
            throw ProbeError("padded timestamp probability influenced text selection")
        }

        let pairLogits = try makeLogits([5, 0, 0, 7, 0, 0, 0.25, 0.5, 100, 90])
        let pairMasked = domainFilter.filterLogits(pairLogits, withTokens: [4, 7])
        let pairResult = timestampRules.filterLogits(pairMasked, withTokens: [4, 7])
        guard pairResult[specialTokens.endToken].floatValue == 7,
              pairResult[8].floatValue == -Float.infinity
        else {
            throw ProbeError("EOT or padded timestamp legality changed after domain masking")
        }

        let illegalHistory = domainFilter.filterLogits(
            try makeLogits(Array(repeating: 1, count: 10)),
            withTokens: [8]
        )
        do {
            try TextDecoder.validateLegalSelection(illegalHistory)
            throw ProbeError("empty legal timestamp path reached sampling")
        } catch is RealAudioSelectionDomainError {
            // Expected.
        }
    }

    private static func proveAlignmentRestrictionBeforeDTW() throws {
        let source = try MLMultiArray(shape: [3, 4], dataType: .float16)
        for row in 0..<3 {
            for column in 0..<4 {
                source[source.linearOffset(for: [row, column])] = NSNumber(
                    value: Float(row * 10 + column)
                )
            }
        }
        let seeker = SegmentSeeker()
        let bounded = try seeker.prepareAlignmentWeightsForDTW(
            alignmentWeights: source,
            filteredRowIndices: [2, 0],
            realAudioSelectionDomain: try makeDomain(samples: 321)
        )
        guard bounded.shape.map(\.intValue) == [2, 2],
              bounded[bounded.linearOffset(for: [0, 0])].floatValue == 20,
              bounded[bounded.linearOffset(for: [0, 1])].floatValue == 21,
              bounded[bounded.linearOffset(for: [1, 0])].floatValue == 0,
              bounded[bounded.linearOffset(for: [1, 1])].floatValue == 1
        else {
            throw ProbeError("pre-DTW row or column restriction changed legal values")
        }
        let path = try seeker.dynamicTimeWarping(withMatrix: bounded)
        guard path.timeIndices.max() == 1 else {
            throw ProbeError("DTW observed a padded alignment column")
        }

        let fullSource = try MLMultiArray(shape: [1, 1_500], dataType: .float16)
        for column in 0..<1_500 {
            fullSource[column] = NSNumber(value: Float(column) / 10)
        }
        let fullBounded = try seeker.prepareAlignmentWeightsForDTW(
            alignmentWeights: fullSource,
            filteredRowIndices: [0],
            realAudioSelectionDomain: try makeDomain(samples: 480_000)
        )
        guard fullBounded.shape.map(\.intValue) == [1, 1_500] else {
            throw ProbeError("full 30-second alignment width changed")
        }
        for column in 0..<1_500 where fullBounded[column] != fullSource[column] {
            throw ProbeError("valid full-window alignment coordinate changed")
        }
    }

    private static func proveEmptyAlignmentRowsFailClosed() throws {
        let seeker = SegmentSeeker()
        let alignmentWeights = try MLMultiArray(shape: [2, 2], dataType: .float16)
        let domain = try makeDomain(samples: 321)
        let emptySegment = TranscriptionSegment(
            start: 0,
            end: 0.02,
            text: "",
            tokens: [],
            tokenLogProbs: []
        )
        let preserved = try seeker.addWordTimestamps(
            segments: [emptySegment],
            alignmentWeights: alignmentWeights,
            tokenizer: ProbeTokenizer(),
            seek: 0,
            segmentSize: 321,
            realAudioSelectionDomain: domain,
            prependPunctuations: Constants.defaultPrependPunctuations,
            appendPunctuations: Constants.defaultAppendPunctuations,
            lastSpeechTimestamp: 0,
            options: DecodingOptions(wordTimestamps: true),
            timings: TranscriptionTimings()
        )
        guard preserved == [emptySegment] else {
            throw ProbeError("empty alignment control changed legal segment values")
        }

        let nonemptySegment = TranscriptionSegment(
            start: 0,
            end: 0.02,
            text: "requires alignment",
            tokens: [],
            tokenLogProbs: []
        )
        let tokenizedSibling = TranscriptionSegment(
            start: 0,
            end: 0.02,
            text: "tokenized",
            tokens: [1],
            tokenLogProbs: [[1: 0]]
        )
        let invalidCases = [
            [nonemptySegment],
            [tokenizedSibling, nonemptySegment],
        ]
        for invalidSegments in invalidCases {
            var rejected = false
            do {
                _ = try seeker.addWordTimestamps(
                    segments: invalidSegments,
                    alignmentWeights: alignmentWeights,
                    tokenizer: ProbeTokenizer(),
                    seek: 0,
                    segmentSize: 321,
                    realAudioSelectionDomain: domain,
                    prependPunctuations: Constants.defaultPrependPunctuations,
                    appendPunctuations: Constants.defaultAppendPunctuations,
                    lastSpeechTimestamp: 0,
                    options: DecodingOptions(wordTimestamps: true),
                    timings: TranscriptionTimings()
                )
            } catch {
                rejected = true
            }
            guard rejected else {
                throw ProbeError("non-empty text without per-segment alignment rows was accepted")
            }
        }
    }

    private static func proveFallbackAndWindowDomains() async throws {
        let fallbackDecoder = ProbeRecordingTextDecoder(fallbackCount: 2)
        let acceptedCallbackRecorder = ProbeSegmentCallbackRecorder()
        let fallbackTask = TranscribeTask(
            currentTimings: TranscriptionTimings(),
            progress: Progress(),
            audioEncoder: ProbeAudioEncoder(),
            featureExtractor: ProbeFeatureExtractor(),
            segmentSeeker: ProbeSegmentSeeker(injectedEnd: 0.02),
            textDecoder: fallbackDecoder,
            tokenizer: ProbeTokenizer()
        )
        fallbackTask.segmentDiscoveryCallback = { segments in
            acceptedCallbackRecorder.record(segments)
        }
        _ = try await fallbackTask.run(
            audioArray: Array(repeating: 0, count: 321),
            decodeOptions: probeDecodingOptions(
                fallbackCount: 2,
                allowedInterval: try AllowedCoordinateInterval(lowerBound: 0, upperBound: 1)
            )
        )
        guard fallbackDecoder.domains.count == 3,
              fallbackDecoder.domains.allSatisfy({ $0 == fallbackDecoder.domains[0] }),
              fallbackDecoder.domains[0].realSampleCount == 321,
              acceptedCallbackRecorder.count == 1
        else {
            throw ProbeError("fallbacks did not retain one immutable real-window domain")
        }

        let windowDecoder = ProbeRecordingTextDecoder(fallbackCount: 0)
        let windowCallbackRecorder = ProbeSegmentCallbackRecorder()
        let windowTask = TranscribeTask(
            currentTimings: TranscriptionTimings(),
            progress: Progress(),
            audioEncoder: ProbeAudioEncoder(),
            featureExtractor: ProbeFeatureExtractor(windowSamples: 500),
            segmentSeeker: ProbeSegmentSeeker(),
            textDecoder: windowDecoder,
            tokenizer: ProbeTokenizer()
        )
        windowTask.segmentDiscoveryCallback = { segments in
            windowCallbackRecorder.record(segments)
        }
        _ = try await windowTask.run(
            audioArray: Array(repeating: 0, count: 1_321),
            decodeOptions: probeDecodingOptions(
                fallbackCount: 0,
                allowedInterval: try AllowedCoordinateInterval(
                    lowerBound: 0,
                    upperBound: Float(1_321) / Float(WhisperKit.sampleRate)
                )
            )
        )
        guard windowDecoder.domains.map(\.seekSampleOffset) == [0, 500, 1_000],
              windowDecoder.domains.map(\.realSampleCount) == [500, 500, 321],
              windowCallbackRecorder.count == 3
        else {
            throw ProbeError("window-local domains leaked across a nonzero seek or odd final tail")
        }

        let rejectedCallbackRecorder = ProbeSegmentCallbackRecorder()
        let rejectedTask = TranscribeTask(
            currentTimings: TranscriptionTimings(),
            progress: Progress(),
            audioEncoder: ProbeAudioEncoder(),
            featureExtractor: ProbeFeatureExtractor(),
            segmentSeeker: ProbeSegmentSeeker(injectedEnd: 0.04),
            textDecoder: ProbeRecordingTextDecoder(fallbackCount: 0),
            tokenizer: ProbeTokenizer()
        )
        rejectedTask.segmentDiscoveryCallback = { segments in
            rejectedCallbackRecorder.record(segments)
        }
        do {
            _ = try await rejectedTask.run(
                audioArray: Array(repeating: 0, count: 321),
                decodeOptions: probeDecodingOptions(
                    fallbackCount: 0,
                    allowedInterval: try AllowedCoordinateInterval(lowerBound: 0, upperBound: 1)
                )
            )
            throw ProbeError("injected coordinate violation was accepted")
        } catch is TranscriptionEmissionBoundError {
            // Expected before segment callback.
        }
        guard rejectedCallbackRecorder.count == 0 else {
            throw ProbeError("segment callback ran before injected coordinate rejection")
        }
    }

    private static func proveSeventeenUnequalWorkers() async throws {
        let worker = try await ProbeWhisperKit(
            WhisperKitConfig(verbose: false, load: false, download: false)
        )
        let audioArrays = (1...17).map { Array(repeating: Float(0), count: $0 * 320) }
        let options = (1...17).map { index in
            DecodingOptions(
                sampleLength: 100 + index,
                concurrentWorkerCount: 16
            )
        }
        let callbackRecorder = ProbeWindowCallbackRecorder()
        let segmentRecorder = ProbeSegmentSeekRecorder()
        let seekOffsets = (0..<17).map { $0 * 10_000 }
        worker.segmentDiscoveryCallback = { segments in
            segmentRecorder.record(segments)
        }
        let results = await worker.transcribeWithOptions(
            audioArrays: audioArrays,
            decodeOptionsArray: options,
            seekOffsets: seekOffsets,
            callback: { progress in
                callbackRecorder.record(progress.windowId)
                return true
            }
        )
        guard results.count == 17 else {
            throw ProbeError("unequal worker result count changed")
        }
        for index in 0..<17 {
            let result = try results[index].get()
            guard result.first?.text == "\((index + 1) * 320):\(101 + index):\(index + 1)" else {
                throw ProbeError("unequal worker reused another child's options or domain")
            }
        }
        guard callbackRecorder.windowIds == Set(0..<17) else {
            throw ProbeError("unequal worker callback IDs are not globally ordered")
        }
        guard segmentRecorder.seeks == Set(seekOffsets) else {
            throw ProbeError("unequal worker seek offsets used a batch-local index")
        }
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

    private static func makeDomain(
        seek: Int = 0,
        samples: Int
    ) throws -> RealAudioSelectionDomain {
        try RealAudioSelectionDomain(
            seekSampleOffset: seek,
            realSampleCount: samples,
            sampleRate: WhisperKit.sampleRate,
            secondsPerTimeToken: WhisperKit.secondsPerTimeToken
        )
    }

    private static func probeDecodingOptions(
        fallbackCount: Int,
        allowedInterval: AllowedCoordinateInterval
    ) -> DecodingOptions {
        DecodingOptions(
            language: "en",
            temperatureFallbackCount: fallbackCount,
            sampleLength: 8,
            usePrefillPrompt: false,
            detectLanguage: false,
            allowedCoordinateInterval: allowedInterval,
            windowClipTime: 0,
            compressionRatioThreshold: nil,
            logProbThreshold: nil,
            firstTokenLogProbThreshold: nil,
            noSpeechThreshold: nil,
            concurrentWorkerCount: 1
        )
    }

    private static func probeSpecialTokens() -> SpecialTokens {
        SpecialTokens(
            endToken: 3,
            englishToken: 1,
            noSpeechToken: 2,
            noTimestampsToken: 2,
            specialTokenBegin: 6,
            startOfPreviousToken: 4,
            startOfTranscriptToken: 4,
            timeTokenBegin: 6,
            transcribeToken: 4,
            translateToken: 5,
            whitespaceToken: 1
        )
    }

    private static func makeLogits(_ values: [Float]) throws -> MLMultiArray {
        let logits = try MLMultiArray(
            shape: [1, 1, NSNumber(value: values.count)],
            dataType: .float16
        )
        for (index, value) in values.enumerated() {
            logits[index] = NSNumber(value: value)
        }
        return logits
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

private final class ProbeRecordingTextDecoder: TextDecoding {
    var tokenizer: WhisperTokenizer?
    var isModelMultilingual = false
    var supportsWordTimestamps = false
    var logitsSize: Int? = 10
    var logitsFilters: [any LogitsFiltering]? = []
    var kvCacheEmbedDim: Int? = 1
    var kvCacheMaxSequenceLength: Int? = 448
    var windowSize: Int? = 1
    var embedSize: Int? = 1
    private let fallbackCount: Int
    private(set) var domains = [RealAudioSelectionDomain]()

    init(fallbackCount: Int) {
        self.fallbackCount = fallbackCount
    }

    func predictLogits(
        _ inputs: any TextDecoderInputType
    ) async throws -> TextDecoderOutputType? {
        nil
    }

    func decodeText(
        from encoderOutput: any AudioEncoderOutputType,
        using decoderInputs: any DecodingInputsType,
        sampler tokenSampler: TokenSampling,
        options decoderOptions: DecodingOptions,
        realAudioSelectionDomain: RealAudioSelectionDomain,
        callback: TranscriptionCallback?
    ) async throws -> DecodingResult {
        domains.append(realAudioSelectionDomain)
        let needsFallback = domains.count <= fallbackCount
        return DecodingResult(
            language: "en",
            languageProbs: ["en": 1],
            tokens: [1],
            tokenLogProbs: [[1: 0]],
            text: "valid",
            avgLogProb: 0,
            noSpeechProb: 0,
            temperature: decoderOptions.temperature,
            compressionRatio: 0,
            fallback: needsFallback
                ? DecodingFallback(needsFallback: true, fallbackReason: "probe")
                : nil
        )
    }

    func detectLanguage(
        from encoderOutput: any AudioEncoderOutputType,
        using decoderInputs: any DecodingInputsType,
        sampler tokenSampler: TokenSampling,
        options: DecodingOptions,
        temperature: FloatType
    ) async throws -> DecodingResult {
        .emptyResults
    }
}

private struct ProbeFeatureExtractor: FeatureExtracting {
    let melCount: Int? = 1
    let windowSamples: Int?

    init(windowSamples: Int = 480_000) {
        self.windowSamples = windowSamples
    }

    func logMelSpectrogram(
        fromAudio inputAudio: any AudioProcessorOutputType
    ) async throws -> (any FeatureExtractorOutputType)? {
        try MLMultiArray(shape: [1], dataType: .float32)
    }
}

private struct ProbeAudioEncoder: AudioEncoding {
    let embedSize: Int? = 1

    func encodeFeatures(
        _ features: any FeatureExtractorOutputType
    ) async throws -> (any AudioEncoderOutputType)? {
        try MLMultiArray(shape: [1], dataType: .float32)
    }
}

private final class ProbeSegmentSeeker: SegmentSeeking {
    private let injectedEnd: Float?

    init(injectedEnd: Float? = nil) {
        self.injectedEnd = injectedEnd
    }

    func findSeekPointAndSegments(
        decodingResult: DecodingResult,
        options: DecodingOptions,
        allSegmentsCount: Int,
        currentSeek seek: Int,
        segmentSize: Int,
        sampleRate: Int,
        timeToken: Int,
        specialToken: Int,
        tokenizer: WhisperTokenizer
    ) -> (Int, [TranscriptionSegment]?) {
        (
            seek + segmentSize,
            [
                TranscriptionSegment(
                    start: Float(seek) / Float(sampleRate),
                    end: injectedEnd
                        ?? Float(seek) / Float(sampleRate)
                        + Float(segmentSize) / Float(sampleRate),
                    text: "valid",
                    tokens: [1],
                    tokenLogProbs: [[1: 0]]
                ),
            ]
        )
    }

    func addWordTimestamps(
        segments: [TranscriptionSegment],
        alignmentWeights: MLMultiArray,
        tokenizer: WhisperTokenizer,
        seek: Int,
        segmentSize: Int,
        realAudioSelectionDomain: RealAudioSelectionDomain,
        prependPunctuations: String,
        appendPunctuations: String,
        lastSpeechTimestamp: Float,
        options: DecodingOptions,
        timings: TranscriptionTimings
    ) throws -> [TranscriptionSegment]? {
        segments
    }
}

private struct ProbeTokenizer: WhisperTokenizer {
    let specialTokens = SpecialTokens(
        endToken: 3,
        englishToken: 1,
        noSpeechToken: 2,
        noTimestampsToken: 2,
        specialTokenBegin: 6,
        startOfPreviousToken: 4,
        startOfTranscriptToken: 4,
        timeTokenBegin: 6,
        transcribeToken: 4,
        translateToken: 5,
        whitespaceToken: 1
    )
    let allLanguageTokens: Set<Int> = []

    func encode(text: String) -> [Int] { [1] }
    func decode(tokens: [Int]) -> String { tokens.isEmpty ? "" : "valid" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    func splitToWordTokens(tokenIds: [Int]) -> (words: [String], wordTokens: [[Int]]) {
        (["valid"], [tokenIds])
    }
}

private final class ProbeSegmentCallbackRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var recordedCount = 0

    var count: Int {
        lock.lock()
        defer { lock.unlock() }
        return recordedCount
    }

    func record(_ segments: [TranscriptionSegment]) {
        lock.lock()
        recordedCount += 1
        lock.unlock()
    }
}

private final class ProbeWindowCallbackRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var recordedWindowIds = Set<Int>()

    var windowIds: Set<Int> {
        lock.lock()
        defer { lock.unlock() }
        return recordedWindowIds
    }

    func record(_ windowId: Int) {
        lock.lock()
        recordedWindowIds.insert(windowId)
        lock.unlock()
    }
}

private final class ProbeSegmentSeekRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var recordedSeeks = Set<Int>()

    var seeks: Set<Int> {
        lock.lock()
        defer { lock.unlock() }
        return recordedSeeks
    }

    func record(_ segments: [TranscriptionSegment]) {
        lock.lock()
        recordedSeeks.formUnion(segments.map(\.seek))
        lock.unlock()
    }
}

private final class ProbeWhisperKit: WhisperKit {
    override func transcribe(
        audioArray: [Float],
        decodeOptions: DecodingOptions? = nil,
        callback: TranscriptionCallback? = nil,
        segmentCallback: SegmentDiscoveryCallback? = nil
    ) async throws -> [TranscriptionResult] {
        let domain = try RealAudioSelectionDomain(
            seekSampleOffset: 0,
            realSampleCount: audioArray.count,
            sampleRate: WhisperKit.sampleRate,
            secondsPerTimeToken: WhisperKit.secondsPerTimeToken
        )
        _ = callback?(
            TranscriptionProgress(
                timings: TranscriptionTimings(),
                text: "",
                tokens: []
            )
        )
        segmentCallback?([
            TranscriptionSegment(
                seek: 0,
                start: 0,
                end: 0,
                text: "worker",
                tokens: [1],
                tokenLogProbs: [[1: 0]]
            ),
        ])
        return [
            TranscriptionResult(
                text: "\(audioArray.count):\(decodeOptions?.sampleLength ?? -1):\(domain.maximumTimestampTokenOffset)",
                segments: [],
                language: "en",
                timings: TranscriptionTimings()
            ),
        ]
    }
}
