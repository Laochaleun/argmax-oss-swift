//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import CoreML
import Foundation
@testable import WhisperKit
import XCTest

final class RealAudioSelectionDomainTests: XCTestCase {
    func testExactSampleBoundariesAndNonGridTails() throws {
        let cases: [(samples: Int, maxOffset: Int, columnCount: Int)] = [
            (1, 0, 1),
            (319, 0, 1),
            (320, 1, 2),
            (321, 1, 2),
            (160_000, 500, 501),
            (379_904, 1_187, 1_188), // 23.744 seconds
            (446_896, 1_396, 1_397), // 27.931 seconds
            (479_984, 1_499, 1_500), // 29.999 seconds
            (480_000, 1_500, 1_500), // full 30-second matrix width is unchanged
        ]

        for testCase in cases {
            let domain = try makeDomain(samples: testCase.samples)
            XCTAssertEqual(domain.samplesPerTimestampToken, 320)
            XCTAssertEqual(domain.maximumTimestampTokenOffset, testCase.maxOffset)
            XCTAssertEqual(
                try domain.validAlignmentColumnCount(matrixColumnCount: 1_500),
                testCase.columnCount
            )
            XCTAssertLessThanOrEqual(
                Float(domain.maximumTimestampTokenOffset) * WhisperKit.secondsPerTimeToken,
                domain.localUpperBoundSeconds
            )
            XCTAssertGreaterThan(
                Float(domain.maximumTimestampTokenOffset + 1) * WhisperKit.secondsPerTimeToken,
                domain.localUpperBoundSeconds
            )
        }
    }

    func testNonzeroSeekAndInvalidDomainsFailClosed() throws {
        let domain = try makeDomain(seek: 68_160, samples: 379_904)
        XCTAssertEqual(domain.globalLowerBoundSeconds, Float(68_160) / 16_000)
        XCTAssertEqual(
            domain.globalUpperBoundSeconds,
            Float(68_160) / 16_000 + Float(379_904) / 16_000
        )
        XCTAssertNoThrow(try domain.validateWindow(seek: 68_160, segmentSize: 379_904))
        XCTAssertThrowsError(try domain.validateWindow(seek: 0, segmentSize: 379_904))

        XCTAssertThrowsError(try makeDomain(samples: 0))
        XCTAssertThrowsError(try makeDomain(samples: -1))
        XCTAssertThrowsError(try makeDomain(seek: -1, samples: 320))
        XCTAssertThrowsError(
            try RealAudioSelectionDomain(
                seekSampleOffset: Int.max,
                realSampleCount: 1,
                sampleRate: 16_000,
                secondsPerTimeToken: 0.02
            )
        )
        XCTAssertThrowsError(
            try RealAudioSelectionDomain(
                seekSampleOffset: 0,
                realSampleCount: 320,
                sampleRate: 16_001,
                secondsPerTimeToken: 0.02
            )
        )
    }

    func testTimestampMaskPrecedesProbabilityAggregationAndPreservesEOT() throws {
        let specialTokens = testSpecialTokens()
        let domain = try makeDomain(samples: 321)
        let domainFilter = ValidTimestampDomainFilter(
            specialTokens: specialTokens,
            realAudioSelectionDomain: domain
        )
        let timestampRules = TimestampRulesFilter(
            specialTokens: specialTokens,
            sampleBegin: 0,
            maxInitialTimestampIndex: nil,
            isModelMultilingual: false
        )

        let logits = try MLMultiArray.logits([5, 0, 0, 0, 0, 0, 0.25, 0.5, 100, 90])
        let masked = domainFilter.filterLogits(logits, withTokens: [4])
        XCTAssertEqual(masked[6].floatValue, 0.25)
        XCTAssertEqual(masked[7].floatValue, 0.5)
        XCTAssertEqual(masked[8].floatValue, -Float.infinity)
        XCTAssertEqual(masked[9].floatValue, -Float.infinity)

        let aggregated = timestampRules.filterLogits(masked, withTokens: [4])
        XCTAssertEqual(aggregated[0].floatValue, 5)
        XCTAssertEqual(aggregated[7].floatValue, 0.5)

        let pairLogits = try MLMultiArray.logits([5, 0, 0, 7, 0, 0, 0.25, 0.5, 100, 90])
        let pairMasked = domainFilter.filterLogits(pairLogits, withTokens: [4, 7])
        let pairResult = timestampRules.filterLogits(pairMasked, withTokens: [4, 7])
        XCTAssertEqual(pairResult[specialTokens.endToken].floatValue, 7)
        XCTAssertEqual(pairResult[8].floatValue, -Float.infinity)
    }

    func testIllegalSampledTimestampLeavesNoLegalSelectionPath() throws {
        let specialTokens = testSpecialTokens()
        let filter = ValidTimestampDomainFilter(
            specialTokens: specialTokens,
            realAudioSelectionDomain: try makeDomain(samples: 321)
        )
        let logits = try MLMultiArray.logits(Array(repeating: FloatType(1), count: 10))
        let result = filter.filterLogits(logits, withTokens: [8])
        XCTAssertThrowsError(try TextDecoder.validateLegalSelection(result)) { error in
            XCTAssertEqual(
                error as? RealAudioSelectionDomainError,
                .noLegalDecoderSelectionPath
            )
        }
    }

    func testRowsAndColumnsAreRestrictedBeforeDTWWithoutChangingLegalValues() throws {
        let source = try makeMatrix(rows: 3, columns: 4) { row, column in
            Float(row * 10 + column)
        }
        let seeker = SegmentSeeker()
        let bounded = try seeker.prepareAlignmentWeightsForDTW(
            alignmentWeights: source,
            filteredRowIndices: [2, 0],
            realAudioSelectionDomain: try makeDomain(samples: 321)
        )

        XCTAssertEqual(bounded.shape.map(\.intValue), [2, 2])
        XCTAssertEqual(bounded[bounded.linearOffset(for: [0, 0])].floatValue, 20)
        XCTAssertEqual(bounded[bounded.linearOffset(for: [0, 1])].floatValue, 21)
        XCTAssertEqual(bounded[bounded.linearOffset(for: [1, 0])].floatValue, 0)
        XCTAssertEqual(bounded[bounded.linearOffset(for: [1, 1])].floatValue, 1)

        let path = try seeker.dynamicTimeWarping(withMatrix: bounded)
        XCTAssertEqual(path.timeIndices.max(), 1)
        XCTAssertThrowsError(
            try seeker.prepareAlignmentWeightsForDTW(
                alignmentWeights: source,
                filteredRowIndices: [3],
                realAudioSelectionDomain: try makeDomain(samples: 321)
            )
        )
    }

    func testThirtySecondControlPreservesFullAlignmentMatrixExactly() throws {
        let source = try makeMatrix(rows: 2, columns: 1_500) { row, column in
            Float(row * 1_500 + column) / 10
        }
        let bounded = try SegmentSeeker().prepareAlignmentWeightsForDTW(
            alignmentWeights: source,
            filteredRowIndices: [0, 1],
            realAudioSelectionDomain: try makeDomain(samples: 480_000)
        )
        XCTAssertEqual(bounded.shape.map(\.intValue), [2, 1_500])
        for row in 0..<2 {
            for column in 0..<1_500 {
                XCTAssertEqual(
                    bounded[bounded.linearOffset(for: [row, column])].floatValue,
                    source[source.linearOffset(for: [row, column])].floatValue
                )
            }
        }
    }

    func testNonemptyTextWithoutAlignmentRowsFailsClosed() throws {
        let seeker = SegmentSeeker()
        let alignmentWeights = try makeMatrix(rows: 2, columns: 2) { _, _ in 0 }
        let domain = try makeDomain(samples: 321)
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

        for invalidSegments in [
            [nonemptySegment],
            [tokenizedSibling, nonemptySegment],
        ] {
            XCTAssertThrowsError(
                try seeker.addWordTimestamps(
                    segments: invalidSegments,
                    alignmentWeights: alignmentWeights,
                    tokenizer: StubTokenizer(),
                    seek: 0,
                    segmentSize: 321,
                    realAudioSelectionDomain: domain,
                    prependPunctuations: Constants.defaultPrependPunctuations,
                    appendPunctuations: Constants.defaultAppendPunctuations,
                    lastSpeechTimestamp: 0,
                    options: DecodingOptions(wordTimestamps: true),
                    timings: TranscriptionTimings()
                )
            )
        }

        let emptySegment = TranscriptionSegment(
            start: 0,
            end: 0.02,
            text: "",
            tokens: [],
            tokenLogProbs: []
        )
        let preserved = try XCTUnwrap(
            seeker.addWordTimestamps(
                segments: [emptySegment],
                alignmentWeights: alignmentWeights,
                tokenizer: StubTokenizer(),
                seek: 0,
                segmentSize: 321,
                realAudioSelectionDomain: domain,
                prependPunctuations: Constants.defaultPrependPunctuations,
                appendPunctuations: Constants.defaultAppendPunctuations,
                lastSpeechTimestamp: 0,
                options: DecodingOptions(wordTimestamps: true),
                timings: TranscriptionTimings()
            )
        )
        XCTAssertEqual(preserved, [emptySegment])
    }

    func testFallbacksReceiveOneImmutableDomainAndGuardRunsBeforeCallback() async throws {
        let decoder = RecordingTextDecoder(fallbackCount: 2)
        let callbackRecorder = SegmentCallbackRecorder()
        let task = TranscribeTask(
            currentTimings: TranscriptionTimings(),
            progress: Progress(),
            audioEncoder: StubAudioEncoder(),
            featureExtractor: StubFeatureExtractor(),
            segmentSeeker: StubSegmentSeeker(injectedEnd: 0.02),
            textDecoder: decoder,
            tokenizer: StubTokenizer()
        )
        task.segmentDiscoveryCallback = { segments in
            callbackRecorder.record(segments)
        }
        let interval = try AllowedCoordinateInterval(lowerBound: 0, upperBound: 1)
        _ = try await task.run(
            audioArray: Array(repeating: 0, count: 321),
            decodeOptions: DecodingOptions(
                language: "en",
                temperatureFallbackCount: 2,
                sampleLength: 8,
                usePrefillPrompt: false,
                detectLanguage: false,
                allowedCoordinateInterval: interval,
                windowClipTime: 0,
                compressionRatioThreshold: nil,
                logProbThreshold: nil,
                firstTokenLogProbThreshold: nil,
                noSpeechThreshold: nil,
                concurrentWorkerCount: 1
            )
        )
        XCTAssertEqual(decoder.domains.count, 3)
        XCTAssertTrue(decoder.domains.allSatisfy { $0 == decoder.domains[0] })
        XCTAssertEqual(decoder.domains[0].realSampleCount, 321)
        XCTAssertEqual(callbackRecorder.count, 1)

        let rejectedCallbackRecorder = SegmentCallbackRecorder()
        let rejectedTask = TranscribeTask(
            currentTimings: TranscriptionTimings(),
            progress: Progress(),
            audioEncoder: StubAudioEncoder(),
            featureExtractor: StubFeatureExtractor(),
            segmentSeeker: StubSegmentSeeker(injectedEnd: 0.04),
            textDecoder: RecordingTextDecoder(fallbackCount: 0),
            tokenizer: StubTokenizer()
        )
        rejectedTask.segmentDiscoveryCallback = { segments in
            rejectedCallbackRecorder.record(segments)
        }
        do {
            _ = try await rejectedTask.run(
                audioArray: Array(repeating: 0, count: 321),
                decodeOptions: DecodingOptions(
                    language: "en",
                    temperatureFallbackCount: 0,
                    sampleLength: 8,
                    usePrefillPrompt: false,
                    detectLanguage: false,
                    allowedCoordinateInterval: interval,
                    windowClipTime: 0,
                    compressionRatioThreshold: nil,
                    logProbThreshold: nil,
                    firstTokenLogProbThreshold: nil,
                    noSpeechThreshold: nil,
                    concurrentWorkerCount: 1
                )
            )
            XCTFail("Expected injected post-selection violation to fail")
        } catch is TranscriptionEmissionBoundError {
            // Expected before callback.
        }
        XCTAssertEqual(rejectedCallbackRecorder.count, 0)
    }

    func testWindowDomainsAreRecreatedForNonzeroSeeksAndOddFinalTail() async throws {
        let decoder = RecordingTextDecoder(fallbackCount: 0)
        let callbackRecorder = SegmentCallbackRecorder()
        let task = TranscribeTask(
            currentTimings: TranscriptionTimings(),
            progress: Progress(),
            audioEncoder: StubAudioEncoder(),
            featureExtractor: StubFeatureExtractor(windowSamples: 500),
            segmentSeeker: StubSegmentSeeker(),
            textDecoder: decoder,
            tokenizer: StubTokenizer()
        )
        task.segmentDiscoveryCallback = { segments in
            callbackRecorder.record(segments)
        }
        _ = try await task.run(
            audioArray: Array(repeating: 0, count: 1_321),
            decodeOptions: DecodingOptions(
                language: "en",
                temperatureFallbackCount: 0,
                sampleLength: 8,
                usePrefillPrompt: false,
                detectLanguage: false,
                allowedCoordinateInterval: try AllowedCoordinateInterval(
                    lowerBound: 0,
                    upperBound: Float(1_321) / Float(WhisperKit.sampleRate)
                ),
                windowClipTime: 0,
                compressionRatioThreshold: nil,
                logProbThreshold: nil,
                firstTokenLogProbThreshold: nil,
                noSpeechThreshold: nil,
                concurrentWorkerCount: 1
            )
        )
        XCTAssertEqual(decoder.domains.map(\.seekSampleOffset), [0, 500, 1_000])
        XCTAssertEqual(decoder.domains.map(\.realSampleCount), [500, 500, 321])
        XCTAssertEqual(callbackRecorder.count, 3)
    }

    func testSeventeenUnequalWorkersKeepPerInputOptionsAndDomains() async throws {
        let worker = try await WorkerIndexProbeWhisperKit(
            WhisperKitConfig(verbose: false, load: false, download: false)
        )
        let audioArrays = (1...17).map { Array(repeating: Float(0), count: $0 * 320) }
        let options = (1...17).map { index in
            DecodingOptions(
                sampleLength: 100 + index,
                concurrentWorkerCount: 16
            )
        }
        let callbackRecorder = WindowCallbackRecorder()
        let segmentRecorder = SegmentSeekRecorder()
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

        XCTAssertEqual(results.count, 17)
        for index in 0..<17 {
            let result = try results[index].get()
            XCTAssertEqual(
                result.first?.text,
                "\((index + 1) * 320):\(101 + index):\(index + 1)"
            )
        }
        XCTAssertEqual(callbackRecorder.windowIds, Set(0..<17))
        XCTAssertEqual(segmentRecorder.seeks, Set(seekOffsets))
    }

    private func makeDomain(seek: Int = 0, samples: Int) throws -> RealAudioSelectionDomain {
        try RealAudioSelectionDomain(
            seekSampleOffset: seek,
            realSampleCount: samples,
            sampleRate: WhisperKit.sampleRate,
            secondsPerTimeToken: WhisperKit.secondsPerTimeToken
        )
    }

    private func testSpecialTokens() -> SpecialTokens {
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

    private func makeMatrix(
        rows: Int,
        columns: Int,
        value: (Int, Int) -> Float
    ) throws -> MLMultiArray {
        let matrix = try MLMultiArray(
            shape: [NSNumber(value: rows), NSNumber(value: columns)],
            dataType: .float32
        )
        for row in 0..<rows {
            for column in 0..<columns {
                matrix[matrix.linearOffset(for: [row, column])] = NSNumber(
                    value: value(row, column)
                )
            }
        }
        return matrix
    }
}

private final class RecordingTextDecoder: TextDecoding {
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

    func predictLogits(_ inputs: any TextDecoderInputType) async throws -> TextDecoderOutputType? {
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

private struct StubFeatureExtractor: FeatureExtracting {
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

private struct StubAudioEncoder: AudioEncoding {
    let embedSize: Int? = 1

    func encodeFeatures(
        _ features: any FeatureExtractorOutputType
    ) async throws -> (any AudioEncoderOutputType)? {
        try MLMultiArray(shape: [1], dataType: .float32)
    }
}

private final class StubSegmentSeeker: SegmentSeeking {
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

private struct StubTokenizer: WhisperTokenizer {
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

private final class SegmentCallbackRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private(set) var count = 0

    func record(_ segments: [TranscriptionSegment]) {
        lock.lock()
        count += 1
        lock.unlock()
    }
}

private final class WindowCallbackRecorder: @unchecked Sendable {
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

private final class SegmentSeekRecorder: @unchecked Sendable {
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

private final class WorkerIndexProbeWhisperKit: WhisperKit {
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
