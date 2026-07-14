//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import Foundation

public enum RealAudioSelectionDomainError: Error, Equatable, LocalizedError {
    case invalidSeekSampleOffset(Int)
    case invalidRealSampleCount(Int)
    case invalidSampleRate(Int)
    case unsupportedTimestampGrid(sampleRate: Int, secondsPerTimeToken: Float)
    case sampleArithmeticOverflow
    case inconsistentWindow(
        expectedSeek: Int,
        actualSeek: Int,
        expectedSampleCount: Int,
        actualSampleCount: Int
    )
    case invalidAlignmentMatrixShape([Int])
    case noLegalDecoderSelectionPath

    public var errorDescription: String? {
        switch self {
            case let .invalidSeekSampleOffset(seek):
                return "Invalid real-audio seek sample offset \(seek)."
            case let .invalidRealSampleCount(sampleCount):
                return "Invalid real-audio sample count \(sampleCount)."
            case let .invalidSampleRate(sampleRate):
                return "Invalid real-audio sample rate \(sampleRate)."
            case let .unsupportedTimestampGrid(sampleRate, secondsPerTimeToken):
                return "Unsupported timestamp grid for sample rate \(sampleRate) and \(secondsPerTimeToken) seconds per timestamp token."
            case .sampleArithmeticOverflow:
                return "Real-audio selection-domain sample arithmetic overflowed."
            case let .inconsistentWindow(
                expectedSeek,
                actualSeek,
                expectedSampleCount,
                actualSampleCount
            ):
                return "Real-audio selection domain [seek \(expectedSeek), samples \(expectedSampleCount)] does not match window [seek \(actualSeek), samples \(actualSampleCount)]."
            case let .invalidAlignmentMatrixShape(shape):
                return "Invalid real-audio alignment matrix shape \(shape)."
            case .noLegalDecoderSelectionPath:
                return "No legal decoder selection path remains inside the real-audio window."
        }
    }
}

/// Immutable, producer-derived selection domain for one real (unpadded) decode window.
///
/// This is a package-internal carrier rather than a decoding option or CLI input. The
/// public read-only shape is required by the existing public decoding protocols; only
/// targets in this package can construct a value.
public struct RealAudioSelectionDomain: Equatable, Sendable {
    public let seekSampleOffset: Int
    public let realSampleCount: Int
    public let sampleRate: Int
    public let samplesPerTimestampToken: Int
    public let maximumTimestampTokenOffset: Int
    public let localLowerBoundSeconds: Float
    public let localUpperBoundSeconds: Float
    public let globalLowerBoundSeconds: Float
    public let globalUpperBoundSeconds: Float

    package init(
        seekSampleOffset: Int,
        realSampleCount: Int,
        sampleRate: Int,
        secondsPerTimeToken: Float
    ) throws {
        guard seekSampleOffset >= 0 else {
            throw RealAudioSelectionDomainError.invalidSeekSampleOffset(seekSampleOffset)
        }
        guard realSampleCount > 0 else {
            throw RealAudioSelectionDomainError.invalidRealSampleCount(realSampleCount)
        }
        guard sampleRate > 0 else {
            throw RealAudioSelectionDomainError.invalidSampleRate(sampleRate)
        }

        let timestampPositionsPerSecond = Float(1) / secondsPerTimeToken
        let roundedTimestampPositionsPerSecond = timestampPositionsPerSecond.rounded()
        guard secondsPerTimeToken.isFinite,
              secondsPerTimeToken > 0,
              timestampPositionsPerSecond.isFinite,
              roundedTimestampPositionsPerSecond == timestampPositionsPerSecond,
              roundedTimestampPositionsPerSecond >= 1
        else {
            throw RealAudioSelectionDomainError.unsupportedTimestampGrid(
                sampleRate: sampleRate,
                secondsPerTimeToken: secondsPerTimeToken
            )
        }

        guard let timestampPositionsPerSecondInteger = Int(
            exactly: roundedTimestampPositionsPerSecond
        ) else {
            throw RealAudioSelectionDomainError.unsupportedTimestampGrid(
                sampleRate: sampleRate,
                secondsPerTimeToken: secondsPerTimeToken
            )
        }
        guard sampleRate.isMultiple(of: timestampPositionsPerSecondInteger) else {
            throw RealAudioSelectionDomainError.unsupportedTimestampGrid(
                sampleRate: sampleRate,
                secondsPerTimeToken: secondsPerTimeToken
            )
        }
        let samplesPerTimestampToken = sampleRate / timestampPositionsPerSecondInteger
        guard samplesPerTimestampToken > 0 else {
            throw RealAudioSelectionDomainError.unsupportedTimestampGrid(
                sampleRate: sampleRate,
                secondsPerTimeToken: secondsPerTimeToken
            )
        }

        let (globalEndSample, overflowed) = seekSampleOffset.addingReportingOverflow(realSampleCount)
        guard !overflowed else {
            throw RealAudioSelectionDomainError.sampleArithmeticOverflow
        }

        let maximumTimestampTokenOffset = realSampleCount / samplesPerTimestampToken
        let (nextTimestampTokenOffset, nextOffsetOverflowed) =
            maximumTimestampTokenOffset.addingReportingOverflow(1)
        guard !nextOffsetOverflowed else {
            throw RealAudioSelectionDomainError.sampleArithmeticOverflow
        }

        let localUpperBoundSeconds = Float(realSampleCount) / Float(sampleRate)
        let maximumTimestampSeconds =
            Float(maximumTimestampTokenOffset) * secondsPerTimeToken
        let nextTimestampSeconds = Float(nextTimestampTokenOffset) * secondsPerTimeToken
        guard localUpperBoundSeconds.isFinite,
              maximumTimestampSeconds <= localUpperBoundSeconds,
              nextTimestampSeconds > localUpperBoundSeconds
        else {
            throw RealAudioSelectionDomainError.unsupportedTimestampGrid(
                sampleRate: sampleRate,
                secondsPerTimeToken: secondsPerTimeToken
            )
        }

        let globalLowerBoundSeconds = Float(seekSampleOffset) / Float(sampleRate)
        let globalUpperBoundSeconds = globalLowerBoundSeconds + localUpperBoundSeconds
        guard globalLowerBoundSeconds.isFinite,
              globalUpperBoundSeconds.isFinite,
              Float(globalEndSample) / Float(sampleRate) >= globalLowerBoundSeconds
        else {
            throw RealAudioSelectionDomainError.sampleArithmeticOverflow
        }

        self.seekSampleOffset = seekSampleOffset
        self.realSampleCount = realSampleCount
        self.sampleRate = sampleRate
        self.samplesPerTimestampToken = samplesPerTimestampToken
        self.maximumTimestampTokenOffset = maximumTimestampTokenOffset
        self.localLowerBoundSeconds = 0
        self.localUpperBoundSeconds = localUpperBoundSeconds
        self.globalLowerBoundSeconds = globalLowerBoundSeconds
        self.globalUpperBoundSeconds = globalUpperBoundSeconds
    }

    package func validateWindow(seek: Int, segmentSize: Int) throws {
        guard seek == seekSampleOffset,
              segmentSize == realSampleCount
        else {
            throw RealAudioSelectionDomainError.inconsistentWindow(
                expectedSeek: seekSampleOffset,
                actualSeek: seek,
                expectedSampleCount: realSampleCount,
                actualSampleCount: segmentSize
            )
        }
    }

    package func validAlignmentColumnCount(matrixColumnCount: Int) throws -> Int {
        guard matrixColumnCount > 0 else {
            throw RealAudioSelectionDomainError.invalidAlignmentMatrixShape([matrixColumnCount])
        }
        let (domainColumnCount, overflowed) = maximumTimestampTokenOffset.addingReportingOverflow(1)
        guard !overflowed else {
            throw RealAudioSelectionDomainError.sampleArithmeticOverflow
        }
        return min(matrixColumnCount, domainColumnCount)
    }

}
