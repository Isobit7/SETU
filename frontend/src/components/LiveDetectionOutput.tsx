import React from 'react';
import { Activity } from 'lucide-react';
import type { RecognitionPayload, Candidate } from '../types';

interface TopCandidate {
  name: string;
  pct: number;
  rank: number;
}

export interface LiveDetectionOutputProps {
  currentResult: RecognitionPayload | null;
  className?: string;
}

export const LiveDetectionOutput: React.FC<LiveDetectionOutputProps> = ({
  currentResult,
  className = '',
}) => {
  const isKnown = currentResult?.is_known_sign || false;
  const isHandDetected = currentResult?.hand_detected || false;
  const label = currentResult?.label || 'no_sign';
  const confidence = currentResult?.confidence || 0;
  const confidencePct = Math.round(confidence * 100);

  const isUncertain =
    (label === 'unknown' || (!isKnown && isHandDetected && label !== 'no_sign')) &&
    label !== 'Awaiting Sign Input';

  // Format main display label: Ensure RealSign letter renders as "LETTER X"
  let displayText: string | null = null;
  if (isKnown && label !== 'Awaiting Sign Input' && label !== 'no_sign') {
    if (currentResult?.display_name) {
      displayText = currentResult.display_name;
    } else if (currentResult?.active_model === 'realsign' && label.length === 1) {
      displayText = `LETTER ${label.toUpperCase()}`;
    } else {
      displayText = label.replace('_', ' ').toUpperCase();
    }
  }

  // Extract candidate rankings directly from currentResult.candidates or model top_3
  let topCandidates: TopCandidate[] = [];

  if (currentResult?.candidates && currentResult.candidates.length > 0) {
    topCandidates = currentResult.candidates
      .filter((c: Candidate) => c.label !== 'no_sign' && c.label !== 'Awaiting Sign Input' && c.label !== 'no_hand')
      .slice(0, 3)
      .map((c: Candidate, idx: number) => ({
        name: currentResult.active_model === 'realsign' && c.label.length === 1
          ? `LETTER ${c.label.toUpperCase()}`
          : c.label.replace('_', ' ').toUpperCase(),
        pct: Math.round(c.confidence * 100),
        rank: idx + 1,
      }));
  } else if (currentResult?.active_model === 'realsign' && currentResult.realsign_prediction?.top_3) {
    topCandidates = currentResult.realsign_prediction.top_3
      .filter(([cand]) => cand !== 'no_sign' && cand !== 'Awaiting Sign Input' && cand !== 'no_hand')
      .slice(0, 3)
      .map(([cand, prob], idx) => ({
        name: cand.length === 1 ? `LETTER ${cand.toUpperCase()}` : cand.toUpperCase(),
        pct: Math.round(prob * 100),
        rank: idx + 1,
      }));
  } else if (currentResult?.active_model === 'isl_sequence_lstm' && currentResult.sequence_lstm_prediction?.top_3) {
    topCandidates = currentResult.sequence_lstm_prediction.top_3
      .filter(([cand]) => cand !== 'no_sign' && cand !== 'Awaiting Sign Input' && cand !== 'no_hand')
      .slice(0, 3)
      .map(([cand, prob], idx) => ({
        name: cand.replace('_', ' ').toUpperCase(),
        pct: Math.round(prob * 100),
        rank: idx + 1,
      }));
  } else if (currentResult?.probabilities) {
    const sorted = Object.entries(currentResult.probabilities).sort((a, b) => b[1] - a[1]);
    topCandidates = sorted
      .filter(([cand]) => cand !== 'no_sign' && cand !== 'Awaiting Sign Input' && cand !== 'no_hand')
      .slice(0, 3)
      .map(([cand, prob], idx) => ({
        name: cand.replace('_', ' ').toUpperCase(),
        pct: Math.round(prob * 100),
        rank: idx + 1,
      }));
  } else if (isKnown && displayText) {
    topCandidates = [
      { name: displayText, pct: confidencePct, rank: 1 }
    ];
  }

  return (
    <div className={`live-detection-output-root ${className}`}>
      {/* Header bar */}
      <div className="unified-header">
        <div className="unified-header-left">
          <Activity size={14} className="unified-header-icon" />
          <h3 className="unified-title font-semibold">Live Detection Output</h3>
        </div>
        <div className="unified-header-right">
          {isKnown && (
            <span className="unified-conf-pill font-semibold" title="Model confidence score">
              {confidencePct}% Match
            </span>
          )}
          {isUncertain && (
            <span className="unified-uncertain-pill font-semibold" title="Low confidence detection">
              Uncertain
            </span>
          )}
          <span className="unified-model-badge font-semibold">
            {currentResult?.active_model === 'realsign'
              ? 'Alphabet'
              : currentResult?.active_model === 'isl_sequence_lstm'
              ? 'Sequence'
              : currentResult?.active_model === 'include_stream'
              ? 'INCLUDE 263'
              : 'Phrases'}
          </span>
        </div>
      </div>

      {/* 1. Live Output Box: Large Bold Typography */}
      <div className="live-output-box" aria-live="polite" aria-atomic="true">
        {isKnown && displayText ? (
          <div className="live-sign-display">
            <span className="live-sign-text">{displayText}</span>
            <span className="live-sign-meta">
              {currentResult?.active_model === 'realsign'
                ? 'RealSign ISL Letter'
                : currentResult?.active_model === 'isl_sequence_lstm'
                ? 'Dynamic ISL Gesture'
                : currentResult?.active_model === 'include_stream'
                ? 'AI4Bharat 263 Sign'
                : 'Common ISL Phrase'}
            </span>
          </div>
        ) : isUncertain ? (
          <div className="live-sign-uncertain">
            <span className="uncertain-sign-text">
              {topCandidates.length > 0 ? `${topCandidates[0].name}?` : 'Detecting...'}
            </span>
            <span className="uncertain-sign-hint">Hold sign steady in camera view</span>
          </div>
        ) : (
          <div className="live-sign-waiting">
            <span className="waiting-dots-pulse"></span>
            <span className="waiting-sign-text">Awaiting Sign Input</span>
            <span className="waiting-sign-hint">Position hands clearly in camera view</span>
          </div>
        )}
      </div>

      {/* 2. Candidate Scores: Clean Horizontal Inline Pill Row */}
      <div className="top3-inline-container" aria-label="Candidate recognition scores">
        <span className="top3-inline-label">Candidate Scores:</span>
        {topCandidates.length > 0 ? (
          <div className="top3-pills-row">
            {topCandidates.map((c) => (
              <span
                key={`${c.name}-${c.rank}`}
                className={`candidate-pill rank-${c.rank}`}
                title={`Candidate rank #${c.rank}: ${c.name} (${c.pct}%)`}
              >
                <span className="pill-rank">Rank {c.rank}:</span>
                <span className="pill-name">{c.name}</span>
                <span className="pill-pct">({c.pct}%)</span>
              </span>
            ))}
          </div>
        ) : (
          <span className="top3-pills-empty">No active candidate scores</span>
        )}
      </div>
    </div>
  );
};

export default LiveDetectionOutput;
