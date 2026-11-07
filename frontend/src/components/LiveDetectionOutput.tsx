import React from 'react';
import { Sparkles, ArrowRight } from 'lucide-react';
import type { RecognitionPayload } from '../types';

export interface LiveDetectionOutputProps {
  currentResult: RecognitionPayload | null;
  className?: string;
  onAcceptPrediction?: (word: string) => void;
  onResolveDisambiguation?: (choice: string) => void;
}

export const LiveDetectionOutput: React.FC<LiveDetectionOutputProps> = ({
  currentResult,
  className = '',
  onAcceptPrediction,
  onResolveDisambiguation,
}) => {
  const isKnown = currentResult?.is_known_sign || false;
  const label = currentResult?.label || 'no_sign';
  const confidence = currentResult?.confidence || 0;
  const confidencePct = Math.round(confidence * 100);

  // Format main display label: Ensure RealSign letter renders cleanly as "LETTER X"
  let displayText: string | null = null;
  const isHolding = !isKnown && Boolean(currentResult?.display_name?.startsWith('HOLDING'));
  if (isKnown && label !== 'Awaiting Sign Input' && label !== 'no_sign' && label !== 'unknown') {
    if (currentResult?.display_name) {
      displayText = currentResult.display_name;
    } else if (currentResult?.active_model === 'realsign' && label.length === 1) {
      displayText = `LETTER ${label.toUpperCase()}`;
    } else {
      displayText = label.replace('_', ' ').toUpperCase();
    }
  } else if (isHolding) {
    displayText = currentResult?.display_name || null;
  }

  const modelBadge =
    currentResult?.active_model === 'realsign'
      ? 'Alphabet (A–Z)'
      : currentResult?.active_model === 'isl_sequence_lstm'
      ? 'Dynamic Sequences'
      : currentResult?.active_model === 'include_stream'
      ? 'INCLUDE 263'
      : currentResult?.active_model === 'isl_numerals'
      ? 'Numbers (0–9 & ₹)'
      : currentResult?.active_model === 'isl_hindi_vowels'
      ? 'Hindi Vowels'
      : 'Conversational Signs';

  const headGesture = currentResult?.head_gesture || currentResult?.non_manual_markers?.head_gesture;
  const predictiveSuggestions = currentResult?.predictive_suggestions || [];
  const disambigActive = currentResult?.disambiguation?.active;
  const disambigOptions = currentResult?.disambiguation?.options || [];

  return (
    <div className={`live-detection-output-root ${className}`}>
      {isKnown && displayText ? (
        /* Active Recognized Sign: Vibrant, clean, high-confidence pill strip */
        <div className="live-detection-active-strip" aria-live="polite" aria-atomic="true">
          <div className="active-strip-left">
            <span className="active-pulse-dot" aria-hidden="true" />
            <span className="active-sign-token">{displayText}</span>
            <span className="active-sign-confidence">{confidencePct}% match</span>
            {headGesture === 'nod' ? (
              <span className="head-intent-badge nod">✓ Locked (Nod)</span>
            ) : headGesture === 'shake' ? (
              <span className="head-intent-badge shake">✕ Cancelled (Shake)</span>
            ) : (
              <span className="head-intent-hint">· Nod to lock</span>
            )}
          </div>
          <div className="active-strip-right">
            <span className="active-model-pill">{modelBadge}</span>
          </div>
        </div>
      ) : isHolding && displayText ? (
        /* Stabilizing / Holding State: Crisp feedback showing sign is being verified */
        <div className="live-detection-holding-strip" aria-live="polite">
          <div className="idle-strip-left">
            <span className="holding-pulse-dot" aria-hidden="true" />
            <span className="holding-status-text">{displayText}</span>
            <span className="holding-status-hint">· Hold steady to lock ({confidencePct}%)</span>
          </div>
          <div className="idle-strip-right">
            <span className="idle-model-pill">{modelBadge}</span>
          </div>
        </div>
      ) : (
        /* Idle / Awaiting State: Subtle, compact 1-line status */
        <div className="live-detection-idle-strip" aria-live="polite">
          <div className="idle-strip-left">
            <span className="idle-status-dot" aria-hidden="true" />
            <span className="idle-status-text">Ready for Sign Input</span>
            <span className="idle-status-hint">· Position hands clearly in camera view</span>
          </div>
          <div className="idle-strip-right">
            <span className="idle-model-pill">{modelBadge}</span>
          </div>
        </div>
      )}

      {/* Intelligent Fallback 1: Disambiguation Pills if top-2 are close */}
      {disambigActive && disambigOptions.length >= 2 && (
        <div className="live-disambiguation-strip" role="region" aria-label="Sign disambiguation choices">
          <span className="disambig-label">Close match — select one:</span>
          <div className="disambig-chips-group">
            {disambigOptions.slice(0, 3).map((opt) => (
              <button
                key={opt.label}
                type="button"
                className="disambig-chip-btn"
                onClick={() => onResolveDisambiguation?.(opt.label)}
                title={`Select ${opt.display}`}
              >
                <span>{opt.display}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Intelligent Fallback 2: Predictive Autocomplete Suggestions */}
      {predictiveSuggestions.length > 0 && (
        <div className="live-predictive-strip" role="region" aria-label="Word completion suggestions">
          <span className="predictive-label">
            <Sparkles size={11} className="text-cyan" />
            <span>Complete Word:</span>
          </span>
          <div className="predictive-chips-group">
            {predictiveSuggestions.slice(0, 4).map((word) => (
              <button
                key={word}
                type="button"
                className="predictive-chip-btn"
                onClick={() => onAcceptPrediction?.(word)}
                title={`Autocomplete with ${word}`}
              >
                <span>{word}</span>
                <ArrowRight size={10} className="chip-arrow" />
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default LiveDetectionOutput;
