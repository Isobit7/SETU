import React from 'react';
import type { RecognitionPayload } from '../types';

export interface LiveDetectionOutputProps {
  currentResult: RecognitionPayload | null;
  className?: string;
}

export const LiveDetectionOutput: React.FC<LiveDetectionOutputProps> = ({
  currentResult,
  className = '',
}) => {
  const isKnown = currentResult?.is_known_sign || false;
  const label = currentResult?.label || 'no_sign';
  const confidence = currentResult?.confidence || 0;
  const confidencePct = Math.round(confidence * 100);

  // Format main display label: Ensure RealSign letter renders cleanly as "LETTER X"
  let displayText: string | null = null;
  if (isKnown && label !== 'Awaiting Sign Input' && label !== 'no_sign' && label !== 'unknown') {
    if (currentResult?.display_name) {
      displayText = currentResult.display_name;
    } else if (currentResult?.active_model === 'realsign' && label.length === 1) {
      displayText = `LETTER ${label.toUpperCase()}`;
    } else {
      displayText = label.replace('_', ' ').toUpperCase();
    }
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
      ) : (
        /* Idle / Awaiting State: Subtle, compact 1-line status */
        <div className="live-detection-idle-strip" aria-live="polite">
          <div className="idle-strip-left">
            <span className="idle-status-dot" aria-hidden="true" />
            <span className="idle-status-text">Ready for Sign Input</span>
            {headGesture === 'shake' ? (
              <span className="head-intent-badge shake">✕ Undone (Shake)</span>
            ) : (
              <span className="idle-status-hint">· Nod to commit word</span>
            )}
          </div>
          <div className="idle-strip-right">
            <span className="idle-model-pill">{modelBadge}</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default LiveDetectionOutput;
