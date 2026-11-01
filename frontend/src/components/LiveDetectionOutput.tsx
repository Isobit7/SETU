import React from 'react';
import { Activity } from 'lucide-react';
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

  return (
    <div className={`live-detection-output-root ${className}`}>
      {/* Header bar */}
      <div className="unified-header">
        <div className="unified-header-left">
          <Activity size={14} className="unified-header-icon" />
          <h3 className="unified-title font-semibold">Live Detection Output</h3>
        </div>
        <div className="unified-header-right">
          {isKnown && displayText && (
            <span className="unified-conf-pill font-semibold" title="Model confidence score">
              {confidencePct}% Match
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

      {/* 1. Live Output Box: Single, Decisive, High-Confidence Output */}
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
        ) : (
          <div className="live-sign-waiting">
            <span className="waiting-dots-pulse"></span>
            <span className="waiting-sign-text">Awaiting Sign Input</span>
            <span className="waiting-sign-hint">Position hands clearly in camera view</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default LiveDetectionOutput;
