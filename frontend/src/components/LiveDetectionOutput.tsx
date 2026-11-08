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

  // Format main display label
  let displayText: string | null = null;
  if (isKnown && label !== 'Awaiting Sign Input' && label !== 'no_sign' && label !== 'unknown') {
    if (currentResult?.display_name) {
      displayText = currentResult.display_name;
    } else if (currentResult?.active_model === 'realsign' && label.length === 1) {
      displayText = `Letter ${label.toUpperCase()}`;
    } else {
      displayText = label.replace('_', ' ').toUpperCase();
    }
  }

  return (
    <div className={`compact-detection-bar ${className}`}>
      <div className="detection-bar-left">
        <Activity size={13} className="text-cyan detection-pulse-icon" />
        <span className="detection-bar-title">Live Sign:</span>
        {displayText ? (
          <span className="detection-active-sign font-semibold">{displayText}</span>
        ) : (
          <span className="detection-idle-sign">Ready</span>
        )}
      </div>

      <div className="detection-bar-right">
        {displayText && confidencePct > 0 && (
          <span className="detection-conf-badge font-semibold">{confidencePct}% Match</span>
        )}
        <span className="detection-model-tag">
          {currentResult?.active_model === 'realsign'
            ? 'Alphabet'
            : currentResult?.active_model === 'isl_sequence_lstm'
            ? 'Sequence'
            : currentResult?.active_model === 'include_stream'
            ? 'INCLUDE'
            : 'Phrases'}
        </span>
      </div>
    </div>
  );
};

export default LiveDetectionOutput;
