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

  const getModelBadge = (model?: string) => {
    switch (model) {
      case 'realsign':
        return 'Alphabet (A–Z)';
      case 'isl_numerals':
        return 'Numbers (0–9)';
      case 'isl_hindi_vowels':
        return 'Hindi Vowels';
      case 'dynamic_unified':
      case 'amrita_phrase':
      case 'amrita_isl_phrase_bilstm':
      case 'isl_dynamic_master_bilstm':
      case 'include_stream':
        return 'Dynamic Signs & Phrases (287)';
      default:
        return 'Sign Detection';
    }
  };

  const nmm = currentResult?.non_manual_markers;
  const isQuestion = Boolean(nmm?.is_question);
  const isNegation = Boolean(nmm?.is_negation || nmm?.head_gesture === 'shake');
  const isAffirmation = Boolean(nmm?.is_affirmation || nmm?.head_gesture === 'nod');

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
        {isQuestion && (
          <span className="nmm-badge nmm-question" title="Eyebrows elevated - question marker (?)">
            ❓ Question
          </span>
        )}
        {isNegation && (
          <span className="nmm-badge nmm-negation" title="Head shake - grammatical negation (NOT)">
            🚫 Negation
          </span>
        )}
        {isAffirmation && (
          <span className="nmm-badge nmm-affirmation" title="Head nod - affirmative confirmation">
            ✓ Confirmed
          </span>
        )}
        {displayText && confidencePct > 0 && (
          <span className="detection-conf-badge font-semibold">{confidencePct}% Match</span>
        )}
        <span className="detection-model-tag">
          {getModelBadge(currentResult?.active_model)}
        </span>
      </div>
    </div>
  );
};

export default LiveDetectionOutput;
