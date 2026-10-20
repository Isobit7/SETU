import React, { useState } from 'react';
import {
  Copy,
  Volume2,
  VolumeX,
  Trash2,
  Check,
  Delete,
  Activity,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import type { RecognitionPayload } from '../types';

interface TopCandidate {
  name: string;
  pct: number;
  rank: number;
}

interface TranslationPanelProps {
  currentResult: RecognitionPayload | null;
  onClearCurrent: () => void;
  onOpenHistory?: () => void;
  historyCount: number;
  sentence?: string;
  onClearSentence?: () => void;
  onBackspaceSentence?: () => void;
  onAddSpaceSentence?: () => void;
  onSpeakSentence?: (text: string) => void;
}

export const TranslationPanel: React.FC<TranslationPanelProps> = ({
  currentResult,
  onClearCurrent,
  historyCount: _historyCount = 0,
  sentence = '',
  onClearSentence,
  onBackspaceSentence,
  onSpeakSentence,
}) => {
  const [copied, setCopied] = useState<boolean>(false);
  const [isSpeaking, setIsSpeaking] = useState<boolean>(false);
  const [diagnosticsOpen, setDiagnosticsOpen] = useState<boolean>(false);

  const isKnown = currentResult?.is_known_sign || false;
  const isHandDetected = currentResult?.hand_detected || false;
  const label = currentResult?.label || 'no_sign';
  const confidence = currentResult?.confidence || 0;
  const confidencePct = Math.round(confidence * 100);

  const isUncertain = (label === 'unknown' || (!isKnown && isHandDetected && label !== 'no_sign')) && label !== 'Awaiting Sign Input';

  const displayText = isKnown && label !== 'Awaiting Sign Input' && label !== 'no_sign'
    ? currentResult?.display_name || label.toUpperCase()
    : null;

  // Unified Text-to-Speech
  const handleSpeak = () => {
    const textToSpeak = sentence.trim() || displayText || '';
    if (!textToSpeak || !('speechSynthesis' in window)) return;

    if (isSpeaking) {
      window.speechSynthesis.cancel();
      setIsSpeaking(false);
      return;
    }

    if (onSpeakSentence && sentence.trim()) {
      onSpeakSentence(sentence);
      return;
    }

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(textToSpeak);
    utterance.lang = 'en-US';
    utterance.rate = 0.95;
    utterance.pitch = 1.0;

    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);

    window.speechSynthesis.speak(utterance);
  };

  // Unified Clipboard copy
  const handleCopy = async () => {
    const textToCopy = sentence.trim() || displayText || '';
    if (!textToCopy) return;
    try {
      await navigator.clipboard.writeText(textToCopy);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy text:', err);
    }
  };

  // Unified Clear All: clears sentence builder and active live sign
  const handleClear = () => {
    if (onClearSentence) onClearSentence();
    if (onClearCurrent) onClearCurrent();
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      setIsSpeaking(false);
    }
  };

  // Extract Top-3 Candidates with percentage
  let top3Candidates: TopCandidate[] = [];
  if (currentResult?.active_model === 'isl_sequence_lstm' && currentResult.sequence_lstm_prediction?.top_3) {
    top3Candidates = currentResult.sequence_lstm_prediction.top_3
      .filter(([cand]) => cand !== 'no_sign' && cand !== 'Awaiting Sign Input')
      .slice(0, 3)
      .map(([cand, prob], idx) => ({
        name: cand.replace('_', ' ').toUpperCase(),
        pct: Math.round(prob * 100),
        rank: idx + 1,
      }));
  } else if (currentResult?.realsign_prediction?.top_3 && currentResult.realsign_prediction.top_3.length > 0) {
    top3Candidates = currentResult.realsign_prediction.top_3
      .filter(([cand]) => cand !== 'no_sign' && cand !== 'Awaiting Sign Input')
      .map(([cand, prob], idx) => ({
        name: cand.toUpperCase(),
        pct: Math.round(prob * 100),
        rank: idx + 1,
      }));
  } else if (currentResult?.probabilities) {
    const sorted = Object.entries(currentResult.probabilities).sort((a, b) => b[1] - a[1]);
    top3Candidates = sorted
      .filter(([cand]) => cand !== 'no_sign' && cand !== 'Awaiting Sign Input')
      .slice(0, 3)
      .map(([cand, prob], idx) => ({
        name: cand.replace('_', ' ').toUpperCase(),
        pct: Math.round(prob * 100),
        rank: idx + 1,
      }));
  } else if (isKnown && displayText) {
    top3Candidates = [
      { name: displayText, pct: confidencePct, rank: 1 }
    ];
  }

  // Word and character stats
  const wordCount = sentence.trim() ? sentence.trim().split(/\s+/).filter(Boolean).length : 0;
  const charCount = sentence.length;
  const hasTextToOperate = Boolean(sentence.trim() || displayText);

  return (
    <div className="unified-translation-container">
      {/* SINGLE UNIFIED TRANSLATION CARD */}
      <div className="unified-card" id="unified-translation-card">
        {/* Card Header */}
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
                  : 'Common ISL Phrase'}
              </span>
            </div>
          ) : isUncertain ? (
            <div className="live-sign-uncertain">
              <span className="uncertain-sign-text">
                {top3Candidates.length > 0 ? `${top3Candidates[0].name}?` : 'Detecting...'}
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

        {/* 2. Top-3 Candidates: Clean Horizontal Inline Pill Row */}
        <div className="top3-inline-container" aria-label="Candidate recognition scores">
          <span className="top3-inline-label">Candidate Scores:</span>
          {top3Candidates.length > 0 ? (
            <div className="top3-pills-row">
              {top3Candidates.map((c) => (
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

        {/* 3. Sentence Builder Text Area */}
        <div className="sentence-integrated-section">
          <div className="sentence-section-header">
            <span className="sentence-section-label">Sentence Buffer</span>
            <div className="sentence-stats-mini">
              <span>{wordCount} {wordCount === 1 ? 'word' : 'words'}</span>
              <span className="stats-dot">·</span>
              <span>{charCount} chars</span>
            </div>
          </div>

          <div
            className="sentence-integrated-box"
            tabIndex={0}
            role="textbox"
            aria-label="Accumulated sentence text"
            aria-multiline="true"
          >
            {sentence && sentence.length > 0 ? (
              <div className="sentence-content-wrap">
                <span className="sentence-text-string">{sentence}</span>
                <span className="sentence-cursor" aria-hidden="true">|</span>
              </div>
            ) : (
              <p className="sentence-placeholder-text">
                Sentence buffer is empty. Recognized signs will append here.
              </p>
            )}
          </div>
        </div>

        {/* 4. Single Action Row: Exactly 4 Compact Icon/Text Buttons */}
        <div className="unified-actions-row" role="toolbar" aria-label="Translation Actions">
          {/* Button 1: Speak */}
          <button
            className={`action-btn speak-action-btn ${isSpeaking ? 'active-speaking' : ''}`}
            disabled={!hasTextToOperate}
            onClick={handleSpeak}
            title={isSpeaking ? 'Stop audio playback' : 'Speak text aloud (TTS)'}
            aria-label={isSpeaking ? 'Stop speech' : 'Speak text'}
          >
            {isSpeaking ? <VolumeX size={15} /> : <Volume2 size={15} />}
            <span>{isSpeaking ? 'Stop' : 'Speak'}</span>
          </button>

          {/* Button 2: Copy */}
          <button
            className="action-btn copy-action-btn"
            disabled={!hasTextToOperate}
            onClick={handleCopy}
            title="Copy text to clipboard"
            aria-label="Copy text"
          >
            {copied ? <Check size={15} className="text-success" /> : <Copy size={15} />}
            <span>{copied ? 'Copied' : 'Copy'}</span>
          </button>

          {/* Button 3: Backspace */}
          <button
            className="action-btn backspace-action-btn"
            disabled={!sentence}
            onClick={onBackspaceSentence}
            title="Delete last letter"
            aria-label="Backspace last character"
          >
            <Delete size={15} />
            <span>Backspace</span>
          </button>

          {/* Button 4: Clear */}
          <button
            className="action-btn clear-action-btn"
            disabled={!hasTextToOperate}
            onClick={handleClear}
            title="Clear text buffer"
            aria-label="Clear all text"
          >
            <Trash2 size={15} />
            <span>Clear</span>
          </button>
        </div>

        {/* Optional Collapsible Diagnostics Footer (Takes zero space when closed) */}
        <div className="unified-diagnostics-drawer">
          <button
            className="diagnostics-mini-trigger"
            onClick={() => setDiagnosticsOpen(!diagnosticsOpen)}
            aria-expanded={diagnosticsOpen}
            title="Toggle raw candidate probabilities"
          >
            <Activity size={13} />
            <span>Raw Candidate Metrics</span>
            {diagnosticsOpen ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
          </button>

          {diagnosticsOpen && currentResult?.probabilities && (
            <div className="diagnostics-mini-body">
              <div className="diagnostics-mini-grid">
                {Object.entries(currentResult.probabilities)
                  .filter(([name]) => name !== 'Awaiting Sign Input')
                  .sort((a, b) => b[1] - a[1])
                  .slice(0, 8)
                  .map(([name, prob]) => (
                    <div key={name} className="diag-chip">
                      <span className="diag-name">{name.replace('_', ' ').toUpperCase()}</span>
                      <span className="diag-val">{Math.round(prob * 100)}%</span>
                    </div>
                  ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
