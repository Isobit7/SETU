import React, { useState } from 'react';
import {
  Copy,
  Volume2,
  VolumeX,
  Trash2,
  Check,
  Delete,
} from 'lucide-react';
import type { RecognitionPayload } from '../types';
import { LiveDetectionOutput } from './LiveDetectionOutput';

interface TranslationPanelProps {
  currentResult: RecognitionPayload | null;
  onClearCurrent: () => void;
  onOpenHistory?: () => void;
  historyCount: number;
  sentence?: string;
  translatedText?: string;
  onClearSentence?: () => void;
  onBackspaceSentence?: () => void;
  onAddSpaceSentence?: () => void;
  onSpeakSentence?: (text: string) => void;
  isSpeaking?: boolean;
  onSpeak?: (text: string) => void;
  onStopSpeak?: () => void;
}

export const TranslationPanel: React.FC<TranslationPanelProps> = ({
  currentResult,
  onClearCurrent,
  historyCount: _historyCount = 0,
  sentence = '',
  translatedText = '',
  onClearSentence,
  onBackspaceSentence,
  onSpeakSentence,
  isSpeaking: propIsSpeaking,
  onSpeak,
  onStopSpeak,
}) => {
  const [copied, setCopied] = useState<boolean>(false);
  const [localIsSpeaking, setLocalIsSpeaking] = useState<boolean>(false);

  const speaking = propIsSpeaking !== undefined ? propIsSpeaking : localIsSpeaking;

  const isKnown = currentResult?.is_known_sign || false;
  const label = currentResult?.label || 'no_sign';

  const displayText = isKnown && label !== 'Awaiting Sign Input' && label !== 'no_sign'
    ? currentResult?.display_name || (currentResult?.active_model === 'realsign' && label.length === 1 ? `LETTER ${label.toUpperCase()}` : label.toUpperCase())
    : null;

  // Active sentence buffer (prefers server accumulator stream, falls back to local prop)
  const activeSentence = (currentResult?.sentence_buffer !== undefined && currentResult.sentence_buffer !== '')
    ? currentResult.sentence_buffer
    : sentence;

  // English grammar translation produced by ISLGlossParser
  const activeTranslation = translatedText || currentResult?.translated_text || '';

  // Unified Text-to-Speech (speaks translated_text || sentence_buffer)
  const handleSpeak = () => {
    const textToSpeak = (activeTranslation.trim() || activeSentence.trim() || displayText || '').trim();
    if (!textToSpeak) return;

    if (speaking) {
      if (onStopSpeak) {
        onStopSpeak();
      } else if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
        setLocalIsSpeaking(false);
      }
      return;
    }

    if (onSpeak) {
      onSpeak(textToSpeak);
      return;
    }

    if (onSpeakSentence) {
      onSpeakSentence(textToSpeak);
      return;
    }

    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(textToSpeak);
      utterance.lang = 'en-US';
      utterance.rate = 1.0;
      utterance.pitch = 1.0;

      utterance.onstart = () => setLocalIsSpeaking(true);
      utterance.onend = () => setLocalIsSpeaking(false);
      utterance.onerror = () => setLocalIsSpeaking(false);

      window.speechSynthesis.speak(utterance);
    }
  };

  // Unified Clipboard copy
  const handleCopy = async () => {
    const textToCopy = activeTranslation.trim()
      ? (activeSentence.trim() ? `${activeTranslation.trim()} (${activeSentence.trim()})` : activeTranslation.trim())
      : (activeSentence.trim() || displayText || '');
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
    if (onStopSpeak) {
      onStopSpeak();
    } else if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      setLocalIsSpeaking(false);
    }
  };

  // Word and character stats
  const wordCount = activeSentence.trim() ? activeSentence.trim().split(/\s+/).filter(Boolean).length : 0;
  const charCount = activeSentence.length;
  const hasSentenceContent = Boolean(activeTranslation.trim() || activeSentence.trim());
  const hasTextToOperate = Boolean(hasSentenceContent || displayText);

  return (
    <div className="unified-translation-container">
      {/* SINGLE UNIFIED TRANSLATION CARD */}
      <div className="unified-card" id="unified-translation-card">
        {/* Live Detection Output & Candidate Scores Component */}
        <LiveDetectionOutput currentResult={currentResult} />

        {/* English Grammar Translation Box (ISL Gloss -> English SVO) */}
        {activeTranslation ? (
          <div className="english-translation-section">
            <div className="translation-section-header">
              <span className="translation-section-badge">English Translation</span>
              <span className="translation-rule-tag">Grammar Parsed</span>
            </div>
            <div className="english-translation-box">
              <p className="english-translation-text">"{activeTranslation}"</p>
            </div>
          </div>
        ) : null}

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
            {activeSentence && activeSentence.length > 0 ? (
              <div className="sentence-content-wrap">
                <span className="sentence-text-string">{activeSentence}</span>
                <span className="sentence-cursor" aria-hidden="true">|</span>
              </div>
            ) : (
              <div className="sentence-calm-empty-state" aria-live="polite">
                <p className="sentence-calm-placeholder">
                  Signed letters and words will assemble here...
                  <span className="sentence-cursor" aria-hidden="true">|</span>
                </p>
                <span className="sentence-shortcut-hint">
                  Press <kbd className="hint-kbd">Space</kbd> for word break · <kbd className="hint-kbd">⌫</kbd> to delete · <kbd className="hint-kbd">S</kbd> to speak
                </span>
              </div>
            )}
          </div>
        </div>

        {/* 4. Single Action Row: Exactly 4 Compact Pill Buttons with Keyboard Badges */}
        <div className="unified-actions-row" role="toolbar" aria-label="Translation Actions">
          {/* Button 1: Speak */}
          <button
            className={`action-btn speak-action-btn ${speaking ? 'active-speaking' : ''}`}
            disabled={!hasSentenceContent}
            onClick={handleSpeak}
            title={speaking ? 'Stop audio playback [S]' : 'Speak text aloud (TTS) [S]'}
            aria-label={speaking ? 'Stop speech' : 'Speak text'}
          >
            {speaking ? (
              <>
                <div className="soundwave-bars" aria-hidden="true">
                  <span className="soundwave-bar" />
                  <span className="soundwave-bar" />
                  <span className="soundwave-bar" />
                  <span className="soundwave-bar" />
                </div>
                <VolumeX size={14} />
                <span>Stop</span>
              </>
            ) : (
              <>
                <Volume2 size={14} />
                <span>Speak</span>
              </>
            )}
            <kbd className="action-kbd-badge">S</kbd>
          </button>

          {/* Button 2: Copy */}
          <button
            className={`action-btn copy-action-btn ${copied ? 'copy-success' : ''}`}
            disabled={!hasTextToOperate}
            onClick={handleCopy}
            title="Copy text to clipboard [C]"
            aria-label="Copy text"
          >
            {copied ? <Check size={14} className="text-success" /> : <Copy size={14} />}
            <span>{copied ? 'Copied!' : 'Copy'}</span>
            <kbd className="action-kbd-badge">C</kbd>
          </button>

          {/* Button 3: Backspace */}
          <button
            className="action-btn backspace-action-btn"
            disabled={!activeSentence}
            onClick={onBackspaceSentence}
            title="Delete last letter or word [Backspace]"
            aria-label="Backspace last character"
          >
            <Delete size={14} />
            <span>Backspace</span>
            <kbd className="action-kbd-badge">⌫</kbd>
          </button>

          {/* Button 4: Clear */}
          <button
            className="action-btn clear-action-btn"
            disabled={!hasTextToOperate}
            onClick={handleClear}
            title="Clear all text from buffer"
            aria-label="Clear all text"
          >
            <Trash2 size={14} />
            <span>Clear</span>
            <kbd className="action-kbd-badge">Del</kbd>
          </button>
        </div>
      </div>
    </div>
  );
};
