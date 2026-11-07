import React, { useState, useRef, useEffect } from 'react';
import {
  Copy,
  Volume2,
  VolumeX,
  Trash2,
  Check,
  Delete,
  Download,
  Printer,
  FileText,
  FileCode,
  ChevronDown,
} from 'lucide-react';
import type { RecognitionPayload } from '../types';
import { LiveDetectionOutput } from './LiveDetectionOutput';

interface IndicLangTab {
  key: string;
  label: string;
  native: string;
  code: string;
}

const INDIC_TABS: IndicLangTab[] = [
  { key: 'en', label: 'English', native: 'English', code: 'en-IN' },
  { key: 'hi', label: 'Hindi', native: 'हिन्दी', code: 'hi-IN' },
  { key: 'ta', label: 'Tamil', native: 'தமிழ்', code: 'ta-IN' },
  { key: 'te', label: 'Telugu', native: 'తెలుగు', code: 'te-IN' },
  { key: 'kn', label: 'Kannada', native: 'ಕನ್ನಡ', code: 'kn-IN' },
];

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
  onSpeakSentence?: (text: string, lang?: string) => void;
  isSpeaking?: boolean;
  onSpeak?: (text: string, lang?: string) => void;
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
  const [selectedLang, setSelectedLang] = useState<string>('en');
  const [isExportMenuOpen, setIsExportMenuOpen] = useState<boolean>(false);

  const exportMenuRef = useRef<HTMLDivElement | null>(null);

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

  // Get translation text for any specified language tab
  const getTranslationForLang = (langKey: string): string => {
    if (langKey === 'en') {
      return activeTranslation || currentResult?.translations?.en || '';
    }
    return currentResult?.translations?.[langKey] || currentResult?.translated_indic?.[langKey] || '';
  };

  const currentTranslationText = getTranslationForLang(selectedLang);
  const currentTab = INDIC_TABS.find((t) => t.key === selectedLang) || INDIC_TABS[0];

  // Close export menu on click outside or Escape
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (exportMenuRef.current && !exportMenuRef.current.contains(e.target as Node)) {
        setIsExportMenuOpen(false);
      }
    };
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setIsExportMenuOpen(false);
    };

    document.addEventListener('mousedown', handleClickOutside);
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, []);

  // Unified Text-to-Speech (speaks regional translation or English translation or buffer)
  const handleSpeak = () => {
    const textToSpeak = (
      currentTranslationText.trim() ||
      activeTranslation.trim() ||
      activeSentence.trim() ||
      displayText ||
      ''
    ).trim();
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
      onSpeak(textToSpeak, currentTab.code);
      return;
    }

    if (onSpeakSentence) {
      onSpeakSentence(textToSpeak, currentTab.code);
      return;
    }

    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(textToSpeak);
      utterance.lang = currentTab.code;
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
    const displayed = currentTranslationText.trim() || activeTranslation.trim();
    const textToCopy = displayed
      ? (activeSentence.trim() ? `${displayed} (${activeSentence.trim()})` : displayed)
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

  // Export 1: Markdown (.md)
  const handleExportMarkdown = () => {
    setIsExportMenuOpen(false);
    const now = new Date();
    const dateStr = now.toISOString().split('T')[0];
    let md = `# SETU — Indian Sign Language Translation Transcript\n\n`;
    md += `**Date:** ${now.toLocaleDateString()} ${now.toLocaleTimeString()}\n\n`;

    md += `## Translations\n`;
    INDIC_TABS.forEach((tab) => {
      const trans = getTranslationForLang(tab.key);
      if (trans.trim()) {
        md += `- **${tab.native} (${tab.label}):** ${trans.trim()}\n`;
      }
    });

    if (activeSentence.trim()) {
      md += `\n## ISL Gloss Tokens\n\`\`\`\n${activeSentence.trim()}\n\`\`\`\n`;
    }
    md += `\n---\n*Exported from SETU ISL Assistive Workstation*\n`;

    const blob = new Blob([md], { type: 'text/markdown;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `SETU_Transcript_${dateStr}.md`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  // Export 2: Plain Text (.txt)
  const handleExportPlainText = () => {
    setIsExportMenuOpen(false);
    const now = new Date();
    const dateStr = now.toISOString().split('T')[0];
    let txt = `SETU — INDIAN SIGN LANGUAGE TRANSLATION TRANSCRIPT\n`;
    txt += `Timestamp: ${now.toLocaleDateString()} ${now.toLocaleTimeString()}\n`;
    txt += `--------------------------------------------------------\n\n`;

    INDIC_TABS.forEach((tab) => {
      const trans = getTranslationForLang(tab.key);
      if (trans.trim()) {
        txt += `${tab.native} (${tab.label}):\n  ${trans.trim()}\n\n`;
      }
    });

    if (activeSentence.trim()) {
      txt += `ISL Gloss Sequence:\n  ${activeSentence.trim()}\n\n`;
    }
    txt += `--------------------------------------------------------\nSETU ISL Platform\n`;

    const blob = new Blob([txt], { type: 'text/plain;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `SETU_Transcript_${dateStr}.txt`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  // Export 3: Print / Save as PDF
  const handlePrint = () => {
    setIsExportMenuOpen(false);
    window.print();
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
  const hasSentenceContent = Boolean(activeTranslation.trim() || activeSentence.trim() || currentTranslationText.trim());
  const hasTextToOperate = Boolean(hasSentenceContent || displayText);

  return (
    <div className="unified-translation-container">
      {/* SINGLE UNIFIED TRANSLATION CARD */}
      <div className="unified-card" id="unified-translation-card">
        {/* Top: Compact Live Detection Status Strip */}
        <LiveDetectionOutput currentResult={currentResult} />

        {/* Middle: Clean Language Tabs without Native Scrollbar */}
        <div className="indic-multilingual-section">
          <div className="indic-lang-tabs-bar" role="tablist" aria-label="Indic Regional Language Selection">
            {INDIC_TABS.map((tab) => {
              const hasContent = Boolean(getTranslationForLang(tab.key).trim());
              const isSelected = selectedLang === tab.key;
              return (
                <button
                  key={tab.key}
                  role="tab"
                  type="button"
                  aria-selected={isSelected}
                  className={`indic-tab-pill ${isSelected ? 'active' : ''} ${hasContent ? 'has-content' : ''}`}
                  onClick={() => setSelectedLang(tab.key)}
                  title={`Switch to ${tab.label} (${tab.native})`}
                >
                  <span className="indic-tab-native">{tab.native}</span>
                  <span className="indic-tab-label">{tab.label}</span>
                  {hasContent && <span className="indic-tab-indicator" />}
                </button>
              );
            })}
          </div>
        </div>

        {/* Main: Unified Translation & Sentence Workspace (NO duplicate boxes!) */}
        <div className="unified-canvas-section">
          <div className="canvas-header-bar">
            <div className="canvas-header-left">
              <span className="canvas-lang-badge">
                {currentTab.native} ({currentTab.label})
              </span>
              <span className="canvas-syntax-tag">
                {selectedLang === 'en' ? 'Natural English' : 'Indic Neural Transfer'}
              </span>
            </div>
            <div className="canvas-header-right">
              {speaking && (
                <div className="tts-live-waveform-indicator" aria-label="Audio speaking">
                  <span className="soundwave-bar bar-1" />
                  <span className="soundwave-bar bar-2" />
                  <span className="soundwave-bar bar-3" />
                  <span className="soundwave-bar bar-4" />
                  <span className="waveform-label">{currentTab.label} TTS</span>
                </div>
              )}
              <div className="sentence-stats-mini">
                <span>{wordCount} {wordCount === 1 ? 'word' : 'words'}</span>
                <span className="stats-dot">·</span>
                <span>{charCount} chars</span>
              </div>
            </div>
          </div>

          <div
            className="unified-canvas-body"
            tabIndex={0}
            role="textbox"
            aria-label="Accumulated translation output"
            aria-multiline="true"
          >
            {hasSentenceContent ? (
              <div className="canvas-content-flow">
                {/* 1. Primary Output String */}
                <div className="canvas-primary-translation">
                  <span className="canvas-main-text">
                    {currentTranslationText.trim()
                      ? currentTranslationText
                      : activeTranslation.trim()
                      ? activeTranslation
                      : activeSentence}
                  </span>
                  <span className="sentence-cursor" aria-hidden="true">|</span>
                </div>

                {/* 2. Secondary Gloss Tokens (Rendered ONLY if translation differs from raw tokens) */}
                {Boolean(activeSentence.trim()) &&
                  Boolean(
                    (currentTranslationText.trim() && currentTranslationText.trim().toLowerCase() !== activeSentence.trim().toLowerCase()) ||
                    (activeTranslation.trim() && activeTranslation.trim().toLowerCase() !== activeSentence.trim().toLowerCase())
                  ) && (
                    <div className="canvas-secondary-gloss">
                      <span className="gloss-tag">ISL Signs:</span>
                      <span className="gloss-text">{activeSentence.trim()}</span>
                    </div>
                  )}
              </div>
            ) : (
              <div className="canvas-empty-state" aria-live="polite">
                <p className="canvas-empty-title">
                  Signed letters and words will assemble here...
                  <span className="sentence-cursor" aria-hidden="true">|</span>
                </p>
                <div className="canvas-shortcut-hints">
                  <span>Press <kbd className="hint-kbd">Space</kbd> for word break</span>
                  <span className="stats-dot">·</span>
                  <span><kbd className="hint-kbd">⌫</kbd> to delete</span>
                  <span className="stats-dot">·</span>
                  <span><kbd className="hint-kbd">S</kbd> to speak</span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Bottom: Streamlined Action Toolbar */}
        <div className="unified-actions-row" role="toolbar" aria-label="Translation Actions">
          {/* Button 1: Speak */}
          <button
            className={`action-btn speak-action-btn ${speaking ? 'active-speaking' : ''}`}
            disabled={!hasSentenceContent}
            onClick={handleSpeak}
            title={speaking ? 'Stop audio playback [S]' : `Speak ${currentTab.label} audio (TTS) [S]`}
            aria-label={speaking ? 'Stop speech' : `Speak ${currentTab.label} audio`}
          >
            {speaking ? (
              <>
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
            title="Copy translation to clipboard [C]"
            aria-label="Copy translated text to clipboard"
          >
            {copied ? (
              <>
                <Check size={14} className="text-emerald" />
                <span className="text-emerald">Copied!</span>
              </>
            ) : (
              <>
                <Copy size={14} />
                <span>Copy</span>
              </>
            )}
            <kbd className="action-kbd-badge">C</kbd>
          </button>

          {/* Button 3: Backspace */}
          {onBackspaceSentence && (
            <button
              className="action-btn backspace-action-btn"
              disabled={!activeSentence}
              onClick={onBackspaceSentence}
              title="Delete previous sign [Backspace]"
              aria-label="Delete last recognized sign"
            >
              <Delete size={14} />
              <span>Backspace</span>
            </button>
          )}

          {/* Button 4: Clear */}
          <button
            className="action-btn clear-action-btn"
            disabled={!hasTextToOperate}
            onClick={handleClear}
            title="Clear all accumulated text and reset"
            aria-label="Clear all sentence text"
          >
            <Trash2 size={14} />
            <span>Clear</span>
          </button>

          {/* Button 5: Multi-Format Export Dropdown */}
          <div className="export-dropdown-wrapper" ref={exportMenuRef}>
            <button
              type="button"
              className={`action-btn export-trigger-btn ${isExportMenuOpen ? 'open' : ''}`}
              disabled={!hasSentenceContent}
              onClick={() => setIsExportMenuOpen((prev) => !prev)}
              title="Export translation transcripts"
              aria-expanded={isExportMenuOpen}
              aria-haspopup="true"
            >
              <Download size={14} />
              <span>Export</span>
              <ChevronDown size={11} className={`export-chevron ${isExportMenuOpen ? 'open' : ''}`} />
            </button>

            {isExportMenuOpen && (
              <div className="export-menu-dropdown" role="menu">
                <button
                  type="button"
                  className="export-menu-item"
                  onClick={handleExportMarkdown}
                  role="menuitem"
                >
                  <FileCode size={14} className="text-accent" />
                  <div className="export-item-text">
                    <span className="export-item-title">Markdown Document (.md)</span>
                    <span className="export-item-desc">Formatted with timestamps and signs</span>
                  </div>
                </button>

                <button
                  type="button"
                  className="export-menu-item"
                  onClick={handleExportPlainText}
                  role="menuitem"
                >
                  <FileText size={14} className="text-muted" />
                  <div className="export-item-text">
                    <span className="export-item-title">Plain Text (.txt)</span>
                    <span className="export-item-desc">Raw multiline text transcript</span>
                  </div>
                </button>

                <button
                  type="button"
                  className="export-menu-item"
                  onClick={handlePrint}
                  role="menuitem"
                >
                  <Printer size={14} className="text-muted" />
                  <div className="export-item-text">
                    <span className="export-item-title">Print / Save as PDF</span>
                    <span className="export-item-desc">Opens system print dialog</span>
                  </div>
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default TranslationPanel;
