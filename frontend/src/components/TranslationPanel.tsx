import React, { useState, useRef, useEffect } from 'react';
import {
  Copy,
  Volume2,
  VolumeX,
  Trash2,
  Check,
  Delete,
  Printer,
  FileText,
  FileCode,
  ChevronDown,
  MoreHorizontal,
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
  const [isMoreMenuOpen, setIsMoreMenuOpen] = useState<boolean>(false);

  const moreMenuRef = useRef<HTMLDivElement | null>(null);

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

  // Close more menu on click outside or Escape
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (moreMenuRef.current && !moreMenuRef.current.contains(e.target as Node)) {
        setIsMoreMenuOpen(false);
      }
    };
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setIsMoreMenuOpen(false);
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
    setIsMoreMenuOpen(false);
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
    setIsMoreMenuOpen(false);
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
    setIsMoreMenuOpen(false);
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

        {/* Modern Segmented Language Switcher (100% width, 5 balanced tabs, zero overflow) */}
        <div className="modern-language-wrapper">
          <div className="modern-language-bar" role="tablist" aria-label="Target Translation Language">
            {INDIC_TABS.map((tab) => {
              const hasContent = Boolean(getTranslationForLang(tab.key).trim());
              const isSelected = selectedLang === tab.key;
              const primaryLabel = tab.key === 'en' ? 'English' : tab.native;
              const secondaryLabel = tab.key === 'en' ? 'EN' : tab.label;

              return (
                <button
                  key={tab.key}
                  role="tab"
                  type="button"
                  aria-selected={isSelected}
                  className={`modern-lang-tab ${isSelected ? 'active' : ''} ${hasContent ? 'has-translation' : ''}`}
                  onClick={() => setSelectedLang(tab.key)}
                  title={`${tab.label} (${tab.native}) · Click to switch language`}
                >
                  <span className="modern-lang-primary">{primaryLabel}</span>
                  <span className="modern-lang-secondary">{secondaryLabel}</span>
                  {hasContent && <span className="modern-lang-dot" aria-hidden="true" />}
                </button>
              );
            })}
          </div>
        </div>

        {/* Main: Unified Translation & Sentence Workspace */}
        <div className="unified-canvas-section">
          <div className="canvas-meta-strip">
            <div className="meta-strip-left">
              {speaking ? (
                <div className="tts-live-waveform-indicator" aria-label="Audio speaking">
                  <span className="soundwave-bar bar-1" />
                  <span className="soundwave-bar bar-2" />
                  <span className="soundwave-bar bar-3" />
                  <span className="soundwave-bar bar-4" />
                  <span className="waveform-label">{currentTab.label} Voice</span>
                </div>
              ) : (
                <span className="meta-target-hint">
                  Target: <strong>{currentTab.label}</strong> ({currentTab.native})
                </span>
              )}
            </div>
            <div className="meta-strip-right">
              <span className="canvas-stats-text">
                {wordCount} {wordCount === 1 ? 'word' : 'words'} · {charCount} chars
              </span>
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

          {/* Button 4: More Options Dropdown (Clear, Export, Print) */}
          <div className="more-actions-wrapper" ref={moreMenuRef}>
            <button
              type="button"
              className={`action-btn more-actions-trigger-btn ${isMoreMenuOpen ? 'open' : ''}`}
              onClick={() => setIsMoreMenuOpen((prev) => !prev)}
              title="More options: Clear buffer, Export transcript, Print"
              aria-expanded={isMoreMenuOpen}
              aria-haspopup="true"
            >
              <MoreHorizontal size={14} />
              <span>More</span>
              <ChevronDown size={11} className={`export-chevron ${isMoreMenuOpen ? 'open' : ''}`} />
            </button>

            {isMoreMenuOpen && (
              <div className="more-menu-dropdown" role="menu">
                <button
                  type="button"
                  className="more-menu-item text-danger"
                  disabled={!hasTextToOperate}
                  onClick={() => {
                    setIsMoreMenuOpen(false);
                    handleClear();
                  }}
                  role="menuitem"
                >
                  <Trash2 size={14} className="text-danger" />
                  <div className="more-item-text">
                    <span className="more-item-title text-danger">Clear Buffer</span>
                    <span className="more-item-desc">Reset assembled signs and text</span>
                  </div>
                </button>

                <div className="dropdown-divider" />

                <button
                  type="button"
                  className="more-menu-item"
                  disabled={!hasSentenceContent}
                  onClick={handleExportMarkdown}
                  role="menuitem"
                >
                  <FileCode size={14} className="text-accent" />
                  <div className="more-item-text">
                    <span className="more-item-title">Export Markdown (.md)</span>
                    <span className="more-item-desc">Formatted transcript with timestamps</span>
                  </div>
                </button>

                <button
                  type="button"
                  className="more-menu-item"
                  disabled={!hasSentenceContent}
                  onClick={handleExportPlainText}
                  role="menuitem"
                >
                  <FileText size={14} className="text-muted" />
                  <div className="more-item-text">
                    <span className="more-item-title">Export Plain Text (.txt)</span>
                    <span className="more-item-desc">Raw multiline text transcript</span>
                  </div>
                </button>

                <button
                  type="button"
                  className="more-menu-item"
                  disabled={!hasSentenceContent}
                  onClick={handlePrint}
                  role="menuitem"
                >
                  <Printer size={14} className="text-muted" />
                  <div className="more-item-text">
                    <span className="more-item-title">Print / Save as PDF</span>
                    <span className="more-item-desc">System print dialog</span>
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
