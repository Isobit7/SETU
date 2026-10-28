import React, { useState } from 'react';
import {
  X,
  Check,
  Sliders,
  Type,
  Layers,
  ShieldCheck,
  Volume2,
  VolumeX,
  Globe,
  Focus,
  Eye,
} from 'lucide-react';
import type { LandmarkStyles, SubtitleSize } from '../types';
import { SUPPORTED_LANGUAGES } from '../hooks/useTTS';

type SettingsTab = 'speech' | 'display' | 'sensitivity';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  landmarkStyles: LandmarkStyles;
  setLandmarkStyles: React.Dispatch<React.SetStateAction<LandmarkStyles>>;
  subtitleSize: SubtitleSize;
  setSubtitleSize: (size: SubtitleSize) => void;
  confidenceThreshold: number;
  setConfidenceThreshold: (val: number) => void;
  autoSpeakEnabled?: boolean;
  onToggleAutoSpeak?: () => void;
  targetLanguage?: string;
  onSelectLanguage?: (lang: string) => void;
  availableVoices?: SpeechSynthesisVoice[];
  selectedVoice?: SpeechSynthesisVoice | null;
  onSelectVoice?: (voice: SpeechSynthesisVoice | null) => void;
  isSpeaking?: boolean;
  enforceFraming?: boolean;
  onToggleEnforceFraming?: () => void;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  landmarkStyles,
  setLandmarkStyles,
  subtitleSize,
  setSubtitleSize,
  confidenceThreshold,
  setConfidenceThreshold,
  autoSpeakEnabled = true,
  onToggleAutoSpeak,
  targetLanguage = 'en-IN',
  onSelectLanguage,
  availableVoices = [],
  selectedVoice = null,
  onSelectVoice,
  isSpeaking: _isSpeaking = false,
  enforceFraming = false,
  onToggleEnforceFraming,
}) => {
  const [activeTab, setActiveTab] = useState<SettingsTab>('speech');

  // Dismiss on Escape key
  React.useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div className="modal-backdrop" onClick={onClose} role="dialog" aria-modal="true">
      <div className="modal-container settings-modal" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="modal-header">
          <div className="modal-title-wrap">
            <Sliders size={20} className="text-primary" />
            <h2>Settings & Preferences</h2>
          </div>
          <button className="modal-close-btn" onClick={onClose} aria-label="Close dialog">
            <X size={20} />
          </button>
        </div>

        {/* 3-Tab Navigation Bar */}
        <div className="settings-tabs-bar" role="tablist" aria-label="Settings Categories">
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === 'speech'}
            className={`settings-tab-btn ${activeTab === 'speech' ? 'active' : ''}`}
            onClick={() => setActiveTab('speech')}
          >
            <Volume2 size={15} />
            <span>Speech & Audio</span>
          </button>

          <button
            type="button"
            role="tab"
            aria-selected={activeTab === 'display'}
            className={`settings-tab-btn ${activeTab === 'display' ? 'active' : ''}`}
            onClick={() => setActiveTab('display')}
          >
            <Eye size={15} />
            <span>Display & View</span>
          </button>

          <button
            type="button"
            role="tab"
            aria-selected={activeTab === 'sensitivity'}
            className={`settings-tab-btn ${activeTab === 'sensitivity' ? 'active' : ''}`}
            onClick={() => setActiveTab('sensitivity')}
          >
            <Sliders size={15} />
            <span>Sensitivity & Privacy</span>
          </button>
        </div>

        {/* Tab Body */}
        <div className="modal-body settings-tab-content">
          {/* TAB 1: Speech & Audio */}
          {activeTab === 'speech' && (
            <div className="settings-section">
              <div className="section-title">
                <Volume2 size={16} className="text-primary" />
                <h3>Text-to-Speech (TTS) & Regional Voices</h3>
              </div>

              {/* Auto-Speak Toggle */}
              <div className="setting-row">
                <div className="setting-label-block">
                  <label>Auto-Speak Sentences</label>
                  <span className="setting-desc">
                    Automatically voice English translations aloud when a sentence completes.
                  </span>
                </div>
                {onToggleAutoSpeak && (
                  <button
                    type="button"
                    className={`toggle-option-btn ${autoSpeakEnabled ? 'selected' : ''}`}
                    onClick={onToggleAutoSpeak}
                    aria-pressed={autoSpeakEnabled}
                  >
                    {autoSpeakEnabled ? (
                      <>
                        <Volume2 size={14} className="text-primary" />
                        <span>Enabled</span>
                      </>
                    ) : (
                      <>
                        <VolumeX size={14} />
                        <span>Muted</span>
                      </>
                    )}
                  </button>
                )}
              </div>

              {/* Regional Voice Language */}
              <div className="setting-row">
                <div className="setting-label-block">
                  <label>
                    <Globe size={14} style={{ display: 'inline', marginRight: 6, verticalAlign: 'middle' }} />
                    Regional Voice Language
                  </label>
                  <span className="setting-desc">
                    Select the regional speech synthesis voice for spoken translations.
                  </span>
                </div>
                <select
                  className="camera-compact-select tts-voice-select"
                  value={targetLanguage}
                  onChange={(e) => {
                    if (onSelectLanguage) {
                      onSelectLanguage(e.target.value);
                    }
                  }}
                  aria-label="Select Speech Synthesis Language"
                >
                  {SUPPORTED_LANGUAGES.map((lang) => (
                    <option key={lang.code} value={lang.code}>
                      {lang.label} ({lang.nativeName})
                    </option>
                  ))}
                </select>
              </div>

              {/* Specific Synthesizer Voice */}
              {availableVoices && availableVoices.length > 0 && (
                <div className="setting-row">
                  <div className="setting-label-block">
                    <label>Speech Synthesizer Voice</label>
                    <span className="setting-desc">
                      Available system voices installed on this device.
                    </span>
                  </div>
                  <select
                    className="camera-compact-select tts-voice-select"
                    value={selectedVoice?.voiceURI || ''}
                    onChange={(e) => {
                      const chosen = availableVoices.find((v) => v.voiceURI === e.target.value);
                      if (onSelectVoice && chosen) {
                        onSelectVoice(chosen);
                      }
                    }}
                    aria-label="Select Text-to-Speech Voice"
                  >
                    {availableVoices.map((voice) => (
                      <option key={voice.voiceURI} value={voice.voiceURI}>
                        {voice.name} ({voice.lang}) {voice.default ? '— Default' : ''}
                      </option>
                    ))}
                  </select>
                </div>
              )}
            </div>
          )}

          {/* TAB 2: Display & View */}
          {activeTab === 'display' && (
            <div className="settings-section">
              {/* Subtitles Font Size */}
              <div className="section-title">
                <Type size={16} className="text-primary" />
                <h3>Live Translation Subtitles</h3>
              </div>
              <div className="setting-row">
                <div className="setting-label-block">
                  <label>Subtitle Font Size</label>
                  <span className="setting-desc">
                    Adjust text scaling for the live camera viewfinder subtitle HUD.
                  </span>
                </div>
                <div className="button-group-toggle">
                  {(['small', 'medium', 'large'] as SubtitleSize[]).map((size) => (
                    <button
                      key={size}
                      type="button"
                      className={`toggle-option-btn ${subtitleSize === size ? 'selected' : ''}`}
                      onClick={() => setSubtitleSize(size)}
                    >
                      {size.charAt(0).toUpperCase() + size.slice(1)}
                    </button>
                  ))}
                </div>
              </div>

              {/* Hand Landmark Skeletons */}
              <div className="section-title mt-4">
                <Layers size={16} className="text-primary" />
                <h3>Hand Landmark Skeletons</h3>
              </div>
              <div className="toggle-list">
                <label className="toggle-checkbox-row">
                  <input
                    type="checkbox"
                    checked={landmarkStyles.showLeftHand}
                    onChange={(e) =>
                      setLandmarkStyles((prev: LandmarkStyles) => ({ ...prev, showLeftHand: e.target.checked }))
                    }
                  />
                  <span className="toggle-label-text">
                    <strong>Left Hand Skeleton</strong> (Calm Sage Green `#81C995`)
                  </span>
                </label>

                <label className="toggle-checkbox-row">
                  <input
                    type="checkbox"
                    checked={landmarkStyles.showRightHand}
                    onChange={(e) =>
                      setLandmarkStyles((prev: LandmarkStyles) => ({ ...prev, showRightHand: e.target.checked }))
                    }
                  />
                  <span className="toggle-label-text">
                    <strong>Right Hand Skeleton</strong> (Slate Cyan `#78A9FF`)
                  </span>
                </label>
              </div>
            </div>
          )}

          {/* TAB 3: Sensitivity & Privacy */}
          {activeTab === 'sensitivity' && (
            <div className="settings-section">
              <div className="section-title">
                <Sliders size={16} className="text-primary" />
                <h3>Detection Sensitivity</h3>
              </div>

              {/* Confidence Threshold Slider */}
              <div className="setting-row">
                <div className="slider-label-row">
                  <label>Confidence Gate Threshold</label>
                  <span className="threshold-val">{Math.round(confidenceThreshold * 100)}%</span>
                </div>
                <input
                  type="range"
                  min="0.20"
                  max="0.90"
                  step="0.05"
                  value={confidenceThreshold}
                  onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                  className="slider-range"
                />
                <span className="setting-desc">
                  Minimum confidence required before a detected sign is registered. Standard default is 35%.
                </span>
              </div>

              {/* Framing Gate Toggle */}
              {onToggleEnforceFraming && (
                <div className="setting-row">
                  <div className="setting-label-block">
                    <label>
                      <Focus size={14} style={{ display: 'inline', marginRight: 6, verticalAlign: 'middle' }} />
                      Enforce Clean Framing Gate
                    </label>
                    <span className="setting-desc">
                      Pause sign detection when user is too close, too far, or hands are dropped.
                    </span>
                  </div>
                  <button
                    type="button"
                    className={`toggle-option-btn ${enforceFraming ? 'selected' : ''}`}
                    onClick={onToggleEnforceFraming}
                    aria-pressed={enforceFraming}
                  >
                    {enforceFraming ? (
                      <>
                        <ShieldCheck size={14} className="text-primary" />
                        <span>Enforced</span>
                      </>
                    ) : (
                      <span>Optional</span>
                    )}
                  </button>
                </div>
              )}

              {/* Privacy Information */}
              <div className="privacy-card-box mt-3">
                <div className="section-title">
                  <ShieldCheck size={16} className="text-success" />
                  <h3>Local Processing Guarantee</h3>
                </div>
                <p className="privacy-desc">
                  All video frames are processed locally on your machine in real-time. No camera frames, video recordings, or biometrics are ever sent to external cloud servers or stored to disk.
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="modal-footer">
          <button className="primary-btn" onClick={onClose}>
            <Check size={16} />
            <span>Save & Close</span>
          </button>
        </div>
      </div>
    </div>
  );
};

export default SettingsModal;
