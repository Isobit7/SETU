import React, { useState } from 'react';
import {
  X,
  Sliders,
  ShieldCheck,
  Volume2,
  Eye,
  Camera,
  Cpu,
  Play,
  Smile,
  CheckCircle2,
  Video,
  Languages,
} from 'lucide-react';
import type { ActiveModelType, BufferInsertionMode, LandmarkStyles, SubtitleSize } from '../types';
import { SUPPORTED_LANGUAGES } from '../hooks/useTTS';

export type SettingsCategory = 'recognition' | 'camera' | 'speech' | 'display';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  // Recognition & Models
  activeModel?: ActiveModelType;
  onSwitchModel?: (model: ActiveModelType) => void;
  confidenceThreshold: number;
  setConfidenceThreshold: (val: number) => void;
  bufferInsertionMode?: BufferInsertionMode;
  onSetBufferInsertionMode?: (mode: BufferInsertionMode) => void;
  // Camera & Feed
  availableCameras?: MediaDeviceInfo[];
  selectedCameraId?: string;
  onSelectCamera?: (deviceId: string) => void;
  landmarkStyles: LandmarkStyles;
  setLandmarkStyles: React.Dispatch<React.SetStateAction<LandmarkStyles>>;
  // Speech & Audio
  autoSpeakEnabled?: boolean;
  onToggleAutoSpeak?: () => void;
  targetLanguage?: string;
  onSelectLanguage?: (lang: string) => void;
  availableVoices?: SpeechSynthesisVoice[];
  selectedVoice?: SpeechSynthesisVoice | null;
  onSelectVoice?: (voice: SpeechSynthesisVoice | null) => void;
  onTestSpeak?: (sampleText: string, langCode?: string) => void;
  isSpeaking?: boolean;
  // Display & Captions
  subtitleSize: SubtitleSize;
  setSubtitleSize: (size: SubtitleSize) => void;
  enforceFraming?: boolean;
  onToggleEnforceFraming?: () => void;
  showFramingGuide?: boolean;
  onToggleFramingGuide?: () => void;
}

const CONFIDENCE_LEVELS = [
  {
    value: 0.25,
    label: '25% Sensitive',
    badge: 'Fast',
    badgeClass: 'badge-fast',
    desc: 'Rapid responsiveness. Great for continuous fingerspelling in clean lighting.',
  },
  {
    value: 0.35,
    label: '35% Balanced',
    badge: 'Recommended',
    badgeClass: 'badge-recommended',
    desc: 'Golden ratio for everyday signing. Eliminates noise while tracking smoothly.',
  },
  {
    value: 0.50,
    label: '50% Confident',
    badge: 'Accurate',
    badgeClass: 'badge-accurate',
    desc: 'Higher confidence requirement. Recommended for formal translation & exams.',
  },
  {
    value: 0.70,
    label: '70% Strict',
    badge: 'Zero-Noise',
    badgeClass: 'badge-strict',
    desc: 'Extreme noise rejection. Requires clean, deliberate holds and direct frontal camera angle.',
  },
];

export const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  activeModel = 'realsign',
  onSwitchModel,
  confidenceThreshold,
  setConfidenceThreshold,
  bufferInsertionMode = 'auto',
  onSetBufferInsertionMode,
  availableCameras = [],
  selectedCameraId = '',
  onSelectCamera,
  landmarkStyles,
  setLandmarkStyles,
  autoSpeakEnabled = true,
  onToggleAutoSpeak,
  targetLanguage = 'en-IN',
  onSelectLanguage,
  onTestSpeak,
  isSpeaking = false,
  subtitleSize,
  setSubtitleSize,
  showFramingGuide = false,
  onToggleFramingGuide,
}) => {
  const [activeCategory, setActiveCategory] = useState<SettingsCategory>('recognition');
  const [testFeedback, setTestFeedback] = useState<string | null>(null);

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

  const handleTestVoice = () => {
    if (!onTestSpeak) return;
    const samplePhrases: Record<string, string> = {
      'en-IN': 'Welcome to SETU. Indian Sign Language bridge is ready.',
      'hi-IN': 'सेतु में आपका स्वागत है। भारतीय सांकेतिक भाषा अनुवाद तैयार है।',
      'ta-IN': 'சேதுவிற்கு வரவேற்கிறோம். சைகை மொழி மொழிபெயர்ப்பு தயாராக உள்ளது.',
      'te-IN': 'సేతుకు స్వాగతం. సైన్ లాంగ్వేజ్ అనువాదం సిద్ధంగా ఉంది.',
      'kn-IN': 'ಸೇತುವಿಗೆ ಸುಸ್ವಾಗತ. ಸಂಜ್ಞಾ ಭಾಷಾ ಅನುವಾದ ಸಿದ್ಧವಾಗಿದೆ.',
    };
    const sample = samplePhrases[targetLanguage] || samplePhrases['en-IN'];
    onTestSpeak(sample, targetLanguage);
    setTestFeedback('Testing voice audio...');
    setTimeout(() => setTestFeedback(null), 2500);
  };

  return (
    <div className="modal-backdrop" onClick={onClose} role="dialog" aria-modal="true">
      <div className="modal-container antigravity-settings-modal" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="modal-header">
          <div className="modal-title-wrap">
            <div className="settings-header-icon-box">
              <Sliders size={20} className="text-cyan" />
            </div>
            <div>
              <h2>Anti-Gravity Studio Settings</h2>
              <p className="modal-subtitle-text">Configure recognition models, camera, audio, and display preferences</p>
            </div>
          </div>
          <button className="modal-close-btn" onClick={onClose} aria-label="Close dialog">
            <X size={20} />
          </button>
        </div>

        {/* 4 Clean Categories Tab Navigation */}
        <div className="antigravity-settings-nav" role="tablist" aria-label="Settings Categories">
          <button
            type="button"
            role="tab"
            aria-selected={activeCategory === 'recognition'}
            className={`antigravity-nav-btn ${activeCategory === 'recognition' ? 'active' : ''}`}
            onClick={() => setActiveCategory('recognition')}
          >
            <ShieldCheck size={16} />
            <span>1. Recognition & Gating</span>
          </button>

          <button
            type="button"
            role="tab"
            aria-selected={activeCategory === 'camera'}
            className={`antigravity-nav-btn ${activeCategory === 'camera' ? 'active' : ''}`}
            onClick={() => setActiveCategory('camera')}
          >
            <Camera size={16} />
            <span>2. Camera & Feed</span>
          </button>

          <button
            type="button"
            role="tab"
            aria-selected={activeCategory === 'speech'}
            className={`antigravity-nav-btn ${activeCategory === 'speech' ? 'active' : ''}`}
            onClick={() => setActiveCategory('speech')}
          >
            <Volume2 size={16} />
            <span>3. Voice & Indic Audio</span>
          </button>

          <button
            type="button"
            role="tab"
            aria-selected={activeCategory === 'display'}
            className={`antigravity-nav-btn ${activeCategory === 'display' ? 'active' : ''}`}
            onClick={() => setActiveCategory('display')}
          >
            <Eye size={16} />
            <span>4. Display & Interface</span>
          </button>
        </div>

        {/* Modal Body / Tab Content */}
        <div className="modal-body antigravity-settings-body">
          {/* TAB 1: Recognition & Gating */}
          {activeCategory === 'recognition' && (
            <div className="settings-section animate-fade-in">
              <div className="section-intro">
                <h3>Recognition Engine & Gating</h3>
                <p>Fine-tune model confidence and choose how signs are inserted into the persistent sentence buffer.</p>
              </div>

              {/* Buffer Insertion Mode: Auto vs Head-Nod */}
              <div className="antigravity-card-setting">
                <div className="card-setting-header">
                  <div className="card-setting-title">
                    <Smile size={18} className="text-cyan" />
                    <h4>Sentence Buffer Insertion Mode</h4>
                  </div>
                  <span className="mode-active-pill">
                    {bufferInsertionMode === 'head_nod' ? '✓ Head-Nod Gated' : '⚡ Auto-Stream'}
                  </span>
                </div>
                <p className="card-setting-desc">
                  Prevent accidental intermediate letters while reaching for a sign.
                </p>

                <div className="insertion-mode-grid">
                  <button
                    type="button"
                    className={`insertion-mode-card ${bufferInsertionMode === 'auto' ? 'active' : ''}`}
                    onClick={() => onSetBufferInsertionMode?.('auto')}
                  >
                    <div className="mode-card-header">
                      <span className="mode-card-name">Auto-Stream (Immediate)</span>
                      <span className="mode-card-badge standard">Standard</span>
                    </div>
                    <p className="mode-card-text">
                      Signs continuously insert into the sentence buffer after holding stably for 2 frames. Best for fast fluent signing.
                    </p>
                  </button>

                  <button
                    type="button"
                    className={`insertion-mode-card ${bufferInsertionMode === 'head_nod' ? 'active' : ''}`}
                    onClick={() => onSetBufferInsertionMode?.('head_nod')}
                  >
                    <div className="mode-card-header">
                      <span className="mode-card-name">Head-Nod Confirmed (Zero-Noise)</span>
                      <span className="mode-card-badge recommended">Recommended</span>
                    </div>
                    <p className="mode-card-text">
                      Signs show live on screen, but <strong>ONLY enter the sentence buffer when you nod your head up and down</strong> (Yes). Shake head (No) to undo/backspace.
                    </p>
                  </button>
                </div>
              </div>

              {/* Discrete Confidence Cards (No Sliders!) */}
              <div className="antigravity-card-setting">
                <div className="card-setting-header">
                  <div className="card-setting-title">
                    <ShieldCheck size={18} className="text-cyan" />
                    <h4>Minimum Confidence Threshold</h4>
                  </div>
                  <span className="confidence-current-badge">
                    {Math.round(confidenceThreshold * 100)}% Selected
                  </span>
                </div>
                <p className="card-setting-desc">
                  Select a calibrated confidence profile. Discrete presets ensure stable, reliable machine learning accuracy.
                </p>

                <div className="confidence-cards-grid">
                  {CONFIDENCE_LEVELS.map((level) => {
                    const isSelected = Math.abs(confidenceThreshold - level.value) < 0.04;
                    return (
                      <button
                        key={level.value}
                        type="button"
                        className={`confidence-card ${isSelected ? 'selected' : ''}`}
                        onClick={() => setConfidenceThreshold(level.value)}
                      >
                        <div className="conf-card-top">
                          <span className="conf-card-label">{level.label}</span>
                          <span className={`conf-card-badge ${level.badgeClass}`}>
                            {level.badge}
                          </span>
                        </div>
                        <p className="conf-card-desc">{level.desc}</p>
                        {isSelected && (
                          <div className="conf-card-check">
                            <CheckCircle2 size={16} />
                            <span>Active</span>
                          </div>
                        )}
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Active Model Selector */}
              {onSwitchModel && (
                <div className="antigravity-card-setting">
                  <div className="card-setting-header">
                    <div className="card-setting-title">
                      <Cpu size={18} className="text-cyan" />
                      <h4>Active Recognition Engine</h4>
                    </div>
                  </div>
                  <div className="model-selector-row">
                    <button
                      type="button"
                      className={`model-pill-btn ${activeModel === 'realsign' ? 'active' : ''}`}
                      onClick={() => onSwitchModel('realsign')}
                    >
                      Alphabet Fingerspelling (A–Z)
                    </button>
                    <button
                      type="button"
                      className={`model-pill-btn ${activeModel === 'isl_mvp' ? 'active' : ''}`}
                      onClick={() => onSwitchModel('isl_mvp')}
                    >
                      Common Signs (5 Phrases)
                    </button>
                    <button
                      type="button"
                      className={`model-pill-btn ${activeModel === 'isl_sequence_lstm' ? 'active' : ''}`}
                      onClick={() => onSwitchModel('isl_sequence_lstm')}
                    >
                      Dynamic Gestures (7 Sequences)
                    </button>
                    <button
                      type="button"
                      className={`model-pill-btn ${activeModel === 'isl_numerals' ? 'active' : ''}`}
                      onClick={() => onSwitchModel('isl_numerals')}
                    >
                      Digits & Currency (0–9 & ₹)
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* TAB 2: Camera & Feed */}
          {activeCategory === 'camera' && (
            <div className="settings-section animate-fade-in">
              <div className="section-intro">
                <h3>Camera & Video Input</h3>
                <p>Manage your input webcam, mirror orientation, and skeletal hand tracking visualization.</p>
              </div>

              {/* Camera Selection */}
              {availableCameras.length > 0 && onSelectCamera && (
                <div className="antigravity-card-setting">
                  <div className="card-setting-header">
                    <div className="card-setting-title">
                      <Video size={18} className="text-cyan" />
                      <h4>Select Video Camera</h4>
                    </div>
                  </div>
                  <select
                    className="antigravity-select"
                    value={selectedCameraId}
                    onChange={(e) => onSelectCamera(e.target.value)}
                  >
                    {availableCameras.map((device, idx) => (
                      <option key={device.deviceId || idx} value={device.deviceId}>
                        {device.label || `Camera ${idx + 1}`}
                      </option>
                    ))}
                  </select>
                </div>
              )}

              {/* Mirror Toggle */}
              <div className="antigravity-toggle-setting">
                <div className="toggle-text-wrap">
                  <h4>Mirror Video Stream</h4>
                  <p>Flips the camera feed horizontally like a physical mirror for natural hand coordination.</p>
                </div>
                <label className="antigravity-switch">
                  <input
                    type="checkbox"
                    checked={landmarkStyles.mirrored}
                    onChange={(e) =>
                      setLandmarkStyles((prev) => ({ ...prev, mirrored: e.target.checked }))
                    }
                  />
                  <span className="slider round" />
                </label>
              </div>

              {/* Landmark Overlays */}
              <div className="antigravity-card-setting">
                <div className="card-setting-header">
                  <div className="card-setting-title">
                    <Eye size={18} className="text-cyan" />
                    <h4>Hand Landmark Visualizations</h4>
                  </div>
                </div>
                <div className="checkbox-row-group">
                  <label className="antigravity-checkbox-label">
                    <input
                      type="checkbox"
                      checked={landmarkStyles.showLeftHand}
                      onChange={(e) =>
                        setLandmarkStyles((prev) => ({ ...prev, showLeftHand: e.target.checked }))
                      }
                    />
                    <span>Show Left Hand Skeleton</span>
                  </label>
                  <label className="antigravity-checkbox-label">
                    <input
                      type="checkbox"
                      checked={landmarkStyles.showRightHand}
                      onChange={(e) =>
                        setLandmarkStyles((prev) => ({ ...prev, showRightHand: e.target.checked }))
                      }
                    />
                    <span>Show Right Hand Skeleton</span>
                  </label>
                </div>
              </div>
            </div>
          )}

          {/* TAB 3: Voice & Indic Audio */}
          {activeCategory === 'speech' && (
            <div className="settings-section animate-fade-in">
              <div className="section-intro">
                <h3>Voice Output & Indic Speech</h3>
                <p>Customize natural speech synthesis for hearing conversation partners.</p>
              </div>

              {/* Auto Speak Toggle */}
              <div className="antigravity-toggle-setting">
                <div className="toggle-text-wrap">
                  <h4>Automatic Text-to-Speech (Auto-Speak)</h4>
                  <p>Speaks recognized signs and completed sentences out loud automatically.</p>
                </div>
                <label className="antigravity-switch">
                  <input
                    type="checkbox"
                    checked={autoSpeakEnabled}
                    onChange={() => onToggleAutoSpeak?.()}
                  />
                  <span className="slider round" />
                </label>
              </div>

              {/* Target Indic Language */}
              {onSelectLanguage && (
                <div className="antigravity-card-setting">
                  <div className="card-setting-header">
                    <div className="card-setting-title">
                      <Languages size={18} className="text-cyan" />
                      <h4>Target Spoken Language</h4>
                    </div>
                  </div>
                  <div className="language-pills-grid">
                    {SUPPORTED_LANGUAGES.map((lang) => (
                      <button
                        key={lang.code}
                        type="button"
                        className={`lang-pill-btn ${targetLanguage === lang.code ? 'active' : ''}`}
                        onClick={() => onSelectLanguage(lang.code)}
                      >
                        <span className="lang-native">{lang.nativeName}</span>
                        <span className="lang-name">{lang.label}</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Test Audio Output */}
              {onTestSpeak && (
                <div className="antigravity-card-setting">
                  <div className="card-setting-header">
                    <div className="card-setting-title">
                      <Volume2 size={18} className="text-cyan" />
                      <h4>Audio Output Verification</h4>
                    </div>
                  </div>
                  <p className="card-setting-desc">
                    Test your browser's speech synthesis engine in the selected language.
                  </p>
                  <button
                    type="button"
                    className="test-voice-btn"
                    onClick={handleTestVoice}
                    disabled={isSpeaking}
                  >
                    <Play size={16} />
                    <span>{testFeedback || 'Test Voice in Selected Language'}</span>
                  </button>
                </div>
              )}
            </div>
          )}

          {/* TAB 4: Display & Interface */}
          {activeCategory === 'display' && (
            <div className="settings-section animate-fade-in">
              <div className="section-intro">
                <h3>Display & Subtitle Formatting</h3>
                <p>Configure on-screen subtitles, framing helpers, and visual accessibility.</p>
              </div>

              {/* Subtitle Size */}
              <div className="antigravity-card-setting">
                <div className="card-setting-header">
                  <div className="card-setting-title">
                    <h4>Subtitle Banner Size</h4>
                  </div>
                </div>
                <div className="size-selector-row">
                  {(['small', 'medium', 'large'] as SubtitleSize[]).map((size) => (
                    <button
                      key={size}
                      type="button"
                      className={`size-pill-btn ${subtitleSize === size ? 'active' : ''}`}
                      onClick={() => setSubtitleSize(size)}
                    >
                      {size.charAt(0).toUpperCase() + size.slice(1)}
                    </button>
                  ))}
                </div>
              </div>

              {/* Framing Guide Toggle */}
              {onToggleFramingGuide && (
                <div className="antigravity-toggle-setting">
                  <div className="toggle-text-wrap">
                    <h4>Visual Signing Guide Overlay</h4>
                    <p>Displays an outline box showing optimal signing space in front of the chest.</p>
                  </div>
                  <label className="antigravity-switch">
                    <input
                      type="checkbox"
                      checked={showFramingGuide}
                      onChange={() => onToggleFramingGuide()}
                    />
                    <span className="slider round" />
                  </label>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="modal-footer antigravity-modal-footer">
          <div className="footer-status-text">
            <span>SETU Anti-Gravity Engine • Zero-Lag 30 FPS Active</span>
          </div>
          <button className="antigravity-done-btn" onClick={onClose}>
            Done
          </button>
        </div>
      </div>
    </div>
  );
};
