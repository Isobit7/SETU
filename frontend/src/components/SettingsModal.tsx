import React, { useState, useEffect } from 'react';
import {
  X,
  Camera,
  SlidersHorizontal,
  Volume2,
  Type,
  Check,
  Play,
} from 'lucide-react';
import type { ActiveModelType, BufferInsertionMode, LandmarkStyles, SubtitleSize } from '../types';
import { SUPPORTED_LANGUAGES } from '../hooks/useTTS';

export type SettingsCategory = 'camera' | 'recognition' | 'speech' | 'display';

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

const SENSITIVITY_OPTIONS = [
  { value: 0.25, label: 'Fast (25%)', hint: 'Quick response for continuous fingerspelling' },
  { value: 0.35, label: 'Balanced (35%)', hint: 'Recommended for everyday signing' },
  { value: 0.50, label: 'Accurate (50%)', hint: 'Higher precision for formal signing' },
  { value: 0.70, label: 'Strict (70%)', hint: 'Requires deliberate, steady holds' },
];

export const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  activeModel = 'realsign',
  onSwitchModel,
  confidenceThreshold,
  setConfidenceThreshold,
  bufferInsertionMode = 'head_nod',
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
  enforceFraming = false,
  onToggleEnforceFraming,
  showFramingGuide = false,
  onToggleFramingGuide,
}) => {
  const [activeCategory, setActiveCategory] = useState<SettingsCategory>('camera');
  const [testFeedback, setTestFeedback] = useState<string | null>(null);

  useEffect(() => {
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
      'en-IN': 'Welcome to SETU. Sign language translation is active.',
      'hi-IN': 'सेतु में आपका स्वागत है। सांकेतिक भाषा अनुवाद तैयार है।',
      'ta-IN': 'சேதுவிற்கு வரவேற்கிறோம். சைகை மொழி மொழிபெயர்ப்பு தயாராக உள்ளது.',
      'te-IN': 'సేతుకు స్వాగతం. సైన్ లాంగ్వేజ్ అనువాదం సిద్ధంగా ఉంది.',
      'kn-IN': 'ಸೇತುವಿಗೆ ಸುಸ್ವಾಗತ. ಸಂಜ್ಞಾ ಭಾಷಾ ಅನುವಾದ ಸಿದ್ಧವಾಗಿದೆ.',
    };
    const sample = samplePhrases[targetLanguage] || samplePhrases['en-IN'];
    onTestSpeak(sample, targetLanguage);
    setTestFeedback('Playing audio...');
    setTimeout(() => setTestFeedback(null), 2500);
  };

  return (
    <div className="modal-backdrop" onClick={onClose} role="dialog" aria-modal="true" aria-labelledby="settings-title">
      <div className="modal-container standard-settings-modal" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="settings-modal-header">
          <div>
            <h2 id="settings-title" className="settings-modal-title">Settings</h2>
            <p className="settings-modal-subtitle">Manage preferences for video, recognition, speech, and captions.</p>
          </div>
          <button className="settings-close-btn" onClick={onClose} aria-label="Close settings">
            <X size={18} />
          </button>
        </div>

        {/* Categories Tab Navigation */}
        <div className="settings-tabs-bar" role="tablist" aria-label="Settings Categories">
          <button
            type="button"
            role="tab"
            aria-selected={activeCategory === 'camera'}
            className={`settings-tab-item ${activeCategory === 'camera' ? 'active' : ''}`}
            onClick={() => setActiveCategory('camera')}
          >
            <Camera size={15} />
            <span>Video & Camera</span>
          </button>

          <button
            type="button"
            role="tab"
            aria-selected={activeCategory === 'recognition'}
            className={`settings-tab-item ${activeCategory === 'recognition' ? 'active' : ''}`}
            onClick={() => setActiveCategory('recognition')}
          >
            <SlidersHorizontal size={15} />
            <span>Recognition</span>
          </button>

          <button
            type="button"
            role="tab"
            aria-selected={activeCategory === 'speech'}
            className={`settings-tab-item ${activeCategory === 'speech' ? 'active' : ''}`}
            onClick={() => setActiveCategory('speech')}
          >
            <Volume2 size={15} />
            <span>Speech & Audio</span>
          </button>

          <button
            type="button"
            role="tab"
            aria-selected={activeCategory === 'display'}
            className={`settings-tab-item ${activeCategory === 'display' ? 'active' : ''}`}
            onClick={() => setActiveCategory('display')}
          >
            <Type size={15} />
            <span>Captions & Display</span>
          </button>
        </div>

        {/* Settings Body */}
        <div className="settings-modal-content">
          {/* TAB 1: Video & Camera */}
          {activeCategory === 'camera' && (
            <div className="settings-pane" role="tabpanel">
              {/* Camera Selection */}
              {availableCameras.length > 0 && onSelectCamera && (
                <div className="setting-row">
                  <div className="setting-info">
                    <label htmlFor="camera-device-select" className="setting-label">Camera device</label>
                    <p className="setting-help">Select the video input source for hand and gesture tracking.</p>
                  </div>
                  <div className="setting-control">
                    <select
                      id="camera-device-select"
                      className="clean-select"
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
                </div>
              )}

              {/* Mirror Video */}
              <div className="setting-row">
                <div className="setting-info">
                  <span className="setting-label">Mirror video feed</span>
                  <p className="setting-help">Flips the camera horizontally for natural hand-eye coordination.</p>
                </div>
                <div className="setting-control">
                  <label className="clean-switch">
                    <input
                      type="checkbox"
                      checked={landmarkStyles.mirrored}
                      onChange={(e) =>
                        setLandmarkStyles((prev) => ({ ...prev, mirrored: e.target.checked }))
                      }
                    />
                    <span className="clean-switch-track" />
                  </label>
                </div>
              </div>

              {/* Skeleton Overlays */}
              <div className="setting-row">
                <div className="setting-info">
                  <span className="setting-label">Hand landmark overlays</span>
                  <p className="setting-help">Show 21-joint skeletal wireframes on top of the live video.</p>
                </div>
                <div className="setting-control setting-checkboxes">
                  <label className="clean-checkbox-label">
                    <input
                      type="checkbox"
                      checked={landmarkStyles.showLeftHand}
                      onChange={(e) =>
                        setLandmarkStyles((prev) => ({ ...prev, showLeftHand: e.target.checked }))
                      }
                    />
                    <span>Left Hand</span>
                  </label>
                  <label className="clean-checkbox-label">
                    <input
                      type="checkbox"
                      checked={landmarkStyles.showRightHand}
                      onChange={(e) =>
                        setLandmarkStyles((prev) => ({ ...prev, showRightHand: e.target.checked }))
                      }
                    />
                    <span>Right Hand</span>
                  </label>
                </div>
              </div>

              {/* Framing Guide */}
              {onToggleFramingGuide && (
                <div className="setting-row">
                  <div className="setting-info">
                    <span className="setting-label">Signing boundary guide</span>
                    <p className="setting-help">Displays a subtle outline on the video indicating optimal signing space.</p>
                  </div>
                  <div className="setting-control">
                    <label className="clean-switch">
                      <input
                        type="checkbox"
                        checked={showFramingGuide}
                        onChange={() => onToggleFramingGuide()}
                      />
                      <span className="clean-switch-track" />
                    </label>
                  </div>
                </div>
              )}

              {/* Enforce Framing */}
              {onToggleEnforceFraming && (
                <div className="setting-row">
                  <div className="setting-info">
                    <span className="setting-label">Require optimal framing</span>
                    <p className="setting-help">Pause sign recognition when hands or torso are out of the recommended area.</p>
                  </div>
                  <div className="setting-control">
                    <label className="clean-switch">
                      <input
                        type="checkbox"
                        checked={enforceFraming}
                        onChange={() => onToggleEnforceFraming()}
                      />
                      <span className="clean-switch-track" />
                    </label>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* TAB 2: Recognition */}
          {activeCategory === 'recognition' && (
            <div className="settings-pane" role="tabpanel">
              {/* Active Vocabulary */}
              {onSwitchModel && (
                <div className="setting-row">
                  <div className="setting-info">
                    <label htmlFor="model-vocab-select" className="setting-label">Active vocabulary</label>
                    <p className="setting-help">Select which gestures or characters the machine learning engine detects.</p>
                  </div>
                  <div className="setting-control">
                    <select
                      id="model-vocab-select"
                      className="clean-select"
                      value={activeModel}
                      onChange={(e) => onSwitchModel(e.target.value as ActiveModelType)}
                    >
                      <option value="realsign">English Alphabet (A–Z)</option>
                      <option value="isl_numerals">Numbers & Currency (0–9, ₹)</option>
                      <option value="isl_hindi_vowels">Hindi Vowels (अ–ऐ)</option>
                      <option value="dynamic_unified">Unified Dynamic Vocabulary & Phrases (287 Signs)</option>
                    </select>
                  </div>
                </div>
              )}

              {/* Confirmation Method */}
              <div className="setting-row vertical-layout">
                <div className="setting-info">
                  <span className="setting-label">Gesture confirmation method</span>
                  <p className="setting-help">Choose how signs are committed to your composed sentence.</p>
                </div>
                <div className="segmented-control">
                  <button
                    type="button"
                    className={`segment-btn ${bufferInsertionMode === 'head_nod' ? 'active' : ''}`}
                    onClick={() => onSetBufferInsertionMode?.('head_nod')}
                  >
                    <span>Head Nod (Recommended)</span>
                  </button>
                  <button
                    type="button"
                    className={`segment-btn ${bufferInsertionMode === 'auto' ? 'active' : ''}`}
                    onClick={() => onSetBufferInsertionMode?.('auto')}
                  >
                    <span>Immediate (Continuous)</span>
                  </button>
                </div>
                <p className="setting-footnote">
                  {bufferInsertionMode === 'head_nod'
                    ? 'Nod head vertically to confirm sign into sentence. Shake head sideways to delete last sign.'
                    : 'Signs automatically commit after being held steady for 2 consecutive video frames.'}
                </p>
              </div>

              {/* Detection Sensitivity */}
              <div className="setting-row vertical-layout">
                <div className="setting-info">
                  <div className="label-with-value">
                    <span className="setting-label">Detection sensitivity</span>
                    <span className="value-pill">{Math.round(confidenceThreshold * 100)}% Confidence</span>
                  </div>
                  <p className="setting-help">Adjust the balance between recognition responsiveness and filtering out noise.</p>
                </div>
                <div className="segmented-control grid-4">
                  {SENSITIVITY_OPTIONS.map((opt) => {
                    const isSelected = Math.abs(confidenceThreshold - opt.value) < 0.04;
                    return (
                      <button
                        key={opt.value}
                        type="button"
                        className={`segment-btn ${isSelected ? 'active' : ''}`}
                        onClick={() => setConfidenceThreshold(opt.value)}
                      >
                        <span>{opt.label}</span>
                      </button>
                    );
                  })}
                </div>
                <p className="setting-footnote">
                  {SENSITIVITY_OPTIONS.find((o) => Math.abs(confidenceThreshold - o.value) < 0.04)?.hint ||
                    'Custom confidence threshold.'}
                </p>
              </div>
            </div>
          )}

          {/* TAB 3: Speech & Audio */}
          {activeCategory === 'speech' && (
            <div className="settings-pane" role="tabpanel">
              {/* Auto Speak */}
              <div className="setting-row">
                <div className="setting-info">
                  <span className="setting-label">Read signs aloud</span>
                  <p className="setting-help">Automatically speak translated sentences through your speakers.</p>
                </div>
                <div className="setting-control">
                  <label className="clean-switch">
                    <input
                      type="checkbox"
                      checked={autoSpeakEnabled}
                      onChange={() => onToggleAutoSpeak?.()}
                    />
                    <span className="clean-switch-track" />
                  </label>
                </div>
              </div>

              {/* Spoken Language */}
              {onSelectLanguage && (
                <div className="setting-row vertical-layout">
                  <div className="setting-info">
                    <span className="setting-label">Voice output language</span>
                    <p className="setting-help">Select the spoken language and dialect for speech synthesis.</p>
                  </div>
                  <div className="clean-lang-grid">
                    {SUPPORTED_LANGUAGES.map((lang) => {
                      const isSelected = targetLanguage === lang.code;
                      return (
                        <button
                          key={lang.code}
                          type="button"
                          className={`clean-lang-card ${isSelected ? 'active' : ''}`}
                          onClick={() => onSelectLanguage(lang.code)}
                        >
                          <div className="lang-card-main">
                            <span className="lang-native-text">{lang.nativeName}</span>
                            <span className="lang-label-text">{lang.label}</span>
                          </div>
                          {isSelected && <Check size={14} className="lang-check" />}
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Test Voice */}
              {onTestSpeak && (
                <div className="setting-row">
                  <div className="setting-info">
                    <span className="setting-label">Voice test</span>
                    <p className="setting-help">Play a sample audio snippet in the selected language.</p>
                  </div>
                  <div className="setting-control">
                    <button
                      type="button"
                      className="clean-secondary-btn"
                      onClick={handleTestVoice}
                      disabled={isSpeaking}
                    >
                      <Play size={13} />
                      <span>{testFeedback || 'Listen to sample'}</span>
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* TAB 4: Captions & Display */}
          {activeCategory === 'display' && (
            <div className="settings-pane" role="tabpanel">
              {/* Subtitle Size */}
              <div className="setting-row">
                <div className="setting-info">
                  <span className="setting-label">Caption size</span>
                  <p className="setting-help">Text size of the live translated subtitle bar on video.</p>
                </div>
                <div className="setting-control">
                  <div className="segmented-control">
                    {(['small', 'medium', 'large'] as SubtitleSize[]).map((size) => (
                      <button
                        key={size}
                        type="button"
                        className={`segment-btn ${subtitleSize === size ? 'active' : ''}`}
                        onClick={() => setSubtitleSize(size)}
                      >
                        <span>{size.charAt(0).toUpperCase() + size.slice(1)}</span>
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="settings-modal-footer">
          <button className="settings-done-btn" onClick={onClose}>
            Done
          </button>
        </div>
      </div>
    </div>
  );
};

export default SettingsModal;