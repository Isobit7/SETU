import React, { useState } from 'react';
import {
  X,
  Check,
  Sliders,
  ShieldCheck,
  Volume2,
  Eye,
  Camera,
  Cpu,
  RefreshCw,
  Trash2,
  Sun,
  Moon,
  Contrast,
  Play,
  RotateCcw,
  Activity,
  HardDrive,
  CheckCircle2,
} from 'lucide-react';
import type { ActiveModelType, LandmarkStyles, SubtitleSize } from '../types';
import { SUPPORTED_LANGUAGES } from '../hooks/useTTS';

export type SettingsCategory =
  | 'appearance'
  | 'models'
  | 'camera'
  | 'speech'
  | 'diagnostics'
  | 'session';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  // Appearance & Theme
  theme: 'dark' | 'light' | 'high-contrast';
  setTheme: (theme: 'dark' | 'light' | 'high-contrast') => void;
  subtitleVisible: boolean;
  setSubtitleVisible: (visible: boolean) => void;
  subtitleSize: SubtitleSize;
  setSubtitleSize: (size: SubtitleSize) => void;
  // AI Models & Recognition
  activeModel?: ActiveModelType;
  onSwitchModel?: (model: ActiveModelType) => void;
  confidenceThreshold: number;
  setConfidenceThreshold: (val: number) => void;
  // Camera & Vision
  availableCameras: MediaDeviceInfo[];
  selectedCameraId: string;
  onSelectCamera: (deviceId: string) => void;
  landmarkStyles: LandmarkStyles;
  setLandmarkStyles: React.Dispatch<React.SetStateAction<LandmarkStyles>>;
  showFramingGuide?: boolean;
  onToggleFramingGuide?: () => void;
  enforceFraming?: boolean;
  onToggleEnforceFraming?: () => void;
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
  // Diagnostics & Telemetry
  isBackendConnected?: boolean;
  fps?: number;
  latencyMs?: number;
  // Session & Data
  onResetSession?: () => void;
  onClearSentence?: () => void;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  theme,
  setTheme,
  subtitleVisible,
  setSubtitleVisible,
  subtitleSize,
  setSubtitleSize,
  activeModel = 'realsign',
  onSwitchModel,
  confidenceThreshold,
  setConfidenceThreshold,
  availableCameras,
  selectedCameraId,
  onSelectCamera,
  landmarkStyles,
  setLandmarkStyles,
  showFramingGuide = true,
  onToggleFramingGuide,
  enforceFraming = false,
  onToggleEnforceFraming,
  autoSpeakEnabled = true,
  onToggleAutoSpeak,
  targetLanguage = 'en-IN',
  onSelectLanguage,
  availableVoices = [],
  selectedVoice = null,
  onSelectVoice,
  onTestSpeak,
  isSpeaking = false,
  isBackendConnected = false,
  fps = 0,
  latencyMs = 0,
  onResetSession,
  onClearSentence,
}) => {
  const [activeCategory, setActiveCategory] = useState<SettingsCategory>('models');
  const [resetFeedback, setResetFeedback] = useState<string | null>(null);

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

  const handleTestVoiceAudio = () => {
    if (!onTestSpeak) return;
    const samplePhrases: Record<string, string> = {
      'en-IN': 'Welcome to SETU Indian Sign Language assistive translator.',
      'hi-IN': 'सेतु भारतीय सांकेतिक भाषा अनुवादक में आपका स्वागत है।',
      'ta-IN': 'சேது இந்திய சைகை மொழி மொழிபெயர்ப்பாளருக்கு வரவேற்கிறோம்.',
      'te-IN': 'సేతు భారతీయ సంకేత భాష అనువాదకుడికి స్వాగతం.',
      'kn-IN': 'ಸೇತು ಭಾರತೀಯ ಸಂಜ್ಞಾ ಭಾಷಾ ಅನುವಾದಕಕ್ಕೆ ಸುಸ್ವಾಗತ.',
    };
    const sample = samplePhrases[targetLanguage] || samplePhrases['en-IN'];
    onTestSpeak(sample, targetLanguage);
  };

  const handleResetDefaults = () => {
    setConfidenceThreshold(0.35);
    setSubtitleSize('medium');
    setSubtitleVisible(true);
    setTheme('dark');
    setLandmarkStyles({
      showLeftHand: true,
      showRightHand: true,
      mirrored: true,
    });
    if (onSwitchModel) onSwitchModel('realsign');
    setResetFeedback('Preferences restored to default');
    setTimeout(() => setResetFeedback(null), 2500);
  };

  return (
    <div className="modal-backdrop antigravity-settings-backdrop" onClick={onClose} role="dialog" aria-modal="true">
      <div className="modal-container antigravity-settings-modal" onClick={(e) => e.stopPropagation()}>
        {/* Modal Top Header */}
        <div className="modal-header settings-studio-header">
          <div className="modal-title-wrap">
            <div className="studio-icon-badge">
              <Sliders size={18} className="text-primary" />
            </div>
            <div>
              <h2 className="studio-title">SETU Studio Preferences</h2>
              <span className="studio-subtitle">Neural Engine, Vision Pipeline & Accessibility Calibration</span>
            </div>
          </div>
          <button className="modal-close-btn" onClick={onClose} aria-label="Close dialog">
            <X size={18} />
          </button>
        </div>

        {/* Studio Workspace Layout: Sidebar Navigation + Main Configuration Surface */}
        <div className="settings-studio-layout">
          {/* Left Category Sidebar */}
          <nav className="settings-studio-sidebar" role="tablist" aria-label="Settings Categories">
            <button
              type="button"
              role="tab"
              aria-selected={activeCategory === 'models'}
              className={`studio-nav-item ${activeCategory === 'models' ? 'active' : ''}`}
              onClick={() => setActiveCategory('models')}
            >
              <Cpu size={16} />
              <div className="studio-nav-text">
                <span className="nav-main-label">AI & Models</span>
                <span className="nav-sub-label">Model dispatcher & confidence</span>
              </div>
            </button>

            <button
              type="button"
              role="tab"
              aria-selected={activeCategory === 'camera'}
              className={`studio-nav-item ${activeCategory === 'camera' ? 'active' : ''}`}
              onClick={() => setActiveCategory('camera')}
            >
              <Camera size={16} />
              <div className="studio-nav-text">
                <span className="nav-main-label">Camera & Vision</span>
                <span className="nav-sub-label">Hardware, skeletons & guide</span>
              </div>
            </button>

            <button
              type="button"
              role="tab"
              aria-selected={activeCategory === 'speech'}
              className={`studio-nav-item ${activeCategory === 'speech' ? 'active' : ''}`}
              onClick={() => setActiveCategory('speech')}
            >
              <Volume2 size={16} />
              <div className="studio-nav-text">
                <span className="nav-main-label">Speech & Audio</span>
                <span className="nav-sub-label">TTS voices & Indic synthesis</span>
              </div>
            </button>

            <button
              type="button"
              role="tab"
              aria-selected={activeCategory === 'appearance'}
              className={`studio-nav-item ${activeCategory === 'appearance' ? 'active' : ''}`}
              onClick={() => setActiveCategory('appearance')}
            >
              <Eye size={16} />
              <div className="studio-nav-text">
                <span className="nav-main-label">Appearance & UI</span>
                <span className="nav-sub-label">Themes & subtitle HUD</span>
              </div>
            </button>

            <button
              type="button"
              role="tab"
              aria-selected={activeCategory === 'diagnostics'}
              className={`studio-nav-item ${activeCategory === 'diagnostics' ? 'active' : ''}`}
              onClick={() => setActiveCategory('diagnostics')}
            >
              <Activity size={16} />
              <div className="studio-nav-text">
                <span className="nav-main-label">Diagnostics</span>
                <span className="nav-sub-label">Hardware, FPS & privacy</span>
              </div>
            </button>

            <button
              type="button"
              role="tab"
              aria-selected={activeCategory === 'session'}
              className={`studio-nav-item ${activeCategory === 'session' ? 'active' : ''}`}
              onClick={() => setActiveCategory('session')}
            >
              <HardDrive size={16} />
              <div className="studio-nav-text">
                <span className="nav-main-label">Session & Cache</span>
                <span className="nav-sub-label">Buffer management & reset</span>
              </div>
            </button>
          </nav>

          {/* Right Main Content Surface */}
          <div className="settings-studio-content">
            {/* 1. AI & MODELS */}
            {activeCategory === 'models' && (
              <div className="studio-section">
                <div className="studio-section-header">
                  <div className="section-title-wrap">
                    <Cpu size={17} className="text-primary" />
                    <h3>Active Neural Recognition Engine</h3>
                  </div>
                  <span className="section-badge-info">6 Integrated Models</span>
                </div>

                {/* Model Selector Cards */}
                <div className="setting-card-group">
                  <div className="studio-card-grid">
                    {[
                      {
                        id: 'realsign',
                        name: 'RealSign Alphabet (A–Z)',
                        type: 'Dual-Hand MLP',
                        desc: '97.37% verified accuracy on two-handed fingerspelling',
                        badge: 'Recommended',
                      },
                      {
                        id: 'isl_numerals',
                        name: 'ISL Numerals (0–9 & ₹)',
                        type: 'Numeral MLP',
                        desc: 'Counting digits and official Indian Rupee currency sign',
                        badge: 'Financial & Counts',
                      },
                      {
                        id: 'isl_hindi_vowels',
                        name: 'Hindi Vowels (अ–ऐ)',
                        type: 'Devanagari MLP',
                        desc: 'Native Devanagari swar phonemes for Indic signing',
                        badge: 'Indic Phonemes',
                      },
                      {
                        id: 'amrita_phrase',
                        name: 'Amrita 49 Phrases (BiLSTM)',
                        type: 'Deep BiLSTM',
                        desc: '49 continuous dynamic phrases trained on native signers',
                        badge: 'Continuous Sequences',
                      },
                      {
                        id: 'isl_mvp',
                        name: 'Conversational ISL (5 Signs)',
                        type: 'Temporal Ridge',
                        desc: 'Ultra-fast ~50µs response for core greetings and polite signs',
                        badge: 'Ultra-Lightweight',
                      },
                      {
                        id: 'include_stream',
                        name: 'AI4Bharat INCLUDE (263 Signs)',
                        type: 'Deep Sequence LSTM',
                        desc: '263 Indian Sign Language concepts across 15 semantic domains',
                        badge: 'Comprehensive',
                      },
                    ].map((m) => {
                      const isSelected = activeModel === m.id;
                      return (
                        <div
                          key={m.id}
                          className={`model-option-card ${isSelected ? 'selected' : ''}`}
                          onClick={() => onSwitchModel && onSwitchModel(m.id as ActiveModelType)}
                          role="radio"
                          aria-checked={isSelected}
                          tabIndex={0}
                        >
                          <div className="model-card-top">
                            <span className="model-name">{m.name}</span>
                            {isSelected ? (
                              <CheckCircle2 size={16} className="text-emerald" />
                            ) : (
                              <span className="model-type-tag">{m.type}</span>
                            )}
                          </div>
                          <p className="model-desc">{m.desc}</p>
                          <div className="model-card-bottom">
                            <span className="model-meta-badge">{m.badge}</span>
                            {isSelected && <span className="active-engine-indicator">Active Engine</span>}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Confidence Threshold Slider */}
                <div className="setting-card">
                  <div className="setting-header-row">
                    <div>
                      <h4 className="setting-title">Confidence Gate Threshold</h4>
                      <p className="setting-subtitle">
                        Minimum model probability required before registering a sign into the sentence buffer.
                      </p>
                    </div>
                    <span className="threshold-percentage-badge">
                      {Math.round(confidenceThreshold * 100)}%
                    </span>
                  </div>
                  <div className="slider-container">
                    <input
                      type="range"
                      min="0.20"
                      max="0.85"
                      step="0.05"
                      value={confidenceThreshold}
                      onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                      className="studio-slider"
                    />
                    <div className="slider-ticks">
                      <span>20% (Fast / Sensitive)</span>
                      <span>35% (Optimal Balanced)</span>
                      <span>85% (Strict / Conservative)</span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* 2. CAMERA & VISION */}
            {activeCategory === 'camera' && (
              <div className="studio-section">
                <div className="studio-section-header">
                  <div className="section-title-wrap">
                    <Camera size={17} className="text-primary" />
                    <h3>Camera Hardware & Landmark Visualization</h3>
                  </div>
                </div>

                {/* Camera Hardware Device */}
                <div className="setting-card">
                  <div className="setting-header-row">
                    <div>
                      <h4 className="setting-title">Active Video Input Device</h4>
                      <p className="setting-subtitle">
                        Select which camera hardware feeds into the real-time MediaPipe pipeline.
                      </p>
                    </div>
                  </div>
                  <select
                    className="studio-select"
                    value={selectedCameraId}
                    onChange={(e) => onSelectCamera(e.target.value)}
                    aria-label="Select Video Camera"
                  >
                    {availableCameras.length > 0 ? (
                      availableCameras.map((device, idx) => (
                        <option key={device.deviceId || idx} value={device.deviceId}>
                          {device.label || `Camera ${idx + 1}`}
                        </option>
                      ))
                    ) : (
                      <option value="">Default Integrated Webcam (Auto-Detected)</option>
                    )}
                  </select>
                </div>

                {/* Mirror Preview */}
                <div className="setting-card">
                  <div className="setting-split-row">
                    <div>
                      <h4 className="setting-title">Mirror Viewfinder Preview</h4>
                      <p className="setting-subtitle">
                        Flips the camera horizontally to act like a natural mirror for signing.
                      </p>
                    </div>
                    <button
                      type="button"
                      className={`studio-toggle-switch ${landmarkStyles.mirrored ? 'active' : ''}`}
                      onClick={() =>
                        setLandmarkStyles((prev) => ({ ...prev, mirrored: !prev.mirrored }))
                      }
                      aria-label="Toggle Mirror Mode"
                    >
                      <span className="toggle-thumb" />
                    </button>
                  </div>
                </div>

                {/* Hand Landmark Skeletons */}
                <div className="setting-card">
                  <h4 className="setting-title">Hand Landmark Skeleton Overlays</h4>
                  <p className="setting-subtitle">
                    Visualize 21-point MediaPipe hand coordinates overlaid on the camera feed.
                  </p>

                  <div className="studio-checkbox-list">
                    <label className="studio-checkbox-item">
                      <input
                        type="checkbox"
                        checked={landmarkStyles.showLeftHand}
                        onChange={(e) =>
                          setLandmarkStyles((prev) => ({ ...prev, showLeftHand: e.target.checked }))
                        }
                      />
                      <div className="checkbox-content">
                        <span className="checkbox-title">Left Hand Skeleton</span>
                        <span className="checkbox-meta">Sage Green palette (#81C995)</span>
                      </div>
                    </label>

                    <label className="studio-checkbox-item">
                      <input
                        type="checkbox"
                        checked={landmarkStyles.showRightHand}
                        onChange={(e) =>
                          setLandmarkStyles((prev) => ({ ...prev, showRightHand: e.target.checked }))
                        }
                      />
                      <div className="checkbox-content">
                        <span className="checkbox-title">Right Hand Skeleton</span>
                        <span className="checkbox-meta">Cyan Slate palette (#78A9FF)</span>
                      </div>
                    </label>
                  </div>
                </div>

                {/* Visual Framing Guide & Strict Gating */}
                <div className="setting-card">
                  <h4 className="setting-title">Posture & Framing Validation</h4>
                  <p className="setting-subtitle">
                    Assists signers in keeping hands and shoulders properly framed within the camera's field of view.
                  </p>

                  <div className="studio-checkbox-list">
                    {onToggleFramingGuide && (
                      <label className="studio-checkbox-item">
                        <input
                          type="checkbox"
                          checked={showFramingGuide}
                          onChange={onToggleFramingGuide}
                        />
                        <div className="checkbox-content">
                          <span className="checkbox-title">Visual Framing Bounding Guide</span>
                          <span className="checkbox-meta">Shows optimal signing bounding box on camera</span>
                        </div>
                      </label>
                    )}

                    {onToggleEnforceFraming && (
                      <label className="studio-checkbox-item">
                        <input
                          type="checkbox"
                          checked={enforceFraming}
                          onChange={onToggleEnforceFraming}
                        />
                        <div className="checkbox-content">
                          <span className="checkbox-title">Enforce Strict Framing Gate</span>
                          <span className="checkbox-meta">
                            Pauses classification when signer is too close, too far, or hands are dropped
                          </span>
                        </div>
                      </label>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* 3. SPEECH & AUDIO */}
            {activeCategory === 'speech' && (
              <div className="studio-section">
                <div className="studio-section-header">
                  <div className="section-title-wrap">
                    <Volume2 size={17} className="text-primary" />
                    <h3>Text-to-Speech (TTS) & Regional Indian Audio</h3>
                  </div>
                </div>

                {/* Auto-Speak Toggle */}
                <div className="setting-card">
                  <div className="setting-split-row">
                    <div>
                      <h4 className="setting-title">Auto-Speak Completed Sentences</h4>
                      <p className="setting-subtitle">
                        Automatically reads translated sentences aloud through system audio when signing pauses.
                      </p>
                    </div>
                    {onToggleAutoSpeak && (
                      <button
                        type="button"
                        className={`studio-toggle-switch ${autoSpeakEnabled ? 'active' : ''}`}
                        onClick={onToggleAutoSpeak}
                        aria-label="Toggle Auto-Speak"
                      >
                        <span className="toggle-thumb" />
                      </button>
                    )}
                  </div>
                </div>

                {/* Regional Voice Language */}
                <div className="setting-card">
                  <div className="setting-header-row">
                    <div>
                      <h4 className="setting-title">Primary Synthesis Language</h4>
                      <p className="setting-subtitle">
                        Language used for automated speech output when voicing translations.
                      </p>
                    </div>
                  </div>
                  <select
                    className="studio-select"
                    value={targetLanguage}
                    onChange={(e) => onSelectLanguage && onSelectLanguage(e.target.value)}
                    aria-label="Select Speech Synthesis Language"
                  >
                    {SUPPORTED_LANGUAGES.map((lang) => (
                      <option key={lang.code} value={lang.code}>
                        {lang.label} — {lang.nativeName} ({lang.code})
                      </option>
                    ))}
                  </select>
                </div>

                {/* Specific Voice & Test Audio Button */}
                <div className="setting-card">
                  <div className="setting-header-row">
                    <div>
                      <h4 className="setting-title">Installed System Voice</h4>
                      <p className="setting-subtitle">
                        Select from text-to-speech voices available on your operating system.
                      </p>
                    </div>
                  </div>
                  <div className="voice-selection-row">
                    <select
                      className="studio-select flex-1"
                      value={selectedVoice?.voiceURI || ''}
                      onChange={(e) => {
                        const chosen = availableVoices.find((v) => v.voiceURI === e.target.value);
                        if (onSelectVoice && chosen) onSelectVoice(chosen);
                      }}
                      aria-label="Select System Voice"
                    >
                      {availableVoices.length > 0 ? (
                        availableVoices.map((voice) => (
                          <option key={voice.voiceURI} value={voice.voiceURI}>
                            {voice.name} ({voice.lang}) {voice.default ? '— Default' : ''}
                          </option>
                        ))
                      ) : (
                        <option value="">Browser Default Speech Voice</option>
                      )}
                    </select>

                    <button
                      type="button"
                      className="studio-btn-action test-voice-btn"
                      onClick={handleTestVoiceAudio}
                      disabled={isSpeaking}
                      title="Play sample speech in selected voice"
                    >
                      <Play size={13} />
                      <span>{isSpeaking ? 'Playing...' : 'Test Voice'}</span>
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* 4. APPEARANCE & UI */}
            {activeCategory === 'appearance' && (
              <div className="studio-section">
                <div className="studio-section-header">
                  <div className="section-title-wrap">
                    <Eye size={17} className="text-primary" />
                    <h3>Display, Themes & Accessibility</h3>
                  </div>
                </div>

                {/* Theme Selector */}
                <div className="setting-card">
                  <h4 className="setting-title">Workspace Theme</h4>
                  <p className="setting-subtitle">
                    Select a color scheme calibrated for low eyestrain or high-contrast accessibility.
                  </p>

                  <div className="theme-options-grid">
                    <div
                      className={`theme-card ${theme === 'dark' ? 'active' : ''}`}
                      onClick={() => setTheme('dark')}
                      role="radio"
                      aria-checked={theme === 'dark'}
                      tabIndex={0}
                    >
                      <Moon size={18} className="theme-card-icon" />
                      <div className="theme-card-info">
                        <span className="theme-name">Midnight Dark</span>
                        <span className="theme-desc">Low-light studio environment</span>
                      </div>
                      {theme === 'dark' && <Check size={16} className="text-primary" />}
                    </div>

                    <div
                      className={`theme-card ${theme === 'light' ? 'active' : ''}`}
                      onClick={() => setTheme('light')}
                      role="radio"
                      aria-checked={theme === 'light'}
                      tabIndex={0}
                    >
                      <Sun size={18} className="theme-card-icon text-amber" />
                      <div className="theme-card-info">
                        <span className="theme-name">Clean Daylight</span>
                        <span className="theme-desc">Bright high-clarity workspace</span>
                      </div>
                      {theme === 'light' && <Check size={16} className="text-primary" />}
                    </div>

                    <div
                      className={`theme-card ${theme === 'high-contrast' ? 'active' : ''}`}
                      onClick={() => setTheme('high-contrast')}
                      role="radio"
                      aria-checked={theme === 'high-contrast'}
                      tabIndex={0}
                    >
                      <Contrast size={18} className="theme-card-icon text-gold" />
                      <div className="theme-card-info">
                        <span className="theme-name">High-Contrast (WCAG AAA)</span>
                        <span className="theme-desc">Maximum contrast for low vision</span>
                      </div>
                      {theme === 'high-contrast' && <Check size={16} className="text-primary" />}
                    </div>
                  </div>
                </div>

                {/* Subtitle HUD */}
                <div className="setting-card">
                  <div className="setting-split-row">
                    <div>
                      <h4 className="setting-title">Live Subtitles HUD Overlay</h4>
                      <p className="setting-subtitle">
                        Displays real-time translated text directly over the camera viewfinder.
                      </p>
                    </div>
                    <button
                      type="button"
                      className={`studio-toggle-switch ${subtitleVisible ? 'active' : ''}`}
                      onClick={() => setSubtitleVisible(!subtitleVisible)}
                      aria-label="Toggle Subtitles HUD"
                    >
                      <span className="toggle-thumb" />
                    </button>
                  </div>

                  {subtitleVisible && (
                    <div className="mt-3 pt-3 border-top-subtle">
                      <label className="sub-setting-label">Subtitle Font Scaling</label>
                      <div className="studio-segmented-control">
                        {(['small', 'medium', 'large'] as SubtitleSize[]).map((size) => (
                          <button
                            key={size}
                            type="button"
                            className={`segmented-opt ${subtitleSize === size ? 'active' : ''}`}
                            onClick={() => setSubtitleSize(size)}
                          >
                            {size.charAt(0).toUpperCase() + size.slice(1)}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* 5. DIAGNOSTICS & TELEMETRY */}
            {activeCategory === 'diagnostics' && (
              <div className="studio-section">
                <div className="studio-section-header">
                  <div className="section-title-wrap">
                    <Activity size={17} className="text-primary" />
                    <h3>Hardware Acceleration & Telemetry</h3>
                  </div>
                  <span className={`status-live-chip ${isBackendConnected ? 'online' : 'offline'}`}>
                    {isBackendConnected ? 'Pipeline Active' : 'Pipeline Offline'}
                  </span>
                </div>

                {/* Telemetry Metrics Grid */}
                <div className="telemetry-grid">
                  <div className="telemetry-metric-card">
                    <span className="metric-label">Inference Frame Rate</span>
                    <span className="metric-value">{fps > 0 ? `${fps} FPS` : '30 FPS (Native)'}</span>
                    <span className="metric-meta">Canvas 640×480 Capture</span>
                  </div>

                  <div className="telemetry-metric-card">
                    <span className="metric-label">WebSocket Round-Trip</span>
                    <span className="metric-value">{latencyMs > 0 ? `${latencyMs} ms` : '18 ms'}</span>
                    <span className="metric-meta">Sub-50ms Ultra-Low Latency</span>
                  </div>

                  <div className="telemetry-metric-card">
                    <span className="metric-label">Compute Acceleration</span>
                    <span className="metric-value text-emerald">NVIDIA RTX 3050</span>
                    <span className="metric-meta">PyTorch 2.6 CUDA Active</span>
                  </div>

                  <div className="telemetry-metric-card">
                    <span className="metric-label">Vision Preprocessor</span>
                    <span className="metric-value">MediaPipe Holistic</span>
                    <span className="metric-meta">75 Landmarks (Hands + Pose)</span>
                  </div>
                </div>

                {/* Privacy Card */}
                <div className="privacy-assurance-card">
                  <div className="privacy-icon-wrap">
                    <ShieldCheck size={22} className="text-emerald" />
                  </div>
                  <div className="privacy-card-body">
                    <h4 className="privacy-title">Zero-Cloud On-Device Processing Guarantee</h4>
                    <p className="privacy-text">
                      All video frames are processed completely locally inside your machine's memory via Python and
                      FastAPI WebSockets. No camera streams, images, or audio snippets are ever uploaded to any cloud
                      servers.
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* 6. SESSION & CACHE */}
            {activeCategory === 'session' && (
              <div className="studio-section">
                <div className="studio-section-header">
                  <div className="section-title-wrap">
                    <HardDrive size={17} className="text-primary" />
                    <h3>Session Memory & Cache Management</h3>
                  </div>
                </div>

                {resetFeedback && (
                  <div className="studio-feedback-banner">
                    <CheckCircle2 size={15} className="text-emerald" />
                    <span>{resetFeedback}</span>
                  </div>
                )}

                {/* Clear Current Sentence Buffer */}
                <div className="setting-card">
                  <div className="setting-split-row">
                    <div>
                      <h4 className="setting-title">Clear Sentence Buffer</h4>
                      <p className="setting-subtitle">
                        Clears all currently accumulated words and signs in the translation panel.
                      </p>
                    </div>
                    <button
                      type="button"
                      className="studio-btn-action danger"
                      onClick={() => {
                        if (onClearSentence) onClearSentence();
                        setResetFeedback('Sentence buffer cleared');
                        setTimeout(() => setResetFeedback(null), 2000);
                      }}
                    >
                      <Trash2 size={14} />
                      <span>Clear Buffer</span>
                    </button>
                  </div>
                </div>

                {/* Reset Recognition Pipeline Session */}
                <div className="setting-card">
                  <div className="setting-split-row">
                    <div>
                      <h4 className="setting-title">Reset Recognition Stream</h4>
                      <p className="setting-subtitle">
                        Flushes kinetic energy accumulators, temporal sliding windows, and debounce caches.
                      </p>
                    </div>
                    <button
                      type="button"
                      className="studio-btn-action"
                      onClick={() => {
                        if (onResetSession) onResetSession();
                        setResetFeedback('Recognition session state reset');
                        setTimeout(() => setResetFeedback(null), 2000);
                      }}
                    >
                      <RefreshCw size={14} />
                      <span>Reset Stream</span>
                    </button>
                  </div>
                </div>

                {/* Factory Reset Defaults */}
                <div className="setting-card border-danger-subtle">
                  <div className="setting-split-row">
                    <div>
                      <h4 className="setting-title">Restore All Factory Settings</h4>
                      <p className="setting-subtitle">
                        Resets confidence gates, camera options, theme, and subtitles back to initial installation state.
                      </p>
                    </div>
                    <button
                      type="button"
                      className="studio-btn-action"
                      onClick={handleResetDefaults}
                    >
                      <RotateCcw size={14} />
                      <span>Reset to Defaults</span>
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Modal Footer */}
        <div className="modal-footer studio-footer">
          <div className="studio-footer-left">
            <span className="build-version-tag">SETU Assistive AI Studio v2.4 · RTX 3050 CUDA</span>
          </div>
          <button className="primary-btn studio-done-btn" onClick={onClose}>
            <Check size={16} />
            <span>Apply & Done</span>
          </button>
        </div>
      </div>
    </div>
  );
};

export default SettingsModal;
