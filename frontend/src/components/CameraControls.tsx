import React, { useState, useRef, useEffect } from 'react';
import {
  Play,
  Pause,
  Square,
  SwitchCamera,
  Layers,
  Subtitles,
  FlipHorizontal,
  Type,
  BookOpen,
  Volume2,
  VolumeX,
  Focus,
  Video,
  ChevronDown,
  Check,
  Hash,
  Languages,
} from 'lucide-react';
import type { ActiveModelType, CameraStatus, LandmarkStyles } from '../types';

interface CameraControlsProps {
  cameraStatus: CameraStatus;
  availableCameras: MediaDeviceInfo[];
  selectedCameraId: string;
  activeModel?: ActiveModelType;
  onSwitchModel?: (model: ActiveModelType) => void;
  confidenceThreshold?: number;
  onThresholdChange?: (val: number) => void;
  autoSpeak?: boolean;
  onToggleAutoSpeak?: () => void;
  landmarkStyles: LandmarkStyles;
  subtitleVisible: boolean;
  showFramingGuide?: boolean;
  onToggleFramingGuide?: () => void;
  enforceFraming?: boolean;
  onToggleEnforceFraming?: () => void;
  onStart?: () => void;
  onPause: () => void;
  onResume: () => void;
  onStop: () => void;
  onReset: () => void;
  onSwitchCamera: (deviceId: string) => void;
  onToggleLandmarks: () => void;
  onToggleMirror: () => void;
  onToggleSubtitles: () => void;
  onOpenIncludeModal?: () => void;
  onOpenRecordModal?: () => void;
}

export const CameraControls: React.FC<CameraControlsProps> = ({
  cameraStatus,
  availableCameras,
  selectedCameraId,
  activeModel = 'realsign',
  onSwitchModel,
  confidenceThreshold: _confidenceThreshold,
  onThresholdChange: _onThresholdChange,
  autoSpeak = false,
  onToggleAutoSpeak,
  landmarkStyles,
  subtitleVisible,
  showFramingGuide = true,
  onToggleFramingGuide,
  enforceFraming: _enforceFraming,
  onToggleEnforceFraming: _onToggleEnforceFraming,
  onStart: _onStart,
  onPause,
  onResume,
  onStop,
  onReset: _onReset,
  onSwitchCamera,
  onToggleLandmarks,
  onToggleMirror,
  onToggleSubtitles,
}) => {
  const [isViewOptionsOpen, setIsViewOptionsOpen] = useState<boolean>(false);
  const [isModelDropdownOpen, setIsModelDropdownOpen] = useState<boolean>(false);

  const viewOptionsRef = useRef<HTMLDivElement | null>(null);
  const modelDropdownRef = useRef<HTMLDivElement | null>(null);

  const isDetecting = cameraStatus === 'detecting';
  const isPaused = cameraStatus === 'paused';
  const isActive = isDetecting || isPaused;
  const handsActive = landmarkStyles.showLeftHand || landmarkStyles.showRightHand;

  // Close popovers on click outside or escape
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (viewOptionsRef.current && !viewOptionsRef.current.contains(e.target as Node)) {
        setIsViewOptionsOpen(false);
      }
      if (modelDropdownRef.current && !modelDropdownRef.current.contains(e.target as Node)) {
        setIsModelDropdownOpen(false);
      }
    };
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setIsViewOptionsOpen(false);
        setIsModelDropdownOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, []);

  const getActiveModelDisplay = (model: ActiveModelType) => {
    switch (model) {
      case 'realsign':
        return { label: 'Letters (A–Z)', icon: Type, sub: 'RealSign Fingerspelling' };
      case 'isl_numerals':
        return { label: 'Numbers (0–9 & ₹)', icon: Hash, sub: 'ISL Counting & Currency' };
      case 'isl_hindi_vowels':
        return { label: 'Hindi Vowels (अ–ऐ)', icon: Languages, sub: 'Devanagari Swar' };
      case 'isl_mvp':
        return { label: 'Conversational (5 Signs)', icon: BookOpen, sub: 'Everyday Phrases' };
      case 'amrita_phrase':
        return { label: 'Amrita 49 Phrases', icon: BookOpen, sub: 'BiLSTM Temporal Network' };
      case 'isl_sequence_lstm':
        return { label: 'Dynamic Sequences', icon: BookOpen, sub: '7 Gestures LSTM' };
      case 'include_stream':
        return { label: 'INCLUDE (263 Signs)', icon: Video, sub: 'AI4Bharat Full Vocabulary' };
      default:
        return { label: 'Letters (A–Z)', icon: Type, sub: 'Alphabet Classifier' };
    }
  };

  const activeDisplay = getActiveModelDisplay(activeModel);
  const ActiveIcon = activeDisplay.icon;

  return (
    <div className="camera-controls-wrapper">
      <div className="controls-bar minimalist-bar" role="toolbar" aria-label="Essential Camera Controls">
        {/* 1. Stream Control: Stop & Pause / Resume */}
        {isActive ? (
          <div className="control-group primary-stream-action">
            <button
              className="ctrl-btn stop-btn"
              onClick={onStop}
              title="Stop camera and recognition"
            >
              <Square size={14} />
              <span>Stop</span>
            </button>
            {isPaused ? (
              <button
                className="ctrl-btn resume-btn"
                onClick={onResume}
                title="Resume live detection"
              >
                <Play size={14} />
                <span>Resume</span>
              </button>
            ) : (
              <button
                className="ctrl-btn pause-btn"
                onClick={onPause}
                title="Pause live detection"
              >
                <Pause size={14} />
                <span>Pause</span>
              </button>
            )}
          </div>
        ) : (
          <div className="control-group camera-standby-indicator">
            <span className="standby-status-dot" />
            <span className="standby-status-text">Standby</span>
          </div>
        )}

        {/* 2. Grouped Model Selector Dropdown (Replaces wide horizontal buttons) */}
        {onSwitchModel && (
          <div className="control-group model-selector-group" ref={modelDropdownRef}>
            <button
              type="button"
              className={`ctrl-btn model-selector-dropdown-trigger ${isModelDropdownOpen ? 'active' : ''}`}
              onClick={() => setIsModelDropdownOpen((prev) => !prev)}
              aria-expanded={isModelDropdownOpen}
              aria-haspopup="true"
              title="Click to switch sign language recognition model"
            >
              <ActiveIcon size={14} className="text-cyan" />
              <span className="model-dropdown-label">
                Mode: <strong className="model-current-name">{activeDisplay.label}</strong>
              </span>
              <ChevronDown size={12} className={`mode-chevron ${isModelDropdownOpen ? 'open' : ''}`} />
            </button>

            {isModelDropdownOpen && (
              <div className="model-dropdown-menu" role="menu">
                {/* Section A: Characters & Fingerspelling */}
                <div className="model-dropdown-header">Fingerspelling & Characters</div>

                <button
                  type="button"
                  role="menuitem"
                  className={`model-menu-item ${activeModel === 'realsign' ? 'selected' : ''}`}
                  onClick={() => {
                    onSwitchModel('realsign');
                    setIsModelDropdownOpen(false);
                  }}
                >
                  <Type size={14} className="menu-item-icon text-cyan" />
                  <div className="menu-item-text">
                    <span className="menu-item-title">Letters (A–Z)</span>
                    <span className="menu-item-desc">RealSign static & continuous fingerspelling</span>
                  </div>
                  {activeModel === 'realsign' && <Check size={14} className="text-cyan menu-item-check" />}
                </button>

                <button
                  type="button"
                  role="menuitem"
                  className={`model-menu-item ${activeModel === 'isl_numerals' ? 'selected' : ''}`}
                  onClick={() => {
                    onSwitchModel('isl_numerals');
                    setIsModelDropdownOpen(false);
                  }}
                >
                  <Hash size={14} className="menu-item-icon text-emerald" />
                  <div className="menu-item-text">
                    <span className="menu-item-title">Numbers (0–9 & ₹)</span>
                    <span className="menu-item-desc">ISL counting and currency signs</span>
                  </div>
                  {activeModel === 'isl_numerals' && <Check size={14} className="text-cyan menu-item-check" />}
                </button>

                <button
                  type="button"
                  role="menuitem"
                  className={`model-menu-item ${activeModel === 'isl_hindi_vowels' ? 'selected' : ''}`}
                  onClick={() => {
                    onSwitchModel('isl_hindi_vowels');
                    setIsModelDropdownOpen(false);
                  }}
                >
                  <Languages size={14} className="menu-item-icon text-amber" />
                  <div className="menu-item-text">
                    <span className="menu-item-title">Hindi Vowels (अ–ऐ)</span>
                    <span className="menu-item-desc">Devanagari ISL swar phonemes</span>
                  </div>
                  {activeModel === 'isl_hindi_vowels' && <Check size={14} className="text-cyan menu-item-check" />}
                </button>

                {/* Section B: Words & Continuous Phrases */}
                <div className="model-dropdown-header">Words & Continuous Phrases</div>

                <button
                  type="button"
                  role="menuitem"
                  className={`model-menu-item ${activeModel === 'isl_mvp' ? 'selected' : ''}`}
                  onClick={() => {
                    onSwitchModel('isl_mvp');
                    setIsModelDropdownOpen(false);
                  }}
                >
                  <BookOpen size={14} className="menu-item-icon text-indigo" />
                  <div className="menu-item-text">
                    <span className="menu-item-title">Everyday Phrases (5 Signs)</span>
                    <span className="menu-item-desc">Hello, Thank You, Good Morning, I, You</span>
                  </div>
                  {activeModel === 'isl_mvp' && <Check size={14} className="text-cyan menu-item-check" />}
                </button>

                <button
                  type="button"
                  role="menuitem"
                  className={`model-menu-item ${activeModel === 'amrita_phrase' ? 'selected' : ''}`}
                  onClick={() => {
                    onSwitchModel('amrita_phrase');
                    setIsModelDropdownOpen(false);
                  }}
                >
                  <BookOpen size={14} className="menu-item-icon text-violet" />
                  <div className="menu-item-text">
                    <span className="menu-item-title">Amrita 49 Phrases (BiLSTM)</span>
                    <span className="menu-item-desc">Verified 49-phrase temporal deep neural network</span>
                  </div>
                  {activeModel === 'amrita_phrase' && <Check size={14} className="text-cyan menu-item-check" />}
                </button>

                <button
                  type="button"
                  role="menuitem"
                  className={`model-menu-item ${activeModel === 'isl_sequence_lstm' ? 'selected' : ''}`}
                  onClick={() => {
                    onSwitchModel('isl_sequence_lstm');
                    setIsModelDropdownOpen(false);
                  }}
                >
                  <BookOpen size={14} className="menu-item-icon text-pink" />
                  <div className="menu-item-text">
                    <span className="menu-item-title">Dynamic Sequences (7 Gestures)</span>
                    <span className="menu-item-desc">Continuous temporal gestures</span>
                  </div>
                  {activeModel === 'isl_sequence_lstm' && <Check size={14} className="text-cyan menu-item-check" />}
                </button>

                <button
                  type="button"
                  role="menuitem"
                  className={`model-menu-item ${activeModel === 'include_stream' ? 'selected' : ''}`}
                  onClick={() => {
                    onSwitchModel('include_stream');
                    setIsModelDropdownOpen(false);
                  }}
                >
                  <Video size={14} className="menu-item-icon text-rose" />
                  <div className="menu-item-text">
                    <span className="menu-item-title">AI4Bharat INCLUDE Full Vocabulary</span>
                    <span className="menu-item-desc">263 comprehensive ISL gestures</span>
                  </div>
                  {activeModel === 'include_stream' && <Check size={14} className="text-cyan menu-item-check" />}
                </button>
              </div>
            )}
          </div>
        )}

        {/* 3. Right Group: View Options Popover (Skeletons, Mirror, Subtitles) */}
        <div className="control-group right-controls-group">
          <div className="view-options-wrapper" ref={viewOptionsRef}>
            <button
              type="button"
              className={`ctrl-btn options-toggle-btn ${isViewOptionsOpen ? 'active' : ''}`}
              onClick={() => setIsViewOptionsOpen((prev) => !prev)}
              title="Camera view overlays, skeletons, and mirror settings"
              aria-expanded={isViewOptionsOpen}
              aria-haspopup="true"
            >
              <Layers size={14} />
              <span>View Options</span>
              <ChevronDown size={11} className={`mode-chevron ${isViewOptionsOpen ? 'open' : ''}`} />
            </button>

            {isViewOptionsOpen && (
              <div className="view-options-popover" role="dialog" aria-label="Camera View Options">
                <div className="popover-header">
                  <span className="popover-title">View & Display Overlays</span>
                </div>

                <div className="popover-body">
                  {/* Hand Skeleton Toggle */}
                  <label className="popover-toggle-item">
                    <div className="popover-toggle-info">
                      <Layers size={14} className="text-muted" />
                      <span>Hand Landmark Skeletons</span>
                    </div>
                    <input
                      type="checkbox"
                      checked={handsActive}
                      onChange={onToggleLandmarks}
                      className="popover-checkbox"
                    />
                  </label>

                  {/* Mirror Preview Toggle */}
                  <label className="popover-toggle-item">
                    <div className="popover-toggle-info">
                      <FlipHorizontal size={14} className="text-muted" />
                      <span>Mirror Preview</span>
                    </div>
                    <input
                      type="checkbox"
                      checked={landmarkStyles.mirrored}
                      onChange={onToggleMirror}
                      className="popover-checkbox"
                    />
                  </label>

                  {/* Subtitles Banner Toggle */}
                  <label className="popover-toggle-item">
                    <div className="popover-toggle-info">
                      <Subtitles size={14} className="text-muted" />
                      <span>Live Subtitles Banner</span>
                    </div>
                    <input
                      type="checkbox"
                      checked={subtitleVisible}
                      onChange={onToggleSubtitles}
                      className="popover-checkbox"
                    />
                  </label>

                  {/* Visual Framing Guide Toggle */}
                  {onToggleFramingGuide && (
                    <label className="popover-toggle-item">
                      <div className="popover-toggle-info">
                        <Focus size={14} className="text-muted" />
                        <span>Visual Framing Guide</span>
                      </div>
                      <input
                        type="checkbox"
                        checked={showFramingGuide}
                        onChange={onToggleFramingGuide}
                        className="popover-checkbox"
                      />
                    </label>
                  )}

                  {/* Auto-Speak Toggle */}
                  {onToggleAutoSpeak && (
                    <label className="popover-toggle-item">
                      <div className="popover-toggle-info">
                        {autoSpeak ? <Volume2 size={14} className="text-cyan" /> : <VolumeX size={14} className="text-muted" />}
                        <span>Auto-Speak Sentences</span>
                      </div>
                      <input
                        type="checkbox"
                        checked={autoSpeak}
                        onChange={onToggleAutoSpeak}
                        className="popover-checkbox"
                      />
                    </label>
                  )}

                  {/* Camera Selection (if multiple devices present) */}
                  {availableCameras.length > 1 && (
                    <div className="popover-camera-select-row">
                      <div className="popover-toggle-info">
                        <SwitchCamera size={14} className="text-muted" />
                        <span>Camera Device</span>
                      </div>
                      <select
                        className="popover-select"
                        value={selectedCameraId}
                        onChange={(e) => onSwitchCamera(e.target.value)}
                        aria-label="Select Camera Device"
                      >
                        {availableCameras.map((device, idx) => (
                          <option key={device.deviceId} value={device.deviceId}>
                            {device.label || `Camera ${idx + 1}`}
                          </option>
                        ))}
                      </select>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default CameraControls;
