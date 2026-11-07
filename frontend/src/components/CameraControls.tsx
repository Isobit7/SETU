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
  MessageSquare,
  Activity,
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
  onOpenIncludeModal: _onOpenIncludeModal,
  onOpenRecordModal,
}) => {
  const [isViewOptionsOpen, setIsViewOptionsOpen] = useState<boolean>(false);
  const [isModeDropdownOpen, setIsModeDropdownOpen] = useState<boolean>(false);

  const viewOptionsRef = useRef<HTMLDivElement | null>(null);
  const modeDropdownRef = useRef<HTMLDivElement | null>(null);

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
      if (modeDropdownRef.current && !modeDropdownRef.current.contains(e.target as Node)) {
        setIsModeDropdownOpen(false);
      }
    };
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setIsViewOptionsOpen(false);
        setIsModeDropdownOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, []);

  const getModeShortLabel = (model: ActiveModelType): string => {
    switch (model) {
      case 'realsign':
        return 'Alphabet (A–Z)';
      case 'isl_numerals':
        return 'Numbers & Currency';
      case 'isl_hindi_vowels':
        return 'Hindi Vowels';
      case 'isl_mvp':
        return 'Everyday Signs (5)';
      case 'amrita_phrase':
        return 'Phrases (49)';
      case 'isl_sequence_lstm':
        return 'Dynamic Gestures';
      case 'include_stream':
        return 'Dictionary (263)';
      default:
        return 'Alphabet (A–Z)';
    }
  };

  const renderModeIcon = (model: ActiveModelType) => {
    switch (model) {
      case 'realsign':
        return <Type size={13} className="mode-btn-icon text-primary" />;
      case 'isl_numerals':
        return <Hash size={13} className="mode-btn-icon text-primary" />;
      case 'isl_hindi_vowels':
        return <Languages size={13} className="mode-btn-icon text-primary" />;
      case 'isl_mvp':
        return <BookOpen size={13} className="mode-btn-icon text-primary" />;
      case 'amrita_phrase':
        return <MessageSquare size={13} className="mode-btn-icon text-primary" />;
      case 'isl_sequence_lstm':
        return <Activity size={13} className="mode-btn-icon text-primary" />;
      case 'include_stream':
        return <Layers size={13} className="mode-btn-icon text-primary" />;
      default:
        return <Type size={13} className="mode-btn-icon text-primary" />;
    }
  };

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

        {/* 2. Unified Recognition Mode Dropdown */}
        {onSwitchModel && (
          <div className="control-group unified-mode-container" ref={modeDropdownRef}>
            <button
              type="button"
              className={`ctrl-btn unified-mode-trigger-btn ${isModeDropdownOpen ? 'active' : ''}`}
              onClick={() => setIsModeDropdownOpen((prev) => !prev)}
              title="Click to switch active recognition mode and vocabulary"
              aria-haspopup="true"
              aria-expanded={isModeDropdownOpen}
            >
              {renderModeIcon(activeModel)}
              <span className="mode-trigger-label">
                Mode: <strong>{getModeShortLabel(activeModel)}</strong>
              </span>
              <ChevronDown size={11} className={`mode-chevron ${isModeDropdownOpen ? 'open' : ''}`} />
            </button>

            {isModeDropdownOpen && (
              <div className="mode-subdropdown-menu unified-mode-menu" role="menu">
                <div className="dropdown-section-title">Letters & Numbers</div>

                <button
                  type="button"
                  role="menuitem"
                  className={`subdropdown-item ${activeModel === 'realsign' ? 'selected' : ''}`}
                  onClick={() => {
                    onSwitchModel('realsign');
                    setIsModeDropdownOpen(false);
                  }}
                >
                  <Type size={14} className="subdropdown-item-icon text-primary" />
                  <div className="subdropdown-text">
                    <span className="subdropdown-title">Alphabet Fingerspelling (A–Z)</span>
                    <span className="subdropdown-desc">RealSign static & continuous alphabet posture recognition</span>
                  </div>
                  {activeModel === 'realsign' && <Check size={14} className="text-primary check-icon" />}
                </button>

                <button
                  type="button"
                  role="menuitem"
                  className={`subdropdown-item ${activeModel === 'isl_numerals' ? 'selected' : ''}`}
                  onClick={() => {
                    onSwitchModel('isl_numerals');
                    setIsModeDropdownOpen(false);
                  }}
                >
                  <Hash size={14} className="subdropdown-item-icon text-primary" />
                  <div className="subdropdown-text">
                    <span className="subdropdown-title">Numbers & Currency (0–9 & ₹)</span>
                    <span className="subdropdown-desc">Indian sign language counting digits and rupee currency</span>
                  </div>
                  {activeModel === 'isl_numerals' && <Check size={14} className="text-primary check-icon" />}
                </button>

                <button
                  type="button"
                  role="menuitem"
                  className={`subdropdown-item ${activeModel === 'isl_hindi_vowels' ? 'selected' : ''}`}
                  onClick={() => {
                    onSwitchModel('isl_hindi_vowels');
                    setIsModeDropdownOpen(false);
                  }}
                >
                  <Languages size={14} className="subdropdown-item-icon text-primary" />
                  <div className="subdropdown-text">
                    <span className="subdropdown-title">Hindi Vowels (अ–ऐ)</span>
                    <span className="subdropdown-desc">Devanagari swar phoneme signs</span>
                  </div>
                  {activeModel === 'isl_hindi_vowels' && <Check size={14} className="text-primary check-icon" />}
                </button>

                <div className="dropdown-divider" />
                <div className="dropdown-section-title">Words & Continuous Phrases</div>

                <button
                  type="button"
                  role="menuitem"
                  className={`subdropdown-item ${activeModel === 'isl_mvp' ? 'selected' : ''}`}
                  onClick={() => {
                    onSwitchModel('isl_mvp');
                    setIsModeDropdownOpen(false);
                  }}
                >
                  <BookOpen size={14} className="subdropdown-item-icon text-primary" />
                  <div className="subdropdown-text">
                    <span className="subdropdown-title">Common Signs (5 Core Signs)</span>
                    <span className="subdropdown-desc">Everyday phrases (Hello, Thank You, Yes, No, Please)</span>
                  </div>
                  {activeModel === 'isl_mvp' && <Check size={14} className="text-primary check-icon" />}
                </button>

                <button
                  type="button"
                  role="menuitem"
                  className={`subdropdown-item ${activeModel === 'amrita_phrase' ? 'selected' : ''}`}
                  onClick={() => {
                    onSwitchModel('amrita_phrase');
                    setIsModeDropdownOpen(false);
                  }}
                >
                  <MessageSquare size={14} className="subdropdown-item-icon text-primary" />
                  <div className="subdropdown-text">
                    <span className="subdropdown-title">Conversational Phrases (Amrita 49)</span>
                    <span className="subdropdown-desc">Everyday conversational dialogs & full sentence queries</span>
                  </div>
                  {activeModel === 'amrita_phrase' && <Check size={14} className="text-primary check-icon" />}
                </button>

                <button
                  type="button"
                  role="menuitem"
                  className={`subdropdown-item ${activeModel === 'isl_sequence_lstm' ? 'selected' : ''}`}
                  onClick={() => {
                    onSwitchModel('isl_sequence_lstm');
                    setIsModeDropdownOpen(false);
                  }}
                >
                  <Activity size={14} className="subdropdown-item-icon text-primary" />
                  <div className="subdropdown-text">
                    <span className="subdropdown-title">Dynamic Gestures (7 Sequences)</span>
                    <span className="subdropdown-desc">Continuous temporal gestures & motion patterns</span>
                  </div>
                  {activeModel === 'isl_sequence_lstm' && <Check size={14} className="text-primary check-icon" />}
                </button>

                <button
                  type="button"
                  role="menuitem"
                  className={`subdropdown-item ${activeModel === 'include_stream' ? 'selected' : ''}`}
                  onClick={() => {
                    onSwitchModel('include_stream');
                    setIsModeDropdownOpen(false);
                  }}
                >
                  <Layers size={14} className="subdropdown-item-icon text-primary" />
                  <div className="subdropdown-text">
                    <span className="subdropdown-title">Extended Dictionary (263 Signs)</span>
                    <span className="subdropdown-desc">Comprehensive AI4Bharat Indian Sign Language library</span>
                  </div>
                  {activeModel === 'include_stream' && <Check size={14} className="text-primary check-icon" />}
                </button>
              </div>
            )}
          </div>
        )}

        {/* 3. Right Group: View Options Popover + Record Sign Button */}
        <div className="control-group right-controls-group">
          {/* View Options Popover Trigger */}
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
              <span>View Options ▾</span>
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
                        {autoSpeak ? <Volume2 size={14} className="text-primary" /> : <VolumeX size={14} className="text-muted" />}
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

                  {/* Record Custom Sign Studio Modal Trigger */}
                  {onOpenRecordModal && (
                    <button
                      type="button"
                      className="popover-action-item"
                      onClick={() => {
                        setIsViewOptionsOpen(false);
                        onOpenRecordModal();
                      }}
                    >
                      <Video size={14} className="text-danger" />
                      <span>Custom Gesture Studio...</span>
                    </button>
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
