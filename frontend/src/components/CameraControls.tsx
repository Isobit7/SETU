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
  const [isPhrasesDropdownOpen, setIsPhrasesDropdownOpen] = useState<boolean>(false);

  const viewOptionsRef = useRef<HTMLDivElement | null>(null);
  const phrasesDropdownRef = useRef<HTMLDivElement | null>(null);

  const isDetecting = cameraStatus === 'detecting';
  const isPaused = cameraStatus === 'paused';
  const isActive = isDetecting || isPaused;
  const handsActive = landmarkStyles.showLeftHand || landmarkStyles.showRightHand;

  const isWordsMode = activeModel !== 'realsign';

  // Close popovers on click outside or escape
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (viewOptionsRef.current && !viewOptionsRef.current.contains(e.target as Node)) {
        setIsViewOptionsOpen(false);
      }
      if (phrasesDropdownRef.current && !phrasesDropdownRef.current.contains(e.target as Node)) {
        setIsPhrasesDropdownOpen(false);
      }
    };
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setIsViewOptionsOpen(false);
        setIsPhrasesDropdownOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, []);

  const getPhraseModeLabel = (model: ActiveModelType) => {
    switch (model) {
      case 'isl_sequence_lstm':
        return 'Dynamic Sequences';
      case 'include_stream':
        return 'INCLUDE (263 Signs)';
      case 'isl_mvp':
      default:
        return 'Conversational Signs';
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

        {/* 2. Primary 2-Tier Segmented Mode Selector */}
        {onSwitchModel && (
          <div className="control-group mode-selector-segmented" role="radiogroup" aria-label="Recognition Mode">
            {/* Mode 1: Letters (A–Z) */}
            <button
              type="button"
              className={`segmented-mode-btn ${activeModel === 'realsign' ? 'active' : ''}`}
              onClick={() => {
                onSwitchModel('realsign');
                setIsPhrasesDropdownOpen(false);
              }}
              title="Alphabet (A–Z): RealSign fingerspelling recognition"
              role="radio"
              aria-checked={activeModel === 'realsign'}
            >
              <Type size={14} />
              <span>Letters (A–Z)</span>
            </button>

            {/* Mode 2: Words & Phrases (with quick sub-tier dropdown) */}
            <div className="segmented-dropdown-container" ref={phrasesDropdownRef}>
              <button
                type="button"
                className={`segmented-mode-btn ${isWordsMode ? 'active' : ''}`}
                onClick={() => {
                  if (activeModel === 'realsign') {
                    onSwitchModel('isl_mvp');
                  }
                  setIsPhrasesDropdownOpen((prev) => !prev);
                }}
                title="Words & Phrases: Common conversational and continuous sign recognition"
                role="radio"
                aria-checked={isWordsMode}
                aria-haspopup="true"
                aria-expanded={isPhrasesDropdownOpen}
              >
                <BookOpen size={14} />
                <span>{isWordsMode ? getPhraseModeLabel(activeModel) : 'Words & Phrases'}</span>
                <ChevronDown size={12} className={`mode-chevron ${isPhrasesDropdownOpen ? 'open' : ''}`} />
              </button>

              {isPhrasesDropdownOpen && (
                <div className="mode-subdropdown-menu" role="menu">
                  <button
                    type="button"
                    role="menuitem"
                    className={`subdropdown-item ${activeModel === 'isl_mvp' ? 'selected' : ''}`}
                    onClick={() => {
                      onSwitchModel('isl_mvp');
                      setIsPhrasesDropdownOpen(false);
                    }}
                  >
                    <div className="subdropdown-text">
                      <span className="subdropdown-title">Conversational Signs</span>
                      <span className="subdropdown-desc">Core everyday phrases (Hello, Thank You, etc.)</span>
                    </div>
                    {activeModel === 'isl_mvp' && <Check size={14} className="text-primary" />}
                  </button>

                  <button
                    type="button"
                    role="menuitem"
                    className={`subdropdown-item ${activeModel === 'isl_sequence_lstm' ? 'selected' : ''}`}
                    onClick={() => {
                      onSwitchModel('isl_sequence_lstm');
                      setIsPhrasesDropdownOpen(false);
                    }}
                  >
                    <div className="subdropdown-text">
                      <span className="subdropdown-title">Dynamic Sequences (LSTM)</span>
                      <span className="subdropdown-desc">Continuous temporal gestures</span>
                    </div>
                    {activeModel === 'isl_sequence_lstm' && <Check size={14} className="text-primary" />}
                  </button>

                  <button
                    type="button"
                    role="menuitem"
                    className={`subdropdown-item ${activeModel === 'include_stream' ? 'selected' : ''}`}
                    onClick={() => {
                      onSwitchModel('include_stream');
                      setIsPhrasesDropdownOpen(false);
                    }}
                  >
                    <div className="subdropdown-text">
                      <span className="subdropdown-title">AI4Bharat Full Vocabulary</span>
                      <span className="subdropdown-desc">263 Comprehensive ISL gestures</span>
                    </div>
                    {activeModel === 'include_stream' && <Check size={14} className="text-primary" />}
                  </button>
                </div>
              )}
            </div>
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
                </div>
              </div>
            )}
          </div>

          {/* 4. Record Sign Button */}
          {onOpenRecordModal && (
            <button
              type="button"
              className="ctrl-btn record-sign-btn"
              onClick={onOpenRecordModal}
              title="Open In-Browser Custom Gesture Recording Studio"
            >
              <Video size={14} className="text-danger" />
              <span>Record Sign</span>
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default CameraControls;
