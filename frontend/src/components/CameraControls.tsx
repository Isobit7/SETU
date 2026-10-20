import React, { useState } from 'react';
import {
  Play,
  Pause,
  Square,
  RotateCcw,
  SwitchCamera,
  Layers,
  Subtitles,
  FlipHorizontal,
  FileVideo,
  Type,
  BookOpen,
  Settings,
  Volume2,
  VolumeX,
  Activity,
  Video,
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
}

export const CameraControls: React.FC<CameraControlsProps> = ({
  cameraStatus,
  availableCameras,
  selectedCameraId,
  activeModel = 'realsign',
  onSwitchModel,
  confidenceThreshold = 0.35,
  onThresholdChange,
  autoSpeak = false,
  onToggleAutoSpeak,
  landmarkStyles,
  subtitleVisible,
  onStart,
  onPause,
  onResume,
  onStop,
  onReset,
  onSwitchCamera,
  onToggleLandmarks,
  onToggleMirror,
  onToggleSubtitles,
  onOpenIncludeModal,
}) => {
  const [isOptionsOpen, setIsOptionsOpen] = useState<boolean>(false);

  const isDetecting = cameraStatus === 'detecting';
  const isPaused = cameraStatus === 'paused';
  const isActive = isDetecting || isPaused;
  const handsActive = landmarkStyles.showLeftHand || landmarkStyles.showRightHand;

  return (
    <div className="camera-controls-wrapper">
      {/* Primary Toolbar: Essential Controls Only */}
      <div className="controls-bar minimalist-bar" role="toolbar" aria-label="Essential Camera Controls">
        {/* 1. Primary Start / Stop / Pause Action */}
        <div className="control-group primary-stream-action">
          {!isActive ? (
            <button
              className="ctrl-btn primary-start-btn"
              onClick={onStart}
              title="Start webcam and live sign detection"
            >
              <Video size={16} />
              <span>Start Camera</span>
            </button>
          ) : (
            <div className="active-stream-btns">
              <button
                className="ctrl-btn stop-btn"
                onClick={onStop}
                title="Stop detection and turn off camera"
              >
                <Square size={15} />
                <span>Stop</span>
              </button>
              {isPaused ? (
                <button
                  className="ctrl-btn resume-btn"
                  onClick={onResume}
                  title="Resume live detection"
                >
                  <Play size={15} />
                  <span>Resume</span>
                </button>
              ) : (
                <button
                  className="ctrl-btn pause-btn"
                  onClick={onPause}
                  title="Pause detection"
                >
                  <Pause size={15} />
                  <span>Pause</span>
                </button>
              )}
            </div>
          )}
        </div>

        {/* 2. Mode Selector: 3-Way Segmented Control */}
        {onSwitchModel && (
          <div className="control-group mode-selector-segmented" role="radiogroup" aria-label="Recognition Mode">
            <button
              className={`segmented-mode-btn ${activeModel === 'realsign' ? 'active' : ''}`}
              onClick={() => onSwitchModel('realsign')}
              title="Alphabet (A–Z): RealSign PyTorch 26-Class MLP"
              role="radio"
              aria-checked={activeModel === 'realsign'}
            >
              <Type size={14} />
              <span>Alphabet (A–Z)</span>
            </button>
            <button
              className={`segmented-mode-btn ${activeModel === 'isl_mvp' ? 'active' : ''}`}
              onClick={() => onSwitchModel('isl_mvp')}
              title="Phrases (Static): 5 Core Conversational Signs"
              role="radio"
              aria-checked={activeModel === 'isl_mvp'}
            >
              <BookOpen size={14} />
              <span>Phrases (Static)</span>
            </button>
            <button
              className={`segmented-mode-btn ${activeModel === 'isl_sequence_lstm' ? 'active' : ''}`}
              onClick={() => onSwitchModel('isl_sequence_lstm')}
              title="Sequence (Dynamic LSTM): 30-Frame Spatiotemporal Model on GPU"
              role="radio"
              aria-checked={activeModel === 'isl_sequence_lstm'}
            >
              <Activity size={14} />
              <span>Sequence (Dynamic LSTM)</span>
            </button>
          </div>
        )}

        {/* 3. Collapsible Options Toggle */}
        <div className="control-group options-group">
          <button
            className={`ctrl-btn options-toggle-btn ${isOptionsOpen ? 'active' : ''}`}
            onClick={() => setIsOptionsOpen((prev) => !prev)}
            title="Toggle auxiliary controls and camera settings"
            aria-expanded={isOptionsOpen}
          >
            <Settings size={14} />
            <span>Options ▾</span>
          </button>
        </div>
      </div>

      {/* Collapsible Options Drawer */}
      {isOptionsOpen && (
        <div className="collapsible-options-drawer" aria-label="Detailed Camera and Detection Settings">
          <div className="options-row toggles-row">
            {/* Hand Landmarks Toggle */}
            <button
              className={`option-chip-btn ${handsActive ? 'active' : ''}`}
              onClick={onToggleLandmarks}
              title={handsActive ? 'Hide hand skeletons' : 'Show hand skeletons'}
              aria-pressed={handsActive}
            >
              <Layers size={14} />
              <span>Hand Skeleton</span>
            </button>

            {/* Mirror Toggle */}
            <button
              className={`option-chip-btn ${landmarkStyles.mirrored ? 'active' : ''}`}
              onClick={onToggleMirror}
              title="Flip camera horizontally"
              aria-pressed={landmarkStyles.mirrored}
            >
              <FlipHorizontal size={14} />
              <span>Mirror Preview</span>
            </button>

            {/* Subtitles Toggle */}
            <button
              className={`option-chip-btn ${subtitleVisible ? 'active' : ''}`}
              onClick={onToggleSubtitles}
              title="Toggle camera subtitle banner"
              aria-pressed={subtitleVisible}
            >
              <Subtitles size={14} />
              <span>Subtitles</span>
            </button>

            {/* Auto-Speak TTS Toggle */}
            {onToggleAutoSpeak && (
              <button
                className={`option-chip-btn ${autoSpeak ? 'active' : ''}`}
                onClick={onToggleAutoSpeak}
                title="Automatically announce recognized signs"
                aria-pressed={autoSpeak}
              >
                {autoSpeak ? <Volume2 size={14} className="text-primary" /> : <VolumeX size={14} />}
                <span>Auto-Speak</span>
              </button>
            )}

            {/* Reset Recognition Buffer */}
            {isActive && (
              <button
                className="option-chip-btn reset-chip-btn"
                onClick={onReset}
                title="Reset temporal feature buffer"
              >
                <RotateCcw size={14} />
                <span>Reset Buffer</span>
              </button>
            )}
          </div>

          <div className="options-row controls-row">
            {/* Confidence Threshold Slider */}
            {onThresholdChange && (
              <div
                className="threshold-slider-group"
                title={`Confidence Threshold: ${Math.round((confidenceThreshold ?? 0.35) * 100)}%`}
              >
                <span className="threshold-title">Confidence Gate:</span>
                <input
                  type="range"
                  min={25}
                  max={90}
                  step={5}
                  value={Math.round((confidenceThreshold ?? 0.35) * 100)}
                  onChange={(e) => onThresholdChange(Number(e.target.value) / 100)}
                  className="threshold-slider-input"
                  aria-label="Detection Confidence Threshold"
                />
                <span className="threshold-pill">{Math.round((confidenceThreshold ?? 0.35) * 100)}%</span>
              </div>
            )}

            {/* Camera Switcher (if multiple cameras available) */}
            {availableCameras.length > 1 && (
              <div className="camera-select-group">
                <SwitchCamera size={14} className="text-muted" />
                <select
                  className="camera-compact-select"
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

            {/* Video File Upload */}
            {onOpenIncludeModal && (
              <button
                className="option-chip-btn include-chip-btn"
                onClick={onOpenIncludeModal}
                title="Upload recorded video for AI4Bharat 263-sign recognition"
              >
                <FileVideo size={14} />
                <span>Upload Video (263 Signs)</span>
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
