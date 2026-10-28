import React, { useState, useEffect, useRef } from 'react';
import {
  X,
  Video,
  Check,
  Loader2,
  Sparkles,
  Database,
  AlertCircle,
  RotateCcw,
  Play,
  Layers,
} from 'lucide-react';
import type { LandmarkGroups } from '../types';

interface RecordSignModalProps {
  isOpen: boolean;
  onClose: () => void;
  currentLandmarks?: LandmarkGroups;
  isCameraActive: boolean;
}

interface CustomSampleItem {
  filename: string;
  label: string;
  shape: number[];
  created_time: number;
}

export const RecordSignModal: React.FC<RecordSignModalProps> = ({
  isOpen,
  onClose,
  currentLandmarks,
  isCameraActive,
}) => {
  const [label, setLabel] = useState<string>('');
  const [recordingState, setRecordingState] = useState<
    'idle' | 'countdown' | 'recording' | 'saving' | 'success' | 'error'
  >('idle');
  const [countdown, setCountdown] = useState<number>(3);
  const [capturedFrames, setCapturedFrames] = useState<number>(0);
  const [statusMessage, setStatusMessage] = useState<string>('');
  const [existingSamples, setExistingSamples] = useState<CustomSampleItem[]>([]);
  const [totalSamplesCount, setTotalSamplesCount] = useState<number>(0);
  const [isLoadingSamples, setIsLoadingSamples] = useState<boolean>(false);

  // Buffer to accumulate the 30 frames of landmarks
  const framesBufferRef = useRef<Array<{
    right_hand: any;
    left_hand: any;
    pose: any;
  }>>([]);
  const recordingTimerRef = useRef<number | null>(null);
  const countdownTimerRef = useRef<number | null>(null);

  // Fetch list of custom samples from backend
  const fetchCustomSamples = async () => {
    setIsLoadingSamples(true);
    try {
      const res = await fetch('/api/v1/dataset/custom_samples');
      if (res.ok) {
        const data = await res.json();
        setExistingSamples(data.samples || []);
        setTotalSamplesCount(data.total_samples || 0);
      }
    } catch {
      // Backend may be offline or in mock test
    } finally {
      setIsLoadingSamples(false);
    }
  };

  useEffect(() => {
    if (isOpen) {
      fetchCustomSamples();
      setRecordingState('idle');
      setCapturedFrames(0);
      setStatusMessage('');
      framesBufferRef.current = [];
    } else {
      if (countdownTimerRef.current) clearInterval(countdownTimerRef.current);
      if (recordingTimerRef.current) clearInterval(recordingTimerRef.current);
    }
  }, [isOpen]);

  // Start 3-2-1 Countdown
  const handleStartCountdown = () => {
    const cleanLabel = label.trim();
    if (!cleanLabel) {
      setStatusMessage('Please enter a gesture label name first.');
      setRecordingState('error');
      return;
    }

    if (!isCameraActive) {
      setStatusMessage('Webcam must be active to record sign keypoints.');
      setRecordingState('error');
      return;
    }

    setRecordingState('countdown');
    setCountdown(3);
    setStatusMessage('Get ready in signing posture...');
    framesBufferRef.current = [];
    setCapturedFrames(0);

    let count = 3;
    if (countdownTimerRef.current) clearInterval(countdownTimerRef.current);

    countdownTimerRef.current = window.setInterval(() => {
      count -= 1;
      if (count > 0) {
        setCountdown(count);
      } else {
        if (countdownTimerRef.current) clearInterval(countdownTimerRef.current);
        startRecordingFrames();
      }
    }, 1000);
  };

  // Record 30 consecutive frames at ~20 FPS (every 50ms)
  const startRecordingFrames = () => {
    setRecordingState('recording');
    setStatusMessage('Recording 30 gesture frames... Perform your sign now!');
    framesBufferRef.current = [];
    setCapturedFrames(0);

    let framesCount = 0;
    if (recordingTimerRef.current) clearInterval(recordingTimerRef.current);

    recordingTimerRef.current = window.setInterval(() => {
      // Snapshot current landmarks from props/stream
      const rh = currentLandmarks?.right_hand || null;
      const lh = currentLandmarks?.left_hand || null;
      const pose = currentLandmarks?.pose || null;

      framesBufferRef.current.push({
        right_hand: rh,
        left_hand: lh,
        pose: pose,
      });

      framesCount += 1;
      setCapturedFrames(framesCount);

      if (framesCount >= 30) {
        if (recordingTimerRef.current) clearInterval(recordingTimerRef.current);
        finishAndSaveRecording();
      }
    }, 50);
  };

  // Submit recorded 30 frames to backend
  const finishAndSaveRecording = async () => {
    setRecordingState('saving');
    setStatusMessage('Compressing and saving 30-frame sequence to data/processed/...');

    try {
      const payload = {
        label: label.trim().toLowerCase().replace(/\s+/g, '_'),
        landmarks_sequence: framesBufferRef.current,
      };

      const res = await fetch('/api/v1/dataset/record_sample', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (res.ok) {
        const data = await res.json();
        setRecordingState('success');
        setStatusMessage(data.message || `Successfully recorded sample for "${payload.label}"!`);
        fetchCustomSamples();
      } else {
        const errData = await res.json().catch(() => ({}));
        setRecordingState('error');
        setStatusMessage(errData.detail || 'Failed to save sample. Check backend connection.');
      }
    } catch (err: any) {
      setRecordingState('error');
      setStatusMessage(err?.message || 'Network error saving sample.');
    }
  };

  const handleReset = () => {
    setRecordingState('idle');
    setCapturedFrames(0);
    setStatusMessage('');
    framesBufferRef.current = [];
  };

  if (!isOpen) return null;

  return (
    <div className="modal-backdrop" onClick={onClose} role="dialog" aria-modal="true">
      <div
        className="modal-container recording-studio-modal"
        onClick={(e) => e.stopPropagation()}
        style={{ maxWidth: 640 }}
      >
        {/* Header */}
        <div className="modal-header">
          <div className="modal-title-wrap">
            <Video size={20} className="text-danger" />
            <h2>Custom Gesture Recording Studio</h2>
          </div>
          <button className="modal-close-btn" onClick={onClose} aria-label="Close dialog">
            <X size={20} />
          </button>
        </div>

        {/* Body */}
        <div className="modal-body">
          {/* Instructions Card */}
          <div className="record-intro-card">
            <Sparkles size={16} className="text-primary" />
            <p>
              Record a custom 30-frame ISL gesture (164-D Spatiotemporal representation).
              Samples are stored in <code>data/processed/</code> for training custom sequence models.
            </p>
          </div>

          {/* Form Input for Sign Label */}
          <div className="record-form-section">
            <label className="record-input-label">
              <span>Sign / Gesture Concept Name</span>
              <span className="record-label-hint">(e.g., DOCTOR, COLLEGE, EXAM, TRAIN)</span>
            </label>
            <div className="record-input-row">
              <input
                type="text"
                value={label}
                onChange={(e) => setLabel(e.target.value)}
                placeholder="Enter sign name..."
                disabled={recordingState === 'countdown' || recordingState === 'recording' || recordingState === 'saving'}
                className="record-label-input"
                maxLength={40}
              />
            </div>

            {/* Quick Idea Chips */}
            <div className="record-quick-chips">
              <span className="quick-chip-hint">Quick ideas:</span>
              {['DOCTOR', 'COLLEGE', 'EXAM', 'FAMILY', 'HOSPITAL', 'TEACHER', 'TRAIN'].map((preset) => (
                <button
                  key={preset}
                  type="button"
                  className="record-preset-chip"
                  onClick={() => setLabel(preset)}
                  disabled={recordingState !== 'idle' && recordingState !== 'error'}
                >
                  {preset}
                </button>
              ))}
            </div>
          </div>

          {/* Status & Recording Viewport Area */}
          <div className="record-viewport-box">
            {recordingState === 'idle' && (
              <div className="record-state-idle">
                <div className="idle-indicator">
                  <Layers size={32} className="text-muted" />
                  <p>
                    {isCameraActive
                      ? 'Camera is active and tracking landmarks. Press Start to begin 3-second countdown.'
                      : 'Camera is currently stopped. Please start the webcam in the main workspace.'}
                  </p>
                </div>
                <button
                  type="button"
                  className="primary-btn start-rec-btn"
                  onClick={handleStartCountdown}
                  disabled={!isCameraActive || !label.trim()}
                >
                  <Play size={16} />
                  <span>Start Recording (3s Countdown)</span>
                </button>
              </div>
            )}

            {recordingState === 'countdown' && (
              <div className="record-state-countdown">
                <span className="countdown-number-display">{countdown}</span>
                <span className="countdown-text">Get into starting gesture position!</span>
              </div>
            )}

            {recordingState === 'recording' && (
              <div className="record-state-recording">
                <div className="recording-header-live">
                  <span className="recording-pulse-dot" />
                  <span className="recording-title-text">CAPTURING GESTURE FRAMES</span>
                </div>
                <div className="progress-bar-track">
                  <div
                    className="progress-bar-fill"
                    style={{ width: `${(capturedFrames / 30) * 100}%` }}
                  />
                </div>
                <span className="frame-counter-text">
                  Frame {capturedFrames} / 30 ({Math.round((capturedFrames / 30) * 100)}%)
                </span>
              </div>
            )}

            {recordingState === 'saving' && (
              <div className="record-state-saving">
                <Loader2 size={32} className="spinner-icon text-primary" />
                <span>{statusMessage}</span>
              </div>
            )}

            {recordingState === 'success' && (
              <div className="record-state-success">
                <div className="success-icon-wrap">
                  <Check size={28} className="text-success" />
                </div>
                <h4>Sample Recorded Successfully!</h4>
                <p className="success-submessage">{statusMessage}</p>
                <div className="success-actions">
                  <button type="button" className="secondary-btn" onClick={handleReset}>
                    <RotateCcw size={14} />
                    <span>Record Another Sample</span>
                  </button>
                  <button type="button" className="primary-btn" onClick={onClose}>
                    <Check size={14} />
                    <span>Done</span>
                  </button>
                </div>
              </div>
            )}

            {recordingState === 'error' && (
              <div className="record-state-error">
                <AlertCircle size={28} className="text-danger" />
                <p className="error-message-text">{statusMessage}</p>
                <button type="button" className="secondary-btn" onClick={handleReset}>
                  <RotateCcw size={14} />
                  <span>Try Again</span>
                </button>
              </div>
            )}
          </div>

          {/* Dataset Status & Recent Samples Drawer */}
          <div className="dataset-status-drawer">
            <div className="dataset-drawer-header">
              <div className="drawer-title-row">
                <Database size={15} className="text-primary" />
                <span className="drawer-title">Dataset Archive:</span>
                <span className="dataset-count-badge">{totalSamplesCount} Recorded Samples</span>
              </div>
              <button
                type="button"
                className="refresh-samples-btn"
                onClick={fetchCustomSamples}
                disabled={isLoadingSamples}
                title="Refresh recorded samples list"
              >
                {isLoadingSamples ? <Loader2 size={12} className="spinner-icon" /> : <RotateCcw size={12} />}
              </button>
            </div>

            {existingSamples.length > 0 ? (
              <div className="samples-mini-list">
                {existingSamples.slice(0, 5).map((sample) => (
                  <div key={sample.filename} className="sample-mini-item">
                    <span className="sample-label-tag">{sample.label}</span>
                    <span className="sample-dim-tag">{sample.shape.join(' × ')}</span>
                    <span className="sample-time-tag">
                      {new Date(sample.created_time * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <span className="empty-samples-hint">No custom samples recorded yet. Record your first gesture above!</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
