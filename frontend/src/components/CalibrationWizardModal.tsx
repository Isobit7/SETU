import React, { useState, useEffect } from 'react';
import { X, CheckCircle2, Sliders, ArrowRight, RotateCcw, Sparkles } from 'lucide-react';
import type { RecognitionPayload } from '../types';

interface CalibrationWizardModalProps {
  isOpen: boolean;
  onClose: () => void;
  currentResult: RecognitionPayload | null;
  onApplyCalibration: (params: {
    rest_y: number;
    pitch_threshold: number;
    yaw_threshold: number;
  }) => void;
}

export const CalibrationWizardModal: React.FC<CalibrationWizardModalProps> = ({
  isOpen,
  onClose,
  currentResult,
  onApplyCalibration,
}) => {
  const [step, setStep] = useState<number>(1);
  const [restY, setRestY] = useState<number>(0.82);
  const [pitchThreshold, setPitchThreshold] = useState<number>(0.048);
  const [yawThreshold, setYawThreshold] = useState<number>(0.048);
  const [detectedPitch, setDetectedPitch] = useState<number>(0.0);
  const [nodSuccess, setNodSuccess] = useState<boolean>(false);

  // Monitor live landmarks during calibration
  useEffect(() => {
    if (!isOpen) return;

    // Track wrist Y coordinate during step 1
    const rh = currentResult?.landmarks?.right_hand;
    const lh = currentResult?.landmarks?.left_hand;
    if (step === 1) {
      if (rh && rh.length > 0) {
        setRestY(Math.max(0.60, Math.min(0.95, rh[0].y)));
      } else if (lh && lh.length > 0) {
        setRestY(Math.max(0.60, Math.min(0.95, lh[0].y)));
      }
    }

    // Monitor head nod in step 3
    if (step === 3) {
      const pitch = (currentResult as any)?.non_manual_markers?.head_pitch_oscillation || 0.0;
      setDetectedPitch(pitch);
      if (pitch >= 0.040 || currentResult?.head_gesture === 'nod') {
        setNodSuccess(true);
      }
    }
  }, [isOpen, step, currentResult]);

  if (!isOpen) return null;

  const handleFinish = () => {
    const config = {
      rest_y: Number(restY.toFixed(2)),
      pitch_threshold: Number(pitchThreshold.toFixed(3)),
      yaw_threshold: Number(yawThreshold.toFixed(3)),
    };
    try {
      localStorage.setItem('setu_user_calibration', JSON.stringify(config));
    } catch {}
    onApplyCalibration(config);
    onClose();
  };

  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="calib-title">
      <div className="modal-card calibration-modal-card">
        {/* Header */}
        <div className="modal-header">
          <div className="header-title-group">
            <Sliders size={20} className="modal-title-icon text-cyan" />
            <div>
              <h2 id="calib-title" className="modal-title">Personal Webcam & Pose Calibration</h2>
              <p className="modal-subtitle">
                Calibrates your rest height, hand bounding zone, and head intent sensitivity in 10 seconds.
              </p>
            </div>
          </div>
          <button type="button" className="modal-close-btn" onClick={onClose} aria-label="Close Calibration">
            <X size={18} />
          </button>
        </div>

        {/* Step Indicator */}
        <div className="calibration-step-bar">
          <div className={`step-dot ${step >= 1 ? 'active' : ''}`}>1. Resting Baseline</div>
          <div className="step-line" />
          <div className={`step-dot ${step >= 2 ? 'active' : ''}`}>2. Signing Space</div>
          <div className="step-line" />
          <div className={`step-dot ${step >= 3 ? 'active' : ''}`}>3. Head Nod Intent</div>
        </div>

        {/* Step Content */}
        <div className="modal-body calibration-body">
          {step === 1 && (
            <div className="calibration-step-content">
              <div className="step-instruction">
                <h3>Step 1: Calibrate Resting Hands Baseline</h3>
                <p>
                  Drop your hands comfortably into your lap or onto your desk. This prevents resting hands from
                  accidentally triggering phantom signs.
                </p>
              </div>

              <div className="calibration-preview-box">
                <div className="calibration-meter-label">
                  <span>Detected Rest Boundary:</span>
                  <span className="calib-val">{Math.round(restY * 100)}% from camera top</span>
                </div>
                <div className="calibration-slider-wrap">
                  <input
                    type="range"
                    min="0.60"
                    max="0.95"
                    step="0.01"
                    value={restY}
                    onChange={(e) => setRestY(parseFloat(e.target.value))}
                    className="calibration-slider"
                  />
                </div>
                <p className="meter-hint">Hands below this height will remain in resting mode.</p>
              </div>

              <div className="step-actions">
                <button type="button" className="primary-btn" onClick={() => setStep(2)}>
                  <span>Looks Good, Next Step</span>
                  <ArrowRight size={16} />
                </button>
              </div>
            </div>
          )}

          {step === 2 && (
            <div className="calibration-step-content">
              <div className="step-instruction">
                <h3>Step 2: Calibrate Active Signing Box</h3>
                <p>
                  Raise your dominant signing hand in front of your chest/face at your natural signing distance (approx
                  50–70cm from screen).
                </p>
              </div>

              <div className="calibration-preview-box">
                <div className="active-hand-indicator">
                  <span
                    className={`indicator-badge ${
                      currentResult?.hand_detected ? 'detected' : 'waiting'
                    }`}
                  >
                    {currentResult?.hand_detected
                      ? '✓ Hand Detected in Optimal Signing Box'
                      : 'Position hand inside camera view'}
                  </span>
                </div>
                <p className="meter-hint">
                  Keep camera at chest/eye level with front lighting for max accuracy.
                </p>
              </div>

              <div className="step-actions">
                <button type="button" className="secondary-btn" onClick={() => setStep(1)}>
                  <span>Back</span>
                </button>
                <button type="button" className="primary-btn" onClick={() => setStep(3)}>
                  <span>Confirm Hand Box, Next</span>
                  <ArrowRight size={16} />
                </button>
              </div>
            </div>
          )}

          {step === 3 && (
            <div className="calibration-step-content">
              <div className="step-instruction">
                <h3>Step 3: Test Head Nod Intent Confirmation</h3>
                <p>
                  Nod your head up and down once (just like saying &ldquo;yes&rdquo;). This tests hands-free instant sign locking.
                </p>
              </div>

              <div className="calibration-preview-box">
                <div className="nod-test-feedback">
                  {nodSuccess ? (
                    <div className="nod-success-badge">
                      <CheckCircle2 size={24} className="text-emerald" />
                      <div>
                        <strong>Head Nod Successfully Detected!</strong>
                        <p>Nodding will instantly lock letters and words without pauses.</p>
                      </div>
                    </div>
                  ) : (
                    <div className="nod-waiting-badge">
                      <Sparkles size={20} className="text-cyan animate-pulse" />
                      <span>Nod head up and down now (Pitch: {(detectedPitch * 100).toFixed(0)}%)</span>
                    </div>
                  )}
                </div>

                <div className="calibration-slider-wrap">
                  <div className="calibration-meter-label">
                    <span>Nod Sensitivity:</span>
                    <span className="calib-val">{pitchThreshold}</span>
                  </div>
                  <input
                    type="range"
                    min="0.02"
                    max="0.08"
                    step="0.005"
                    value={pitchThreshold}
                    onChange={(e) => setPitchThreshold(parseFloat(e.target.value))}
                    className="calibration-slider"
                  />
                </div>
              </div>

              <div className="step-actions">
                <button type="button" className="secondary-btn" onClick={() => setStep(2)}>
                  <span>Back</span>
                </button>
                <button type="button" className="primary-btn finish-btn" onClick={handleFinish}>
                  <CheckCircle2 size={16} />
                  <span>Save Personal Calibration</span>
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="modal-footer">
          <button
            type="button"
            className="secondary-btn btn-sm"
            onClick={() => {
              setRestY(0.82);
              setPitchThreshold(0.048);
              setYawThreshold(0.048);
              setStep(1);
            }}
          >
            <RotateCcw size={12} />
            <span>Reset to Factory Defaults</span>
          </button>
          <span className="calib-storage-note">Saved locally to this device</span>
        </div>
      </div>
    </div>
  );
};
export default CalibrationWizardModal;
