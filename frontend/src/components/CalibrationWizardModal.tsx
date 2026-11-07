import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  X,
  CheckCircle2,
  Sliders,
  ArrowRight,
  RotateCcw,
  Sparkles,
  Hand,
  Activity,
  Check,
  RefreshCw,
} from 'lucide-react';
import type { RecognitionPayload } from '../types';

interface CalibrationWizardModalProps {
  isOpen: boolean;
  onClose: () => void;
  currentResult: RecognitionPayload | null;
  onApplyCalibration: (params: {
    rest_y: number;
    pitch_threshold: number;
    yaw_threshold: number;
    motion_threshold?: number;
    hand_scale?: number;
  }) => void;
}

interface HandSample {
  palmSpan: number;
  handWidth: number;
  fingerRatios: number[];
  wristX: number;
  wristY: number;
}

function dist3d(
  p1: { x: number; y: number; z?: number },
  p2: { x: number; y: number; z?: number }
): number {
  const dx = p1.x - p2.x;
  const dy = p1.y - p2.y;
  const dz = (p1.z ?? 0) - (p2.z ?? 0);
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

export const CalibrationWizardModal: React.FC<CalibrationWizardModalProps> = ({
  isOpen,
  onClose,
  currentResult,
  onApplyCalibration,
}) => {
  const [step, setStep] = useState<number>(1);
  const [showManualOverride, setShowManualOverride] = useState<boolean>(false);

  // Auto-calibrated results
  const [handScale, setHandScale] = useState<number>(1.0);
  const [palmSpan, setPalmSpan] = useState<number>(0.20);
  const [handWidth, setHandWidth] = useState<number>(0.24);
  const [fingerRatios, setFingerRatios] = useState<number[]>([0.9, 1.4, 1.5, 1.4, 1.1]);
  const [jitterNoiseFloor, setJitterNoiseFloor] = useState<number>(0.0035);
  const [motionThreshold, setMotionThreshold] = useState<number>(0.016);
  const [restY, setRestY] = useState<number>(0.82);
  const [pitchThreshold, setPitchThreshold] = useState<number>(0.035);
  const [yawThreshold, setYawThreshold] = useState<number>(0.035);

  // Live progress metrics
  const [handFramesCount, setHandFramesCount] = useState<number>(0);
  const [restFramesCount, setRestFramesCount] = useState<number>(0);
  const [liveYawOscillation, setLiveYawOscillation] = useState<number>(0.0);
  const [livePitchOscillation, setLivePitchOscillation] = useState<number>(0.0);
  const [shakeDetected, setShakeDetected] = useState<boolean>(false);
  const [nodDetected, setNodDetected] = useState<boolean>(false);

  // Frame accumulators using refs to avoid excessive re-render churn
  const handSamplesRef = useRef<HandSample[]>([]);
  const restSamplesRef = useRef<number[]>([]);
  const peakYawRef = useRef<number>(0.0);
  const peakPitchRef = useRef<number>(0.0);
  const autoAdvanceTimerRef = useRef<number | null>(null);

  // Reset accumulators when reopening
  useEffect(() => {
    if (isOpen) {
      setStep(1);
      setHandFramesCount(0);
      setRestFramesCount(0);
      setShakeDetected(false);
      setNodDetected(false);
      handSamplesRef.current = [];
      restSamplesRef.current = [];
      peakYawRef.current = 0.0;
      peakPitchRef.current = 0.0;

      // Load existing calibration from localStorage if available
      try {
        const saved = localStorage.getItem('setu_user_calibration');
        if (saved) {
          const parsed = JSON.parse(saved);
          if (parsed.rest_y) setRestY(parsed.rest_y);
          if (parsed.motion_threshold) setMotionThreshold(parsed.motion_threshold);
          if (parsed.pitch_threshold) setPitchThreshold(parsed.pitch_threshold);
          if (parsed.yaw_threshold) setYawThreshold(parsed.yaw_threshold);
          if (parsed.hand_scale) setHandScale(parsed.hand_scale);
        }
      } catch {}
    } else {
      if (autoAdvanceTimerRef.current) {
        clearTimeout(autoAdvanceTimerRef.current);
      }
    }
  }, [isOpen]);

  // Main automatic calibration sampling loop driven by incoming recognition results
  useEffect(() => {
    if (!isOpen || !currentResult) return;

    const rh = currentResult.landmarks?.right_hand;
    const lh = currentResult.landmarks?.left_hand;
    const dominantHand = (rh && rh.length === 21) ? rh : (lh && lh.length === 21 ? lh : null);

    // =========================================================================
    // STEP 1: AUTOMATIC HAND DETECTION & GEOMETRIC NORMALIZATION (30 frames)
    // =========================================================================
    if (step === 1 && dominantHand) {
      const wrist = dominantHand[0];
      const middleMcp = dominantHand[9];
      const thumbTip = dominantHand[4];
      const indexTip = dominantHand[8];
      const middleTip = dominantHand[12];
      const ringTip = dominantHand[16];
      const pinkyTip = dominantHand[20];

      const pSpan = dist3d(wrist, middleMcp);
      const hWidth = dist3d(thumbTip, pinkyTip);

      if (pSpan > 0.05) {
        const ratios = [
          dist3d(wrist, thumbTip) / pSpan,
          dist3d(wrist, indexTip) / pSpan,
          dist3d(wrist, middleTip) / pSpan,
          dist3d(wrist, ringTip) / pSpan,
          dist3d(wrist, pinkyTip) / pSpan,
        ];

        handSamplesRef.current.push({
          palmSpan: pSpan,
          handWidth: hWidth,
          fingerRatios: ratios,
          wristX: wrist.x,
          wristY: wrist.y,
        });

        const count = handSamplesRef.current.length;
        setHandFramesCount(count);

        if (count >= 30) {
          // Compute averages and noise floor
          const samples = handSamplesRef.current;
          const avgPalm = samples.reduce((s, x) => s + x.palmSpan, 0) / count;
          const avgWidth = samples.reduce((s, x) => s + x.handWidth, 0) / count;
          const avgRatios = [0, 1, 2, 3, 4].map(
            (idx) => samples.reduce((s, x) => s + x.fingerRatios[idx], 0) / count
          );

          // Sensor/jitter noise floor (standard deviation of wrist coordinate)
          const avgWristX = samples.reduce((s, x) => s + x.wristX, 0) / count;
          const avgWristY = samples.reduce((s, x) => s + x.wristY, 0) / count;
          const variance =
            samples.reduce((s, x) => s + (x.wristX - avgWristX) ** 2 + (x.wristY - avgWristY) ** 2, 0) / count;
          const jitterStd = Math.sqrt(variance);

          // Calibrated scale factor relative to canonical 0.20 normalized palm
          const derivedScale = Math.max(0.70, Math.min(1.40, avgPalm / 0.20));
          // Calibrated motion threshold: 3.8x webcam noise floor (immune to stationary tremor)
          const derivedMotion = Math.max(0.012, Math.min(0.028, jitterStd * 3.8));

          setPalmSpan(Number(avgPalm.toFixed(3)));
          setHandWidth(Number(avgWidth.toFixed(3)));
          setHandScale(Number(derivedScale.toFixed(2)));
          setFingerRatios(avgRatios.map((r) => Number(r.toFixed(2))));
          setJitterNoiseFloor(Number(jitterStd.toFixed(4)));
          setMotionThreshold(Number(derivedMotion.toFixed(3)));

          // Auto-advance to Step 2
          if (!autoAdvanceTimerRef.current) {
            autoAdvanceTimerRef.current = window.setTimeout(() => {
              setStep(2);
              autoAdvanceTimerRef.current = null;
            }, 650);
          }
        }
      }
    }

    // =========================================================================
    // STEP 2: AUTOMATIC RESTING HANDS BASELINE (20 frames)
    // =========================================================================
    if (step === 2) {
      let restingWristY: number | null = null;
      if (dominantHand) {
        restingWristY = dominantHand[0].y;
      }

      // Collect lower/rest poses (natural resting positions are in lower 40% of camera)
      if (restingWristY !== null && restingWristY >= 0.60) {
        restSamplesRef.current.push(restingWristY);
        const count = restSamplesRef.current.length;
        setRestFramesCount(count);

        if (count >= 20) {
          const sorted = [...restSamplesRef.current].sort((a, b) => a - b);
          const medianY = sorted[Math.floor(sorted.length / 2)];
          // Place boundary slightly above resting position for safety
          const calibratedRestY = Math.max(0.65, Math.min(0.92, medianY - 0.04));
          setRestY(Number(calibratedRestY.toFixed(2)));

          // Auto-advance to Step 3
          if (!autoAdvanceTimerRef.current) {
            autoAdvanceTimerRef.current = window.setTimeout(() => {
              setStep(3);
              autoAdvanceTimerRef.current = null;
            }, 650);
          }
        }
      }
    }

    // =========================================================================
    // STEP 3: AUTOMATIC HEAD MOTION & INTENT CONFIRMATION (Left/Right & Up/Down)
    // =========================================================================
    if (step === 3) {
      const yaw = currentResult.non_manual_markers?.head_yaw_oscillation ?? 0.0;
      const pitch = currentResult.non_manual_markers?.head_pitch_oscillation ?? 0.0;
      const gesture = currentResult.head_gesture || currentResult.non_manual_markers?.head_gesture;

      setLiveYawOscillation(yaw);
      setLivePitchOscillation(pitch);

      if (yaw > peakYawRef.current) {
        peakYawRef.current = yaw;
      }
      if (pitch > peakPitchRef.current) {
        peakPitchRef.current = pitch;
      }

      if (yaw >= 0.028 || gesture === 'shake') {
        setShakeDetected(true);
      }
      if (pitch >= 0.028 || gesture === 'nod') {
        setNodDetected(true);
      }

      // Once natural head motion is confirmed, compute optimal personal thresholds
      if ((yaw >= 0.028 || gesture === 'shake') || (pitch >= 0.028 || gesture === 'nod')) {
        const bestYaw = Math.max(peakYawRef.current, 0.035);
        const bestPitch = Math.max(peakPitchRef.current, 0.035);
        const derivedYawThresh = Math.max(0.022, Math.min(0.055, bestYaw * 0.70));
        const derivedPitchThresh = Math.max(0.022, Math.min(0.055, bestPitch * 0.70));

        setYawThreshold(Number(derivedYawThresh.toFixed(3)));
        setPitchThreshold(Number(derivedPitchThresh.toFixed(3)));

        // Automatically finalize and save
        if (!autoAdvanceTimerRef.current) {
          autoAdvanceTimerRef.current = window.setTimeout(() => {
            setStep(4);
            autoAdvanceTimerRef.current = null;
          }, 800);
        }
      }
    }
  }, [isOpen, step, currentResult]);

  // Apply calibration and save to localStorage
  const handleApplyAndFinish = useCallback(() => {
    const config = {
      rest_y: Number(restY.toFixed(2)),
      pitch_threshold: Number(pitchThreshold.toFixed(3)),
      yaw_threshold: Number(yawThreshold.toFixed(3)),
      motion_threshold: Number(motionThreshold.toFixed(3)),
      hand_scale: Number(handScale.toFixed(2)),
    };
    try {
      localStorage.setItem('setu_user_calibration', JSON.stringify(config));
    } catch {}
    onApplyCalibration(config);
    onClose();
  }, [restY, pitchThreshold, yawThreshold, motionThreshold, handScale, onApplyCalibration, onClose]);

  // Auto-apply calibration when Step 4 is reached
  useEffect(() => {
    if (step === 4) {
      const config = {
        rest_y: Number(restY.toFixed(2)),
        pitch_threshold: Number(pitchThreshold.toFixed(3)),
        yaw_threshold: Number(yawThreshold.toFixed(3)),
        motion_threshold: Number(motionThreshold.toFixed(3)),
        hand_scale: Number(handScale.toFixed(2)),
      };
      try {
        localStorage.setItem('setu_user_calibration', JSON.stringify(config));
      } catch {}
      onApplyCalibration(config);
    }
  }, [step, restY, pitchThreshold, yawThreshold, motionThreshold, handScale, onApplyCalibration]);

  const handleRestart = () => {
    if (autoAdvanceTimerRef.current) {
      clearTimeout(autoAdvanceTimerRef.current);
      autoAdvanceTimerRef.current = null;
    }
    setStep(1);
    setHandFramesCount(0);
    setRestFramesCount(0);
    setShakeDetected(false);
    setNodDetected(false);
    handSamplesRef.current = [];
    restSamplesRef.current = [];
    peakYawRef.current = 0.0;
    peakPitchRef.current = 0.0;
  };

  const handleResetDefaults = () => {
    setRestY(0.82);
    setMotionThreshold(0.016);
    setPitchThreshold(0.035);
    setYawThreshold(0.035);
    setHandScale(1.0);
    setPalmSpan(0.20);
    setHandWidth(0.24);
    setFingerRatios([0.9, 1.4, 1.5, 1.4, 1.1]);
    const defaultConfig = {
      rest_y: 0.82,
      pitch_threshold: 0.035,
      yaw_threshold: 0.035,
      motion_threshold: 0.016,
      hand_scale: 1.0,
    };
    try {
      localStorage.setItem('setu_user_calibration', JSON.stringify(defaultConfig));
    } catch {}
    onApplyCalibration(defaultConfig);
  };

  if (!isOpen) return null;

  const fingerNames = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky'];

  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="calib-title">
      <div className="modal-card calibration-modal-card">
        {/* Header */}
        <div className="modal-header">
          <div className="header-title-group">
            <Sparkles size={20} className="modal-title-icon text-cyan" />
            <div>
              <h2 id="calib-title" className="modal-title">Automatic Hands-Free Calibration</h2>
              <p className="modal-subtitle">
                Zero sliders required. Simply follow the 3 natural visual prompts.
              </p>
            </div>
          </div>
          <button type="button" className="modal-close-btn" onClick={onClose} aria-label="Close Calibration">
            <X size={18} />
          </button>
        </div>

        {/* Step Progress Bar */}
        <div className="calibration-step-bar">
          <div className={`step-dot ${step === 1 ? 'active' : step > 1 ? 'completed' : ''}`}>
            {step > 1 ? '✓ 1. Hand Geometry' : '1. Hand Geometry'}
          </div>
          <div className="step-line" />
          <div className={`step-dot ${step === 2 ? 'active' : step > 2 ? 'completed' : ''}`}>
            {step > 2 ? '✓ 2. Resting Baseline' : '2. Resting Baseline'}
          </div>
          <div className="step-line" />
          <div className={`step-dot ${step === 3 ? 'active' : step > 3 ? 'completed' : ''}`}>
            {step > 3 ? '✓ 3. Head Turning' : '3. Head Turning'}
          </div>
          <div className="step-line" />
          <div className={`step-dot ${step === 4 ? 'active' : ''}`}>
            4. Profile Ready
          </div>
        </div>

        {/* Step Body */}
        <div className="modal-body calibration-body">
          {/* STEP 1: HAND GEOMETRY & NORMALIZATION */}
          {step === 1 && (
            <div className="calibration-step-content">
              <div className="step-instruction">
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <Hand size={18} className="text-cyan animate-pulse" />
                  <h3>Step 1: Show Hand to Camera</h3>
                </div>
                <p>
                  Hold your open signing hand steadily in front of the camera. The system will measure your palm span,
                  finger extension ratios, and webcam noise floor automatically.
                </p>
              </div>

              <div className="calibration-preview-box">
                {/* Progress Meter */}
                <div className="calibration-meter-label">
                  <span>Capturing Hand Landmark Ratios:</span>
                  <span className="calib-val">{handFramesCount} / 30 Frames</span>
                </div>
                <div className="calib-progress-track">
                  <div
                    className="calib-progress-fill"
                    style={{ width: `${Math.min(100, (handFramesCount / 30) * 100)}%` }}
                  />
                </div>

                {/* Live Detected Metrics Grid */}
                <div className="calib-metrics-grid">
                  <div className="calib-metric-card">
                    <span className="calib-metric-lbl">Palm Span</span>
                    <strong className="calib-metric-val">{palmSpan}</strong>
                    <span className="calib-metric-sub">Wrist to MCP</span>
                  </div>
                  <div className="calib-metric-card">
                    <span className="calib-metric-lbl">Hand Scale</span>
                    <strong className="calib-metric-val">{handScale}x</strong>
                    <span className="calib-metric-sub">Span: {handWidth}</span>
                  </div>
                  <div className="calib-metric-card">
                    <span className="calib-metric-lbl">Webcam Noise Floor</span>
                    <strong className="calib-metric-val">{(jitterNoiseFloor * 1000).toFixed(1)}‰</strong>
                    <span className="calib-metric-sub">Motion threshold: {motionThreshold}</span>
                  </div>
                </div>

                {/* 5 Finger Ratios Visual */}
                <div className="calib-finger-strip">
                  <span className="finger-strip-title">Normalized Finger Extension Ratios:</span>
                  <div className="finger-chips-row">
                    {fingerRatios.map((ratio, idx) => (
                      <div key={fingerNames[idx]} className="finger-ratio-chip">
                        <span className="finger-name">{fingerNames[idx]}</span>
                        <strong className="finger-ratio">{ratio}x</strong>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="step-actions">
                <button
                  type="button"
                  className="primary-btn"
                  disabled={handFramesCount < 10}
                  onClick={() => setStep(2)}
                >
                  <span>Skip to Step 2</span>
                  <ArrowRight size={16} />
                </button>
              </div>
            </div>
          )}

          {/* STEP 2: NATURAL RESTING BASELINE */}
          {step === 2 && (
            <div className="calibration-step-content">
              <div className="step-instruction">
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <Activity size={18} className="text-cyan animate-pulse" />
                  <h3>Step 2: Rest Hands on Lap or Desk</h3>
                </div>
                <p>
                  Lower your hands down comfortably. The system will detect your natural resting boundary to prevent
                  phantom signs when your hands are relaxed.
                </p>
              </div>

              <div className="calibration-preview-box">
                <div className="calibration-meter-label">
                  <span>Measuring Resting Wrist Y-Level:</span>
                  <span className="calib-val">{restFramesCount} / 20 Frames</span>
                </div>
                <div className="calib-progress-track">
                  <div
                    className="calib-progress-fill"
                    style={{ width: `${Math.min(100, (restFramesCount / 20) * 100)}%` }}
                  />
                </div>

                <div className="rest-boundary-display">
                  <div className="rest-line-visual">
                    <span className="active-zone-label">Active Signing Zone (Top 75%)</span>
                    <div className="rest-threshold-divider" style={{ top: `${restY * 100}%` }}>
                      <span className="rest-tag">Rest Threshold: {Math.round(restY * 100)}%</span>
                    </div>
                    <span className="rest-zone-label">Resting / Inactive Zone</span>
                  </div>
                </div>
              </div>

              <div className="step-actions">
                <button type="button" className="secondary-btn" onClick={() => setStep(1)}>
                  <span>Back</span>
                </button>
                <button type="button" className="primary-btn" onClick={() => setStep(3)}>
                  <span>Next: Head Turning</span>
                  <ArrowRight size={16} />
                </button>
              </div>
            </div>
          )}

          {/* STEP 3: HEAD MOTION / TURNING INTENT */}
          {step === 3 && (
            <div className="calibration-step-content">
              <div className="step-instruction">
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <Sparkles size={18} className="text-cyan animate-pulse" />
                  <h3>Step 3: Turn Head or Nod Once</h3>
                </div>
                <p>
                  Turn your head gently left-and-right (shake) or nod up-and-down (nod). This calibrates your natural
                  head range for hands-free letter locking and question/negation grammar.
                </p>
              </div>

              <div className="calibration-preview-box">
                <div className="head-meters-container">
                  {/* Left / Right Shake */}
                  <div className="head-meter-row">
                    <div className="head-meter-info">
                      <span className="head-meter-lbl">Left / Right Shake (Yaw):</span>
                      <span className="calib-val">{(liveYawOscillation * 100).toFixed(1)}%</span>
                    </div>
                    <div className="calib-progress-track">
                      <div
                        className="calib-progress-fill"
                        style={{
                          width: `${Math.min(100, (liveYawOscillation / (yawThreshold * 1.5)) * 100)}%`,
                          background: shakeDetected ? '#10b981' : '#0ea5e9',
                        }}
                      />
                    </div>
                    <span className={`status-pill ${shakeDetected ? 'confirmed' : 'pending'}`}>
                      {shakeDetected ? '✓ Left/Right Calibrated' : 'Turn head left or right'}
                    </span>
                  </div>

                  {/* Up / Down Nod */}
                  <div className="head-meter-row">
                    <div className="head-meter-info">
                      <span className="head-meter-lbl">Up / Down Nod (Pitch):</span>
                      <span className="calib-val">{(livePitchOscillation * 100).toFixed(1)}%</span>
                    </div>
                    <div className="calib-progress-track">
                      <div
                        className="calib-progress-fill"
                        style={{
                          width: `${Math.min(100, (livePitchOscillation / (pitchThreshold * 1.5)) * 100)}%`,
                          background: nodDetected ? '#10b981' : '#0ea5e9',
                        }}
                      />
                    </div>
                    <span className={`status-pill ${nodDetected ? 'confirmed' : 'pending'}`}>
                      {nodDetected ? '✓ Nod Calibrated' : 'Nod head up or down'}
                    </span>
                  </div>
                </div>
              </div>

              <div className="step-actions">
                <button type="button" className="secondary-btn" onClick={() => setStep(2)}>
                  <span>Back</span>
                </button>
                <button type="button" className="primary-btn" onClick={() => setStep(4)}>
                  <span>Confirm Profile</span>
                  <ArrowRight size={16} />
                </button>
              </div>
            </div>
          )}

          {/* STEP 4: CALIBRATION COMPLETE & PROFILE APPLIED */}
          {step === 4 && (
            <div className="calibration-step-content">
              <div className="step-instruction">
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <CheckCircle2 size={20} className="text-emerald" />
                  <h3 style={{ color: '#34d399' }}>Personal Calibration Profile Active!</h3>
                </div>
                <p>
                  Your individual hand anatomy, motion thresholds, and camera baseline are now locked and saved to your device.
                </p>
              </div>

              <div className="calibration-preview-box">
                <div className="calib-summary-grid">
                  <div className="calib-summary-card">
                    <span className="card-label">Hand Scale</span>
                    <strong className="card-value">{handScale}x</strong>
                    <span className="card-desc">Palm: {palmSpan} | Span: {handWidth}</span>
                  </div>
                  <div className="calib-summary-card">
                    <span className="card-label">Motion Sensitivity</span>
                    <strong className="card-value">{motionThreshold}</strong>
                    <span className="card-desc">Webcam jitter adapted</span>
                  </div>
                  <div className="calib-summary-card">
                    <span className="card-label">Rest Boundary</span>
                    <strong className="card-value">{Math.round(restY * 100)}%</strong>
                    <span className="card-desc">From camera top</span>
                  </div>
                  <div className="calib-summary-card">
                    <span className="card-label">Head Nod & Turn</span>
                    <strong className="card-value">{pitchThreshold} / {yawThreshold}</strong>
                    <span className="card-desc">Pitch / Yaw sensitivity</span>
                  </div>
                </div>

                {/* Optional Manual Fine-Tuning Drawer */}
                <div className="manual-override-section">
                  <button
                    type="button"
                    className="manual-toggle-btn"
                    onClick={() => setShowManualOverride(!showManualOverride)}
                  >
                    <Sliders size={14} />
                    <span>{showManualOverride ? 'Hide Fine-Tuning' : 'Fine-Tune Sliders Manually (Optional)'}</span>
                  </button>

                  {showManualOverride && (
                    <div className="manual-sliders-group">
                      <div className="slider-row">
                        <label>Rest Boundary: {Math.round(restY * 100)}%</label>
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
                      <div className="slider-row">
                        <label>Motion Gating: {motionThreshold}</label>
                        <input
                          type="range"
                          min="0.008"
                          max="0.040"
                          step="0.002"
                          value={motionThreshold}
                          onChange={(e) => setMotionThreshold(parseFloat(e.target.value))}
                          className="calibration-slider"
                        />
                      </div>
                      <div className="slider-row">
                        <label>Head Turning Sensitivity: {yawThreshold}</label>
                        <input
                          type="range"
                          min="0.02"
                          max="0.08"
                          step="0.005"
                          value={yawThreshold}
                          onChange={(e) => setYawThreshold(parseFloat(e.target.value))}
                          className="calibration-slider"
                        />
                      </div>
                    </div>
                  )}
                </div>
              </div>

              <div className="step-actions">
                <button type="button" className="secondary-btn" onClick={handleRestart}>
                  <RefreshCw size={14} />
                  <span>Calibrate Again</span>
                </button>
                <button type="button" className="primary-btn finish-btn" onClick={handleApplyAndFinish}>
                  <Check size={16} />
                  <span>Done & Start Signing</span>
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="modal-footer">
          <button type="button" className="secondary-btn btn-sm" onClick={handleResetDefaults}>
            <RotateCcw size={12} />
            <span>Reset Factory Defaults</span>
          </button>
          <span className="calib-storage-note">Automatically synced over WebSocket</span>
        </div>
      </div>
    </div>
  );
};

export default CalibrationWizardModal;
