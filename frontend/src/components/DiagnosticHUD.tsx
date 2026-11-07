import React, { useState } from 'react';
import { Activity, Eye, Zap, Sun, Minimize2, Maximize2, ShieldCheck, AlertCircle } from 'lucide-react';
import type { RecognitionPayload } from '../types';

interface DiagnosticHUDProps {
  currentResult: RecognitionPayload | null;
  activeThreshold: number;
  visible: boolean;
  onClose?: () => void;
}

export const DiagnosticHUD: React.FC<DiagnosticHUDProps> = ({
  currentResult,
  activeThreshold,
  visible,
}) => {
  const [collapsed, setCollapsed] = useState<boolean>(false);

  if (!visible) return null;

  const leftHandPresent = (currentResult?.landmarks?.left_hand?.length || 0) > 0;
  const rightHandPresent = (currentResult?.landmarks?.right_hand?.length || 0) > 0;
  const handDetected = currentResult?.hand_detected ?? (leftHandPresent || rightHandPresent);

  const confidence = currentResult?.confidence ?? 0;
  const confPercent = Math.round(confidence * 100);
  const thresholdPercent = Math.round(activeThreshold * 100);
  const isConfident = confPercent >= thresholdPercent;

  // Kinetic energy from backend (or fallback 0.0)
  const kinetic = (currentResult as any)?.kinetic_energy ?? 0.0;
  const motionState = (currentResult as any)?.motion_state || (kinetic > 0.055 ? 'moving' : 'settled');
  const isHoldingStatic = motionState === 'settled' || kinetic <= 0.045;

  // Lighting
  const lightScore = currentResult?.lighting?.score ?? 1.0;
  const lightPercent = Math.round(lightScore * 100);
  const isLowLight = lightScore < 0.38;

  // Head gesture
  const headGesture = currentResult?.head_gesture || (currentResult as any)?.non_manual_markers?.head_gesture;

  return (
    <div
      className={`diagnostic-hud-root ${collapsed ? 'collapsed' : ''}`}
      role="region"
      aria-label="Real-Time Computer Vision & Model Diagnostics"
    >
      <div className="hud-header">
        <div className="hud-title">
          <Activity size={14} className="hud-icon text-cyan" />
          <span>Live Vision & Model Diagnostics</span>
        </div>
        <button
          type="button"
          className="hud-toggle-btn"
          onClick={() => setCollapsed(!collapsed)}
          title={collapsed ? 'Expand HUD' : 'Collapse HUD'}
          aria-label={collapsed ? 'Expand HUD' : 'Collapse HUD'}
        >
          {collapsed ? <Maximize2 size={12} /> : <Minimize2 size={12} />}
        </button>
      </div>

      {!collapsed && (
        <div className="hud-body">
          {/* Row 1: Hand Tracking Status */}
          <div className="hud-metric-row">
            <span className="metric-label">
              <Eye size={12} />
              <span>Hand Tracking:</span>
            </span>
            <div className="metric-values">
              <span className={`hud-pill ${leftHandPresent ? 'active' : 'inactive'}`}>
                LH: {leftHandPresent ? 'Tracked' : 'None'}
              </span>
              <span className={`hud-pill ${rightHandPresent ? 'active' : 'inactive'}`}>
                RH: {rightHandPresent ? 'Tracked' : 'None'}
              </span>
            </div>
          </div>

          {/* Row 2: Kinetic Motion / Holding State */}
          <div className="hud-metric-row">
            <span className="metric-label">
              <Zap size={12} />
              <span>Motion State:</span>
            </span>
            <div className="metric-values">
              <span className={`hud-pill ${isHoldingStatic && handDetected ? 'success' : 'warn'}`}>
                {!handDetected
                  ? 'Awaiting Hands'
                  : isHoldingStatic
                  ? 'Holding (Static)'
                  : 'Moving (Transit)'}
              </span>
              <span className="kinetic-val" title="Hand kinetic velocity">
                {kinetic.toFixed(3)}
              </span>
            </div>
          </div>
          {/* Kinetic Energy Bar */}
          <div className="hud-progress-track" title="Lower kinetic energy means steady hold for recognition">
            <div
              className={`hud-progress-fill ${kinetic > 0.055 ? 'fill-warn' : 'fill-good'}`}
              style={{ width: `${Math.min(100, (kinetic / 0.10) * 100)}%` }}
            />
          </div>

          {/* Row 3: Confidence vs Threshold Gauge */}
          <div className="hud-metric-row">
            <span className="metric-label">
              <ShieldCheck size={12} />
              <span>Model Confidence:</span>
            </span>
            <div className="metric-values">
              <span className={`hud-pill ${isConfident && handDetected ? 'success' : 'neutral'}`}>
                {confPercent}% (Min: {thresholdPercent}%)
              </span>
            </div>
          </div>
          {/* Confidence Bar */}
          <div className="hud-progress-track">
            <div
              className={`hud-progress-fill ${isConfident ? 'fill-good' : 'fill-dim'}`}
              style={{ width: `${confPercent}%` }}
            />
          </div>

          {/* Row 4: Lighting & Distance */}
          <div className="hud-metric-row">
            <span className="metric-label">
              <Sun size={12} />
              <span>Lighting & Framing:</span>
            </span>
            <div className="metric-values">
              <span className={`hud-pill ${isLowLight ? 'warn' : 'success'}`}>
                {isLowLight ? 'Low Light' : 'Optimal Light'} ({lightPercent}%)
              </span>
            </div>
          </div>

          {/* Row 5: Intent Recognition Status */}
          <div className="hud-metric-row hud-intent-row">
            <span className="metric-label">
              <AlertCircle size={12} />
              <span>Head Intent:</span>
            </span>
            <div className="metric-values">
              {headGesture === 'nod' ? (
                <span className="hud-pill success pulse">✓ Nod Detected (Locking)</span>
              ) : headGesture === 'shake' ? (
                <span className="hud-pill warn pulse">✕ Shake Detected (Cancel)</span>
              ) : (
                <span className="hud-pill inactive">Nod = Lock | Shake = Cancel</span>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
export default DiagnosticHUD;

