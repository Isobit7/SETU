import React from 'react';
import { Camera, ShieldCheck, Cpu } from 'lucide-react';
import type { CameraStatus } from '../types';

interface StatusStripProps {
  cameraStatus: CameraStatus;
  isBackendConnected: boolean;
  guidanceText?: string;
  fps: number;
  latencyMs: number;
}

export const StatusStrip: React.FC<StatusStripProps> = ({
  cameraStatus,
  isBackendConnected,
  guidanceText,
  fps,
  latencyMs,
}) => {
  return (
    <footer className="status-strip" role="contentinfo">
      <div className="status-item">
        <Camera size={15} className="status-icon" />
        <span className="status-label">Camera:</span>
        <span className={`status-val ${cameraStatus}`}>
          {cameraStatus === 'detecting'
            ? 'Active (Tracking)'
            : cameraStatus === 'paused'
            ? 'Paused'
            : cameraStatus === 'stopped'
            ? 'Hardware Released'
            : 'Idle'}
        </span>
      </div>

      <div className="status-separator">|</div>

      <div className="status-item guidance-item">
        <span className="status-label">Framing:</span>
        <span className="guidance-msg">
          {guidanceText || 'Position upper body and both hands clearly in front of camera'}
        </span>
      </div>

      <div className="status-separator">|</div>

      <div className="status-item">
        <Cpu size={15} className="status-icon" />
        <span className="status-label">Pipeline:</span>
        <span className="status-val">
          {isBackendConnected ? `158-D Holistic (${fps} FPS • ${latencyMs}ms)` : 'Offline'}
        </span>
      </div>

      <div className="status-separator">|</div>

      <div className="status-item">
        <ShieldCheck size={15} className="status-icon text-success" />
        <span className="status-val privacy-text">Local Processing • Zero Cloud Storage</span>
      </div>
    </footer>
  );
};
