import React from 'react';
import { ArrowRightLeft, ShieldCheck, Wifi, WifiOff, Eye, Sliders } from 'lucide-react';

interface TopBarProps {
  isBackendConnected: boolean;
  fps: number;
  latencyMs: number;
  onOpenSettings: () => void;
  onOpenVocabulary: () => void;
}

export const TopBar: React.FC<TopBarProps> = ({
  isBackendConnected,
  fps,
  latencyMs,
  onOpenSettings,
  onOpenVocabulary,
}) => {
  return (
    <header className="topbar" role="banner">
      <div className="topbar-left">
        {/* Supported translation pair */}
        <div className="language-selector-badge" title="Active Recognition Direction">
          <span className="lang-pill primary">Indian Sign Language (ISL)</span>
          <ArrowRightLeft size={16} className="direction-icon" />
          <span className="lang-pill secondary">English (Text & Speech)</span>
        </div>

        <button
          className="vocab-quick-btn"
          onClick={onOpenVocabulary}
          title="View 5 supported ISL vocabulary signs"
        >
          <Eye size={15} />
          <span>5 Signs Supported</span>
        </button>
      </div>

      <div className="topbar-right">
        {/* Backend health status */}
        <div
          className={`connection-pill ${isBackendConnected ? 'connected' : 'disconnected'}`}
          title={
            isBackendConnected
              ? `Connected to SETU Engine (${latencyMs}ms latency, ~${fps} FPS)`
              : 'Backend disconnected (ws://localhost:8000)'
          }
        >
          {isBackendConnected ? (
            <>
              <Wifi size={15} />
              <span>SETU Engine Online</span>
              <span className="fps-counter">{fps > 0 ? `${fps} FPS` : ''}</span>
            </>
          ) : (
            <>
              <WifiOff size={15} />
              <span>Backend Offline</span>
            </>
          )}
        </div>

        {/* Privacy badge */}
        <div className="privacy-pill" title="Local frame processing. No external cloud recording.">
          <ShieldCheck size={15} />
          <span>Local Engine</span>
        </div>

        {/* Accessibility & Settings quick button */}
        <button
          className="icon-settings-btn"
          onClick={onOpenSettings}
          aria-label="Open settings and accessibility preferences"
          title="Settings & Accessibility"
        >
          <Sliders size={18} />
        </button>
      </div>
    </header>
  );
};
