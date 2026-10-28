import React from 'react';
import { Hand, Sparkles } from 'lucide-react';

export interface DisambiguationOption {
  label: string;
  display: string;
  hand: 'left' | 'right';
}

export interface DisambiguationOverlayProps {
  options: DisambiguationOption[];
  onSelect: (label: string) => void;
  active: boolean;
}

export const DisambiguationOverlay: React.FC<DisambiguationOverlayProps> = ({
  options,
  onSelect,
  active,
}) => {
  if (!active || options.length < 2) return null;

  const leftOption = options.find((o) => o.hand === 'left') || options[0];
  const rightOption = options.find((o) => o.hand === 'right') || options[1];

  return (
    <div className="disambiguation-hud-container" role="region" aria-label="Sign Disambiguation Options">
      {/* Top Banner Indicator */}
      <div className="disambiguation-banner">
        <Sparkles size={14} className="disambiguation-sparkle" />
        <span className="disambiguation-title">Low-Margin Sign Detected — Choose One</span>
      </div>

      <div className="disambiguation-cards-grid">
        {/* Left Option Card */}
        <button
          type="button"
          className="disambiguation-card left-card"
          onClick={() => onSelect(leftOption.label)}
          aria-label={`Select Option 1: ${leftOption.display}`}
        >
          <div className="card-badge left-badge">
            <Hand size={13} className="hand-lift-icon" />
            <span>Option 1 (Lift Left Hand)</span>
          </div>
          <span className="disambiguation-sign-label">{leftOption.display}</span>
          <span className="card-instruction">Click or raise left hand high</span>
        </button>

        {/* Right Option Card */}
        <button
          type="button"
          className="disambiguation-card right-card"
          onClick={() => onSelect(rightOption.label)}
          aria-label={`Select Option 2: ${rightOption.display}`}
        >
          <div className="card-badge right-badge">
            <Hand size={13} className="hand-lift-icon" />
            <span>Option 2 (Lift Right Hand)</span>
          </div>
          <span className="disambiguation-sign-label">{rightOption.display}</span>
          <span className="card-instruction">Click or raise right hand high</span>
        </button>
      </div>
    </div>
  );
};

