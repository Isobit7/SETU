import React from 'react';
import { Sparkles } from 'lucide-react';

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
    <div className="disambiguation-top-pill" role="region" aria-label="Sign Suggestions">
      <div className="disambiguation-pill-title">
        <Sparkles size={12} className="disambiguation-sparkle" />
        <span>Suggested:</span>
      </div>
      <div className="disambiguation-pill-options">
        <button
          type="button"
          className="disambiguation-mini-chip chip-left"
          onClick={() => onSelect(leftOption.label)}
          title={`Select ${leftOption.display}`}
        >
          <span className="hand-tag">L</span>
          <span className="chip-sign">{leftOption.display}</span>
        </button>
        <span className="disambiguation-divider">/</span>
        <button
          type="button"
          className="disambiguation-mini-chip chip-right"
          onClick={() => onSelect(rightOption.label)}
          title={`Select ${rightOption.display}`}
        >
          <span className="hand-tag">R</span>
          <span className="chip-sign">{rightOption.display}</span>
        </button>
      </div>
    </div>
  );
};
