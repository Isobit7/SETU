import React from 'react';
import type { FramingFeedback } from '../types';

interface VisualFramingGuideProps {
  feedback: FramingFeedback;
  visible: boolean;
  mirrored?: boolean;
  enforceFraming?: boolean;
  onToggleEnforceFraming?: () => void;
  onToggleVisible?: () => void;
}

/**
 * VisualFramingGuide
 * Renders an unobtrusive, studio-grade viewfinder framing boundary.
 * Never obstructs or covers the user's face, neck, or signing space.
 */
export const VisualFramingGuide: React.FC<VisualFramingGuideProps> = ({
  feedback,
  visible,
}) => {
  if (!visible) return null;

  const isOptimal = feedback.isOptimal;

  return (
    <div className="visual-framing-overlay" aria-hidden="true">
      <svg
        className="framing-svg-canvas"
        viewBox="0 0 1000 750"
        preserveAspectRatio="none"
      >
        {/* Safe Signing Perimeter */}
        <rect
          x="70"
          y="60"
          width="860"
          height="620"
          rx="18"
          className="viewfinder-safe-zone"
          strokeWidth="1"
        />

        {/* Studio Viewfinder Corner Reticles */}
        <g
          className={`viewfinder-brackets ${isOptimal ? 'optimal' : ''}`}
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          fill="none"
        >
          {/* Top-Left Corner */}
          <path d="M 130 60 L 88 60 A 18 18 0 0 0 70 78 L 70 120" />
          {/* Top-Right Corner */}
          <path d="M 870 60 L 912 60 A 18 18 0 0 1 930 78 L 930 120" />
          {/* Bottom-Left Corner */}
          <path d="M 70 622 L 70 664 A 18 18 0 0 0 88 682 L 130 682" />
          {/* Bottom-Right Corner */}
          <path d="M 930 622 L 930 664 A 18 18 0 0 1 912 682 L 870 682" />
        </g>
      </svg>
    </div>
  );
};
