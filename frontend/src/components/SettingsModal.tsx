import React from 'react';
import { X, Check, Sliders, Type, Layers, ShieldCheck } from 'lucide-react';
import type { LandmarkStyles, SubtitleSize } from '../types';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  landmarkStyles: LandmarkStyles;
  setLandmarkStyles: React.Dispatch<React.SetStateAction<LandmarkStyles>>;
  subtitleSize: SubtitleSize;
  setSubtitleSize: (size: SubtitleSize) => void;
  confidenceThreshold: number;
  setConfidenceThreshold: (val: number) => void;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  landmarkStyles,
  setLandmarkStyles,
  subtitleSize,
  setSubtitleSize,
  confidenceThreshold,
  setConfidenceThreshold,
}) => {
  // Dismiss on Escape key
  React.useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div className="modal-backdrop" onClick={onClose} role="dialog" aria-modal="true">
      <div className="modal-container settings-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-title-wrap">
            <Sliders size={20} className="text-primary" />
            <h2>Settings & Accessibility</h2>
          </div>
          <button className="modal-close-btn" onClick={onClose} aria-label="Close dialog">
            <X size={20} />
          </button>
        </div>

        <div className="modal-body">
          {/* Subtitle Preferences */}
          <div className="settings-section">
            <div className="section-title">
              <Type size={16} />
              <h3>Movie Subtitle Preferences</h3>
            </div>
            <div className="setting-row">
              <label>Subtitle Font Size</label>
              <div className="button-group-toggle">
                {(['small', 'medium', 'large'] as SubtitleSize[]).map((size) => (
                  <button
                    key={size}
                    className={`toggle-option-btn ${subtitleSize === size ? 'selected' : ''}`}
                    onClick={() => setSubtitleSize(size)}
                  >
                    {size.charAt(0).toUpperCase() + size.slice(1)}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Hand Landmark Display Preferences */}
          <div className="settings-section">
            <div className="section-title">
              <Layers size={16} />
              <h3>Hand Landmark Layers (Hand-Only Overlay)</h3>
            </div>
            <div className="toggle-list">
              <label className="toggle-checkbox-row">
                <input
                  type="checkbox"
                  checked={landmarkStyles.showLeftHand}
                  onChange={(e) =>
                    setLandmarkStyles((prev: LandmarkStyles) => ({ ...prev, showLeftHand: e.target.checked }))
                  }
                />
                <span className="toggle-label-text">
                  <strong>Left Hand Skeletons</strong> (Emerald Green #10B981)
                </span>
              </label>

              <label className="toggle-checkbox-row">
                <input
                  type="checkbox"
                  checked={landmarkStyles.showRightHand}
                  onChange={(e) =>
                    setLandmarkStyles((prev: LandmarkStyles) => ({ ...prev, showRightHand: e.target.checked }))
                  }
                />
                <span className="toggle-label-text">
                  <strong>Right Hand Skeletons</strong> (Amber Orange #F59E0B)
                </span>
              </label>

              <p className="setting-desc mt-1">
                * Note: Body, face, and torso landmarks are permanently disabled from visual overlay for optimal performance and privacy.
              </p>
            </div>
          </div>

          {/* Model Sensitivity */}
          <div className="settings-section">
            <div className="section-title">
              <Sliders size={16} />
              <h3>Detection Sensitivity Threshold</h3>
            </div>
            <div className="setting-row">
              <div className="slider-label-row">
                <label>Confidence Gate</label>
                <span className="threshold-val">{Math.round(confidenceThreshold * 100)}%</span>
              </div>
              <input
                type="range"
                min="0.20"
                max="0.90"
                step="0.05"
                value={confidenceThreshold}
                onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                className="slider-range"
              />
              <span className="setting-desc">
                Minimum score required to trigger a recognized sign instead of uncertain. Default is 50%.
              </span>
            </div>
          </div>

          {/* Privacy Information */}
          <div className="settings-section privacy-card-box">
            <div className="section-title">
              <ShieldCheck size={16} className="text-success" />
              <h3>Local Processing Guarantee</h3>
            </div>
            <p className="privacy-desc">
              All video frames are processed locally on your machine in real-time. No webcam frames, video recordings, or biometrics are ever sent to the cloud or saved to disk.
            </p>
          </div>
        </div>

        <div className="modal-footer">
          <button className="primary-btn" onClick={onClose}>
            <Check size={16} />
            <span>Save & Close</span>
          </button>
        </div>
      </div>
    </div>
  );
};
