import React from 'react';
import { HelpCircle, Hand, Camera, Sliders, ShieldCheck, X } from 'lucide-react';

interface HelpModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const HelpModal: React.FC<HelpModalProps> = ({ isOpen, onClose }) => {
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
      <div className="modal-container help-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-title-wrap">
            <HelpCircle size={20} className="text-primary" />
            <h2>SETU Help & User Guide</h2>
          </div>
          <button className="modal-close-btn" onClick={onClose} aria-label="Close dialog">
            <X size={20} />
          </button>
        </div>

        <div className="modal-body">
          <div className="help-section">
            <div className="help-icon-title">
              <Camera size={18} className="text-primary" />
              <h4>1. Camera & Lighting</h4>
            </div>
            <p>
              Position your webcam so your hands and chest are clearly visible. Ensure good frontal lighting so MediaPipe can detect wrist and finger joints accurately.
            </p>
          </div>

          <div className="help-section">
            <div className="help-icon-title">
              <Hand size={18} className="text-secondary" />
              <h4>2. Hand-Only Landmarks</h4>
            </div>
            <p>
              SETU renders only your left hand (emerald green) and right hand (amber orange) skeleton joints. To protect privacy and reduce clutter, facial mesh and torso outlines are not displayed.
            </p>
          </div>

          <div className="help-section">
            <div className="help-icon-title">
              <Sliders size={18} className="text-warning" />
              <h4>3. Signing & Resting Behavior</h4>
            </div>
            <p>
              When your hands are relaxed or resting below your chest, the system shows <strong>Waiting for a sign</strong>. When you raise your hands to sign one of the 5 supported gestures, the recognized text appears immediately in the movie subtitle and translation card.
            </p>
          </div>

          <div className="help-section">
            <div className="help-icon-title">
              <ShieldCheck size={18} className="text-success" />
              <h4>4. Local Processing Privacy</h4>
            </div>
            <p>
              All video frames are processed entirely on your local machine using the Python backend. No webcam frames, video recordings, or biometrics leave your computer.
            </p>
          </div>
        </div>

        <div className="modal-footer">
          <button className="primary-btn" onClick={onClose}>
            <span>Got It</span>
          </button>
        </div>
      </div>
    </div>
  );
};
