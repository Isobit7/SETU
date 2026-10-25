import React, { useState } from 'react';
import { X, Check, Type, Cpu, Camera, BookOpen } from 'lucide-react';
import { WebcamAlphabetRecognition } from './WebcamAlphabetRecognition';

interface AlphabetModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const AlphabetModal: React.FC<AlphabetModalProps> = ({ isOpen, onClose }) => {
  const [activeTab, setActiveTab] = useState<'live' | 'reference'>('live');

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

  const alphabet = Array.from({ length: 26 }, (_, i) => String.fromCharCode(65 + i));

  return (
    <div className="modal-backdrop" onClick={onClose} role="dialog" aria-modal="true">
      <div
        className="modal-container alphabet-modal"
        onClick={(e) => e.stopPropagation()}
        style={{ maxWidth: activeTab === 'live' ? '920px' : '780px', transition: 'max-width 0.2s ease' }}
      >
        <div className="modal-header">
          <div className="modal-title-wrap">
            <Type size={20} className="text-primary" />
            <h2>ISL 26-Class Alphabet Recognition</h2>
          </div>
          <button className="modal-close-btn" onClick={onClose} aria-label="Close dialog">
            <X size={20} />
          </button>
        </div>

        {/* Tab Switcher */}
        <div className="alphabet-modal-tabs">
          <button
            onClick={() => setActiveTab('live')}
            className={`alphabet-tab-btn ${activeTab === 'live' ? 'active' : ''}`}
          >
            <Camera size={14} />
            <span>Live Camera Recognition</span>
          </button>
          <button
            onClick={() => setActiveTab('reference')}
            className={`alphabet-tab-btn ${activeTab === 'reference' ? 'active' : ''}`}
          >
            <BookOpen size={14} />
            <span>Alphabet Reference Chart</span>
          </button>
        </div>

        <div className="modal-body">
          {activeTab === 'live' ? (
            <WebcamAlphabetRecognition apiBaseUrl="http://localhost:8000" />
          ) : (
            <>
              {/* Model Metrics Callout */}
              <div className="alphabet-engine-card">
                <Cpu size={22} className="text-primary" style={{ flexShrink: 0 }} />
                <div>
                  <strong>Active PyTorch MLP Engine:</strong> RealSign 26-Class Alphabet Classifier (63-D → 128 → 64 → 26).
                  <div className="alphabet-engine-meta">
                    <span style={{ color: 'var(--color-emerald)', fontWeight: 600 }}>91.05% Test Accuracy</span> • <span style={{ color: 'var(--color-slate-blue)', fontWeight: 600 }}>90.80% Macro F1</span> • 22,580 MediaPipe 3D Landmark Vectors.
                  </div>
                </div>
              </div>

              <p className="modal-intro" style={{ marginBottom: '14px' }}>
                Position either hand in front of your camera. SETU's dedicated HandLandmarker tracks 21 3D joints, normalizes coordinates by wrist origin and palm span, and classifies letters A through Z with sub-15ms latency.
              </p>

              {/* Alphabet Grid */}
              <div className="alphabet-grid">
                {alphabet.map((letter) => (
                  <div key={letter} className="alphabet-tile">
                    <div className="alphabet-tile-letter">{letter}</div>
                    <div className="alphabet-tile-sub">Letter {letter}</div>
                  </div>
                ))}
              </div>


            <div className="vocab-notice-box">
              <h4>Real-Time Signing Tips</h4>
              <p>
                Keep your fingers facing the camera lens. You can toggle between <strong>Alphabet (A–Z)</strong> and <strong>Phrases (5 Signs)</strong> using the toggle bar below the video stream. Subtitles will show <strong>LETTER [X]</strong> along with real-time confidence scores and landmark skeleton overlays.
              </p>
            </div>
          </>
        )}
      </div>

        <div className="modal-footer">
          <button className="primary-btn" onClick={onClose}>
            <Check size={16} />
            <span>Ready to Sign</span>
          </button>
        </div>
      </div>
    </div>
  );
};
