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
        <div style={{
          display: 'flex',
          gap: '8px',
          padding: '8px 20px',
          borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
          background: 'rgba(0, 0, 0, 0.2)',
        }}>
          <button
            onClick={() => setActiveTab('live')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '6px 14px',
              borderRadius: '6px',
              border: activeTab === 'live' ? '1px solid #38BDF8' : '1px solid transparent',
              background: activeTab === 'live' ? 'rgba(56, 189, 248, 0.15)' : 'transparent',
              color: activeTab === 'live' ? '#38BDF8' : '#94A3B8',
              fontWeight: 600,
              fontSize: '13px',
              cursor: 'pointer',
            }}
          >
            <Camera size={14} />
            <span>Live Camera Recognition</span>
          </button>
          <button
            onClick={() => setActiveTab('reference')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '6px 14px',
              borderRadius: '6px',
              border: activeTab === 'reference' ? '1px solid #38BDF8' : '1px solid transparent',
              background: activeTab === 'reference' ? 'rgba(56, 189, 248, 0.15)' : 'transparent',
              color: activeTab === 'reference' ? '#38BDF8' : '#94A3B8',
              fontWeight: 600,
              fontSize: '13px',
              cursor: 'pointer',
            }}
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
          <div style={{
            display: 'flex',
            gap: '1rem',
            background: 'rgba(59, 130, 246, 0.08)',
            border: '1px solid rgba(59, 130, 246, 0.25)',
            borderRadius: '8px',
            padding: '12px 16px',
            marginBottom: '18px',
            alignItems: 'center',
            fontSize: '13px',
          }}>
            <Cpu size={22} className="text-primary" style={{ flexShrink: 0 }} />
            <div>
              <strong>Active PyTorch MLP Engine:</strong> RealSign 26-Class Alphabet Classifier (63-D → 128 → 64 → 26).
              <div style={{ color: 'var(--text-muted, #94A3B8)', marginTop: '2px' }}>
                <span style={{ color: '#10B981', fontWeight: 600 }}>91.05% Test Accuracy</span> • <span style={{ color: '#38BDF8', fontWeight: 600 }}>90.80% Macro F1</span> • 22,580 MediaPipe 3D Landmark Vectors.
              </div>
            </div>
          </div>

          <p className="modal-intro" style={{ marginBottom: '14px' }}>
            Position either hand in front of your camera. SETU's dedicated HandLandmarker tracks 21 3D joints, normalizes coordinates by wrist origin and palm span, and classifies letters A through Z with sub-15ms latency.
          </p>

          {/* Alphabet Grid */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(90px, 1fr))',
            gap: '10px',
            marginBottom: '18px',
          }}>
            {alphabet.map((letter) => (
              <div
                key={letter}
                style={{
                  background: 'rgba(255, 255, 255, 0.03)',
                  border: '1px solid rgba(255, 255, 255, 0.08)',
                  borderRadius: '8px',
                  padding: '10px 8px',
                  textAlign: 'center',
                  transition: 'transform 0.15s ease, border-color 0.15s ease',
                  cursor: 'default',
                }}
              >
                <div style={{
                  fontSize: '24px',
                  fontWeight: 700,
                  color: '#38BDF8',
                  lineHeight: '1.2',
                }}>
                  {letter}
                </div>
                <div style={{
                  fontSize: '11px',
                  color: 'var(--text-muted, #94A3B8)',
                  marginTop: '2px',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px',
                }}>
                  Letter {letter}
                </div>
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
