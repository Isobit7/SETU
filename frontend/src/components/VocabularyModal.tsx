import React from 'react';
import { X, Check, BookOpen, Hand } from 'lucide-react';

interface VocabularyModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const VocabularyModal: React.FC<VocabularyModalProps> = ({ isOpen, onClose }) => {
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

  const signs = [
    {
      sign: 'HELLO',
      category: 'Greeting',
      description: 'Open hand salute/wave near forehead or shoulder.',
      tips: 'Show palm clearly to camera and make a natural open hand gesture.',
    },
    {
      sign: 'THANK YOU',
      category: 'Courtesy',
      description: 'Flat hand starting from chin/lips and extending outwards forward.',
      tips: 'Ensure fingers are extended straight as the hand moves toward the camera.',
    },
    {
      sign: 'GOOD MORNING',
      category: 'Greeting',
      description: 'Compound gesture: sign for "good" followed by "morning".',
      tips: 'Keep motion within the signing frame; both hands can be involved.',
    },
    {
      sign: 'I (ME)',
      category: 'Pronoun',
      description: 'Index finger pointing inward to chest.',
      tips: 'Keep hand at chest level; index clearly separated from other fingers.',
    },
    {
      sign: 'YOU',
      category: 'Pronoun',
      description: 'Index finger pointing outward towards your conversation partner.',
      tips: 'Extend index finger forward directly towards the webcam.',
    },
  ];

  return (
    <div className="modal-backdrop" onClick={onClose} role="dialog" aria-modal="true">
      <div className="modal-container" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-title-wrap">
            <BookOpen size={20} className="text-primary" />
            <h2>Supported ISL Vocabulary (5-Sign MVP)</h2>
          </div>
          <button className="modal-close-btn" onClick={onClose} aria-label="Close dialog">
            <X size={20} />
          </button>
        </div>

        <div className="modal-body">
          <p className="modal-intro">
            SETU recognizes core Indian Sign Language (ISL) gestures using precise hand tracking and spatiotemporal sequence classification.
          </p>

          <div className="vocab-cards-grid">
            {signs.map((item) => (
              <div key={item.sign} className="vocab-card">
                <div className="vocab-card-header">
                  <span className="vocab-sign-name">{item.sign}</span>
                  <span className="vocab-badge">{item.category}</span>
                </div>
                <p className="vocab-desc">{item.description}</p>
                <div className="vocab-tip">
                  <Hand size={14} className="text-secondary" />
                  <span>{item.tips}</span>
                </div>
              </div>
            ))}
          </div>

          <div className="vocab-notice-box">
            <h4>Resting & Continuous Signing Notes</h4>
            <p>
              When hands are lowered or resting by your side, the model automatically switches to <strong>Waiting for a sign...</strong> (no_sign class), preventing false triggers.
            </p>
          </div>
        </div>

        <div className="modal-footer">
          <button className="primary-btn" onClick={onClose}>
            <Check size={16} />
            <span>Got It</span>
          </button>
        </div>
      </div>
    </div>
  );
};
