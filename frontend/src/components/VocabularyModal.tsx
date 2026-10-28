import React, { useState, useEffect } from 'react';
import { X, Check, BookOpen, Hand, RefreshCw } from 'lucide-react';
import { ApiService, type VocabularyItem } from '../services/apiService';

interface VocabularyModalProps {
  isOpen: boolean;
  onClose: () => void;
  activeModel?: string;
}

const DEFAULT_SIGNS: Array<{ sign: string; category: string; description: string; tips: string }> = [
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

export const VocabularyModal: React.FC<VocabularyModalProps> = ({ isOpen, onClose, activeModel: _activeModel }) => {
  const [vocabItems, setVocabItems] = useState<Array<{ sign: string; category: string; description: string; tips: string }>>(DEFAULT_SIGNS);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Dismiss on Escape key
  useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  // Dynamically load vocabulary on open
  useEffect(() => {
    if (!isOpen) return;

    let isMounted = true;
    setIsLoading(true);
    setErrorMsg(null);

    ApiService.fetchVocabulary()
      .then((res) => {
        if (!isMounted) return;
        if (res && res.vocabulary && res.vocabulary.length > 0) {
          const mapped = res.vocabulary.map((item: VocabularyItem) => ({
            sign: item.display || item.key.toUpperCase().replace('_', ' '),
            category: item.category || 'General',
            description: item.description || 'Natural Indian Sign Language gesture.',
            tips: 'Position hands clearly within camera framing zone.',
          }));
          setVocabItems(mapped);
        }
      })
      .catch((err) => {
        if (!isMounted) return;
        console.warn('Could not fetch dynamic vocabulary from backend, using default catalog:', err);
        setErrorMsg('Operating with local cached vocabulary.');
      })
      .finally(() => {
        if (isMounted) setIsLoading(false);
      });

    return () => {
      isMounted = false;
    };
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div className="modal-backdrop" onClick={onClose} role="dialog" aria-modal="true">
      <div className="modal-container" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-title-wrap">
            <BookOpen size={20} className="text-primary" />
            <h2>Supported ISL Vocabulary</h2>
            {isLoading && <RefreshCw size={14} className="animate-spin text-secondary" />}
          </div>
          <button className="modal-close-btn" onClick={onClose} aria-label="Close dialog">
            <X size={20} />
          </button>
        </div>

        <div className="modal-body">
          <p className="modal-intro">
            SETU recognizes core Indian Sign Language (ISL) gestures using precise hand tracking and spatiotemporal sequence classification.
          </p>

          {errorMsg && (
            <p className="text-xs text-amber-400 mb-2">{errorMsg}</p>
          )}

          <div className="vocab-cards-grid">
            {vocabItems.map((item) => (
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
