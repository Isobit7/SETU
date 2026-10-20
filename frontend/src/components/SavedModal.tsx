import React, { useState } from 'react';
import { Bookmark, Copy, Volume2, Check, X } from 'lucide-react';
import type { HistoryItem } from '../types';

interface SavedModalProps {
  isOpen: boolean;
  onClose: () => void;
  history: HistoryItem[];
}

export const SavedModal: React.FC<SavedModalProps> = ({ isOpen, onClose, history }) => {
  const [copiedId, setCopiedId] = useState<string | null>(null);

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

  // Items bookmarked in session
  const bookmarked = history.filter((h) => h.isBookmarked);

  const handleCopy = async (id: string, text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 2000);
    } catch (err) {
      console.error('Failed to copy text:', err);
    }
  };

  const handleSpeak = (text: string) => {
    if (!('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'en-US';
    window.speechSynthesis.speak(utterance);
  };

  return (
    <div className="modal-backdrop" onClick={onClose} role="dialog" aria-modal="true">
      <div className="modal-container saved-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-title-wrap">
            <Bookmark size={20} className="text-warning" />
            <h2>Saved Signs & Bookmarks</h2>
          </div>
          <button className="modal-close-btn" onClick={onClose} aria-label="Close dialog">
            <X size={20} />
          </button>
        </div>

        <div className="modal-body">
          {bookmarked.length === 0 ? (
            <div className="empty-history-text">
              <p>No saved signs yet.</p>
              <small>Use the bookmark icon next to any translation in History to save it here for quick reference.</small>
            </div>
          ) : (
            <div className="history-modal-list">
              {bookmarked.map((item) => (
                <div key={item.id} className="history-row">
                  <div className="history-content">
                    <span className="history-text">{item.text}</span>
                    <span className="history-meta">
                      {item.timestamp.toLocaleTimeString([], {
                        hour: '2-digit',
                        minute: '2-digit',
                      })}
                    </span>
                  </div>
                  <div className="history-actions">
                    <button
                      className="icon-mini-btn"
                      onClick={() => handleCopy(item.id, item.text)}
                      title="Copy"
                    >
                      {copiedId === item.id ? <Check size={14} className="text-success" /> : <Copy size={14} />}
                    </button>
                    <button
                      className="icon-mini-btn"
                      onClick={() => handleSpeak(item.text)}
                      title="Speak"
                    >
                      <Volume2 size={14} />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="modal-footer">
          <button className="primary-btn" onClick={onClose}>
            <span>Close</span>
          </button>
        </div>
      </div>
    </div>
  );
};
