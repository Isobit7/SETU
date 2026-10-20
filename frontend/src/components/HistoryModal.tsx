import React, { useState } from 'react';
import { History, Copy, Volume2, Bookmark, Trash2, Check, X } from 'lucide-react';
import type { HistoryItem } from '../types';

interface HistoryModalProps {
  isOpen: boolean;
  onClose: () => void;
  history: HistoryItem[];
  onClearHistory: () => void;
}

export const HistoryModal: React.FC<HistoryModalProps> = ({
  isOpen,
  onClose,
  history,
  onClearHistory,
}) => {
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [bookmarkedIds, setBookmarkedIds] = useState<Set<string>>(new Set());

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

  const toggleBookmark = (id: string) => {
    setBookmarkedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  return (
    <div className="modal-backdrop" onClick={onClose} role="dialog" aria-modal="true">
      <div className="modal-container history-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-title-wrap">
            <History size={20} className="text-primary" />
            <h2>Session Translation History ({history.length})</h2>
          </div>
          <button className="modal-close-btn" onClick={onClose} aria-label="Close dialog">
            <X size={20} />
          </button>
        </div>

        <div className="modal-body">
          {history.length === 0 ? (
            <div className="empty-history-text">
              <p>No translations recorded in this session yet.</p>
              <small>When you sign before the camera, recognized signs will appear here automatically.</small>
            </div>
          ) : (
            <div className="history-modal-list">
              {history.map((item) => (
                <div key={item.id} className="history-row">
                  <div className="history-content">
                    <span className="history-text">{item.text}</span>
                    <span className="history-meta">
                      {item.timestamp.toLocaleTimeString([], {
                        hour: '2-digit',
                        minute: '2-digit',
                        second: '2-digit',
                      })}{' '}
                      • {Math.round(item.confidence * 100)}% match
                    </span>
                  </div>

                  <div className="history-actions">
                    <button
                      className="icon-mini-btn"
                      onClick={() => handleCopy(item.id, item.text)}
                      title="Copy text"
                    >
                      {copiedId === item.id ? <Check size={14} className="text-success" /> : <Copy size={14} />}
                    </button>
                    <button
                      className="icon-mini-btn"
                      onClick={() => handleSpeak(item.text)}
                      title="Play audio"
                    >
                      <Volume2 size={14} />
                    </button>
                    <button
                      className={`icon-mini-btn ${bookmarkedIds.has(item.id) ? 'bookmarked' : ''}`}
                      onClick={() => toggleBookmark(item.id)}
                      title="Bookmark sign"
                    >
                      <Bookmark size={14} />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="modal-footer history-modal-footer">
          {history.length > 0 && (
            <button className="danger-text-btn" onClick={onClearHistory}>
              <Trash2 size={15} />
              <span>Clear History</span>
            </button>
          )}
          <button className="primary-btn" onClick={onClose}>
            <span>Close</span>
          </button>
        </div>
      </div>
    </div>
  );
};
