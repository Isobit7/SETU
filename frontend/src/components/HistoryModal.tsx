import React, { useState } from 'react';
import { History, Copy, Volume2, Bookmark, Trash2, Check, X, Download } from 'lucide-react';
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

  const handleExportTranscript = () => {
    if (history.length === 0) return;
    const now = new Date();
    const dateStr = now.toISOString().split('T')[0];
    let content = `# SETU — ISL Translation Session Transcript\n`;
    content += `**Date:** ${now.toLocaleDateString()} ${now.toLocaleTimeString()}\n`;
    content += `**Total Recorded Signs:** ${history.length}\n\n`;
    content += `---\n\n`;
    content += `## Chronological Dialogue\n\n`;
    history.forEach((item, idx) => {
      const timeStr = item.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
      const conf = Math.round(item.confidence * 100);
      content += `${idx + 1}. **[${timeStr}]** ${item.text} *(Confidence: ${conf}%)*\n`;
    });
    content += `\n---\n*Exported from SETU — Authentic Indian Sign Language Real-Time AI Workstation*\n`;

    const blob = new Blob([content], { type: 'text/markdown;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `SETU_Session_Transcript_${dateStr}.md`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
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
            <div className="flex items-center gap-2">
              <button
                className="secondary-btn flex items-center gap-1.5 px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 rounded-lg text-xs font-semibold transition"
                onClick={handleExportTranscript}
                title="Download full session transcript as Markdown"
              >
                <Download size={14} />
                <span>Export (.md)</span>
              </button>
              <button className="danger-text-btn" onClick={onClearHistory}>
                <Trash2 size={15} />
                <span>Clear</span>
              </button>
            </div>
          )}
          <button className="primary-btn" onClick={onClose}>
            <span>Close</span>
          </button>
        </div>
      </div>
    </div>
  );
};
