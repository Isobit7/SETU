import React, { useState, useEffect, useRef } from 'react';
import {
  Mic,
  MicOff,
  Volume2,
  Send,
  Download,
  Trash2,
  Video,
  User,
} from 'lucide-react';
import { Avatar3DViewer } from './Avatar3DViewer';
import type { RecognitionPayload } from '../types';

interface Message {
  id: string;
  sender: 'signer' | 'speaker';
  text: string;
  indicText?: string;
  timestamp: string;
}

interface ConversationStudioProps {
  currentResult: RecognitionPayload | null;
  onSpeak?: (text: string) => void;
  videoRef: React.RefObject<HTMLVideoElement | null>;
  isCameraActive: boolean;
  onStartCamera: () => Promise<void>;
  onStopCamera: () => void;
}

export const ConversationStudio: React.FC<ConversationStudioProps> = ({
  currentResult,
  onSpeak,
  videoRef,
  isCameraActive,
  onStartCamera,
  onStopCamera,
}) => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      sender: 'speaker',
      text: 'Hello! How can I help you today?',
      indicText: 'नमस्ते! आज मैं आपकी क्या मदद कर सकता हूँ?',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    },
  ]);

  const [speechInput, setSpeechInput] = useState<string>('');
  const [isListening, setIsListening] = useState<boolean>(false);
  const [activeAvatarSign, setActiveAvatarSign] = useState<string>('GOOD_MORNING');
  const [showAvatar, setShowAvatar] = useState<boolean>(true);
  const recognitionRef = useRef<any>(null);

  // Active signer sentence from live recognition
  const activeSignerSentence =
    currentResult?.translated_text || currentResult?.sentence_buffer || '';

  // Setup Web Speech API for voice listening
  useEffect(() => {
    const SpeechRecognition =
      (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = 'en-IN';

      recognition.onresult = (event: any) => {
        let finalTranscript = '';
        for (let i = event.resultIndex; i < event.results.length; ++i) {
          if (event.results[i].isFinal) {
            finalTranscript += event.results[i][0].transcript;
          }
        }
        if (finalTranscript) {
          setSpeechInput((prev) => (prev ? `${prev} ${finalTranscript}` : finalTranscript));
        }
      };

      recognition.onerror = () => {
        setIsListening(false);
      };

      recognition.onend = () => {
        setIsListening(false);
      };

      recognitionRef.current = recognition;
    }
  }, []);

  const toggleListening = () => {
    if (!recognitionRef.current) return;
    if (isListening) {
      recognitionRef.current.stop();
      setIsListening(false);
    } else {
      try {
        recognitionRef.current.start();
        setIsListening(true);
      } catch {
        setIsListening(false);
      }
    }
  };

  // Signer commits their sentence into the conversation
  const handleCommitSignerMessage = () => {
    if (!activeSignerSentence.trim()) return;

    const newMsg: Message = {
      id: String(Date.now()),
      sender: 'signer',
      text: activeSignerSentence.trim(),
      indicText: currentResult?.translated_indic?.hi,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, newMsg]);

    // Automatically speak message to the hearing participant
    if (onSpeak) {
      onSpeak(activeSignerSentence.trim());
    }
  };

  // Hearing participant sends spoken/typed response
  const handleSendSpeakerMessage = () => {
    if (!speechInput.trim()) return;

    const upperText = speechInput.trim().toUpperCase();
    if (upperText.includes('GOOD MORNING') || upperText.includes('MORNING')) {
      setActiveAvatarSign('GOOD_MORNING');
    } else if (upperText.includes('THANK YOU') || upperText.includes('THANKS') || upperText.includes('THANK')) {
      setActiveAvatarSign('THANK_YOU');
    } else if (upperText.includes('NAMASTE') || upperText.includes('VANAKKAM')) {
      setActiveAvatarSign('NAMASTE');
    } else if (upperText.includes('HELP')) {
      setActiveAvatarSign('HELP');
    } else if (upperText.includes('PLEASE') || upperText.includes('PLZ')) {
      setActiveAvatarSign('PLEASE');
    } else if (upperText.includes('WELCOME')) {
      setActiveAvatarSign('WELCOME');
    } else if (upperText.includes('WATER') || upperText.includes('DRINK')) {
      setActiveAvatarSign('WATER');
    } else if (upperText.includes('FOOD') || upperText.includes('EAT')) {
      setActiveAvatarSign('FOOD');
    } else if (upperText.includes('YES') || upperText.includes('YEAH')) {
      setActiveAvatarSign('YES');
    } else if (upperText.includes('NO') || upperText.includes('NOT')) {
      setActiveAvatarSign('NO');
    } else if (upperText.includes('SORRY')) {
      setActiveAvatarSign('SORRY');
    } else if (upperText.includes('STOP')) {
      setActiveAvatarSign('STOP');
    } else if (upperText.includes('HELLO') || upperText.includes('HI') || upperText.includes('HEY')) {
      setActiveAvatarSign('HELLO');
    } else {
      const words = upperText.split(/\s+/).filter(Boolean);
      if (words.length > 0) {
        setActiveAvatarSign(words[0]);
      }
    }

    const newMsg: Message = {
      id: String(Date.now()),
      sender: 'speaker',
      text: speechInput.trim(),
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, newMsg]);
    setSpeechInput('');
  };

  const exportTranscript = () => {
    const text = messages
      .map(
        (m) =>
          `[${m.timestamp}] ${m.sender === 'signer' ? '🤟 Signer (ISL)' : '🎙️ Speaker (Voice)'}: ${m.text}`
      )
      .join('\n');

    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `SETU_Conversation_${new Date().toISOString().slice(0, 10)}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="conversation-studio-container">
      {/* LEFT PANE: Signer Camera & Live Recognition */}
      <div className="conversation-pane conversation-pane-left">
        <div className="conversation-pane-header">
          <div className="conversation-header-info">
            <span className="conversation-icon-badge badge-indigo">
              <Video size={18} />
            </span>
            <div>
              <h2 className="conversation-pane-title">Signer View (ISL → Speech)</h2>
              <p className="conversation-pane-desc">Live Indian Sign Language Camera</p>
            </div>
          </div>
          <button
            type="button"
            onClick={isCameraActive ? onStopCamera : onStartCamera}
            className={`conversation-camera-btn ${isCameraActive ? 'btn-stop' : 'btn-start'}`}
          >
            {isCameraActive ? 'Stop Camera' : 'Start Camera'}
          </button>
        </div>

        {/* Camera Container */}
        <div className="conversation-camera-view">
          <video
            ref={videoRef as any}
            autoPlay
            playsInline
            muted
            className={`conversation-video-element ${isCameraActive ? 'active' : 'hidden'}`}
          />
          {!isCameraActive && (
            <div className="conversation-camera-idle">
              <Video size={40} className="conversation-camera-idle-icon" />
              <p className="conversation-camera-idle-text">Camera preview off. Click "Start Camera" above.</p>
            </div>
          )}

          {/* Current Live Sign Overlay */}
          {currentResult?.is_known_sign && (
            <div className="conversation-live-sign-pill">
              {currentResult?.label ? currentResult.label.toUpperCase() : ''}
            </div>
          )}
        </div>

        {/* Live Signer Sentence Box */}
        <div className="conversation-sentence-card">
          <span className="conversation-sentence-label">
            Accumulated Sentence
          </span>
          <p className="conversation-sentence-text">
            {activeSignerSentence || (
              <span className="conversation-sentence-placeholder">Waiting for sign gestures...</span>
            )}
          </p>

          {currentResult?.translated_indic?.hi && (
            <p className="conversation-indic-text">
              {currentResult.translated_indic.hi}
            </p>
          )}

          <div className="conversation-signer-actions">
            <button
              type="button"
              onClick={handleCommitSignerMessage}
              disabled={!activeSignerSentence.trim()}
              className="conversation-speak-btn"
            >
              <Send size={14} /> Speak to Participant
            </button>
            <button
              type="button"
              onClick={() => onSpeak && onSpeak(activeSignerSentence)}
              disabled={!activeSignerSentence.trim()}
              className="conversation-quick-audio-btn"
              title="Speak immediately"
            >
              <Volume2 size={16} />
            </button>
          </div>
        </div>
      </div>

      {/* RIGHT PANE: Hearing Participant Voice & Dialogue History */}
      <div className="conversation-pane conversation-pane-right">
        {/* Header Actions */}
        <div className="conversation-pane-header">
          <div className="conversation-header-info">
            <span className="conversation-icon-badge badge-emerald">
              <User size={18} />
            </span>
            <div>
              <h2 className="conversation-pane-title">Conversation Dialogue</h2>
              <p className="conversation-pane-desc">Bilingual Sign ↔ Voice Transcript</p>
            </div>
          </div>
          <div className="conversation-dialogue-actions">
            <button
              type="button"
              onClick={() => setShowAvatar(!showAvatar)}
              className="conversation-header-btn"
              style={{ background: showAvatar ? '#4f46e5' : 'rgba(255,255,255,0.08)', color: '#ffffff' }}
              title="Toggle 3D Signer Avatar for Deaf participant"
            >
              👤 {showAvatar ? '3D Avatar Active' : 'Show 3D Avatar'}
            </button>
            <button
              type="button"
              onClick={exportTranscript}
              className="conversation-header-btn"
              title="Download Transcript"
            >
              <Download size={13} /> Export
            </button>
            <button
              type="button"
              onClick={() => setMessages([])}
              className="conversation-header-btn btn-danger"
              title="Clear Transcript"
            >
              <Trash2 size={13} />
            </button>
          </div>
        </div>

        {/* Hearing -> Deaf 3D Humanoid Signer Viewport */}
        {showAvatar && (
          <div style={{ margin: '0 0 12px 0', borderRadius: '10px', overflow: 'hidden', border: '1px solid rgba(99, 102, 241, 0.35)', background: '#0b1329', height: '280px', flexShrink: 0 }}>
            <Avatar3DViewer currentSign={activeAvatarSign} />
          </div>
        )}

        {/* Message Feed */}
        <div className="conversation-feed">
          {messages.map((m) => (
            <div
              key={m.id}
              className={`conversation-bubble-row ${
                m.sender === 'signer' ? 'row-signer' : 'row-speaker'
              }`}
            >
              <div
                className={`conversation-bubble ${
                  m.sender === 'signer' ? 'bubble-signer' : 'bubble-speaker'
                }`}
              >
                <div className="conversation-bubble-meta">
                  <span>{m.sender === 'signer' ? '🤟 Deaf Participant (ISL)' : '🎙️ Hearing Participant (Voice)'}</span>
                  <span>•</span>
                  <span>{m.timestamp}</span>
                </div>
                <p className="conversation-bubble-text">{m.text}</p>
                {m.indicText && (
                  <p className="conversation-bubble-indic">{m.indicText}</p>
                )}
              </div>
            </div>
          ))}
        </div>

        {/* Speaker Voice Input Box */}
        <div className="conversation-input-box">
          <div className="conversation-input-row">
            <input
              type="text"
              value={speechInput}
              onChange={(e) => setSpeechInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSendSpeakerMessage()}
              placeholder="Speak or type your response for the signer..."
              className="conversation-input-field"
            />
            <button
              type="button"
              onClick={toggleListening}
              className={`conversation-mic-btn ${isListening ? 'listening' : ''}`}
              title={isListening ? 'Stop Mic' : 'Listen with Mic'}
            >
              {isListening ? <MicOff size={16} /> : <Mic size={16} />}
            </button>
            <button
              type="button"
              onClick={handleSendSpeakerMessage}
              disabled={!speechInput.trim()}
              className="conversation-send-btn"
              title="Send reply"
            >
              <Send size={15} />
            </button>
          </div>
          <div className="conversation-input-footer">
            <span>Avatar translating word: <strong className="active-sign-token">{activeAvatarSign}</strong></span>
            <span>Web Speech Voice Enabled</span>
          </div>
        </div>
      </div>
    </div>
  );
};
