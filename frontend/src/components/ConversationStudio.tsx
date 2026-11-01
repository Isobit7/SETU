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
  const [activeAvatarSign, setActiveAvatarSign] = useState<string>('HELLO');
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

    const words = speechInput.trim().toUpperCase().split(/\s+/);
    if (words.length > 0) {
      setActiveAvatarSign(words[0]);
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
    <div className="flex-1 flex flex-col md:flex-row h-[calc(100vh-4rem)] bg-slate-950 text-slate-100 overflow-hidden">
      {/* LEFT PANE: Signer Camera & Live Recognition */}
      <div className="w-full md:w-1/2 flex flex-col border-b md:border-b-0 md:border-r border-slate-800 p-4 gap-4 overflow-y-auto">
        <div className="flex items-center justify-between pb-2 border-b border-slate-800">
          <div className="flex items-center gap-2">
            <span className="p-2 rounded-lg bg-indigo-500/10 text-indigo-400">
              <Video className="w-5 h-5" />
            </span>
            <div>
              <h2 className="text-base font-semibold">Signer View (ISL $\to$ Speech)</h2>
              <p className="text-xs text-slate-400">Live Indian Sign Language Camera</p>
            </div>
          </div>
          <button
            onClick={isCameraActive ? onStopCamera : onStartCamera}
            className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-colors ${
              isCameraActive
                ? 'bg-rose-500/20 text-rose-300 hover:bg-rose-500/30'
                : 'bg-emerald-500/20 text-emerald-300 hover:bg-emerald-500/30'
            }`}
          >
            {isCameraActive ? 'Stop Camera' : 'Start Camera'}
          </button>
        </div>

        {/* Camera Container */}
        <div className="relative aspect-video bg-slate-900 rounded-xl overflow-hidden border border-slate-800 flex items-center justify-center">
          <video
            ref={videoRef as any}
            autoPlay
            playsInline
            muted
            className={`w-full h-full object-cover ${isCameraActive ? 'block' : 'hidden'}`}
          />
          {!isCameraActive && (
            <div className="text-center p-6 text-slate-500">
              <Video className="w-10 h-10 mx-auto mb-2 opacity-50" />
              <p className="text-xs">Camera preview off. Click "Start Camera" above.</p>
            </div>
          )}

          {/* Current Live Sign Overlay */}
          {currentResult?.is_known_sign && (
            <div className="absolute top-3 left-3 px-3 py-1 bg-slate-900/80 backdrop-blur rounded-lg border border-indigo-500/40 text-indigo-300 font-mono text-sm font-semibold">
              {currentResult?.label ? currentResult.label.toUpperCase() : ''}
            </div>
          )}
        </div>

        {/* Live Signer Sentence Box */}
        <div className="bg-slate-900/80 rounded-xl p-4 border border-slate-800 flex flex-col gap-2">
          <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
            Accumulated Sentence
          </span>
          <p className="text-lg font-medium text-slate-100 min-h-[2rem]">
            {activeSignerSentence || (
              <span className="text-slate-600 italic">Waiting for sign gestures...</span>
            )}
          </p>

          {currentResult?.translated_indic?.hi && (
            <p className="text-sm font-medium text-amber-400">
              {currentResult.translated_indic.hi}
            </p>
          )}

          <div className="flex gap-2 mt-2">
            <button
              onClick={handleCommitSignerMessage}
              disabled={!activeSignerSentence.trim()}
              className="flex-1 py-2 px-4 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-xs font-semibold text-white transition-colors flex items-center justify-center gap-2"
            >
              <Send className="w-3.5 h-3.5" /> Speak to Participant
            </button>
            <button
              onClick={() => onSpeak && onSpeak(activeSignerSentence)}
              disabled={!activeSignerSentence.trim()}
              className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-slate-300 transition-colors"
              title="Speak immediately"
            >
              <Volume2 className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* RIGHT PANE: Hearing Participant Voice & Dialogue History */}
      <div className="w-full md:w-1/2 flex flex-col p-4 gap-4 overflow-hidden">
        {/* Header Actions */}
        <div className="flex items-center justify-between pb-2 border-b border-slate-800">
          <div className="flex items-center gap-2">
            <span className="p-2 rounded-lg bg-emerald-500/10 text-emerald-400">
              <User className="w-5 h-5" />
            </span>
            <div>
              <h2 className="text-base font-semibold">Conversation Dialogue</h2>
              <p className="text-xs text-slate-400">Bilingual Sign $\leftrightarrow$ Voice Transcript</p>
            </div>
          </div>
          <div className="flex gap-2">
            <button
              onClick={exportTranscript}
              className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors text-xs flex items-center gap-1.5"
              title="Download Transcript"
            >
              <Download className="w-3.5 h-3.5" /> Export
            </button>
            <button
              onClick={() => setMessages([])}
              className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-rose-400 transition-colors text-xs"
              title="Clear Transcript"
            >
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>

        {/* Message Feed */}
        <div className="flex-1 overflow-y-auto space-y-3 pr-2 scrollbar-thin">
          {messages.map((m) => (
            <div
              key={m.id}
              className={`flex flex-col ${
                m.sender === 'signer' ? 'items-start' : 'items-end'
              }`}
            >
              <div
                className={`max-w-[85%] rounded-2xl p-3.5 text-sm ${
                  m.sender === 'signer'
                    ? 'bg-indigo-600/20 text-indigo-100 border border-indigo-500/30'
                    : 'bg-emerald-600/20 text-emerald-100 border border-emerald-500/30'
                }`}
              >
                <div className="flex items-center gap-1.5 mb-1 text-[11px] font-semibold opacity-70">
                  {m.sender === 'signer' ? '🤟 Deaf Participant (ISL)' : '🎙️ Hearing Participant (Voice)'}
                  <span>•</span>
                  <span>{m.timestamp}</span>
                </div>
                <p className="font-medium">{m.text}</p>
                {m.indicText && (
                  <p className="text-xs text-amber-300 mt-1 font-medium">{m.indicText}</p>
                )}
              </div>
            </div>
          ))}
        </div>

        {/* Speaker Voice Input Box */}
        <div className="bg-slate-900 rounded-xl p-3 border border-slate-800 flex flex-col gap-2">
          <div className="flex items-center gap-2">
            <input
              type="text"
              value={speechInput}
              onChange={(e) => setSpeechInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSendSpeakerMessage()}
              placeholder="Speak or type your response for the signer..."
              className="flex-1 bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-emerald-500"
            />
            <button
              onClick={toggleListening}
              className={`p-2.5 rounded-lg transition-colors ${
                isListening
                  ? 'bg-rose-500 text-white animate-pulse'
                  : 'bg-slate-800 hover:bg-slate-700 text-slate-300'
              }`}
              title={isListening ? 'Stop Mic' : 'Listen with Mic'}
            >
              {isListening ? <MicOff className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
            </button>
            <button
              onClick={handleSendSpeakerMessage}
              disabled={!speechInput.trim()}
              className="p-2.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 text-white transition-colors"
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
          <div className="flex items-center justify-between text-[11px] text-slate-500 px-1">
            <span>Avatar translating word: <strong className="text-emerald-400">{activeAvatarSign}</strong></span>
            <span>Web Speech Voice Enabled</span>
          </div>
        </div>
      </div>
    </div>
  );
};
