import { useState, useEffect, useCallback, useRef } from 'react';
import type { TTSState, TTSControls } from '../types';

/**
 * Helper to select the most appropriate English voice.
 * Prioritizes: Indian English (en-IN), US English (en-US), British English (en-GB), or any English.
 */
function selectBestVoice(voices: SpeechSynthesisVoice[]): SpeechSynthesisVoice | null {
  if (!voices || voices.length === 0) return null;

  const inVoice = voices.find(
    (v) => v.lang === 'en-IN' || v.lang.replace('_', '-').toLowerCase() === 'en-in'
  );
  if (inVoice) return inVoice;

  const usVoice = voices.find(
    (v) => v.lang === 'en-US' || v.lang.replace('_', '-').toLowerCase() === 'en-us'
  );
  if (usVoice) return usVoice;

  const gbVoice = voices.find(
    (v) => v.lang === 'en-GB' || v.lang.replace('_', '-').toLowerCase() === 'en-gb'
  );
  if (gbVoice) return gbVoice;

  const anyEn = voices.find((v) => v.lang.toLowerCase().startsWith('en'));
  if (anyEn) return anyEn;

  return voices[0] || null;
}

export interface UseTTSReturn extends TTSState, TTSControls {
  setAutoSpeakEnabled: React.Dispatch<React.SetStateAction<boolean>>;
}

/**
 * Custom hook for browser-native Text-to-Speech (TTS) integration using window.speechSynthesis.
 */
export function useTTS(): UseTTSReturn {
  const [isSpeaking, setIsSpeaking] = useState<boolean>(false);
  const [availableVoices, setAvailableVoices] = useState<SpeechSynthesisVoice[]>([]);
  const [selectedVoice, setSelectedVoice] = useState<SpeechSynthesisVoice | null>(null);

  // Initialize autoSpeakEnabled from localStorage, defaulting to true
  const [autoSpeakEnabled, setAutoSpeakEnabled] = useState<boolean>(() => {
    if (typeof window === 'undefined') return true;
    try {
      const saved = localStorage.getItem('setu_auto_speak');
      if (saved !== null) {
        return saved === 'true';
      }
    } catch {
      // localStorage may fail in restricted sandboxes
    }
    return true;
  });

  const selectedVoiceRef = useRef<SpeechSynthesisVoice | null>(selectedVoice);
  const availableVoicesRef = useRef<SpeechSynthesisVoice[]>(availableVoices);

  useEffect(() => {
    selectedVoiceRef.current = selectedVoice;
  }, [selectedVoice]);

  useEffect(() => {
    availableVoicesRef.current = availableVoices;
  }, [availableVoices]);

  // Load and listen for system voices
  useEffect(() => {
    if (typeof window === 'undefined' || !('speechSynthesis' in window)) return;

    const updateVoices = () => {
      const voices = window.speechSynthesis.getVoices();
      if (voices && voices.length > 0) {
        setAvailableVoices(voices);
        setSelectedVoice((current) => current || selectBestVoice(voices));
      }
    };

    updateVoices();

    if (window.speechSynthesis.onvoiceschanged !== undefined) {
      window.speechSynthesis.onvoiceschanged = updateVoices;
    }

    return () => {
      if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
        if (window.speechSynthesis.onvoiceschanged !== undefined) {
          window.speechSynthesis.onvoiceschanged = null;
        }
      }
    };
  }, []);

  const activeUtteranceRef = useRef<SpeechSynthesisUtterance | null>(null);

  // Stop active speech
  const stop = useCallback(() => {
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      try {
        window.speechSynthesis.cancel();
      } catch (err) {
        console.warn('Speech cancel error:', err);
      }
    }
    activeUtteranceRef.current = null;
    setIsSpeaking(false);
  }, []);

  // Speak specified text string
  const speak = useCallback(
    (text: string) => {
      if (!text || !text.trim() || typeof window === 'undefined' || !('speechSynthesis' in window)) {
        return;
      }

      try {
        // Cancel any active speech
        window.speechSynthesis.cancel();

        const cleanText = text.trim();
        const utterance = new SpeechSynthesisUtterance(cleanText);
        activeUtteranceRef.current = utterance;

        utterance.pitch = 1.0;
        utterance.rate = 1.0;

        const voiceToUse = selectedVoiceRef.current || selectBestVoice(availableVoicesRef.current);
        if (voiceToUse) {
          utterance.voice = voiceToUse;
          utterance.lang = voiceToUse.lang;
        } else {
          utterance.lang = 'en-US';
        }

        utterance.onstart = () => {
          setIsSpeaking(true);
        };

        utterance.onend = () => {
          activeUtteranceRef.current = null;
          setIsSpeaking(false);
        };

        utterance.onerror = (e) => {
          activeUtteranceRef.current = null;
          if (e.error !== 'interrupted' && e.error !== 'canceled') {
            console.warn('SpeechSynthesis error:', e.error);
          }
          setIsSpeaking(false);
        };

        window.speechSynthesis.speak(utterance);
      } catch (err) {
        activeUtteranceRef.current = null;
        console.error('Failed to execute speak utterance:', err);
        setIsSpeaking(false);
      }
    },
    []
  );

  // Toggle autoSpeakEnabled with persistent localStorage
  const toggleAutoSpeak = useCallback(() => {
    setAutoSpeakEnabled((prev) => {
      const next = !prev;
      try {
        localStorage.setItem('setu_auto_speak', String(next));
      } catch (e) {
        console.warn('Could not save auto-speak preference to localStorage:', e);
      }
      return next;
    });
  }, []);

  return {
    isSpeaking,
    autoSpeakEnabled,
    availableVoices,
    selectedVoice,
    speak,
    stop,
    toggleAutoSpeak,
    setSelectedVoice,
    setAutoSpeakEnabled,
  };
}
