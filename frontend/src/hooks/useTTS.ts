import { useState, useEffect, useCallback, useRef } from 'react';
import type { TTSState, TTSControls } from '../types';

export interface SupportedLanguage {
  code: string;
  label: string;
  nativeName: string;
}

export const SUPPORTED_LANGUAGES: SupportedLanguage[] = [
  { code: 'en-IN', label: 'Indian English', nativeName: 'English (India)' },
  { code: 'hi-IN', label: 'Hindi', nativeName: 'हिन्दी' },
  { code: 'ta-IN', label: 'Tamil', nativeName: 'தமிழ்' },
  { code: 'te-IN', label: 'Telugu', nativeName: 'తెలుగు' },
  { code: 'kn-IN', label: 'Kannada', nativeName: 'ಕನ್ನಡ' },
];

/**
 * Helper to select the most appropriate regional voice.
 * Prioritizes: requested target language (e.g. hi-IN, ta-IN, te-IN, kn-IN, en-IN), then prefix match, then English fallback.
 */
export function selectBestVoice(
  voices: SpeechSynthesisVoice[],
  targetLang: string = 'en-IN'
): SpeechSynthesisVoice | null {
  if (!voices || voices.length === 0) return null;

  const targetPrefix = targetLang.split('-')[0].toLowerCase();

  // 1. Exact language match
  const exact = voices.find(
    (v) => v.lang.replace('_', '-').toLowerCase() === targetLang.toLowerCase()
  );
  if (exact) return exact;

  // 2. Prefix match (e.g. hi, ta, te, kn, en)
  const prefixMatch = voices.find((v) =>
    v.lang.toLowerCase().startsWith(targetPrefix)
  );
  if (prefixMatch) return prefixMatch;

  // 3. Fallback to Indian English
  const inVoice = voices.find(
    (v) => v.lang === 'en-IN' || v.lang.replace('_', '-').toLowerCase() === 'en-in'
  );
  if (inVoice) return inVoice;

  // 4. Any English
  const anyEn = voices.find((v) => v.lang.toLowerCase().startsWith('en'));
  if (anyEn) return anyEn;

  return voices[0] || null;
}

export interface UseTTSReturn extends TTSState, TTSControls {
  targetLanguage: string;
  setTargetLanguage: (lang: string) => void;
  setAutoSpeakEnabled: React.Dispatch<React.SetStateAction<boolean>>;
}

/**
 * Custom hook for browser-native Text-to-Speech (TTS) integration using window.speechSynthesis
 * with full multi-lingual regional Indian language voice synthesis support.
 */
export function useTTS(): UseTTSReturn {
  const [isSpeaking, setIsSpeaking] = useState<boolean>(false);
  const [availableVoices, setAvailableVoices] = useState<SpeechSynthesisVoice[]>([]);
  const [selectedVoice, setSelectedVoice] = useState<SpeechSynthesisVoice | null>(null);
  const [targetLanguage, setTargetLanguageState] = useState<string>(() => {
    if (typeof window === 'undefined') return 'en-IN';
    try {
      return localStorage.getItem('setu_tts_lang') || 'en-IN';
    } catch {
      return 'en-IN';
    }
  });

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
        setSelectedVoice((current) => current || selectBestVoice(voices, targetLanguage));
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
  }, [targetLanguage]);

  const setTargetLanguage = useCallback((lang: string) => {
    setTargetLanguageState(lang);
    try {
      localStorage.setItem('setu_tts_lang', lang);
    } catch (e) {
      console.warn('Could not save TTS language:', e);
    }
    const best = selectBestVoice(availableVoicesRef.current, lang);
    if (best) {
      setSelectedVoice(best);
    }
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

  // Speak specified text string with regional language voice
  const speak = useCallback(
    (text: string, language?: string) => {
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

        const langToUse = language || targetLanguage;
        const currentVoice = selectedVoiceRef.current;
        const matchesTarget = currentVoice && currentVoice.lang.toLowerCase().startsWith(langToUse.split('-')[0].toLowerCase());

        const voiceToUse = matchesTarget
          ? currentVoice
          : selectBestVoice(availableVoicesRef.current, langToUse);

        if (voiceToUse) {
          utterance.voice = voiceToUse;
          utterance.lang = voiceToUse.lang;
        } else {
          utterance.lang = langToUse;
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
    [targetLanguage]
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
    targetLanguage,
    setTargetLanguage,
    speak,
    stop,
    toggleAutoSpeak,
    setSelectedVoice,
    setAutoSpeakEnabled,
  };
}

