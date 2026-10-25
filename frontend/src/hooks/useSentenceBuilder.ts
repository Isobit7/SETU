/**
 * SETU — Custom Hook: useSentenceBuilder
 * Decouples temporal streak debouncing, duplicate prevention, and sentence accumulation
 * business logic from the React UI view layer.
 */

import { useRef, useState, useCallback } from 'react';

interface UseSentenceBuilderOptions {
  confidenceThreshold: number;
  autoSpeakEnabled: boolean;
  speak: (text: string) => void;
  stop: () => void;
  onClearBackend?: () => void;
  onBackspaceBackend?: () => void;
}

export function useSentenceBuilder({
  confidenceThreshold,
  autoSpeakEnabled,
  speak,
  stop,
  onClearBackend,
  onBackspaceBackend,
}: UseSentenceBuilderOptions) {
  const [sentence, setSentence] = useState<string>('');

  // Temporal Debouncing Refs
  const candidateStreakRef = useRef<{ sign: string; count: number }>({ sign: '', count: 0 });
  const lastAppendedSignRef = useRef<string>('');
  const lastAppendedTimeRef = useRef<number>(0);
  const hadHandDropRef = useRef<boolean>(false);
  const lastSpokenSentenceRef = useRef<string>('');

  const handlePredictionChange = useCallback((data: any) => {
    if (!data) return;

    // Directly synchronize sentence buffer from backend SentenceAccumulator if provided
    if (data.sentence_buffer !== undefined) {
      setSentence(data.sentence_buffer);
    }

    // Auto-Speak trigger on sentence completion:
    // When translated_text updates and differs from lastSpokenSentenceRef.current
    if (data.translated_text && typeof data.translated_text === 'string') {
      const translated = data.translated_text.trim();
      if (translated && translated !== lastSpokenSentenceRef.current) {
        lastSpokenSentenceRef.current = translated;
        if (autoSpeakEnabled) {
          speak(translated);
        }
      }
    } else if (!data.sentence_buffer && !data.translated_text) {
      lastSpokenSentenceRef.current = '';
    }

    const isHand = data.hand_detected ?? false;
    const isKnown = data.is_known_sign ?? false;
    const conf = data.confidence ?? 0;
    const label = data.label ?? '';
    const noSignProb = data.probabilities?.['no_sign'] ?? 0;

    // If hand dropped or no sign, note the drop and reset streak
    if (!isHand || label === 'no_sign' || label === 'unknown' || label === 'Awaiting Sign Input' || noSignProb > 0.70 || !isKnown) {
      candidateStreakRef.current = { sign: '', count: 0 };
      hadHandDropRef.current = true;
      return;
    }

    // Check user confidence threshold gate
    if (conf < confidenceThreshold) {
      candidateStreakRef.current = { sign: '', count: 0 };
      return;
    }

    // Determine the sign/character to append
    let signText = '';
    if (data.active_model === 'realsign') {
      signText = (data.label || '').toUpperCase();
    } else {
      signText = (data.display_name || label.replace('_', ' ')).toUpperCase();
    }

    if (!signText || signText === 'NO_SIGN' || signText === 'UNKNOWN' || signText === 'AWAITING SIGN INPUT') return;

    // Temporal Streak Counter: 2 consecutive frames for Alphabet, 3 for Sequence
    const requiredStreak = data.active_model === 'realsign' ? 2 : 3;

    if (candidateStreakRef.current.sign === signText) {
      candidateStreakRef.current.count += 1;
    } else {
      candidateStreakRef.current = { sign: signText, count: 1 };
      return;
    }

    if (candidateStreakRef.current.count < requiredStreak) {
      return;
    }

    // Refractory Cooldown & Duplicate Prevention
    const now = Date.now();
    const timeSinceLast = now - lastAppendedTimeRef.current;
    const isSameSign = signText === lastAppendedSignRef.current;

    // Duplicate lockout: 350ms lockout AND requires idle baseline transition
    if (isSameSign && (timeSinceLast < 350 || !hadHandDropRef.current)) {
      return;
    }

    // Confirmed append!
    lastAppendedSignRef.current = signText;
    lastAppendedTimeRef.current = now;
    hadHandDropRef.current = false;

    setSentence((prev) => {
      // If single-character alphabet letter (A-Z)
      if (signText.length === 1) {
        return prev + signText;
      }
      // If phrase (e.g., "HELLO", "THANK YOU")
      const trimmed = prev.trim();
      if (!trimmed) {
        return signText + ' ';
      }
      return prev.endsWith(' ') ? `${prev}${signText} ` : `${prev} ${signText} `;
    });
  }, [confidenceThreshold, autoSpeakEnabled, speak]);

  const handleClear = useCallback(() => {
    setSentence('');
    lastAppendedSignRef.current = '';
    candidateStreakRef.current = { sign: '', count: 0 };
    lastSpokenSentenceRef.current = '';
    stop();
    if (onClearBackend) onClearBackend();
  }, [stop, onClearBackend]);

  const handleBackspace = useCallback(() => {
    setSentence((prev) => {
      if (!prev) return '';
      const trimmed = prev.trimEnd();
      const lastSpaceIndex = trimmed.lastIndexOf(' ');
      if (lastSpaceIndex === -1) {
        return prev.length === 1 ? '' : prev.slice(0, -1);
      }
      return prev.slice(0, -1);
    });
    if (onBackspaceBackend) onBackspaceBackend();
  }, [onBackspaceBackend]);

  return {
    sentence,
    setSentence,
    handlePredictionChange,
    handleClear,
    handleBackspace,
  };
}
