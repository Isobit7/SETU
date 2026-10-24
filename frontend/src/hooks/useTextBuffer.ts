/**
 * SETU — Real-Time Text Buffer & Letter Accumulator Hook
 *
 * Smoothly accumulates recognized alphabet letters into words and sentences:
 * 1. Hold Duration Gate: User must hold sign continuously for 600ms.
 * 2. Confidence Threshold: Only predictions with confidence >= 0.85 are considered.
 * 3. Live Hold Progress: Computes 0-100% progress for responsive UI progress bars.
 * 4. Lockout Cooldown: 400ms lockout after committing a character to prevent unintentional repeats.
 * 5. Gesture & Manual Actions: Supports SPACE, BACKSPACE gestures and manual control buttons.
 */

import { useState, useRef, useCallback, useEffect } from 'react';

export interface UseTextBufferOptions {
  /** Duration in milliseconds a letter must be continuously held (default: 600ms) */
  holdDurationMs?: number;
  /** Minimum model confidence required to progress hold (default: 0.85) */
  confidenceThreshold?: number;
  /** Post-commit cooldown lockout in milliseconds (default: 400ms) */
  lockoutDurationMs?: number;
  /** Maximum idle time before resetting active hold if no prediction arrives (default: 300ms) */
  idleResetMs?: number;
  /** Optional callback fired when a letter is committed */
  onCharacterCommitted?: (character: string, currentBuffer: string) => void;
}

export interface UseTextBufferReturn {
  /** The accumulated text string */
  text: string;
  /** Setter to directly modify text if needed */
  setText: React.Dispatch<React.SetStateAction<string>>;
  /** Currently active letter candidate being held */
  currentCandidate: string | null;
  /** Confidence score of the current candidate */
  candidateConfidence: number;
  /** Hold completion percentage (0 - 100) */
  holdProgress: number;
  /** Whether a letter is actively being held with valid confidence */
  isHolding: boolean;
  /** Whether the system is in post-commit lockout cooldown */
  isLockedOut: boolean;
  /** Time remaining in lockout cooldown (ms) */
  lockoutRemainingMs: number;
  /** Ingestion method to feed predictions from API */
  processPrediction: (letter: string, confidence: number) => void;
  /** Resets active hold timer and candidate without altering buffer */
  resetHold: () => void;
  /** Appends a space character */
  manualSpace: () => void;
  /** Removes the last character from buffer */
  manualBackspace: () => void;
  /** Completely clears the text buffer */
  clearBuffer: () => void;
}

export function useTextBuffer({
  holdDurationMs = 600,
  confidenceThreshold = 0.85,
  lockoutDurationMs = 400,
  idleResetMs = 300,
  onCharacterCommitted,
}: UseTextBufferOptions = {}): UseTextBufferReturn {
  const [text, setText] = useState<string>('');
  const [currentCandidate, setCurrentCandidate] = useState<string | null>(null);
  const [candidateConfidence, setCandidateConfidence] = useState<number>(0);
  const [holdProgress, setHoldProgress] = useState<number>(0);
  const [isHolding, setIsHolding] = useState<boolean>(false);
  const [isLockedOut, setIsLockedOut] = useState<boolean>(false);
  const [lockoutRemainingMs, setLockoutRemainingMs] = useState<number>(0);

  // References to avoid stale state in high-frequency camera loops
  const candidateRef = useRef<string | null>(null);
  const holdStartTimeRef = useRef<number | null>(null);
  const lastPredictionTimeRef = useRef<number>(0);
  const lockoutEndTimeRef = useRef<number>(0);
  const committedTextRef = useRef<string>('');

  // Keep committed text ref synchronized
  useEffect(() => {
    committedTextRef.current = text;
  }, [text]);

  const resetHold = useCallback(() => {
    candidateRef.current = null;
    holdStartTimeRef.current = null;
    setCurrentCandidate(null);
    setCandidateConfidence(0);
    setHoldProgress(0);
    setIsHolding(false);
  }, []);

  const commitCharacter = useCallback(
    (char: string) => {
      const now = performance.now();
      let newText = committedTextRef.current;

      if (char === 'SPACE' || char === ' ') {
        newText += ' ';
      } else if (char === 'BACKSPACE' || char === 'DELETE') {
        newText = newText.slice(0, -1);
      } else {
        newText += char;
      }

      committedTextRef.current = newText;
      setText(newText);

      // Trigger cooldown lockout to prevent repeat commits
      lockoutEndTimeRef.current = now + lockoutDurationMs;
      setIsLockedOut(true);

      // Reset hold state
      resetHold();

      if (onCharacterCommitted) {
        onCharacterCommitted(char, newText);
      }
    },
    [lockoutDurationMs, onCharacterCommitted, resetHold]
  );

  const processPrediction = useCallback(
    (rawLetter: string, confidence: number) => {
      const now = performance.now();
      lastPredictionTimeRef.current = now;

      // 1. Check if currently in post-commit lockout cooldown
      if (now < lockoutEndTimeRef.current) {
        setIsLockedOut(true);
        setLockoutRemainingMs(Math.max(0, Math.round(lockoutEndTimeRef.current - now)));
        resetHold();
        return;
      } else {
        setIsLockedOut(false);
        setLockoutRemainingMs(0);
      }

      // 2. Validate letter string
      if (!rawLetter || typeof rawLetter !== 'string') {
        resetHold();
        return;
      }

      const letter = rawLetter.trim().toUpperCase();

      // Ignore placeholder labels
      if (
        letter === 'AWAITING SIGN INPUT' ||
        letter === 'NO_SIGN' ||
        letter === 'UNKNOWN' ||
        letter === ''
      ) {
        resetHold();
        return;
      }

      // 3. Check Confidence Gate
      if (confidence < confidenceThreshold) {
        resetHold();
        return;
      }

      // 4. Update or Progress Hold
      if (candidateRef.current !== letter) {
        // New letter started
        candidateRef.current = letter;
        holdStartTimeRef.current = now;
        setCurrentCandidate(letter);
        setCandidateConfidence(confidence);
        setHoldProgress(0);
        setIsHolding(true);
      } else {
        // Same letter maintained: compute progress
        const startTime = holdStartTimeRef.current ?? now;
        const elapsed = now - startTime;
        const progress = Math.min(100, Math.round((elapsed / holdDurationMs) * 100));

        setCandidateConfidence(confidence);
        setHoldProgress(progress);
        setIsHolding(true);

        // Check if hold duration threshold is reached
        if (elapsed >= holdDurationMs) {
          commitCharacter(letter);
        }
      }
    },
    [confidenceThreshold, holdDurationMs, commitCharacter, resetHold]
  );

  // Background monitor: decays hold if predictions stop arriving (e.g. hand out of frame)
  useEffect(() => {
    const timer = setInterval(() => {
      const now = performance.now();

      // Update lockout countdown
      if (now < lockoutEndTimeRef.current) {
        setIsLockedOut(true);
        setLockoutRemainingMs(Math.max(0, Math.round(lockoutEndTimeRef.current - now)));
      } else if (lockoutEndTimeRef.current > 0) {
        lockoutEndTimeRef.current = 0;
        setIsLockedOut(false);
        setLockoutRemainingMs(0);
      }

      // Reset hold if camera feed went idle
      if (
        holdStartTimeRef.current !== null &&
        now - lastPredictionTimeRef.current > idleResetMs
      ) {
        resetHold();
      }
    }, 50);

    return () => clearInterval(timer);
  }, [idleResetMs, resetHold]);

  // Manual button actions
  const manualSpace = useCallback(() => {
    setText((prev) => prev + ' ');
  }, []);

  const manualBackspace = useCallback(() => {
    setText((prev) => prev.slice(0, -1));
  }, []);

  const clearBuffer = useCallback(() => {
    setText('');
    resetHold();
  }, [resetHold]);

  return {
    text,
    setText,
    currentCandidate,
    candidateConfidence,
    holdProgress,
    isHolding,
    isLockedOut,
    lockoutRemainingMs,
    processPrediction,
    resetHold,
    manualSpace,
    manualBackspace,
    clearBuffer,
  };
}

export default useTextBuffer;
