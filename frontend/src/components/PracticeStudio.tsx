import React, { useState, useEffect, useRef } from 'react';
import {
  Award,
  CheckCircle2,
  ChevronRight,
  RotateCcw,
  Target,
  Sparkles,
  Volume2,
  X,
  PlayCircle,
  PauseCircle,
  Shuffle,
  AlertCircle,
} from 'lucide-react';
import type { RecognitionPayload } from '../types';
import { Practice3DHandViewer } from './Practice3DHandViewer';

export interface PracticeStudioProps {
  currentResult: RecognitionPayload | null;
  onSpeak?: (text: string) => void;
  onClose?: () => void;
}

const ALPHABET_LIST = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');

const ALPHABET_GUIDES: Record<string, { desc: string; tips: string; isTwoHanded?: boolean }> = {
  A: {
    desc: 'Form a closed fist with your dominant hand. Keep thumb straight upright against index knuckle.',
    tips: 'Ensure other four fingers are completely folded into palm.',
  },
  B: {
    desc: 'Dual-Hand Sign: Hold left hand flat upright with palm facing you; tap flat right hand fingertips against left palm.',
    tips: 'Bring both hands into camera view so contact is visible.',
    isTwoHanded: true,
  },
  C: {
    desc: 'Curve thumb and four fingers into an open semi-circular "C" arch facing forward.',
    tips: 'Keep a clear 2-inch gap between fingertips and thumb tip. Use Side Profile view on 3D model to inspect curve!',
  },
  D: {
    desc: 'Dual-Hand Sign: Left index points straight up (stem); right index and thumb touch left index to form the round loop.',
    tips: 'Hold both hands steady at chest height.',
    isTwoHanded: true,
  },
  E: {
    desc: 'Curl all four fingers downward so tips rest lightly on top edge of folded thumb.',
    tips: 'Keep fingers tightly grouped together.',
  },
  F: {
    desc: 'Touch index fingertip to thumb tip forming a circle; keep middle, ring, and pinky standing tall.',
    tips: 'Similar to the classic "OK" gesture.',
  },
  G: {
    desc: 'Extend index and thumb horizontally forward in parallel, pointing sideways.',
    tips: 'Keep middle, ring, and pinky folded.',
  },
  H: {
    desc: 'Extend index and middle fingers together horizontally forward like a horizontal bar.',
    tips: 'Keep both fingers closed side-by-side.',
  },
  I: {
    desc: 'Form a closed fist with dominant hand and extend only the pinky straight up.',
    tips: 'Thumb folds across index and middle fingers.',
  },
  J: {
    desc: 'Extend pinky straight up, then trace a gentle "J" curve in the air.',
    tips: 'Keep wrist relaxed.',
  },
  K: {
    desc: 'Extend index straight up; angle middle finger forward with thumb placed between them.',
    tips: 'Thumb supports middle finger joint.',
  },
  L: {
    desc: 'Extend index finger straight up and thumb outward at 90 degrees to form an "L".',
    tips: 'Fingers should form a sharp right angle.',
  },
  M: {
    desc: 'Fold three fingers (index, middle, ring) tightly over tucked thumb.',
    tips: 'Only thumb tip peeks out beneath ring finger.',
  },
  N: {
    desc: 'Fold two fingers (index and middle) tightly over tucked thumb.',
    tips: 'Thumb tip peeks out between middle and ring finger.',
  },
  O: {
    desc: 'Curve all fingertips to meet thumb tip, forming a hollow circle ("O").',
    tips: 'Fingers should touch gently without collapsing.',
  },
  P: {
    desc: 'Dual-Hand Sign: Left index points straight up; right index and thumb form an upper loop touching left fingertip.',
    tips: 'Keep left index steady as the vertical anchor.',
    isTwoHanded: true,
  },
  Q: {
    desc: 'Dual-Hand Sign: Left hand forms a circle; right index points straight down through the circle.',
    tips: 'Align right finger through the center of left loop.',
    isTwoHanded: true,
  },
  R: {
    desc: 'Cross middle finger over index finger; fold ring and pinky into palm.',
    tips: 'Keep thumb folded over ring finger.',
  },
  S: {
    desc: 'Make a tight fist with thumb wrapped across the front of all folded fingers.',
    tips: 'Contrast with "A" where thumb is beside the fingers.',
  },
  T: {
    desc: 'Dual-Hand Sign: Left index stands straight up; right index rests horizontally across top of left index.',
    tips: 'Forms a clean cross/T-bar.',
    isTwoHanded: true,
  },
  U: {
    desc: 'Hold index and middle fingers together pointing straight up, closed side-by-side.',
    tips: 'Ensure fingers are touching, not spread.',
  },
  V: {
    desc: 'Extend index and middle fingers straight up and spread apart in a "V" shape (peace sign).',
    tips: 'Spread fingers wide apart.',
  },
  W: {
    desc: 'Extend index, middle, and ring fingers upright and spread apart in a "W" shape.',
    tips: 'Thumb holds pinky tip down.',
  },
  X: {
    desc: 'Curl index finger into a hooked shape; keep other fingers closed.',
    tips: 'Looks like a small pirate hook.',
  },
  Y: {
    desc: 'Extend thumb and pinky outwards (shaka sign); fold middle three fingers into palm.',
    tips: 'Keep wrist centered in camera.',
  },
  Z: {
    desc: 'Extend index finger and trace the letter "Z" in the air.',
    tips: 'Draw two horizontal lines connected by a diagonal.',
  },
};

const COMMON_SIGNS = [
  { id: 'hello', label: 'HELLO', desc: 'Raise dominant hand near temple and wave gently outward.' },
  { id: 'thank_you', label: 'THANK YOU', desc: 'Touch fingertips of flat hand to chin and move forward toward other person.' },
  { id: 'namaste', label: 'NAMASTE', desc: 'Join palms together in front of chest in traditional respectful gesture.' },
  { id: 'water', label: 'WATER', desc: 'Hold W-hand or cup gesture near mouth and tilt toward lips.' },
  { id: 'food', label: 'FOOD', desc: 'Bring fingertips together and tap lightly toward lips twice.' },
  { id: 'help', label: 'HELP', desc: 'Place closed fist with thumb up onto flat open palm of other hand.' },
];

let sharedAudioCtx: AudioContext | null = null;
const getAudioContext = (): AudioContext | null => {
  try {
    if (typeof window === 'undefined') return null;
    const AudioContextClass =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    if (!AudioContextClass) return null;
    if (!sharedAudioCtx || sharedAudioCtx.state === 'closed') {
      sharedAudioCtx = new AudioContextClass();
    }
    if (sharedAudioCtx.state === 'suspended') {
      void sharedAudioCtx.resume();
    }
    return sharedAudioCtx;
  } catch {
    return null;
  }
};

/** Synthesize a gentle celebratory chime using browser native Web Audio API */
const playSuccessChime = () => {
  try {
    const ctx = getAudioContext();
    if (!ctx) return;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(523.25, ctx.currentTime); // C5
    osc.frequency.exponentialRampToValueAtTime(659.25, ctx.currentTime + 0.12); // E5
    osc.frequency.exponentialRampToValueAtTime(783.99, ctx.currentTime + 0.25); // G5
    gain.gain.setValueAtTime(0.18, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.4);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.4);
  } catch {
    // AudioContext blocked or unsupported, fail silently
  }
};

export const PracticeStudio: React.FC<PracticeStudioProps> = ({
  currentResult,
  onSpeak,
  onClose,
}) => {
  const [mode, setMode] = useState<'alphabet' | 'phrases'>('alphabet');
  const [selectedTarget, setSelectedTarget] = useState<string>('A');
  const [streak, setStreak] = useState<number>(0);
  const [holdProgress, setHoldProgress] = useState<number>(0);
  const [isSuccess, setIsSuccess] = useState<boolean>(false);
  const [autoAdvance, setAutoAdvance] = useState<boolean>(true);
  const [isShuffle, setIsShuffle] = useState<boolean>(false);

  const autoAdvanceTimeoutRef = useRef<number | null>(null);

  const rawLabel = currentResult?.label || 'no_sign';
  const currentDetected = rawLabel.replace('_', ' ').toUpperCase();
  const currentConfidence = currentResult?.confidence || 0;

  // Flexible target match (letter "A" or "LETTER A" or phrase match)
  const isMatch = Boolean(
    currentResult?.is_known_sign &&
      (currentDetected === selectedTarget.toUpperCase() ||
        currentDetected === `LETTER ${selectedTarget.toUpperCase()}` ||
        rawLabel.toLowerCase() === selectedTarget.toLowerCase()) &&
      currentConfidence >= 0.45
  );

  // Hold timer: User must hold the correct sign steady for ~1.0s to score
  useEffect(() => {
    let timer: number | null = null;
    if (isMatch && !isSuccess) {
      timer = window.setInterval(() => {
        setHoldProgress((prev) => {
          if (prev >= 100) {
            setIsSuccess(true);
            setStreak((s) => s + 1);
            playSuccessChime();
            if (onSpeak) onSpeak(`Correct! That is ${selectedTarget}!`);
            return 100;
          }
          return prev + 20;
        });
      }, 140);
    } else if (!isMatch && !isSuccess) {
      setHoldProgress(0);
    }
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [isMatch, isSuccess, selectedTarget, onSpeak]);

  const handleNext = () => {
    if (autoAdvanceTimeoutRef.current) {
      clearTimeout(autoAdvanceTimeoutRef.current);
      autoAdvanceTimeoutRef.current = null;
    }
    setIsSuccess(false);
    setHoldProgress(0);

    if (mode === 'alphabet') {
      if (isShuffle) {
        const nextLetters = ALPHABET_LIST.filter((l) => l !== selectedTarget);
        const randomLetter = nextLetters[Math.floor(Math.random() * nextLetters.length)];
        setSelectedTarget(randomLetter);
      } else {
        const idx = ALPHABET_LIST.indexOf(selectedTarget);
        const nextIdx = (idx + 1) % ALPHABET_LIST.length;
        setSelectedTarget(ALPHABET_LIST[nextIdx]);
      }
    } else {
      if (isShuffle) {
        const nextSigns = COMMON_SIGNS.filter((s) => s.label !== selectedTarget);
        const randomSign = nextSigns[Math.floor(Math.random() * nextSigns.length)];
        setSelectedTarget(randomSign.label);
      } else {
        const idx = COMMON_SIGNS.findIndex((s) => s.label === selectedTarget);
        const nextIdx = (idx + 1) % COMMON_SIGNS.length;
        setSelectedTarget(COMMON_SIGNS[nextIdx].label);
      }
    }
  };

  // Auto-advance trigger after success
  useEffect(() => {
    if (isSuccess && autoAdvance) {
      autoAdvanceTimeoutRef.current = window.setTimeout(() => {
        handleNext();
      }, 1200);
    }
    return () => {
      if (autoAdvanceTimeoutRef.current) {
        clearTimeout(autoAdvanceTimeoutRef.current);
      }
    };
  }, [isSuccess, autoAdvance]);

  const handleReset = () => {
    setStreak(0);
    setIsSuccess(false);
    setHoldProgress(0);
  };

  const isDualHandedAlphabet = mode === 'alphabet' && 'BDPTQ'.includes(selectedTarget);

  return (
    <div className={`practice-studio-card ${isSuccess ? 'practice-success-state' : ''}`}>
      {/* 1. Header Toolbar */}
      <div className="practice-header">
        <div className="practice-header-left">
          <div className="practice-icon-wrap" aria-hidden="true">
            <Target size={16} className="text-cyan" />
          </div>
          <div className="practice-title-group">
            <h3 className="practice-title">Practice Studio</h3>
            <span className="practice-badge">Live AI Scoring</span>
          </div>
        </div>

        <div className="practice-header-right">
          <div className="practice-streak-pill" title="Consecutive correct signs held">
            <Award size={13} className="text-amber" />
            <span>Streak: <strong>{streak}</strong></span>
          </div>

          <button
            type="button"
            className="practice-mini-btn"
            onClick={handleReset}
            title="Reset streak to 0"
            aria-label="Reset streak"
          >
            <RotateCcw size={13} />
          </button>

          {onClose && (
            <button
              type="button"
              className="practice-mini-btn practice-close-btn"
              onClick={onClose}
              title="Return to live translator"
              aria-label="Close practice studio"
            >
              <X size={14} />
            </button>
          )}
        </div>
      </div>

      {/* 2. Mode Selector & Exercise Options */}
      <div className="practice-controls-strip">
        <div className="practice-mode-pills" role="tablist">
          <button
            type="button"
            role="tab"
            aria-selected={mode === 'alphabet'}
            className={`practice-tab-btn ${mode === 'alphabet' ? 'active' : ''}`}
            onClick={() => {
              setMode('alphabet');
              setSelectedTarget('A');
              setIsSuccess(false);
              setHoldProgress(0);
            }}
          >
            Alphabet (A–Z)
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={mode === 'phrases'}
            className={`practice-tab-btn ${mode === 'phrases' ? 'active' : ''}`}
            onClick={() => {
              setMode('phrases');
              setSelectedTarget(COMMON_SIGNS[0].label);
              setIsSuccess(false);
              setHoldProgress(0);
            }}
          >
            Conversational Signs
          </button>
        </div>

        <div className="practice-toggles-group">
          {/* Auto-Next Toggle */}
          <button
            type="button"
            className={`practice-pill-toggle ${autoAdvance ? 'active' : ''}`}
            onClick={() => setAutoAdvance(!autoAdvance)}
            title="Auto-advance to next sign upon successful hold"
          >
            {autoAdvance ? <PlayCircle size={12} /> : <PauseCircle size={12} />}
            <span>Auto: {autoAdvance ? 'ON' : 'OFF'}</span>
          </button>

          {/* Shuffle Toggle */}
          <button
            type="button"
            className={`practice-pill-toggle ${isShuffle ? 'active' : ''}`}
            onClick={() => setIsShuffle(!isShuffle)}
            title="Shuffle sign sequence"
          >
            <Shuffle size={12} />
            <span>{isShuffle ? 'Random' : 'Linear'}</span>
          </button>
        </div>
      </div>

      {/* 3. Horizontal Target Selector Carousel */}
      <div className="practice-chips-scroll" role="radiogroup" aria-label="Select Target Sign">
        {mode === 'alphabet'
          ? ALPHABET_LIST.map((letter) => (
              <button
                key={letter}
                type="button"
                role="radio"
                aria-checked={selectedTarget === letter}
                onClick={() => {
                  setSelectedTarget(letter);
                  setIsSuccess(false);
                  setHoldProgress(0);
                }}
                className={`practice-chip ${selectedTarget === letter ? 'active' : ''}`}
              >
                {letter}
              </button>
            ))
          : COMMON_SIGNS.map((s) => (
              <button
                key={s.id}
                type="button"
                role="radio"
                aria-checked={selectedTarget === s.label}
                onClick={() => {
                  setSelectedTarget(s.label);
                  setIsSuccess(false);
                  setHoldProgress(0);
                }}
                className={`practice-chip phrase-chip ${selectedTarget === s.label ? 'active' : ''}`}
              >
                {s.label}
              </button>
            ))}
      </div>

      {/* 4. Target Challenge Details Card */}
      <div className="practice-challenge-card">
        <div className="challenge-card-header">
          <span className="challenge-label">Target Challenge</span>
          {isDualHandedAlphabet && (
            <span className="challenge-dual-badge">
              <AlertCircle size={11} />
              <span>Two Hands Required</span>
            </span>
          )}
        </div>

        <div className="challenge-target-name">
          {mode === 'alphabet' ? `Letter ${selectedTarget}` : selectedTarget}
        </div>

        {/* 3D Skeletal Perspective Rigged Hand Model Demonstration */}
        <Practice3DHandViewer sign={selectedTarget} />

        <p className="challenge-instruction">
          {mode === 'alphabet'
            ? ALPHABET_GUIDES[selectedTarget]?.desc || `Hold your hand in canonical ISL pose for "${selectedTarget}".`
            : COMMON_SIGNS.find((s) => s.label === selectedTarget)?.desc}
        </p>

        {mode === 'alphabet' && ALPHABET_GUIDES[selectedTarget]?.tips && (
          <div className="challenge-pro-tip">
            <strong>Pro Tip:</strong> {ALPHABET_GUIDES[selectedTarget].tips}
          </div>
        )}

        <div className="challenge-actions-row">
          {onSpeak && (
            <button
              type="button"
              className="practice-action-btn speak-btn"
              onClick={() => onSpeak(selectedTarget)}
              title="Hear pronunciation"
            >
              <Volume2 size={13} />
              <span>Hear Sign</span>
            </button>
          )}

          <button
            type="button"
            className="practice-action-btn next-btn"
            onClick={handleNext}
            title="Skip to next sign"
          >
            <span>Next Sign</span>
            <ChevronRight size={14} />
          </button>
        </div>
      </div>

      {/* 5. Live AI Match Gauge & Verification Box */}
      <div className={`practice-feedback-card ${isSuccess ? 'success' : isMatch ? 'matching' : 'idle'}`}>
        <div className="feedback-header">
          <span className="feedback-label">Live Webcam Recognition</span>
          <span className="feedback-conf">
            {currentConfidence > 0 ? `${Math.round(currentConfidence * 100)}% Match` : 'Awaiting'}
          </span>
        </div>

        <div className="feedback-detected-value">
          {currentDetected !== 'NO SIGN' && currentDetected !== 'UNKNOWN' && currentDetected !== 'NO_SIGN' ? (
            <span className={isMatch ? 'text-emerald' : 'text-main'}>{currentDetected}</span>
          ) : (
            <span className="text-muted">Show hands to camera</span>
          )}
        </div>

        {/* Dynamic Hold Gauge Progress Bar */}
        <div className="practice-hold-bar" role="progressbar" aria-valuenow={holdProgress} aria-valuemin={0} aria-valuemax={100}>
          <div
            className={`practice-hold-fill ${isSuccess ? 'completed' : isMatch ? 'filling' : 'empty'}`}
            style={{ width: `${holdProgress}%` }}
          />
        </div>

        {/* Dynamic Status Text */}
        <div className="practice-status-hint">
          {isSuccess ? (
            <div className="status-success-banner">
              <CheckCircle2 size={15} className="text-emerald" />
              <span>Perfect! Streak +1 {autoAdvance ? '• Next sign in 1s...' : ''}</span>
            </div>
          ) : isMatch ? (
            <div className="status-matching-banner">
              <Sparkles size={14} className="text-cyan animate-spin" />
              <span>Matching! Hold steady to score ({holdProgress}%)...</span>
            </div>
          ) : (
            <span className="status-waiting-text">
              Hold the sign steady in camera view for 1 second
            </span>
          )}
        </div>
      </div>
    </div>
  );
};

export default PracticeStudio;
