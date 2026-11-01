import React, { useState, useEffect } from 'react';
import { Award, CheckCircle2, ChevronRight, RotateCcw, Target, Sparkles, Volume2, X } from 'lucide-react';
import type { RecognitionPayload } from '../types';

export interface PracticeStudioProps {
  currentResult: RecognitionPayload | null;
  onSpeak?: (text: string) => void;
  onClose?: () => void;
}

const ALPHABET_LIST = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');

const COMMON_SIGNS = [
  { id: 'hello', label: 'HELLO', desc: 'Raise dominant hand near temple and wave gently outward.' },
  { id: 'thank_you', label: 'THANK YOU', desc: 'Touch fingertips of flat hand to chin and move forward toward other person.' },
  { id: 'namaste', label: 'NAMASTE', desc: 'Join palms together in front of chest in traditional respectful gesture.' },
  { id: 'water', label: 'WATER', desc: 'Hold W-hand or cup gesture near mouth and tilt toward lips.' },
  { id: 'food', label: 'FOOD', desc: 'Bring fingertips together and tap lightly toward lips twice.' },
  { id: 'help', label: 'HELP', desc: 'Place closed fist with thumb up onto flat open palm of other hand.' },
];

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

  const currentDetected = currentResult?.label?.toUpperCase() || 'NO_SIGN';
  const currentConfidence = currentResult?.confidence || 0;
  const isMatch = currentDetected === selectedTarget.toUpperCase() && currentConfidence >= 0.60;

  // Hold timer: User must hold the correct sign steady for 1.2s to score
  useEffect(() => {
    let timer: any = null;
    if (isMatch && !isSuccess) {
      timer = setInterval(() => {
        setHoldProgress((prev) => {
          if (prev >= 100) {
            setIsSuccess(true);
            setStreak((s) => s + 1);
            if (onSpeak) onSpeak(`Excellent! You signed ${selectedTarget}!`);
            return 100;
          }
          return prev + 15;
        });
      }, 150);
    } else if (!isMatch && !isSuccess) {
      setHoldProgress(0);
    }
    return () => clearInterval(timer);
  }, [isMatch, isSuccess, selectedTarget, onSpeak]);

  const handleNext = () => {
    setIsSuccess(false);
    setHoldProgress(0);
    if (mode === 'alphabet') {
      const idx = ALPHABET_LIST.indexOf(selectedTarget);
      const nextIdx = (idx + 1) % ALPHABET_LIST.length;
      setSelectedTarget(ALPHABET_LIST[nextIdx]);
    } else {
      const idx = COMMON_SIGNS.findIndex((s) => s.label === selectedTarget);
      const nextIdx = (idx + 1) % COMMON_SIGNS.length;
      setSelectedTarget(COMMON_SIGNS[nextIdx].label);
    }
  };

  const handleReset = () => {
    setStreak(0);
    setIsSuccess(false);
    setHoldProgress(0);
  };

  return (
    <div className="practice-studio-root bg-slate-900/95 border border-slate-700/80 rounded-2xl p-6 text-white shadow-2xl backdrop-blur-md">
      {/* Studio Header */}
      <div className="flex items-center justify-between border-b border-slate-700/60 pb-4 mb-6">
        <div className="flex items-center gap-3">
          <div className="p-2.5 bg-indigo-500/20 text-indigo-400 rounded-xl border border-indigo-500/30">
            <Target className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
              ISL Interactive Practice Studio
              <span className="text-xs px-2.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 font-medium">
                Live AI Scoring
              </span>
            </h2>
            <p className="text-sm text-slate-400">
              Practice authentic Indian Sign Language gestures with real-time accuracy scoring.
            </p>
          </div>
        </div>

        {/* Score & Streak */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 px-3.5 py-1.5 bg-amber-500/10 border border-amber-500/30 rounded-xl">
            <Award className="w-5 h-5 text-amber-400" />
            <span className="text-sm font-semibold text-amber-300">Streak: {streak}</span>
          </div>
          <button
            onClick={handleReset}
            className="p-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition"
            title="Reset Score"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
          {onClose && (
            <button
              onClick={onClose}
              className="p-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition"
              title="Close Practice Studio"
            >
              <X className="w-5 h-5" />
            </button>
          )}
        </div>
      </div>

      {/* Mode Selector Tabs */}
      <div className="flex items-center gap-3 mb-6">
        <button
          onClick={() => {
            setMode('alphabet');
            setSelectedTarget('A');
            setIsSuccess(false);
            setHoldProgress(0);
          }}
          className={`px-4 py-2 rounded-xl text-sm font-medium transition ${
            mode === 'alphabet'
              ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/30'
              : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
          }`}
        >
          Dual-Hand Alphabet (A–Z)
        </button>
        <button
          onClick={() => {
            setMode('phrases');
            setSelectedTarget(COMMON_SIGNS[0].label);
            setIsSuccess(false);
            setHoldProgress(0);
          }}
          className={`px-4 py-2 rounded-xl text-sm font-medium transition ${
            mode === 'phrases'
              ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/30'
              : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
          }`}
        >
          Common Phrases & Signs
        </button>
      </div>

      {/* Target Selector Bar */}
      <div className="flex items-center gap-2 overflow-x-auto pb-3 mb-6 scrollbar-thin scrollbar-thumb-slate-700">
        {mode === 'alphabet'
          ? ALPHABET_LIST.map((letter) => (
              <button
                key={letter}
                onClick={() => {
                  setSelectedTarget(letter);
                  setIsSuccess(false);
                  setHoldProgress(0);
                }}
                className={`w-10 h-10 rounded-xl font-bold text-sm shrink-0 transition ${
                  selectedTarget === letter
                    ? 'bg-indigo-500 text-white ring-2 ring-indigo-400 shadow-md'
                    : 'bg-slate-800/80 text-slate-400 hover:bg-slate-700 hover:text-white'
                }`}
              >
                {letter}
              </button>
            ))
          : COMMON_SIGNS.map((s) => (
              <button
                key={s.id}
                onClick={() => {
                  setSelectedTarget(s.label);
                  setIsSuccess(false);
                  setHoldProgress(0);
                }}
                className={`px-3 py-2 rounded-xl font-medium text-xs shrink-0 transition ${
                  selectedTarget === s.label
                    ? 'bg-indigo-500 text-white ring-2 ring-indigo-400 shadow-md'
                    : 'bg-slate-800/80 text-slate-400 hover:bg-slate-700 hover:text-white'
                }`}
              >
                {s.label}
              </button>
            ))}
      </div>

      {/* Practice Challenge Center */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 bg-slate-800/50 rounded-2xl p-6 border border-slate-700/50">
        {/* Left: Target Sign & Guidance */}
        <div className="flex flex-col justify-between">
          <div>
            <div className="text-xs uppercase tracking-wider text-slate-400 mb-1 font-semibold">
              Target Sign
            </div>
            <div className="text-4xl font-extrabold text-indigo-400 tracking-wide mb-3">
              {mode === 'alphabet' ? `LETTER ${selectedTarget}` : selectedTarget}
            </div>

            {mode === 'alphabet' && 'BDPTQ'.includes(selectedTarget) && (
              <div className="inline-flex items-center gap-1.5 px-3 py-1 bg-amber-500/10 border border-amber-500/30 rounded-lg text-xs text-amber-300 font-medium mb-3">
                <span>⚠️ Dual-Handed Sign: Both hands required in frame!</span>
              </div>
            )}

            <p className="text-sm text-slate-300 leading-relaxed mb-4">
              {mode === 'alphabet'
                ? `Hold your hands in the canonical ISL pose for "${selectedTarget}". Keep both wrists centered within the camera view.`
                : COMMON_SIGNS.find((s) => s.label === selectedTarget)?.desc}
            </p>
          </div>

          <div className="flex items-center gap-3">
            {onSpeak && (
              <button
                onClick={() => onSpeak(selectedTarget)}
                className="flex items-center gap-2 px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded-xl text-xs font-semibold text-slate-200 transition"
              >
                <Volume2 className="w-4 h-4" /> Speak Sign
              </button>
            )}
            <button
              onClick={handleNext}
              className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 rounded-xl text-xs font-bold text-white transition ml-auto"
            >
              Next Sign <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Right: Live AI Feedback & Hold Gauge */}
        <div className="flex flex-col justify-center items-center p-6 bg-slate-900/80 rounded-xl border border-slate-700/60 text-center">
          <div className="text-xs uppercase tracking-wider text-slate-400 mb-2 font-semibold">
            Live Webcam Detection
          </div>

          <div className="text-3xl font-black mb-1">
            {currentDetected !== 'NO_SIGN' && currentDetected !== 'UNKNOWN' ? (
              <span className={isMatch ? 'text-emerald-400' : 'text-slate-300'}>
                {currentDetected}
              </span>
            ) : (
              <span className="text-slate-500">Awaiting Sign</span>
            )}
          </div>

          <div className="text-xs text-slate-400 mb-4">
            Confidence: {Math.round(currentConfidence * 100)}%
          </div>

          {/* Progress / Hold Ring Gauge */}
          <div className="w-full bg-slate-800 rounded-full h-3 mb-3 overflow-hidden border border-slate-700">
            <div
              className={`h-full transition-all duration-150 ${
                isSuccess
                  ? 'bg-emerald-400'
                  : isMatch
                  ? 'bg-gradient-to-r from-indigo-500 to-emerald-400'
                  : 'bg-transparent'
              }`}
              style={{ width: `${holdProgress}%` }}
            />
          </div>

          {/* Dynamic Status Guidance */}
          {isSuccess ? (
            <div className="flex items-center gap-2 text-emerald-400 font-bold text-sm animate-bounce">
              <CheckCircle2 className="w-5 h-5" /> Completed! Streak +1
            </div>
          ) : isMatch ? (
            <div className="flex items-center gap-2 text-indigo-400 font-semibold text-xs">
              <Sparkles className="w-4 h-4 animate-spin" /> Perfect match! Hold steady to score...
            </div>
          ) : (
            <div className="text-xs text-slate-400">
              Form the sign and hold steady in front of camera
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
