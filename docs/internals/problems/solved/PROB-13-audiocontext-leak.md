---
id: PROB-13
title: "AudioContext Leak in PracticeStudio Causing Browser Chime Freeze"
stage: not_solved
status: solved
created: 2026-11-01
updated: 2026-11-01 12:18
severity: Medium
category: Frontend Web Audio & Resource Leaks
---

# PROB-13: AudioContext Leak in PracticeStudio Causing Browser Chime Freeze

## 1. Problem Description
In the Practice Studio, after completing several successful signs (between 6 and 32 items depending on the browser), the success audio chime permanently stops playing, and the browser console logs `DOMException: Only 6 AudioContexts are allowed at the same time` or `The AudioContext was not allowed to start`.

## 2. Root Cause Analysis
In `frontend/src/components/PracticeStudio.tsx` (lines 23–43):
```typescript
const playSuccessChime = () => {
  try {
    const AudioContextClass = window.AudioContext || ...;
    if (!AudioContextClass) return;
    const ctx = new AudioContextClass(); // <--- LEAK
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    ...
    osc.start();
    osc.stop(ctx.currentTime + 0.40);
  } catch { ... }
};
```
Every time `playSuccessChime` is called, a brand new `AudioContext` is instantiated. When `osc.stop` triggers, the oscillator stops, but the `AudioContext` remains active in browser memory. Web browsers enforce a strict quota on simultaneous open AudioContexts. Once this limit is reached, all subsequent audio calls fail silently.

## 3. Implementation Plan & Proposed Fix
1. Reuse a shared singleton `AudioContext` instance across chimes:
   ```typescript
   let sharedAudioCtx: AudioContext | null = null;
   const getAudioContext = (): AudioContext | null => {
     if (typeof window === 'undefined') return null;
     const AudioContextClass = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
     if (!AudioContextClass) return null;
     if (!sharedAudioCtx || sharedAudioCtx.state === 'closed') {
       sharedAudioCtx = new AudioContextClass();
     }
     if (sharedAudioCtx.state === 'suspended') {
       void sharedAudioCtx.resume();
     }
     return sharedAudioCtx;
   };
   ```
2. In `playSuccessChime`, acquire `ctx` from `getAudioContext()`.

## 4. Acceptance Criteria & Verification
- [x] 50+ consecutive success chimes trigger cleanly without throwing `AudioContext` quota errors.
- [x] TypeScript compilation passes cleanly.
