import React, { useRef, useEffect, useState, useCallback } from 'react';
import * as THREE from 'three';
import { Sparkles, Activity } from 'lucide-react';

export interface HumanoidAvatar3DProps {
  currentSign: string;
  isTwoHanded?: boolean;
  className?: string;
}

// Finger joint angles in radians
export interface FingerPose {
  mcp: number; // Metacarpophalangeal flexion (0: straight upright, 1.57: 90 deg curl into palm)
  pip: number; // Proximal interphalangeal flexion
  dip: number; // Distal interphalangeal flexion
  spread: number; // Abduction / lateral spread angle (-0.35 to 0.35 rad)
}

// 5-finger hand state
export interface HandPoseState {
  wristRot: [number, number, number]; // pitch, yaw, roll
  wristPos: [number, number, number]; // x, y, z target position in front of chest
  thumb: FingerPose;
  index: FingerPose;
  middle: FingerPose;
  ring: FingerPose;
  pinky: FingerPose;
}

export interface BodyPoseState {
  headRot: [number, number, number]; // pitch, yaw, roll
  leftArm: HandPoseState;
  rightArm: HandPoseState;
  isTwoHanded: boolean;
}

// Helper to create finger flexion pose
function f(mcp: number, pip: number, dip: number, spread: number = 0): FingerPose {
  return { mcp, pip, dip, spread };
}

// Extended upright finger
const OPEN_FINGER = f(0.02, 0.02, 0.02, 0.0);
// Closed fist finger (curled inward into palm)
const FIST_FINGER = f(1.48, 1.55, 1.25, 0.0);
// Semi-curled finger (C or O curve)
const CURLED_FINGER = f(0.85, 0.95, 0.75, 0.0);
// Hooked finger (X or claw)
const HOOK_FINGER = f(0.40, 1.25, 0.95, 0.0);

// Library of Authentic Indian Sign Language (ISL) Canonical 3D Poses
export function getSignPose(rawSign: string, progress: number = 0.5): BodyPoseState {
  const s = (rawSign || 'GOOD_MORNING').toUpperCase().replace(/\s+/g, '_');

  // Natural resting ready pose in front of lower chest
  const neutralLeft: HandPoseState = {
    wristRot: [0.15, 0.3, -0.2],
    wristPos: [-0.18, 0.08, 0.24],
    thumb: f(0.3, 0.4, 0.2, -0.3),
    index: f(0.35, 0.45, 0.25, -0.08),
    middle: f(0.4, 0.5, 0.3, 0.0),
    ring: f(0.4, 0.5, 0.3, 0.08),
    pinky: f(0.35, 0.45, 0.25, 0.16),
  };

  const neutralRight: HandPoseState = {
    wristRot: [0.15, -0.3, 0.2],
    wristPos: [0.18, 0.08, 0.24],
    thumb: f(0.3, 0.4, 0.2, 0.3),
    index: f(0.35, 0.45, 0.25, 0.08),
    middle: f(0.4, 0.5, 0.3, 0.0),
    ring: f(0.4, 0.5, 0.3, -0.08),
    pinky: f(0.35, 0.45, 0.25, -0.16),
  };

  const pose: BodyPoseState = {
    headRot: [0, 0, 0],
    leftArm: { ...neutralLeft },
    rightArm: { ...neutralRight },
    isTwoHanded: false,
  };

  // -------------------------------------------------------------
  // Conversational Phrases & Greetings with Fluid Dynamic Phases
  // -------------------------------------------------------------
  if (s === 'GOOD_MORNING' || s === 'GOOD MORNING') {
    pose.isTwoHanded = true;
    if (progress < 0.46) {
      // Phase 1: "GOOD" -> Crisp Thumbs-Up with dominant hand at chest + affirmative nod
      const p1 = progress / 0.46;
      const nod = Math.sin(p1 * Math.PI) * 0.06;
      pose.headRot = [0.08 + nod, 0, 0];
      pose.rightArm = {
        wristRot: [0.15, -0.20, 0.10],
        wristPos: [0.12, 0.20 + nod * 0.5, 0.28],
        thumb: f(0.0, 0.0, 0.0, 0.55), // High vertical thumbs-up
        index: FIST_FINGER,
        middle: FIST_FINGER,
        ring: FIST_FINGER,
        pinky: FIST_FINGER,
      };
      pose.leftArm = {
        wristRot: [0.15, 0.25, -0.15],
        wristPos: [-0.16, 0.08, 0.24],
        thumb: f(0.2, 0.3, 0.2, -0.25),
        index: f(0.3, 0.4, 0.2, -0.05),
        middle: f(0.3, 0.4, 0.2, 0.0),
        ring: f(0.3, 0.4, 0.2, 0.05),
        pinky: f(0.3, 0.4, 0.2, 0.1),
      };
    } else {
      // Phase 2: "MORNING" -> Non-dominant arm forms horizontal horizon, dominant hand rises like sun
      const p2 = (progress - 0.46) / 0.54;
      const riseY = 0.14 + p2 * 0.18;
      const sunSpread = 0.18 + p2 * 0.22;
      pose.headRot = [-0.08, 0, 0]; // Approving gaze towards morning sunrise
      // Left arm horizontal horizon across chest
      pose.leftArm = {
        wristRot: [1.45, 0.0, 0.0],
        wristPos: [-0.06, 0.12, 0.28],
        thumb: f(0.05, 0.05, 0.05, -0.30),
        index: OPEN_FINGER,
        middle: OPEN_FINGER,
        ring: OPEN_FINGER,
        pinky: OPEN_FINGER,
      };
      // Right hand rising up with fingers spreading like sun rays
      pose.rightArm = {
        wristRot: [0.10, -0.15, 0.14],
        wristPos: [0.10 + p2 * 0.06, riseY, 0.28],
        thumb: f(0.0, 0.0, 0.0, 0.45 + p2 * 0.15),
        index: f(0.0, 0.0, 0.0, sunSpread),
        middle: f(0.0, 0.0, 0.0, 0.0),
        ring: f(0.0, 0.0, 0.0, -sunSpread),
        pinky: f(0.0, 0.0, 0.0, -sunSpread * 1.4),
      };
    }
  } else if (s === 'HELLO' || s === 'HI' || s === 'WAVE') {
    // Right hand raised to temple, fingers upright, waving side-to-side
    const wave = Math.sin(progress * Math.PI * 4) * 0.22;
    pose.headRot = [0.02, -0.05, wave * 0.12];
    pose.rightArm = {
      wristRot: [0.12, -0.18, 0.15 + wave],
      wristPos: [0.20, 0.35, 0.24],
      thumb: f(0.08, 0.05, 0.05, 0.45),
      index: f(0.02, 0.02, 0.02, 0.18),
      middle: f(0.02, 0.02, 0.02, 0.0),
      ring: f(0.02, 0.02, 0.02, -0.18),
      pinky: f(0.02, 0.02, 0.02, -0.30),
    };
  } else if (s === 'NAMASTE' || s === 'THANK_YOU' || s === 'THANK YOU' || s === 'THANKS') {
    // Both palms pressed flat together at center chest, respectful head bow
    pose.isTwoHanded = true;
    const bow = Math.sin(progress * Math.PI * 2) * 0.06;
    pose.headRot = [0.12 + bow, 0.0, 0.0];
    pose.rightArm = {
      wristRot: [0.0, -1.45, 0.0],
      wristPos: [0.032, 0.18 - bow * 0.15, 0.26],
      thumb: f(0.08, 0.05, 0.05, 0.18),
      index: f(0.02, 0.02, 0.02, 0.04),
      middle: f(0.02, 0.02, 0.02, 0.0),
      ring: f(0.02, 0.02, 0.02, -0.04),
      pinky: f(0.02, 0.02, 0.02, -0.08),
    };
    pose.leftArm = {
      wristRot: [0.0, 1.45, 0.0],
      wristPos: [-0.032, 0.18 - bow * 0.15, 0.26],
      thumb: f(0.08, 0.05, 0.05, -0.18),
      index: f(0.02, 0.02, 0.02, -0.04),
      middle: f(0.02, 0.02, 0.02, 0.0),
      ring: f(0.02, 0.02, 0.02, 0.04),
      pinky: f(0.02, 0.02, 0.02, 0.08),
    };
  } else if (s === 'HELP') {
    // Left palm horizontal facing up; right fist with thumb up placed on left palm, lifting together
    pose.isTwoHanded = true;
    const lift = Math.sin(progress * Math.PI * 2) * 0.035;
    pose.headRot = [0.04, 0, 0];
    pose.leftArm = {
      wristRot: [1.45, 0.0, 0.0],
      wristPos: [-0.04, 0.11 + lift, 0.28],
      thumb: f(0.1, 0.1, 0.1, -0.28),
      index: OPEN_FINGER,
      middle: OPEN_FINGER,
      ring: OPEN_FINGER,
      pinky: OPEN_FINGER,
    };
    pose.rightArm = {
      wristRot: [0.15, -0.2, 0.0],
      wristPos: [0.03, 0.19 + lift, 0.28],
      thumb: f(0.0, 0.0, 0.0, 0.55),
      index: FIST_FINGER,
      middle: FIST_FINGER,
      ring: FIST_FINGER,
      pinky: FIST_FINGER,
    };
  } else if (s === 'PLEASE' || s === 'WELCOME') {
    // Both open hands sweeping gently inwards over chest
    pose.isTwoHanded = true;
    const sweep = Math.sin(progress * Math.PI * 2) * 0.03;
    pose.rightArm = {
      wristRot: [0.25, -0.35 + sweep, 0.1],
      wristPos: [0.16 - sweep, 0.16, 0.26],
      thumb: f(0.08, 0.05, 0.05, 0.30),
      index: f(0.05, 0.05, 0.05, 0.12),
      middle: f(0.05, 0.05, 0.05, 0.0),
      ring: f(0.05, 0.05, 0.05, -0.12),
      pinky: f(0.05, 0.05, 0.05, -0.22),
    };
    pose.leftArm = {
      wristRot: [0.25, 0.35 - sweep, -0.1],
      wristPos: [-0.16 + sweep, 0.16, 0.26],
      thumb: f(0.08, 0.05, 0.05, -0.30),
      index: f(0.05, 0.05, 0.05, -0.12),
      middle: f(0.05, 0.05, 0.05, 0.0),
      ring: f(0.05, 0.05, 0.05, 0.12),
      pinky: f(0.05, 0.05, 0.05, 0.22),
    };
  } else if (s === 'WATER' || s === 'DRINK') {
    // C-shaped hand brought near mouth
    const sip = Math.sin(progress * Math.PI * 2) * 0.03;
    pose.headRot = [-0.08 - sip * 0.5, 0, 0];
    pose.rightArm = {
      wristRot: [0.35 + sip * 0.2, -0.25, 0.15],
      wristPos: [0.08, 0.31 + sip, 0.22],
      thumb: f(0.40, 0.50, 0.35, 0.38),
      index: f(0.65, 0.75, 0.55, 0.05),
      middle: f(0.70, 0.80, 0.60, 0.0),
      ring: f(0.75, 0.85, 0.65, -0.05),
      pinky: f(0.80, 0.90, 0.70, -0.10),
    };
  } else if (s === 'FOOD' || s === 'EAT') {
    // Bunched fingertips tapping near lips
    const tap = Math.abs(Math.sin(progress * Math.PI * 4)) * 0.022;
    pose.headRot = [0.04, 0, 0];
    pose.rightArm = {
      wristRot: [0.42, -0.18, 0.10],
      wristPos: [0.04, 0.33, 0.21 + tap],
      thumb: f(0.85, 0.95, 0.70, 0.12),
      index: f(0.95, 1.15, 0.85, 0.0),
      middle: f(0.95, 1.15, 0.85, 0.0),
      ring: f(0.95, 1.15, 0.85, 0.0),
      pinky: f(0.95, 1.15, 0.85, 0.0),
    };
  } else if (s === 'YES') {
    // Fist nodding at shoulder height
    const nod = Math.sin(progress * Math.PI * 4) * 0.25;
    pose.headRot = [0.10 + nod * 0.3, 0, 0];
    pose.rightArm = {
      wristRot: [0.45 + nod, 0.0, 0.1],
      wristPos: [0.18, 0.24, 0.26],
      thumb: f(0.6, 1.2, 1.0, 0.2),
      index: FIST_FINGER,
      middle: FIST_FINGER,
      ring: FIST_FINGER,
      pinky: FIST_FINGER,
    };
  } else if (s === 'NO') {
    // Index and middle extended, snapping down; head shaking
    const shake = Math.sin(progress * Math.PI * 4) * 0.15;
    const snap = Math.abs(Math.sin(progress * Math.PI * 2)) * 0.4;
    pose.headRot = [0, shake, 0];
    pose.rightArm = {
      wristRot: [0.20, 0.1, 0.0],
      wristPos: [0.18, 0.22, 0.28],
      thumb: f(0.6, 0.7, 0.5, 0.2),
      index: f(0.55 + snap, 0.75, 0.45, 0.05),
      middle: f(0.55 + snap, 0.75, 0.45, -0.05),
      ring: FIST_FINGER,
      pinky: FIST_FINGER,
    };
  } else if (s === 'YOU') {
    // Index pointing directly forward to partner
    pose.rightArm = {
      wristRot: [1.45, 0.0, 0.0],
      wristPos: [0.12, 0.18, 0.38],
      thumb: f(0.8, 1.0, 0.8, 0.2),
      index: f(0.0, 0.0, 0.0, 0.0),
      middle: FIST_FINGER,
      ring: FIST_FINGER,
      pinky: FIST_FINGER,
    };
  } else if (s === 'I' || s === 'ME') {
    // Index pointing gently to center chest
    pose.rightArm = {
      wristRot: [0.55, -0.4, 0.1],
      wristPos: [0.05, 0.18, 0.18],
      thumb: f(0.8, 1.0, 0.8, 0.2),
      index: f(0.0, 0.0, 0.0, 0.0),
      middle: FIST_FINGER,
      ring: FIST_FINGER,
      pinky: FIST_FINGER,
    };
  } else if (s === 'SORRY') {
    // Fist rubbing circular motion on chest
    const rub = progress * Math.PI * 4;
    pose.headRot = [0.08, 0, 0];
    pose.rightArm = {
      wristRot: [0.35, -0.25, 0.1],
      wristPos: [0.04 + Math.cos(rub) * 0.03, 0.18 + Math.sin(rub) * 0.03, 0.20],
      thumb: f(0.8, 1.0, 0.8, 0.2),
      index: FIST_FINGER,
      middle: FIST_FINGER,
      ring: FIST_FINGER,
      pinky: FIST_FINGER,
    };
  } else if (s === 'STOP') {
    // Open palm pushed forward firmly
    pose.rightArm = {
      wristRot: [0.0, 0.0, 0.0],
      wristPos: [0.18, 0.22, 0.35],
      thumb: f(0.0, 0.0, 0.0, 0.45),
      index: OPEN_FINGER,
      middle: OPEN_FINGER,
      ring: OPEN_FINGER,
      pinky: OPEN_FINGER,
    };
  }
  // -------------------------------------------------------------
  // ISL Alphabets (A–Z) with authentic upright finger configurations
  // -------------------------------------------------------------
  else if (s === 'A') {
    pose.rightArm.wristPos = [0.16, 0.22, 0.26];
    pose.rightArm.thumb = f(0.0, 0.0, 0.0, 0.45);
    pose.rightArm.index = FIST_FINGER;
    pose.rightArm.middle = FIST_FINGER;
    pose.rightArm.ring = FIST_FINGER;
    pose.rightArm.pinky = FIST_FINGER;
  } else if (s === 'B') {
    pose.isTwoHanded = true;
    pose.leftArm = {
      wristRot: [0.1, 0.35, -0.1],
      wristPos: [-0.06, 0.22, 0.26],
      thumb: f(0.1, 0.1, 0.1, -0.2),
      index: OPEN_FINGER,
      middle: OPEN_FINGER,
      ring: OPEN_FINGER,
      pinky: OPEN_FINGER,
    };
    pose.rightArm = {
      wristRot: [0.1, -0.35, 0.1],
      wristPos: [0.06, 0.22, 0.26],
      thumb: f(0.1, 0.1, 0.1, 0.2),
      index: OPEN_FINGER,
      middle: OPEN_FINGER,
      ring: OPEN_FINGER,
      pinky: OPEN_FINGER,
    };
  } else if (s === 'C') {
    pose.rightArm.wristPos = [0.16, 0.24, 0.26];
    pose.rightArm.thumb = f(0.45, 0.55, 0.35, 0.35);
    pose.rightArm.index = CURLED_FINGER;
    pose.rightArm.middle = CURLED_FINGER;
    pose.rightArm.ring = CURLED_FINGER;
    pose.rightArm.pinky = CURLED_FINGER;
  } else if (s === 'D') {
    pose.isTwoHanded = true;
    pose.leftArm = {
      wristRot: [0.0, 0.1, 0.0],
      wristPos: [-0.05, 0.24, 0.26],
      thumb: FIST_FINGER,
      index: f(0.0, 0.0, 0.0, 0.0),
      middle: FIST_FINGER,
      ring: FIST_FINGER,
      pinky: FIST_FINGER,
    };
    pose.rightArm = {
      wristRot: [0.1, -0.4, 0.2],
      wristPos: [0.04, 0.24, 0.26],
      thumb: f(0.4, 0.5, 0.3, 0.3),
      index: f(0.6, 0.8, 0.6, 0.0),
      middle: FIST_FINGER,
      ring: FIST_FINGER,
      pinky: FIST_FINGER,
    };
  } else if (s === 'E') {
    pose.rightArm.wristPos = [0.16, 0.22, 0.26];
    pose.rightArm.thumb = f(0.7, 0.8, 0.6, 0.1);
    pose.rightArm.index = f(1.1, 1.2, 0.9, 0.0);
    pose.rightArm.middle = f(1.1, 1.2, 0.9, 0.0);
    pose.rightArm.ring = f(1.1, 1.2, 0.9, 0.0);
    pose.rightArm.pinky = f(1.1, 1.2, 0.9, 0.0);
  } else if (s === 'F') {
    pose.rightArm.wristPos = [0.16, 0.24, 0.26];
    pose.rightArm.thumb = f(0.6, 0.7, 0.4, 0.2);
    pose.rightArm.index = f(0.8, 0.9, 0.6, 0.0);
    pose.rightArm.middle = OPEN_FINGER;
    pose.rightArm.ring = OPEN_FINGER;
    pose.rightArm.pinky = OPEN_FINGER;
  } else if (s === 'G') {
    pose.rightArm.wristRot = [0.0, 0.5, 0.8];
    pose.rightArm.wristPos = [0.16, 0.22, 0.26];
    pose.rightArm.thumb = f(0.1, 0.0, 0.0, 0.25);
    pose.rightArm.index = f(0.0, 0.0, 0.0, 0.0);
    pose.rightArm.middle = FIST_FINGER;
    pose.rightArm.ring = FIST_FINGER;
    pose.rightArm.pinky = FIST_FINGER;
  } else if (s === 'H') {
    pose.rightArm.wristRot = [0.0, 0.5, 0.8];
    pose.rightArm.wristPos = [0.16, 0.22, 0.26];
    pose.rightArm.thumb = f(0.8, 1.0, 0.8, 0.1);
    pose.rightArm.index = f(0.0, 0.0, 0.0, 0.02);
    pose.rightArm.middle = f(0.0, 0.0, 0.0, -0.02);
    pose.rightArm.ring = FIST_FINGER;
    pose.rightArm.pinky = FIST_FINGER;
  } else if (s === 'I') {
    pose.rightArm.wristPos = [0.16, 0.24, 0.26];
    pose.rightArm.thumb = f(0.9, 1.1, 0.8, 0.2);
    pose.rightArm.index = FIST_FINGER;
    pose.rightArm.middle = FIST_FINGER;
    pose.rightArm.ring = FIST_FINGER;
    pose.rightArm.pinky = f(0.0, 0.0, 0.0, 0.0);
  } else if (s === 'J') {
    pose.rightArm.wristPos = [0.16, 0.24, 0.26];
    const jSwing = Math.sin(progress * Math.PI * 2) * 0.25;
    pose.rightArm.wristRot = [0.1, 0.0, jSwing];
    pose.rightArm.thumb = f(0.9, 1.1, 0.8, 0.2);
    pose.rightArm.index = FIST_FINGER;
    pose.rightArm.middle = FIST_FINGER;
    pose.rightArm.ring = FIST_FINGER;
    pose.rightArm.pinky = f(0.0, 0.0, 0.0, 0.0);
  } else if (s === 'K') {
    pose.rightArm.wristPos = [0.16, 0.24, 0.26];
    pose.rightArm.thumb = f(0.2, 0.2, 0.1, 0.35);
    pose.rightArm.index = f(0.0, 0.0, 0.0, 0.0);
    pose.rightArm.middle = f(0.7, 0.2, 0.1, 0.0);
    pose.rightArm.ring = FIST_FINGER;
    pose.rightArm.pinky = FIST_FINGER;
  } else if (s === 'L') {
    pose.rightArm.wristPos = [0.16, 0.24, 0.26];
    pose.rightArm.thumb = f(0.0, 0.0, 0.0, 0.55);
    pose.rightArm.index = f(0.0, 0.0, 0.0, 0.0);
    pose.rightArm.middle = FIST_FINGER;
    pose.rightArm.ring = FIST_FINGER;
    pose.rightArm.pinky = FIST_FINGER;
  } else if (s === 'M') {
    pose.rightArm.wristPos = [0.16, 0.22, 0.26];
    pose.rightArm.thumb = f(0.8, 1.0, 0.8, 0.1);
    pose.rightArm.index = f(1.2, 1.1, 0.9, 0.0);
    pose.rightArm.middle = f(1.2, 1.1, 0.9, 0.0);
    pose.rightArm.ring = f(1.2, 1.1, 0.9, 0.0);
    pose.rightArm.pinky = FIST_FINGER;
  } else if (s === 'N') {
    pose.rightArm.wristPos = [0.16, 0.22, 0.26];
    pose.rightArm.thumb = f(0.8, 1.0, 0.8, 0.1);
    pose.rightArm.index = f(1.2, 1.1, 0.9, 0.0);
    pose.rightArm.middle = f(1.2, 1.1, 0.9, 0.0);
    pose.rightArm.ring = FIST_FINGER;
    pose.rightArm.pinky = FIST_FINGER;
  } else if (s === 'O') {
    pose.rightArm.wristPos = [0.16, 0.22, 0.26];
    pose.rightArm.thumb = f(0.8, 0.8, 0.6, 0.2);
    pose.rightArm.index = f(0.9, 0.9, 0.8, 0.0);
    pose.rightArm.middle = f(0.9, 0.9, 0.8, 0.0);
    pose.rightArm.ring = f(0.9, 0.9, 0.8, 0.0);
    pose.rightArm.pinky = f(0.9, 0.9, 0.8, 0.0);
  } else if (s === 'P') {
    pose.rightArm.wristRot = [0.65, -0.2, 0.0];
    pose.rightArm.wristPos = [0.16, 0.20, 0.26];
    pose.rightArm.thumb = f(0.2, 0.2, 0.1, 0.35);
    pose.rightArm.index = f(0.0, 0.0, 0.0, 0.0);
    pose.rightArm.middle = f(0.7, 0.2, 0.1, 0.0);
    pose.rightArm.ring = FIST_FINGER;
    pose.rightArm.pinky = FIST_FINGER;
  } else if (s === 'Q') {
    pose.rightArm.wristRot = [0.85, 0.3, 0.2];
    pose.rightArm.wristPos = [0.16, 0.20, 0.26];
    pose.rightArm.thumb = f(0.1, 0.0, 0.0, 0.25);
    pose.rightArm.index = f(0.0, 0.0, 0.0, 0.0);
    pose.rightArm.middle = FIST_FINGER;
    pose.rightArm.ring = FIST_FINGER;
    pose.rightArm.pinky = FIST_FINGER;
  } else if (s === 'R') {
    pose.rightArm.wristPos = [0.16, 0.24, 0.26];
    pose.rightArm.thumb = f(0.8, 1.0, 0.8, 0.2);
    pose.rightArm.index = f(0.0, 0.0, 0.0, -0.10);
    pose.rightArm.middle = f(0.0, 0.0, 0.0, 0.10);
    pose.rightArm.ring = FIST_FINGER;
    pose.rightArm.pinky = FIST_FINGER;
  } else if (s === 'S') {
    pose.rightArm.wristPos = [0.16, 0.22, 0.26];
    pose.rightArm.thumb = f(1.1, 1.1, 0.9, 0.0); // Folded tight over front of fingers
    pose.rightArm.index = FIST_FINGER;
    pose.rightArm.middle = FIST_FINGER;
    pose.rightArm.ring = FIST_FINGER;
    pose.rightArm.pinky = FIST_FINGER;
  } else if (s === 'T') {
    pose.rightArm.wristPos = [0.16, 0.22, 0.26];
    pose.rightArm.thumb = f(0.5, 0.5, 0.4, 0.2);
    pose.rightArm.index = f(1.2, 1.1, 0.9, 0.0);
    pose.rightArm.middle = FIST_FINGER;
    pose.rightArm.ring = FIST_FINGER;
    pose.rightArm.pinky = FIST_FINGER;
  } else if (s === 'U') {
    pose.rightArm.wristPos = [0.16, 0.24, 0.26];
    pose.rightArm.thumb = f(0.8, 1.0, 0.8, 0.2);
    pose.rightArm.index = f(0.0, 0.0, 0.0, 0.01);
    pose.rightArm.middle = f(0.0, 0.0, 0.0, -0.01);
    pose.rightArm.ring = FIST_FINGER;
    pose.rightArm.pinky = FIST_FINGER;
  } else if (s === 'V') {
    pose.rightArm.wristPos = [0.16, 0.24, 0.26];
    pose.rightArm.thumb = f(0.8, 1.0, 0.8, 0.2);
    pose.rightArm.index = f(0.0, 0.0, 0.0, 0.22);
    pose.rightArm.middle = f(0.0, 0.0, 0.0, -0.22);
    pose.rightArm.ring = FIST_FINGER;
    pose.rightArm.pinky = FIST_FINGER;
  } else if (s === 'W') {
    pose.rightArm.wristPos = [0.16, 0.24, 0.26];
    pose.rightArm.thumb = f(0.8, 1.0, 0.8, 0.2);
    pose.rightArm.index = f(0.0, 0.0, 0.0, 0.25);
    pose.rightArm.middle = f(0.0, 0.0, 0.0, 0.0);
    pose.rightArm.ring = f(0.0, 0.0, 0.0, -0.25);
    pose.rightArm.pinky = FIST_FINGER;
  } else if (s === 'X') {
    pose.rightArm.wristPos = [0.16, 0.22, 0.26];
    pose.rightArm.thumb = f(0.7, 0.9, 0.7, 0.2);
    pose.rightArm.index = HOOK_FINGER;
    pose.rightArm.middle = FIST_FINGER;
    pose.rightArm.ring = FIST_FINGER;
    pose.rightArm.pinky = FIST_FINGER;
  } else if (s === 'Y') {
    pose.rightArm.wristPos = [0.16, 0.22, 0.26];
    pose.rightArm.thumb = f(0.0, 0.0, 0.0, 0.55);
    pose.rightArm.index = FIST_FINGER;
    pose.rightArm.middle = FIST_FINGER;
    pose.rightArm.ring = FIST_FINGER;
    pose.rightArm.pinky = f(0.0, 0.0, 0.0, -0.42);
  } else if (s === 'Z') {
    const zPhase = (progress * 3) % 1;
    pose.rightArm.wristPos = [0.14 + zPhase * 0.06, 0.24 - zPhase * 0.04, 0.26];
    pose.rightArm.thumb = f(0.8, 1.0, 0.8, 0.2);
    pose.rightArm.index = f(0.0, 0.0, 0.0, 0.0);
    pose.rightArm.middle = FIST_FINGER;
    pose.rightArm.ring = FIST_FINGER;
    pose.rightArm.pinky = FIST_FINGER;
  }

  return pose;
}

// -------------------------------------------------------------
// Component
// -------------------------------------------------------------
export const HumanoidAvatar3D: React.FC<HumanoidAvatar3DProps> = ({
  currentSign,
  className = '',
}) => {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const reqIdRef = useRef<number | null>(null);
  const [cameraMode, setCameraMode] = useState<'front' | 'hands' | 'side'>('front');
  const [showTelemetry, setShowTelemetry] = useState<boolean>(false);
  const [jointGlow, setJointGlow] = useState<boolean>(false);

  // Live real-time telemetry state for smallest bend verification
  const [fingerDegrees, setFingerDegrees] = useState<{
    thumb: { angle: number; status: string };
    index: { angle: number; status: string };
    middle: { angle: number; status: string };
    ring: { angle: number; status: string };
    pinky: { angle: number; status: string };
  }>({
    thumb: { angle: 0, status: 'Extended' },
    index: { angle: 0, status: 'Extended' },
    middle: { angle: 0, status: 'Extended' },
    ring: { angle: 0, status: 'Extended' },
    pinky: { angle: 0, status: 'Extended' },
  });

  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const jointMarkersRef = useRef<THREE.Mesh[]>([]);

  // Smooth interpolation state
  const currentPoseRef = useRef<BodyPoseState>(getSignPose('GOOD_MORNING', 0));
  const targetPoseRef = useRef<BodyPoseState>(getSignPose(currentSign, 0));

  // Dynamic 3D Human body components
  const bodyRef = useRef<{
    head: THREE.Group;
    torso: THREE.Group;
    leftArm: {
      shoulder: THREE.Group;
      upperArm: THREE.Mesh;
      elbow: THREE.Group;
      forearm: THREE.Mesh;
      wrist: THREE.Group;
      palm: THREE.Group;
      fingers: {
        thumb: THREE.Group[];
        index: THREE.Group[];
        middle: THREE.Group[];
        ring: THREE.Group[];
        pinky: THREE.Group[];
      };
    };
    rightArm: {
      shoulder: THREE.Group;
      upperArm: THREE.Mesh;
      elbow: THREE.Group;
      forearm: THREE.Mesh;
      wrist: THREE.Group;
      palm: THREE.Group;
      fingers: {
        thumb: THREE.Group[];
        index: THREE.Group[];
        middle: THREE.Group[];
        ring: THREE.Group[];
        pinky: THREE.Group[];
      };
    };
    leftEye: THREE.Group;
    rightEye: THREE.Group;
  } | null>(null);

  // Camera presets
  const applyCameraPreset = useCallback((preset: 'front' | 'hands' | 'side') => {
    setCameraMode(preset);
    const camera = cameraRef.current;
    if (!camera) return;

    if (preset === 'front') {
      // Natural portrait framing: Head at top third, hands centered in chest signing box
      camera.position.set(0, 0.26, 1.25);
      camera.lookAt(0, 0.20, 0.05);
    } else if (preset === 'hands') {
      // Macro zoom centered on signing space framing both hands and all 10 fingers
      camera.position.set(0, 0.24, 0.74);
      camera.lookAt(0, 0.22, 0.24);
    } else if (preset === 'side') {
      // 45 degree depth angle
      camera.position.set(0.68, 0.28, 0.95);
      camera.lookAt(0.05, 0.22, 0.12);
    }
  }, []);

  // Toggle highlight on joint articulation rings
  useEffect(() => {
    jointMarkersRef.current.forEach((mesh) => {
      mesh.visible = jointGlow;
    });
  }, [jointGlow]);

  // Three.js Scene Setup & Render Loop
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const width = mount.clientWidth || 600;
    const height = mount.clientHeight || 500;

    // 1. Scene
    const scene = new THREE.Scene();
    sceneRef.current = scene;
    scene.background = new THREE.Color(0x0a0f1d);

    // 2. Camera: Framed so head is in top third and hands are centered in signing space
    const camera = new THREE.PerspectiveCamera(38, width / height, 0.1, 100);
    camera.position.set(0, 0.26, 1.25);
    camera.lookAt(0, 0.20, 0.05);
    cameraRef.current = camera;

    // 3. WebGL Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.15;
    rendererRef.current = renderer;
    mount.appendChild(renderer.domElement);

    // 4. Lighting: Three-point studio lighting + Hand illumination key light
    const keyLight = new THREE.DirectionalLight(0xfff5eb, 1.6);
    keyLight.position.set(1.5, 2.0, 2.0);
    scene.add(keyLight);

    const fillLight = new THREE.DirectionalLight(0xa5c4e8, 0.85);
    fillLight.position.set(-1.8, 1.2, 1.5);
    scene.add(fillLight);

    const rimLight = new THREE.DirectionalLight(0x818cf8, 1.4);
    rimLight.position.set(0, 2.2, -2.0);
    scene.add(rimLight);

    const ambientLight = new THREE.AmbientLight(0x283048, 1.2);
    scene.add(ambientLight);

    const handSpotlight = new THREE.PointLight(0xfff8f0, 1.5, 4.0);
    handSpotlight.position.set(0, 0.22, 0.65);
    scene.add(handSpotlight);

    // 5. Materials
    const skinMaterial = new THREE.MeshStandardMaterial({
      color: 0xdf986c, // Warm natural Indian skin tone
      roughness: 0.52,
      metalness: 0.02,
    });


    const knuckleMaterial = new THREE.MeshStandardMaterial({
      color: 0xc87d4e,
      roughness: 0.42,
      metalness: 0.04,
    });

    const nailMaterial = new THREE.MeshStandardMaterial({
      color: 0xffeedd,
      roughness: 0.20,
      metalness: 0.08,
    });

    const jointGlowMaterial = new THREE.MeshStandardMaterial({
      color: 0x38bdf8,
      emissive: 0x0284c7,
      emissiveIntensity: 0.8,
      roughness: 0.2,
    });

    const suitMaterial = new THREE.MeshStandardMaterial({
      color: 0x1e293b, // Modern navy/slate blazer
      roughness: 0.75,
      metalness: 0.08,
    });

    const collarMaterial = new THREE.MeshStandardMaterial({
      color: 0x6366f1, // Neon indigo collar accent
      roughness: 0.30,
      metalness: 0.35,
    });

    const hairMaterial = new THREE.MeshStandardMaterial({
      color: 0x1a1a20, // Dark natural styled hair
      roughness: 0.85,
    });

    const eyeWhiteMaterial = new THREE.MeshBasicMaterial({ color: 0xf8fafc });
    const irisMaterial = new THREE.MeshBasicMaterial({ color: 0x3e2312 });
    const pupilMaterial = new THREE.MeshBasicMaterial({ color: 0x09090b });
    const lipMaterial = new THREE.MeshStandardMaterial({ color: 0xc56e65, roughness: 0.42 });

    // 6. Build Humanoid Rig
    const humanRoot = new THREE.Group();
    scene.add(humanRoot);

    // Torso / Chest Group
    const torsoGroup = new THREE.Group();
    humanRoot.add(torsoGroup);

    // Sculpted Suit Body
    const chestGeo = new THREE.CylinderGeometry(0.23, 0.17, 0.46, 24);
    const chestMesh = new THREE.Mesh(chestGeo, suitMaterial);
    chestMesh.scale.set(1.0, 1.0, 0.62);
    torsoGroup.add(chestMesh);

    // V-Neck Lapel Trim
    const collarGeo = new THREE.TorusGeometry(0.085, 0.014, 12, 24);
    const collarMesh = new THREE.Mesh(collarGeo, collarMaterial);
    collarMesh.rotation.x = Math.PI / 2;
    collarMesh.position.set(0, 0.22, 0.01);
    torsoGroup.add(collarMesh);

    // Neck
    const neckGeo = new THREE.CylinderGeometry(0.065, 0.075, 0.11, 16);
    const neckMesh = new THREE.Mesh(neckGeo, skinMaterial);
    neckMesh.position.set(0, 0.27, 0);
    torsoGroup.add(neckMesh);

    // Head Group
    const headGroup = new THREE.Group();
    headGroup.position.set(0, 0.41, 0.02);
    torsoGroup.add(headGroup);

    // Skull
    const headGeo = new THREE.SphereGeometry(0.115, 32, 32);
    const headMesh = new THREE.Mesh(headGeo, skinMaterial);
    headMesh.scale.set(0.94, 1.16, 0.98);
    headGroup.add(headMesh);

    // Sculpted Hair: Seated cleanly on upper skull and back, preventing any forehead clipping
    const hairCrownGeo = new THREE.SphereGeometry(0.120, 32, 24, 0, Math.PI * 2, 0, Math.PI * 0.38);
    const hairCrown = new THREE.Mesh(hairCrownGeo, hairMaterial);
    hairCrown.scale.set(0.96, 1.16, 1.02);
    hairCrown.position.set(0, 0.042, -0.024);
    headGroup.add(hairCrown);

    const hairBackGeo = new THREE.SphereGeometry(0.120, 32, 24, -Math.PI * 0.8, Math.PI * 1.6, Math.PI * 0.22, Math.PI * 0.44);
    const hairBack = new THREE.Mesh(hairBackGeo, hairMaterial);
    hairBack.scale.set(0.98, 1.17, 1.04);
    hairBack.position.set(0, 0.008, -0.018);
    headGroup.add(hairBack);

    // Clean hairline fringe: Smooth curved torus band across hairline (above brow, zero facial clipping)
    const fringeGeo = new THREE.TorusGeometry(0.106, 0.012, 10, 24, Math.PI * 0.65);
    const fringe = new THREE.Mesh(fringeGeo, hairMaterial);
    fringe.rotation.x = Math.PI * 0.28;
    fringe.rotation.z = -Math.PI * 0.325;
    fringe.position.set(0, 0.076, 0.048);
    headGroup.add(fringe);

    // Ears
    const createEar = (isRight: boolean) => {
      const ear = new THREE.Mesh(new THREE.SphereGeometry(0.022, 12, 12), skinMaterial);
      ear.scale.set(0.4, 1.0, 0.6);
      ear.position.set(isRight ? 0.11 : -0.11, 0.01, -0.01);
      return ear;
    };
    headGroup.add(createEar(true));
    headGroup.add(createEar(false));

    // Expressive 3D Eyes with Almond Shape, Iris, Pupil, and Specular Catchlight
    const createEye = (isRight: boolean) => {
      const eyeG = new THREE.Group();
      // Almond eye sclera
      const whiteGeo = new THREE.SphereGeometry(0.015, 16, 16);
      const whiteMesh = new THREE.Mesh(whiteGeo, eyeWhiteMaterial);
      whiteMesh.scale.set(1.25, 0.85, 0.60);
      eyeG.add(whiteMesh);

      // Warm hazel/amber iris
      const irisGeo = new THREE.CylinderGeometry(0.0075, 0.0075, 0.003, 16);
      const irisMesh = new THREE.Mesh(irisGeo, irisMaterial);
      irisMesh.rotation.x = Math.PI / 2;
      irisMesh.position.set(0, 0, 0.008);
      eyeG.add(irisMesh);

      // Deep black pupil
      const pupilGeo = new THREE.CylinderGeometry(0.0042, 0.0042, 0.0032, 16);
      const pupilMesh = new THREE.Mesh(pupilGeo, pupilMaterial);
      pupilMesh.rotation.x = Math.PI / 2;
      pupilMesh.position.set(0, 0, 0.0088);
      eyeG.add(pupilMesh);

      // Specular catchlight (adds vivid lifelike gaze)
      const catchlight = new THREE.Mesh(
        new THREE.SphereGeometry(0.0018, 8, 8),
        new THREE.MeshBasicMaterial({ color: 0xffffff })
      );
      catchlight.position.set(0.0025, 0.0025, 0.010);
      eyeG.add(catchlight);

      // Upper eyelid fold
      const lidGeo = new THREE.TorusGeometry(0.014, 0.0022, 8, 16, Math.PI * 0.9);
      const lidMesh = new THREE.Mesh(lidGeo, knuckleMaterial);
      lidMesh.position.set(0, 0.004, 0.007);
      eyeG.add(lidMesh);

      eyeG.position.set(isRight ? 0.040 : -0.040, 0.030, 0.098);
      return eyeG;
    };
    const rEye = createEye(true);
    const lEye = createEye(false);
    headGroup.add(rEye);
    headGroup.add(lEye);

    // Arched Eyebrows cleanly positioned above eyes
    const createEyebrow = (isRight: boolean) => {
      const browG = new THREE.Group();
      const browMesh = new THREE.Mesh(
        new THREE.CylinderGeometry(0.0028, 0.0016, 0.034, 10),
        hairMaterial
      );
      browMesh.rotation.z = Math.PI / 2 + (isRight ? -0.15 : 0.15);
      browG.add(browMesh);
      browG.position.set(isRight ? 0.042 : -0.042, 0.054, 0.106);
      return browG;
    };
    headGroup.add(createEyebrow(true));
    headGroup.add(createEyebrow(false));

    // Natural Human Nose
    const noseG = new THREE.Group();
    const bridgeMesh = new THREE.Mesh(
      new THREE.CylinderGeometry(0.007, 0.011, 0.036, 12),
      skinMaterial
    );
    bridgeMesh.rotation.x = -0.22;
    bridgeMesh.position.set(0, 0.012, 0.006);
    noseG.add(bridgeMesh);

    const tipMesh = new THREE.Mesh(new THREE.SphereGeometry(0.009, 12, 12), skinMaterial);
    tipMesh.position.set(0, -0.006, 0.013);
    noseG.add(tipMesh);

    const nostrilL = new THREE.Mesh(new THREE.SphereGeometry(0.0055, 10, 10), skinMaterial);
    nostrilL.position.set(-0.008, -0.008, 0.008);
    noseG.add(nostrilL);

    const nostrilR = new THREE.Mesh(new THREE.SphereGeometry(0.0055, 10, 10), skinMaterial);
    nostrilR.position.set(0.008, -0.008, 0.008);
    noseG.add(nostrilR);

    noseG.position.set(0, 0.008, 0.108);
    headGroup.add(noseG);

    // Expressive Friendly Lips
    const mouthG = new THREE.Group();
    const upperLip = new THREE.Mesh(
      new THREE.TorusGeometry(0.018, 0.0036, 8, 16, Math.PI * 0.8),
      lipMaterial
    );
    upperLip.rotation.z = Math.PI * 1.1;
    upperLip.position.set(0, 0.002, 0);
    mouthG.add(upperLip);

    const lowerLip = new THREE.Mesh(
      new THREE.TorusGeometry(0.016, 0.0048, 8, 16, Math.PI * 0.7),
      lipMaterial
    );
    lowerLip.rotation.z = Math.PI * 0.15;
    lowerLip.position.set(0, -0.003, 0);
    mouthG.add(lowerLip);

    mouthG.position.set(0, -0.038, 0.106);
    headGroup.add(mouthG);

    // ---------------------------------------------------------
    // Anatomical 5-Finger Hand & Arm Builder
    // ---------------------------------------------------------
    jointMarkersRef.current = [];

    const buildArm = (isRight: boolean) => {
      const dir = isRight ? 1 : -1;

      // Shoulder Socket
      const shoulder = new THREE.Group();
      shoulder.position.set(dir * 0.23, 0.18, 0);
      torsoGroup.add(shoulder);

      const shoulderCap = new THREE.Mesh(new THREE.SphereGeometry(0.048, 14, 14), suitMaterial);
      shoulder.add(shoulderCap);

      // Upper Arm Mesh
      const upperArmMesh = new THREE.Mesh(
        new THREE.CylinderGeometry(0.042, 0.036, 1.0, 14),
        suitMaterial
      );
      torsoGroup.add(upperArmMesh);

      // Elbow Joint
      const elbow = new THREE.Group();
      torsoGroup.add(elbow);

      const elbowCap = new THREE.Mesh(new THREE.SphereGeometry(0.036, 12, 12), skinMaterial);
      elbow.add(elbowCap);

      // Forearm Mesh
      const forearmMesh = new THREE.Mesh(
        new THREE.CylinderGeometry(0.034, 0.028, 1.0, 14),
        skinMaterial
      );
      torsoGroup.add(forearmMesh);

      // Wrist Joint (Seamless transition to palm)
      const wrist = new THREE.Group();
      torsoGroup.add(wrist);

      const wristCap = new THREE.Mesh(new THREE.CylinderGeometry(0.026, 0.024, 0.022, 14), skinMaterial);
      wristCap.position.set(0, 0.011, 0);
      wrist.add(wristCap);

      // Palm Group
      const palmGroup = new THREE.Group();
      wrist.add(palmGroup);

      // 1. Central Metacarpal Palm (softly rounded)
      const palmGeo = new THREE.BoxGeometry(0.070, 0.074, 0.020);
      const palmMesh = new THREE.Mesh(palmGeo, skinMaterial);
      palmMesh.position.set(0, 0.045, 0);
      palmGroup.add(palmMesh);

      // 2. Thenar Eminence (Thumb base muscle mount on radial side)
      const thenarGeo = new THREE.SphereGeometry(0.020, 14, 14);
      const thenarMesh = new THREE.Mesh(thenarGeo, skinMaterial);
      thenarMesh.scale.set(1.15, 1.35, 0.85);
      thenarMesh.position.set(dir * 0.032, 0.022, 0.005);
      palmGroup.add(thenarMesh);

      // 3. Hypothenar Eminence (Pinky muscle mount on ulnar side)
      const hypoGeo = new THREE.SphereGeometry(0.016, 12, 12);
      const hypoMesh = new THREE.Mesh(hypoGeo, skinMaterial);
      hypoMesh.scale.set(0.95, 1.25, 0.80);
      hypoMesh.position.set(-dir * 0.028, 0.020, 0.004);
      palmGroup.add(hypoMesh);

      // Build 5 Articulated 3D Phalanges extending UPWARDS (+Y)
      const buildFingerPhalanges = (
        baseX: number,
        baseY: number,
        baseZ: number,
        pLen: number,
        iLen: number,
        dLen: number,
        fRadius: number,
        isThumb: boolean = false
      ): THREE.Group[] => {
        // Joint 1: MCP Knuckle
        const mcpJoint = new THREE.Group();
        mcpJoint.position.set(baseX, baseY, baseZ);
        palmGroup.add(mcpJoint);

        const knuckle1 = new THREE.Mesh(new THREE.SphereGeometry(fRadius * 1.05, 12, 12), knuckleMaterial);
        mcpJoint.add(knuckle1);

        const ring1 = new THREE.Mesh(new THREE.TorusGeometry(fRadius * 1.25, 0.002, 8, 16), jointGlowMaterial);
        ring1.rotation.x = Math.PI / 2;
        ring1.visible = false;
        mcpJoint.add(ring1);
        jointMarkersRef.current.push(ring1);

        // Proximal Phalanx
        const proxPhalanx = new THREE.Mesh(
          new THREE.CylinderGeometry(fRadius * 0.86, fRadius * 0.95, pLen, 12),
          skinMaterial
        );
        proxPhalanx.position.set(0, pLen / 2, 0);
        mcpJoint.add(proxPhalanx);

        // Joint 2: PIP Knuckle
        const pipJoint = new THREE.Group();
        pipJoint.position.set(0, pLen, 0);
        mcpJoint.add(pipJoint);

        const knuckle2 = new THREE.Mesh(new THREE.SphereGeometry(fRadius * 0.96, 12, 12), knuckleMaterial);
        pipJoint.add(knuckle2);

        const ring2 = new THREE.Mesh(new THREE.TorusGeometry(fRadius * 1.18, 0.002, 8, 16), jointGlowMaterial);
        ring2.rotation.x = Math.PI / 2;
        ring2.visible = false;
        pipJoint.add(ring2);
        jointMarkersRef.current.push(ring2);

        // Intermediate Phalanx (or Distal for thumb)
        const interPhalanx = new THREE.Mesh(
          new THREE.CylinderGeometry(fRadius * 0.78, fRadius * 0.86, iLen, 12),
          skinMaterial
        );
        interPhalanx.position.set(0, iLen / 2, 0);
        pipJoint.add(interPhalanx);

        if (isThumb) {
          // Thumb Distal Tip & Fingernail
          const thumbTip = new THREE.Mesh(new THREE.SphereGeometry(fRadius * 0.86, 12, 12), skinMaterial);
          thumbTip.position.set(0, iLen, 0);
          pipJoint.add(thumbTip);

          const thumbNailGeo = new THREE.CylinderGeometry(fRadius * 0.65, fRadius * 0.58, iLen * 0.6, 12, 1, false, 0, Math.PI);
          const thumbNail = new THREE.Mesh(thumbNailGeo, nailMaterial);
          thumbNail.rotation.y = Math.PI / 2;
          thumbNail.position.set(0, iLen * 0.65, -fRadius * 0.48);
          pipJoint.add(thumbNail);

          return [mcpJoint, pipJoint];
        }

        // Joint 3: DIP Knuckle
        const dipJoint = new THREE.Group();
        dipJoint.position.set(0, iLen, 0);
        pipJoint.add(dipJoint);

        const knuckle3 = new THREE.Mesh(new THREE.SphereGeometry(fRadius * 0.90, 12, 12), knuckleMaterial);
        dipJoint.add(knuckle3);

        const ring3 = new THREE.Mesh(new THREE.TorusGeometry(fRadius * 1.08, 0.0018, 8, 16), jointGlowMaterial);
        ring3.rotation.x = Math.PI / 2;
        ring3.visible = false;
        dipJoint.add(ring3);
        jointMarkersRef.current.push(ring3);

        // Distal Phalanx
        const distPhalanx = new THREE.Mesh(
          new THREE.CylinderGeometry(fRadius * 0.68, fRadius * 0.76, dLen, 12),
          skinMaterial
        );
        distPhalanx.position.set(0, dLen / 2, 0);
        dipJoint.add(distPhalanx);

        // Fingertip rounded cap
        const tip = new THREE.Mesh(new THREE.SphereGeometry(fRadius * 0.76, 12, 12), skinMaterial);
        tip.position.set(0, dLen, 0);
        dipJoint.add(tip);

        // Keratin Fingernail on Dorsal Face (-Z)
        const nailGeo = new THREE.CylinderGeometry(fRadius * 0.62, fRadius * 0.55, dLen * 0.65, 12, 1, false, 0, Math.PI);
        const nail = new THREE.Mesh(nailGeo, nailMaterial);
        nail.rotation.y = Math.PI / 2;
        nail.position.set(0, dLen * 0.62, -fRadius * 0.48);
        dipJoint.add(nail);

        return [mcpJoint, pipJoint, dipJoint];
      };

      // Thumb attaches at thenar eminence on radial side (dir * 0.038, y = 0.024)
      const thumb = buildFingerPhalanges(dir * 0.038, 0.024, 0.006, 0.038, 0.030, 0, 0.0125, true);

      // 4 fingers attach along natural transverse knuckle arch
      const index = buildFingerPhalanges(dir * 0.026, 0.080, 0.0, 0.042, 0.028, 0.022, 0.0112);
      const middle = buildFingerPhalanges(dir * 0.008, 0.086, 0.0, 0.046, 0.032, 0.024, 0.0118);
      const ring = buildFingerPhalanges(-dir * 0.009, 0.082, 0.0, 0.042, 0.028, 0.022, 0.0110);
      const pinky = buildFingerPhalanges(-dir * 0.026, 0.072, 0.0, 0.032, 0.022, 0.018, 0.0098);

      return {
        shoulder,
        upperArm: upperArmMesh,
        elbow,
        forearm: forearmMesh,
        wrist,
        palm: palmGroup,
        fingers: { thumb, index, middle, ring, pinky },
      };
    };

    const rightArm = buildArm(true);
    const leftArm = buildArm(false);

    bodyRef.current = {
      head: headGroup,
      torso: torsoGroup,
      leftArm,
      rightArm,
      leftEye: lEye,
      rightEye: rEye,
    };

    // 7. Mouse Orbit Controls for 3D Viewport Interaction
    let isMouseDown = false;
    let prevMouseX = 0;
    let prevMouseY = 0;
    let targetRotY = 0;
    let targetRotX = 0;

    const handleMouseDown = (e: MouseEvent) => {
      isMouseDown = true;
      prevMouseX = e.clientX;
      prevMouseY = e.clientY;
    };

    const handleMouseMove = (e: MouseEvent) => {
      if (!isMouseDown) return;
      const dx = e.clientX - prevMouseX;
      const dy = e.clientY - prevMouseY;
      prevMouseX = e.clientX;
      prevMouseY = e.clientY;

      targetRotY += dx * 0.008;
      targetRotX = Math.max(-0.35, Math.min(0.35, targetRotX + dy * 0.008));
    };

    const handleMouseUp = () => {
      isMouseDown = false;
    };

    const dom = renderer.domElement;
    dom.addEventListener('mousedown', handleMouseDown);
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);

    // 8. Animation Loop
    let prevTimestamp = performance.now();
    let blinkTimer = 0;
    let telemetryTimer = 0;

    const animate = () => {
      reqIdRef.current = requestAnimationFrame(animate);

      const now = performance.now();
      const delta = Math.min((now - prevTimestamp) / 1000, 0.1);
      prevTimestamp = now;
      const time = now / 1000;

      // Smooth camera orbit damping
      humanRoot.rotation.y += (targetRotY - humanRoot.rotation.y) * 0.1;
      humanRoot.rotation.x += (targetRotX - humanRoot.rotation.x) * 0.1;

      // Natural breathing motion
      torsoGroup.position.y = Math.sin(time * 1.8) * 0.005;

      // Eye blinking
      blinkTimer += delta;
      if (blinkTimer > 3.2) {
        lEye.scale.y = 0.1;
        rEye.scale.y = 0.1;
        if (blinkTimer > 3.35) {
          lEye.scale.y = 1.0;
          rEye.scale.y = 1.0;
          blinkTimer = 0;
        }
      }

      // Dynamic Sign Language Movement: Continuously computes progress-driven gesture
      const cycleDuration = 2.4;
      const signProgress = (time % cycleDuration) / cycleDuration;
      targetPoseRef.current = getSignPose(currentSign, signProgress);

      // Smooth Damping to Target ISL Pose
      const cur = currentPoseRef.current;
      const tar = targetPoseRef.current;
      const lerpSpeed = delta * 9.5;

      const lerpVal = (c: number, t: number) => c + (t - c) * lerpSpeed;
      const lerpVec3 = (c: [number, number, number], t: [number, number, number]) => {
        c[0] = lerpVal(c[0], t[0]);
        c[1] = lerpVal(c[1], t[1]);
        c[2] = lerpVal(c[2], t[2]);
      };

      lerpVec3(cur.headRot, tar.headRot);
      lerpVec3(cur.rightArm.wristPos, tar.rightArm.wristPos);
      lerpVec3(cur.rightArm.wristRot, tar.rightArm.wristRot);
      lerpVec3(cur.leftArm.wristPos, tar.leftArm.wristPos);
      lerpVec3(cur.leftArm.wristRot, tar.leftArm.wristRot);

      // Interpolate all 5 fingers for both hands
      const lerpFinger = (cF: FingerPose, tF: FingerPose) => {
        cF.mcp = lerpVal(cF.mcp, tF.mcp);
        cF.pip = lerpVal(cF.pip, tF.pip);
        cF.dip = lerpVal(cF.dip, tF.dip);
        cF.spread = lerpVal(cF.spread, tF.spread);
      };

      lerpFinger(cur.rightArm.thumb, tar.rightArm.thumb);
      lerpFinger(cur.rightArm.index, tar.rightArm.index);
      lerpFinger(cur.rightArm.middle, tar.rightArm.middle);
      lerpFinger(cur.rightArm.ring, tar.rightArm.ring);
      lerpFinger(cur.rightArm.pinky, tar.rightArm.pinky);

      lerpFinger(cur.leftArm.thumb, tar.leftArm.thumb);
      lerpFinger(cur.leftArm.index, tar.leftArm.index);
      lerpFinger(cur.leftArm.middle, tar.leftArm.middle);
      lerpFinger(cur.leftArm.ring, tar.leftArm.ring);
      lerpFinger(cur.leftArm.pinky, tar.leftArm.pinky);

      // Apply to Head
      headGroup.rotation.set(cur.headRot[0], cur.headRot[1], cur.headRot[2]);

      // Dynamic Arm & Elbow Kinematic Solver
      const solveArmKinematics = (
        armRig: any,
        wristPos: [number, number, number],
        wristRot: [number, number, number],
        isRight: boolean
      ) => {
        const dir = isRight ? 1 : -1;
        const S = new THREE.Vector3(dir * 0.23, 0.18, 0);
        const W = new THREE.Vector3(wristPos[0], wristPos[1], wristPos[2]);

        // Place wrist
        armRig.wrist.position.copy(W);
        armRig.wrist.rotation.set(wristRot[0], wristRot[1], wristRot[2]);

        // Solve Elbow Position naturally flexing down and out
        const SW = new THREE.Vector3().subVectors(W, S);
        const d = Math.min(SW.length(), 0.44);
        const mid = new THREE.Vector3().addVectors(S, W).multiplyScalar(0.5);

        const naturalSag = new THREE.Vector3(dir * 0.35, -0.65, -0.2).normalize();
        const swUnit = SW.clone().normalize();
        const perp = naturalSag.clone().sub(swUnit.clone().multiplyScalar(naturalSag.dot(swUnit))).normalize();
        const bendSag = Math.sqrt(Math.max(0.001, 0.22 * 0.22 - (d * 0.5) * (d * 0.5)));

        const E = mid.clone().add(perp.multiplyScalar(bendSag));
        armRig.elbow.position.copy(E);

        // Orient Upper Arm from S to E
        const SE = new THREE.Vector3().subVectors(E, S);
        const upperLen = SE.length();
        armRig.upperArm.scale.set(1, upperLen, 1);
        armRig.upperArm.position.copy(new THREE.Vector3().addVectors(S, E).multiplyScalar(0.5));
        armRig.upperArm.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), SE.clone().normalize());

        // Orient Forearm from E to W
        const EW = new THREE.Vector3().subVectors(W, E);
        const foreLen = EW.length();
        armRig.forearm.scale.set(1, foreLen, 1);
        armRig.forearm.position.copy(new THREE.Vector3().addVectors(E, W).multiplyScalar(0.5));
        armRig.forearm.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), EW.clone().normalize());
      };

      const rRig = bodyRef.current!.rightArm;
      const lRig = bodyRef.current!.leftArm;

      solveArmKinematics(rRig, cur.rightArm.wristPos, cur.rightArm.wristRot, true);
      solveArmKinematics(lRig, cur.leftArm.wristPos, cur.leftArm.wristRot, false);

      // Apply 4-Finger Articulation (MCP, PIP, DIP)
      const applyFingerPhalanges = (joints: THREE.Group[], poseF: FingerPose) => {
        joints[0].rotation.set(poseF.mcp, 0, -poseF.spread);
        joints[1].rotation.x = poseF.pip;
        if (joints[2]) joints[2].rotation.x = poseF.dip;
      };

      // Right Thumb (dir = 1): Branches outward to +X in open pose, upright in thumbs-up, across in fist
      const rA = cur.rightArm;
      const rThumbSpread = rA.thumb.spread;
      const rThumbMCP = rA.thumb.mcp;
      if (rThumbMCP < 0.15 && rThumbSpread > 0.45) {
        // High vertical thumbs-up (straight upright)
        rRig.fingers.thumb[0].rotation.set(0.08, -0.10, -0.12);
        rRig.fingers.thumb[1].rotation.x = 0.02;
      } else if (rThumbMCP > 1.0) {
        // Fist / clenched: thumb wraps inward across front of fingers
        rRig.fingers.thumb[0].rotation.set(0.85, -0.40, 0.35);
        rRig.fingers.thumb[1].rotation.x = 0.75;
      } else {
        // Natural open / semi-curled thumb branching gracefully outward (+X)
        rRig.fingers.thumb[0].rotation.set(0.20 + rThumbMCP * 0.4, -0.15, -(0.42 + rThumbSpread * 0.45));
        rRig.fingers.thumb[1].rotation.x = rA.thumb.pip;
      }

      applyFingerPhalanges(rRig.fingers.index, rA.index);
      applyFingerPhalanges(rRig.fingers.middle, rA.middle);
      applyFingerPhalanges(rRig.fingers.ring, rA.ring);
      applyFingerPhalanges(rRig.fingers.pinky, rA.pinky);

      // Left Thumb (dir = -1): Branches outward to -X in open pose
      const lA = cur.leftArm;
      const lThumbSpread = lA.thumb.spread;
      const lThumbMCP = lA.thumb.mcp;
      if (lThumbMCP < 0.15 && Math.abs(lThumbSpread) > 0.45) {
        lRig.fingers.thumb[0].rotation.set(0.08, 0.10, 0.12);
        lRig.fingers.thumb[1].rotation.x = 0.02;
      } else if (lThumbMCP > 1.0) {
        lRig.fingers.thumb[0].rotation.set(0.85, 0.40, -0.35);
        lRig.fingers.thumb[1].rotation.x = 0.75;
      } else {
        // Natural open thumb branching gracefully outward (-X)
        lRig.fingers.thumb[0].rotation.set(0.20 + lThumbMCP * 0.4, 0.15, +(0.42 + Math.abs(lThumbSpread) * 0.45));
        lRig.fingers.thumb[1].rotation.x = lA.thumb.pip;
      }

      applyFingerPhalanges(lRig.fingers.index, lA.index);
      applyFingerPhalanges(lRig.fingers.middle, lA.middle);
      applyFingerPhalanges(lRig.fingers.ring, lA.ring);
      applyFingerPhalanges(lRig.fingers.pinky, lA.pinky);

      // Periodic Telemetry Update (5Hz)
      telemetryTimer += delta;
      if (telemetryTimer > 0.2) {
        telemetryTimer = 0;
        const radToDeg = (rad: number) => Math.round((rad * 180) / Math.PI);
        const getStatus = (deg: number) => {
          if (deg < 15) return 'Extended';
          if (deg < 45) return 'Slight Bend';
          if (deg < 75) return 'Hooked';
          return 'Curled';
        };

        const tDeg = radToDeg(rA.thumb.mcp + rA.thumb.pip);
        const iDeg = radToDeg(rA.index.mcp + rA.index.pip + rA.index.dip);
        const mDeg = radToDeg(rA.middle.mcp + rA.middle.pip + rA.middle.dip);
        const rDeg = radToDeg(rA.ring.mcp + rA.ring.pip + rA.ring.dip);
        const pDeg = radToDeg(rA.pinky.mcp + rA.pinky.pip + rA.pinky.dip);

        setFingerDegrees({
          thumb: { angle: tDeg, status: getStatus(tDeg / 2) },
          index: { angle: iDeg, status: getStatus(iDeg / 3) },
          middle: { angle: mDeg, status: getStatus(mDeg / 3) },
          ring: { angle: rDeg, status: getStatus(rDeg / 3) },
          pinky: { angle: pDeg, status: getStatus(pDeg / 3) },
        });
      }

      renderer.render(scene, camera);
    };

    animate();

    const ro = new ResizeObserver((entries) => {
      for (let entry of entries) {
        const { width: w, height: h } = entry.contentRect;
        if (w > 0 && h > 0) {
          camera.aspect = w / h;
          camera.updateProjectionMatrix();
          renderer.setSize(w, h);
        }
      }
    });
    ro.observe(mount);

    return () => {
      if (reqIdRef.current) cancelAnimationFrame(reqIdRef.current);
      ro.disconnect();
      dom.removeEventListener('mousedown', handleMouseDown);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
      if (mount && dom) mount.removeChild(dom);
      renderer.dispose();
    };
  }, [currentSign]);

  return (
    <div
      ref={mountRef}
      className={`humanoid-avatar-wrap relative w-full h-full overflow-hidden select-none ${className}`}
      style={{
        background: 'radial-gradient(circle at 50% 30%, #1e1b4b 0%, #0f172a 60%, #020617 100%)',
        minHeight: '440px',
      }}
    >
      {/* Sleek Camera Preset Controls (Top Right) */}
      <div
        style={{
          position: 'absolute',
          top: '12px',
          right: '12px',
          display: 'flex',
          gap: '6px',
          zIndex: 10,
        }}
      >
        <button
          type="button"
          onClick={() => applyCameraPreset('front')}
          style={{
            background: cameraMode === 'front' ? '#4f46e5' : 'rgba(15, 23, 42, 0.75)',
            color: '#ffffff',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '6px',
            padding: '5px 10px',
            fontSize: '11px',
            fontWeight: 600,
            cursor: 'pointer',
            backdropFilter: 'blur(8px)',
          }}
          title="Full Upper Body Signer View"
        >
          👤 Body
        </button>
        <button
          type="button"
          onClick={() => applyCameraPreset('hands')}
          style={{
            background: cameraMode === 'hands' ? '#4f46e5' : 'rgba(15, 23, 42, 0.75)',
            color: '#ffffff',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '6px',
            padding: '5px 10px',
            fontSize: '11px',
            fontWeight: 600,
            cursor: 'pointer',
            backdropFilter: 'blur(8px)',
          }}
          title="Macro Zoom to Hands & Fingers"
        >
          🤲 Hands Zoom
        </button>
        <button
          type="button"
          onClick={() => applyCameraPreset('side')}
          style={{
            background: cameraMode === 'side' ? '#4f46e5' : 'rgba(15, 23, 42, 0.75)',
            color: '#ffffff',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '6px',
            padding: '5px 10px',
            fontSize: '11px',
            fontWeight: 600,
            cursor: 'pointer',
            backdropFilter: 'blur(8px)',
          }}
          title="3D Depth Angle"
        >
          📐 Depth
        </button>
        <button
          type="button"
          onClick={() => setJointGlow(!jointGlow)}
          style={{
            background: jointGlow ? '#10b981' : 'rgba(15, 23, 42, 0.75)',
            color: '#ffffff',
            border: '1px solid rgba(16, 185, 129, 0.4)',
            borderRadius: '6px',
            padding: '5px 10px',
            fontSize: '11px',
            fontWeight: 600,
            cursor: 'pointer',
            backdropFilter: 'blur(8px)',
          }}
          title="Highlight Knuckle Joint Rings"
        >
          ✨ Joint Glow
        </button>
      </div>

      {/* Collapsible Telemetry Chip (Top Left, Non-Intrusive) */}
      <div
        style={{
          position: 'absolute',
          top: '12px',
          left: '12px',
          zIndex: 10,
        }}
      >
        <button
          type="button"
          onClick={() => setShowTelemetry(!showTelemetry)}
          style={{
            background: showTelemetry ? 'rgba(15, 23, 42, 0.92)' : 'rgba(15, 23, 42, 0.75)',
            color: '#38bdf8',
            border: '1px solid rgba(56, 189, 248, 0.35)',
            borderRadius: '6px',
            padding: '5px 10px',
            fontSize: '11px',
            fontWeight: 600,
            cursor: 'pointer',
            backdropFilter: 'blur(8px)',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
          }}
        >
          <Activity size={13} />
          <span>{showTelemetry ? 'Hide Telemetry' : 'Finger Telemetry'}</span>
        </button>

        {showTelemetry && (
          <div
            style={{
              marginTop: '6px',
              background: 'rgba(10, 15, 29, 0.94)',
              backdropFilter: 'blur(10px)',
              border: '1px solid rgba(56, 189, 248, 0.35)',
              borderRadius: '8px',
              padding: '8px 12px',
              width: '180px',
              display: 'flex',
              flexDirection: 'column',
              gap: '4px',
            }}
          >
            {[
              { label: 'Thumb', val: fingerDegrees.thumb },
              { label: 'Index', val: fingerDegrees.index },
              { label: 'Middle', val: fingerDegrees.middle },
              { label: 'Ring', val: fingerDegrees.ring },
              { label: 'Pinky', val: fingerDegrees.pinky },
            ].map(({ label, val }) => (
              <div key={label} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px' }}>
                <span style={{ color: '#cbd5e1', fontWeight: 600 }}>{label}:</span>
                <span style={{ color: val.angle < 30 ? '#34d399' : val.angle < 100 ? '#fbbf24' : '#f87171', fontWeight: 700 }}>
                  {val.status} ({val.angle}°)
                </span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Subtle Drag Hint */}
      <div
        style={{
          position: 'absolute',
          bottom: '10px',
          right: '12px',
          color: 'rgba(165, 180, 252, 0.6)',
          fontSize: '10px',
          display: 'flex',
          alignItems: 'center',
          gap: '4px',
          pointerEvents: 'none',
          zIndex: 5,
        }}
      >
        <Sparkles size={11} />
        <span>Drag to rotate 3D view</span>
      </div>
    </div>
  );
};
