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
  mcp: number; // Metacarpophalangeal flexion (0: straight, 1.57: 90 deg curl)
  pip: number; // Proximal interphalangeal flexion
  dip: number; // Distal interphalangeal flexion
  spread: number; // Abduction / spread angle (-0.4 to 0.4 rad)
}

// 5-finger hand state
export interface HandPoseState {
  wristRot: [number, number, number]; // pitch, yaw, roll
  wristPos: [number, number, number]; // x, y, z target position relative to chest
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

// Fully open finger
const OPEN_FINGER = f(0.04, 0.04, 0.04, 0.0);
// Fully closed fist finger
const FIST_FINGER = f(1.48, 1.55, 1.25, 0.0);
// Semi-curled finger (C or O shape)
const CURLED_FINGER = f(0.85, 0.95, 0.75, 0.0);
// Hook finger (X or claw)
const HOOK_FINGER = f(0.35, 1.20, 0.90, 0.0);

// Library of Authentic Indian Sign Language (ISL) Canonical 3D Poses
export function getSignPose(rawSign: string): BodyPoseState {
  const s = (rawSign || 'GOOD_MORNING').toUpperCase().replace(/\s+/g, '_');

  // Neutral rest pose (arms at resting chest ready position)
  const neutralLeft: HandPoseState = {
    wristRot: [0.1, -0.2, -0.1],
    wristPos: [-0.22, -0.14, 0.22],
    thumb: f(0.3, 0.4, 0.2, -0.3),
    index: f(0.4, 0.5, 0.3, -0.1),
    middle: f(0.45, 0.55, 0.3, 0.0),
    ring: f(0.45, 0.55, 0.3, 0.1),
    pinky: f(0.4, 0.5, 0.3, 0.2),
  };

  const neutralRight: HandPoseState = {
    wristRot: [0.1, 0.2, 0.1],
    wristPos: [0.22, -0.14, 0.22],
    thumb: f(0.3, 0.4, 0.2, 0.3),
    index: f(0.4, 0.5, 0.3, 0.1),
    middle: f(0.45, 0.55, 0.3, 0.0),
    ring: f(0.45, 0.55, 0.3, -0.1),
    pinky: f(0.4, 0.5, 0.3, -0.2),
  };

  const pose: BodyPoseState = {
    headRot: [0, 0, 0],
    leftArm: { ...neutralLeft },
    rightArm: { ...neutralRight },
    isTwoHanded: false,
  };

  // -------------------------------------------------------------
  // Conversational Phrases & Greetings
  // -------------------------------------------------------------
  if (s === 'GOOD_MORNING' || s === 'GOOD MORNING') {
    // Both hands sweep up from chest into open morning sun salute
    pose.isTwoHanded = true;
    pose.headRot = [-0.08, 0, 0];
    pose.rightArm = {
      wristRot: [-0.25, -0.1, 0.25],
      wristPos: [0.22, 0.24, 0.26],
      thumb: f(0.0, 0.0, 0.0, 0.4),
      index: f(0.0, 0.0, 0.0, 0.18),
      middle: f(0.0, 0.0, 0.0, 0.0),
      ring: f(0.0, 0.0, 0.0, -0.18),
      pinky: f(0.0, 0.0, 0.0, -0.32),
    };
    pose.leftArm = {
      wristRot: [-0.25, 0.1, -0.25],
      wristPos: [-0.22, 0.24, 0.26],
      thumb: f(0.0, 0.0, 0.0, -0.4),
      index: f(0.0, 0.0, 0.0, -0.18),
      middle: f(0.0, 0.0, 0.0, 0.0),
      ring: f(0.0, 0.0, 0.0, 0.18),
      pinky: f(0.0, 0.0, 0.0, 0.32),
    };
  } else if (s === 'HELLO' || s === 'HI' || s === 'WAVE') {
    // Right hand near temple in open friendly salute
    pose.headRot = [0.02, -0.05, 0.03];
    pose.rightArm = {
      wristRot: [-0.15, 0.15, 0.18],
      wristPos: [0.25, 0.32, 0.24],
      thumb: f(0.08, 0.05, 0.05, 0.35),
      index: f(0.02, 0.02, 0.02, 0.12),
      middle: f(0.02, 0.02, 0.02, 0.0),
      ring: f(0.02, 0.02, 0.02, -0.12),
      pinky: f(0.02, 0.02, 0.02, -0.24),
    };
  } else if (s === 'NAMASTE' || s === 'THANK_YOU' || s === 'THANK YOU' || s === 'THANKS') {
    // Both palms pressed flat together at center chest with respectful head bow
    pose.isTwoHanded = true;
    pose.headRot = [0.16, 0.0, 0.0];
    pose.rightArm = {
      wristRot: [0.15, -0.35, 0.2],
      wristPos: [0.042, 0.10, 0.26],
      thumb: f(0.1, 0.05, 0.05, 0.15),
      index: f(0.03, 0.03, 0.03, 0.04),
      middle: f(0.03, 0.03, 0.03, 0.0),
      ring: f(0.03, 0.03, 0.03, -0.04),
      pinky: f(0.03, 0.03, 0.03, -0.08),
    };
    pose.leftArm = {
      wristRot: [0.15, 0.35, -0.2],
      wristPos: [-0.042, 0.10, 0.26],
      thumb: f(0.1, 0.05, 0.05, -0.15),
      index: f(0.03, 0.03, 0.03, -0.04),
      middle: f(0.03, 0.03, 0.03, 0.0),
      ring: f(0.03, 0.03, 0.03, 0.04),
      pinky: f(0.03, 0.03, 0.03, 0.08),
    };
  } else if (s === 'HELP') {
    // ISL Two-Handed: Left palm flat facing up; right fist with thumb up placed on left palm
    pose.isTwoHanded = true;
    pose.leftArm = {
      wristRot: [0.35, 0.0, 0.0],
      wristPos: [-0.06, -0.02, 0.28],
      thumb: f(0.1, 0.1, 0.1, -0.3),
      index: f(0.04, 0.04, 0.04, -0.1),
      middle: f(0.04, 0.04, 0.04, 0.0),
      ring: f(0.04, 0.04, 0.04, 0.1),
      pinky: f(0.04, 0.04, 0.04, 0.2),
    };
    pose.rightArm = {
      wristRot: [0.1, -0.15, 0.0],
      wristPos: [0.06, 0.06, 0.28],
      thumb: f(0.0, 0.0, 0.0, 0.45),
      index: FIST_FINGER,
      middle: FIST_FINGER,
      ring: FIST_FINGER,
      pinky: FIST_FINGER,
    };
  } else if (s === 'PLEASE' || s === 'WELCOME') {
    // Both open hands flat palm welcoming sweep
    pose.isTwoHanded = true;
    pose.rightArm = {
      wristRot: [-0.1, -0.2, 0.1],
      wristPos: [0.18, 0.08, 0.26],
      thumb: f(0.1, 0.05, 0.05, 0.25),
      index: f(0.04, 0.04, 0.04, 0.1),
      middle: f(0.04, 0.04, 0.04, 0.0),
      ring: f(0.04, 0.04, 0.04, -0.1),
      pinky: f(0.04, 0.04, 0.04, -0.2),
    };
    pose.leftArm = {
      wristRot: [-0.1, 0.2, -0.1],
      wristPos: [-0.18, 0.08, 0.26],
      thumb: f(0.1, 0.05, 0.05, -0.25),
      index: f(0.04, 0.04, 0.04, -0.1),
      middle: f(0.04, 0.04, 0.04, 0.0),
      ring: f(0.04, 0.04, 0.04, 0.1),
      pinky: f(0.04, 0.04, 0.04, 0.2),
    };
  } else if (s === 'WATER' || s === 'DRINK') {
    // C-shaped cup brought to mouth
    pose.rightArm = {
      wristRot: [0.35, -0.2, 0.4],
      wristPos: [0.12, 0.26, 0.22],
      thumb: f(0.5, 0.6, 0.4, 0.35),
      index: f(0.7, 0.8, 0.6, 0.05),
      middle: f(0.75, 0.85, 0.65, 0.0),
      ring: f(0.8, 0.9, 0.7, -0.05),
      pinky: f(0.85, 0.95, 0.75, -0.1),
    };
  } else if (s === 'FOOD' || s === 'EAT') {
    // Bunched fingertips tapping near lips
    pose.rightArm = {
      wristRot: [0.45, -0.1, 0.2],
      wristPos: [0.08, 0.28, 0.20],
      thumb: f(0.8, 0.9, 0.6, 0.1),
      index: f(0.9, 1.1, 0.8, 0.0),
      middle: f(0.9, 1.1, 0.8, 0.0),
      ring: f(0.9, 1.1, 0.8, 0.0),
      pinky: f(0.9, 1.1, 0.8, 0.0),
    };
  } else if (s === 'YES') {
    // Fist nodding at shoulder height
    pose.headRot = [0.12, 0, 0];
    pose.rightArm = {
      wristRot: [0.35, 0.0, 0.1],
      wristPos: [0.20, 0.18, 0.26],
      thumb: f(0.6, 1.2, 1.0, 0.2),
      index: FIST_FINGER,
      middle: FIST_FINGER,
      ring: FIST_FINGER,
      pinky: FIST_FINGER,
    };
  } else if (s === 'NO') {
    // Index and middle extended, snapping down
    pose.headRot = [0, 0.12, 0];
    pose.rightArm = {
      wristRot: [0.1, 0.1, 0.0],
      wristPos: [0.20, 0.16, 0.28],
      thumb: f(0.6, 0.7, 0.5, 0.2),
      index: f(0.55, 0.75, 0.45, 0.05),
      middle: f(0.55, 0.75, 0.45, -0.05),
      ring: FIST_FINGER,
      pinky: FIST_FINGER,
    };
  } else if (s === 'YOU') {
    // Index pointing directly forward to conversational partner
    pose.rightArm = {
      wristRot: [-0.05, 0.0, 0.1],
      wristPos: [0.14, 0.10, 0.38],
      thumb: f(0.8, 1.1, 0.8, 0.2),
      index: f(0.0, 0.0, 0.0, 0.0),
      middle: FIST_FINGER,
      ring: FIST_FINGER,
      pinky: FIST_FINGER,
    };
  } else if (s === 'I' || s === 'ME') {
    // Index pointing gently to center chest
    pose.rightArm = {
      wristRot: [0.65, -0.3, 0.1],
      wristPos: [0.06, 0.08, 0.15],
      thumb: f(0.8, 1.0, 0.8, 0.2),
      index: f(0.0, 0.0, 0.0, 0.0),
      middle: FIST_FINGER,
      ring: FIST_FINGER,
      pinky: FIST_FINGER,
    };
  } else if (s === 'SORRY') {
    // Fist rubbing circular chest
    pose.headRot = [0.06, 0.0, 0.0];
    pose.rightArm = {
      wristRot: [0.4, -0.2, 0.1],
      wristPos: [0.05, 0.12, 0.18],
      thumb: f(0.6, 1.2, 1.0, 0.2),
      index: FIST_FINGER,
      middle: FIST_FINGER,
      ring: FIST_FINGER,
      pinky: FIST_FINGER,
    };
  } else if (s === 'STOP') {
    // Vertical right hand chopped down on horizontal left palm
    pose.isTwoHanded = true;
    pose.leftArm = {
      wristRot: [0.3, 0.0, 0.0],
      wristPos: [-0.05, -0.05, 0.26],
      thumb: f(0.1, 0.1, 0.1, -0.2),
      index: OPEN_FINGER,
      middle: OPEN_FINGER,
      ring: OPEN_FINGER,
      pinky: OPEN_FINGER,
    };
    pose.rightArm = {
      wristRot: [0.0, -1.3, 0.0],
      wristPos: [0.03, 0.01, 0.26],
      thumb: f(0.1, 0.1, 0.1, 0.1),
      index: OPEN_FINGER,
      middle: OPEN_FINGER,
      ring: OPEN_FINGER,
      pinky: OPEN_FINGER,
    };
  } else if (s === 'LOVE' || s === 'FRIEND') {
    // Both arms crossed gently over chest
    pose.isTwoHanded = true;
    pose.headRot = [0.04, 0.0, 0.0];
    pose.rightArm = {
      wristRot: [0.4, -0.3, 0.2],
      wristPos: [-0.08, 0.14, 0.20],
      thumb: f(0.1, 0.1, 0.1, 0.2),
      index: OPEN_FINGER,
      middle: OPEN_FINGER,
      ring: OPEN_FINGER,
      pinky: OPEN_FINGER,
    };
    pose.leftArm = {
      wristRot: [0.4, 0.3, -0.2],
      wristPos: [0.08, 0.12, 0.22],
      thumb: f(0.1, 0.1, 0.1, -0.2),
      index: OPEN_FINGER,
      middle: OPEN_FINGER,
      ring: OPEN_FINGER,
      pinky: OPEN_FINGER,
    };
  }
  // -------------------------------------------------------------
  // ISL Alphabet (A–Z) with authentic bimanual letters
  // -------------------------------------------------------------
  else if (s === 'A') {
    // Fist with thumb extended upright alongside index
    pose.rightArm.wristPos = [0.18, 0.14, 0.28];
    pose.rightArm.thumb = f(0.0, 0.0, 0.0, 0.45);
    pose.rightArm.index = FIST_FINGER;
    pose.rightArm.middle = FIST_FINGER;
    pose.rightArm.ring = FIST_FINGER;
    pose.rightArm.pinky = FIST_FINGER;
  } else if (s === 'B') {
    // ISL Two-Handed: Left palm flat open; right palm open touching left palm
    pose.isTwoHanded = true;
    pose.leftArm = {
      wristRot: [0.1, 0.35, -0.1],
      wristPos: [-0.06, 0.14, 0.25],
      thumb: f(0.1, 0.1, 0.1, -0.2),
      index: OPEN_FINGER,
      middle: OPEN_FINGER,
      ring: OPEN_FINGER,
      pinky: OPEN_FINGER,
    };
    pose.rightArm = {
      wristRot: [0.1, -0.35, 0.1],
      wristPos: [0.06, 0.14, 0.25],
      thumb: f(0.1, 0.1, 0.1, 0.2),
      index: OPEN_FINGER,
      middle: OPEN_FINGER,
      ring: OPEN_FINGER,
      pinky: OPEN_FINGER,
    };
  } else if (s === 'C') {
    // Curved hand forming a clear C arc facing forward
    pose.rightArm.wristPos = [0.18, 0.16, 0.28];
    pose.rightArm.thumb = f(0.5, 0.6, 0.4, 0.35);
    pose.rightArm.index = CURLED_FINGER;
    pose.rightArm.middle = CURLED_FINGER;
    pose.rightArm.ring = CURLED_FINGER;
    pose.rightArm.pinky = CURLED_FINGER;
  } else if (s === 'D') {
    // ISL Two-Handed: Left index upright; right index & thumb curved to meet left index tip & base
    pose.isTwoHanded = true;
    pose.leftArm = {
      wristRot: [0.0, 0.1, 0.0],
      wristPos: [-0.05, 0.18, 0.26],
      thumb: FIST_FINGER,
      index: f(0.0, 0.0, 0.0, 0.0),
      middle: FIST_FINGER,
      ring: FIST_FINGER,
      pinky: FIST_FINGER,
    };
    pose.rightArm = {
      wristRot: [0.1, -0.4, 0.2],
      wristPos: [0.04, 0.18, 0.26],
      thumb: f(0.4, 0.5, 0.3, 0.3),
      index: f(0.6, 0.8, 0.6, 0.0),
      middle: FIST_FINGER,
      ring: FIST_FINGER,
      pinky: FIST_FINGER,
    };
  } else if (s === 'E') {
    // All fingers curled with fingertips touching tucked thumb
    pose.rightArm.wristPos = [0.18, 0.14, 0.28];
    pose.rightArm.thumb = f(0.7, 0.8, 0.6, 0.1);
    pose.rightArm.index = f(1.1, 1.2, 0.9, 0.0);
    pose.rightArm.middle = f(1.1, 1.2, 0.9, 0.0);
    pose.rightArm.ring = f(1.1, 1.2, 0.9, 0.0);
    pose.rightArm.pinky = f(1.1, 1.2, 0.9, 0.0);
  } else if (s === 'F') {
    // Thumb and index touching in circle, middle/ring/pinky straight upright
    pose.rightArm.wristPos = [0.18, 0.18, 0.28];
    pose.rightArm.thumb = f(0.6, 0.7, 0.4, 0.2);
    pose.rightArm.index = f(0.8, 0.9, 0.6, 0.0);
    pose.rightArm.middle = OPEN_FINGER;
    pose.rightArm.ring = OPEN_FINGER;
    pose.rightArm.pinky = OPEN_FINGER;
  } else if (s === 'G') {
    // Index and thumb parallel pointing horizontally
    pose.rightArm.wristRot = [0.0, 0.5, 0.8];
    pose.rightArm.wristPos = [0.18, 0.16, 0.28];
    pose.rightArm.thumb = f(0.1, 0.0, 0.0, 0.25);
    pose.rightArm.index = f(0.0, 0.0, 0.0, 0.0);
    pose.rightArm.middle = FIST_FINGER;
    pose.rightArm.ring = FIST_FINGER;
    pose.rightArm.pinky = FIST_FINGER;
  } else if (s === 'H') {
    // Index and middle parallel pointing horizontally
    pose.rightArm.wristRot = [0.0, 0.5, 0.8];
    pose.rightArm.wristPos = [0.18, 0.16, 0.28];
    pose.rightArm.thumb = f(0.8, 1.0, 0.8, 0.1);
    pose.rightArm.index = f(0.0, 0.0, 0.0, 0.02);
    pose.rightArm.middle = f(0.0, 0.0, 0.0, -0.02);
    pose.rightArm.ring = FIST_FINGER;
    pose.rightArm.pinky = FIST_FINGER;
  } else if (s === 'I') {
    // Pinky upright, all other fingers closed
    pose.rightArm.wristPos = [0.18, 0.18, 0.28];
    pose.rightArm.thumb = f(0.9, 1.1, 0.8, 0.2);
    pose.rightArm.index = FIST_FINGER;
    pose.rightArm.middle = FIST_FINGER;
    pose.rightArm.ring = FIST_FINGER;
    pose.rightArm.pinky = f(0.0, 0.0, 0.0, 0.0);
  } else if (s === 'J') {
    // Pinky upright tracing J curve
    pose.rightArm.wristRot = [0.15, 0.3, -0.25];
    pose.rightArm.wristPos = [0.18, 0.18, 0.28];
    pose.rightArm.thumb = f(0.9, 1.1, 0.8, 0.2);
    pose.rightArm.index = FIST_FINGER;
    pose.rightArm.middle = FIST_FINGER;
    pose.rightArm.ring = FIST_FINGER;
    pose.rightArm.pinky = f(0.0, 0.0, 0.0, 0.0);
  } else if (s === 'K') {
    // Index upright, middle angled forward 45 deg, thumb between them
    pose.rightArm.wristPos = [0.18, 0.18, 0.28];
    pose.rightArm.thumb = f(0.2, 0.2, 0.1, 0.35);
    pose.rightArm.index = f(0.0, 0.0, 0.0, 0.0);
    pose.rightArm.middle = f(0.7, 0.2, 0.1, 0.0);
    pose.rightArm.ring = FIST_FINGER;
    pose.rightArm.pinky = FIST_FINGER;
  } else if (s === 'L') {
    // Index and thumb form sharp 90-degree L
    pose.rightArm.wristPos = [0.18, 0.18, 0.28];
    pose.rightArm.thumb = f(0.0, 0.0, 0.0, 0.55);
    pose.rightArm.index = f(0.0, 0.0, 0.0, 0.0);
    pose.rightArm.middle = FIST_FINGER;
    pose.rightArm.ring = FIST_FINGER;
    pose.rightArm.pinky = FIST_FINGER;
  } else if (s === 'M') {
    // Three fingers folded over thumb
    pose.rightArm.wristPos = [0.18, 0.14, 0.28];
    pose.rightArm.thumb = f(0.9, 1.0, 0.8, 0.0);
    pose.rightArm.index = f(1.3, 1.4, 0.9, 0.0);
    pose.rightArm.middle = f(1.3, 1.4, 0.9, 0.0);
    pose.rightArm.ring = f(1.3, 1.4, 0.9, 0.0);
    pose.rightArm.pinky = FIST_FINGER;
  } else if (s === 'N') {
    // Two fingers folded over thumb
    pose.rightArm.wristPos = [0.18, 0.14, 0.28];
    pose.rightArm.thumb = f(0.9, 1.0, 0.8, 0.0);
    pose.rightArm.index = f(1.3, 1.4, 0.9, 0.0);
    pose.rightArm.middle = f(1.3, 1.4, 0.9, 0.0);
    pose.rightArm.ring = FIST_FINGER;
    pose.rightArm.pinky = FIST_FINGER;
  } else if (s === 'O') {
    // All 5 fingers touching in closed O circle
    pose.rightArm.wristPos = [0.18, 0.16, 0.28];
    pose.rightArm.thumb = f(0.8, 0.8, 0.6, 0.2);
    pose.rightArm.index = f(0.9, 0.9, 0.8, 0.0);
    pose.rightArm.middle = f(0.9, 0.9, 0.8, 0.0);
    pose.rightArm.ring = f(0.9, 0.9, 0.8, 0.0);
    pose.rightArm.pinky = f(0.9, 0.9, 0.8, 0.0);
  } else if (s === 'P') {
    // ISL Two-Handed: Left index upright; right index & thumb form upper circular loop
    pose.isTwoHanded = true;
    pose.leftArm = {
      wristRot: [0.0, 0.1, 0.0],
      wristPos: [-0.05, 0.16, 0.26],
      thumb: FIST_FINGER,
      index: f(0.0, 0.0, 0.0, 0.0),
      middle: FIST_FINGER,
      ring: FIST_FINGER,
      pinky: FIST_FINGER,
    };
    pose.rightArm = {
      wristRot: [0.1, -0.3, 0.2],
      wristPos: [0.03, 0.22, 0.26],
      thumb: f(0.5, 0.6, 0.4, 0.3),
      index: f(0.8, 0.9, 0.6, 0.0),
      middle: FIST_FINGER,
      ring: FIST_FINGER,
      pinky: FIST_FINGER,
    };
  } else if (s === 'Q') {
    // ISL Two-Handed: Left hand forms O circle, right index rests inside as Q tail
    pose.isTwoHanded = true;
    pose.leftArm = {
      wristRot: [0.1, 0.2, 0.0],
      wristPos: [-0.05, 0.14, 0.26],
      thumb: f(0.8, 0.8, 0.6, 0.2),
      index: f(0.9, 0.9, 0.8, 0.0),
      middle: f(0.9, 0.9, 0.8, 0.0),
      ring: f(0.9, 0.9, 0.8, 0.0),
      pinky: f(0.9, 0.9, 0.8, 0.0),
    };
    pose.rightArm = {
      wristRot: [0.2, -0.2, 0.3],
      wristPos: [0.02, 0.10, 0.26],
      thumb: FIST_FINGER,
      index: f(0.0, 0.0, 0.0, 0.0),
      middle: FIST_FINGER,
      ring: FIST_FINGER,
      pinky: FIST_FINGER,
    };
  } else if (s === 'R') {
    // Index and middle fingers crossed
    pose.rightArm.wristPos = [0.18, 0.18, 0.28];
    pose.rightArm.thumb = f(0.8, 1.0, 0.8, 0.2);
    pose.rightArm.index = f(0.0, 0.0, 0.0, -0.08);
    pose.rightArm.middle = f(0.0, 0.0, 0.0, 0.08);
    pose.rightArm.ring = FIST_FINGER;
    pose.rightArm.pinky = FIST_FINGER;
  } else if (s === 'S') {
    // Tight fist with thumb firmly crossed across front of fingers
    pose.rightArm.wristPos = [0.18, 0.14, 0.28];
    pose.rightArm.thumb = f(0.9, 1.1, 0.8, -0.15);
    pose.rightArm.index = FIST_FINGER;
    pose.rightArm.middle = FIST_FINGER;
    pose.rightArm.ring = FIST_FINGER;
    pose.rightArm.pinky = FIST_FINGER;
  } else if (s === 'T') {
    // ISL Two-Handed: Left index vertical stem; right index horizontal crossbar
    pose.isTwoHanded = true;
    pose.leftArm = {
      wristRot: [0.0, 0.1, 0.0],
      wristPos: [-0.05, 0.14, 0.26],
      thumb: FIST_FINGER,
      index: f(0.0, 0.0, 0.0, 0.0),
      middle: FIST_FINGER,
      ring: FIST_FINGER,
      pinky: FIST_FINGER,
    };
    pose.rightArm = {
      wristRot: [0.0, 0.0, 1.57],
      wristPos: [0.02, 0.22, 0.26],
      thumb: FIST_FINGER,
      index: f(0.0, 0.0, 0.0, 0.0),
      middle: FIST_FINGER,
      ring: FIST_FINGER,
      pinky: FIST_FINGER,
    };
  } else if (s === 'U') {
    // Index and middle upright touching together
    pose.rightArm.wristPos = [0.18, 0.18, 0.28];
    pose.rightArm.thumb = f(0.8, 1.0, 0.8, 0.2);
    pose.rightArm.index = f(0.0, 0.0, 0.0, 0.01);
    pose.rightArm.middle = f(0.0, 0.0, 0.0, -0.01);
    pose.rightArm.ring = FIST_FINGER;
    pose.rightArm.pinky = FIST_FINGER;
  } else if (s === 'V') {
    // Index and middle extended wide in V peace sign
    pose.rightArm.wristPos = [0.18, 0.18, 0.28];
    pose.rightArm.thumb = f(0.8, 1.0, 0.8, 0.2);
    pose.rightArm.index = f(0.0, 0.0, 0.0, 0.22);
    pose.rightArm.middle = f(0.0, 0.0, 0.0, -0.22);
    pose.rightArm.ring = FIST_FINGER;
    pose.rightArm.pinky = FIST_FINGER;
  } else if (s === 'W') {
    // Index, middle, ring upright in W
    pose.rightArm.wristPos = [0.18, 0.18, 0.28];
    pose.rightArm.thumb = f(0.8, 1.0, 0.8, 0.2);
    pose.rightArm.index = f(0.0, 0.0, 0.0, 0.25);
    pose.rightArm.middle = f(0.0, 0.0, 0.0, 0.0);
    pose.rightArm.ring = f(0.0, 0.0, 0.0, -0.25);
    pose.rightArm.pinky = FIST_FINGER;
  } else if (s === 'X') {
    // Hooked index finger (visible sharp PIP/DIP bend)
    pose.rightArm.wristPos = [0.18, 0.16, 0.28];
    pose.rightArm.thumb = f(0.7, 0.9, 0.7, 0.2);
    pose.rightArm.index = HOOK_FINGER;
    pose.rightArm.middle = FIST_FINGER;
    pose.rightArm.ring = FIST_FINGER;
    pose.rightArm.pinky = FIST_FINGER;
  } else if (s === 'Y') {
    // Thumb and pinky extended wide, middle 3 closed
    pose.rightArm.wristPos = [0.18, 0.16, 0.28];
    pose.rightArm.thumb = f(0.0, 0.0, 0.0, 0.5);
    pose.rightArm.index = FIST_FINGER;
    pose.rightArm.middle = FIST_FINGER;
    pose.rightArm.ring = FIST_FINGER;
    pose.rightArm.pinky = f(0.0, 0.0, 0.0, -0.4);
  } else if (s === 'Z') {
    // Index finger pointing and tracing Z in air
    pose.rightArm.wristRot = [0.1, 0.2, 0.1];
    pose.rightArm.wristPos = [0.18, 0.16, 0.28];
    pose.rightArm.thumb = f(0.8, 1.0, 0.8, 0.2);
    pose.rightArm.index = f(0.0, 0.0, 0.0, 0.0);
    pose.rightArm.middle = FIST_FINGER;
    pose.rightArm.ring = FIST_FINGER;
    pose.rightArm.pinky = FIST_FINGER;
  }
  // -------------------------------------------------------------
  // Digits (0–9)
  // -------------------------------------------------------------
  else if (s === '0') {
    pose.rightArm.wristPos = [0.18, 0.16, 0.28];
    pose.rightArm.thumb = f(0.8, 0.8, 0.6, 0.2);
    pose.rightArm.index = f(0.9, 0.9, 0.8, 0.0);
    pose.rightArm.middle = f(0.9, 0.9, 0.8, 0.0);
    pose.rightArm.ring = f(0.9, 0.9, 0.8, 0.0);
    pose.rightArm.pinky = f(0.9, 0.9, 0.8, 0.0);
  } else if (s === '1') {
    pose.rightArm.wristPos = [0.18, 0.18, 0.28];
    pose.rightArm.thumb = FIST_FINGER;
    pose.rightArm.index = f(0.0, 0.0, 0.0, 0.0);
    pose.rightArm.middle = FIST_FINGER;
    pose.rightArm.ring = FIST_FINGER;
    pose.rightArm.pinky = FIST_FINGER;
  } else if (s === '2') {
    pose.rightArm.wristPos = [0.18, 0.18, 0.28];
    pose.rightArm.thumb = FIST_FINGER;
    pose.rightArm.index = f(0.0, 0.0, 0.0, 0.15);
    pose.rightArm.middle = f(0.0, 0.0, 0.0, -0.15);
    pose.rightArm.ring = FIST_FINGER;
    pose.rightArm.pinky = FIST_FINGER;
  } else if (s === '3') {
    pose.rightArm.wristPos = [0.18, 0.18, 0.28];
    pose.rightArm.thumb = FIST_FINGER;
    pose.rightArm.index = f(0.0, 0.0, 0.0, 0.2);
    pose.rightArm.middle = f(0.0, 0.0, 0.0, 0.0);
    pose.rightArm.ring = f(0.0, 0.0, 0.0, -0.2);
    pose.rightArm.pinky = FIST_FINGER;
  } else if (s === '4') {
    pose.rightArm.wristPos = [0.18, 0.18, 0.28];
    pose.rightArm.thumb = FIST_FINGER;
    pose.rightArm.index = f(0.0, 0.0, 0.0, 0.25);
    pose.rightArm.middle = f(0.0, 0.0, 0.0, 0.08);
    pose.rightArm.ring = f(0.0, 0.0, 0.0, -0.08);
    pose.rightArm.pinky = f(0.0, 0.0, 0.0, -0.25);
  } else if (s === '5') {
    pose.rightArm.wristPos = [0.18, 0.18, 0.28];
    pose.rightArm.thumb = f(0.0, 0.0, 0.0, 0.45);
    pose.rightArm.index = f(0.0, 0.0, 0.0, 0.25);
    pose.rightArm.middle = f(0.0, 0.0, 0.0, 0.08);
    pose.rightArm.ring = f(0.0, 0.0, 0.0, -0.08);
    pose.rightArm.pinky = f(0.0, 0.0, 0.0, -0.25);
  } else if (s === '6') {
    // Thumb and pinky touching, 3 middle extended
    pose.rightArm.wristPos = [0.18, 0.18, 0.28];
    pose.rightArm.thumb = f(0.6, 0.7, 0.4, 0.15);
    pose.rightArm.index = OPEN_FINGER;
    pose.rightArm.middle = OPEN_FINGER;
    pose.rightArm.ring = OPEN_FINGER;
    pose.rightArm.pinky = f(0.8, 0.9, 0.7, 0.0);
  } else if (s === '7') {
    // Thumb and ring touching, others extended
    pose.rightArm.wristPos = [0.18, 0.18, 0.28];
    pose.rightArm.thumb = f(0.6, 0.7, 0.4, 0.15);
    pose.rightArm.index = OPEN_FINGER;
    pose.rightArm.middle = OPEN_FINGER;
    pose.rightArm.ring = f(0.8, 0.9, 0.7, 0.0);
    pose.rightArm.pinky = OPEN_FINGER;
  } else if (s === '8') {
    // Thumb and middle touching, others extended
    pose.rightArm.wristPos = [0.18, 0.18, 0.28];
    pose.rightArm.thumb = f(0.6, 0.7, 0.4, 0.15);
    pose.rightArm.index = OPEN_FINGER;
    pose.rightArm.middle = f(0.8, 0.9, 0.7, 0.0);
    pose.rightArm.ring = OPEN_FINGER;
    pose.rightArm.pinky = OPEN_FINGER;
  } else if (s === '9') {
    // Thumb and index touching in circle, middle/ring/pinky extended
    pose.rightArm.wristPos = [0.18, 0.18, 0.28];
    pose.rightArm.thumb = f(0.6, 0.7, 0.4, 0.15);
    pose.rightArm.index = f(0.8, 0.9, 0.7, 0.0);
    pose.rightArm.middle = OPEN_FINGER;
    pose.rightArm.ring = OPEN_FINGER;
    pose.rightArm.pinky = OPEN_FINGER;
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
  const [cameraMode, setCameraMode] = useState<'front' | 'hands' | 'rightHand' | 'leftHand' | 'side'>('front');
  const [showJointHUD, setShowJointHUD] = useState<boolean>(true);
  const [highlightJoints, setHighlightJoints] = useState<boolean>(false);

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

  // References for scene & animation loop
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);

  // Joint indicator ring meshes for toggleable high-contrast visualization
  const jointMarkersRef = useRef<THREE.Mesh[]>([]);

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
      palm: THREE.Mesh;
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
      palm: THREE.Mesh;
      fingers: {
        thumb: THREE.Group[];
        index: THREE.Group[];
        middle: THREE.Group[];
        ring: THREE.Group[];
        pinky: THREE.Group[];
      };
    };
    leftEye: THREE.Mesh;
    rightEye: THREE.Mesh;
    mouth: THREE.Mesh;
  } | null>(null);

  // Smooth interpolation state
  const currentPoseRef = useRef<BodyPoseState>(getSignPose('GOOD_MORNING'));
  const targetPoseRef = useRef<BodyPoseState>(getSignPose(currentSign));

  // Update target pose when currentSign changes
  useEffect(() => {
    targetPoseRef.current = getSignPose(currentSign);
  }, [currentSign]);

  // Camera presets
  const applyCameraPreset = useCallback((preset: 'front' | 'hands' | 'rightHand' | 'leftHand' | 'side') => {
    setCameraMode(preset);
    const camera = cameraRef.current;
    if (!camera) return;

    if (preset === 'front') {
      camera.position.set(0, 0.16, 1.08);
      camera.lookAt(0, 0.12, 0.12);
    } else if (preset === 'hands') {
      camera.position.set(0, 0.14, 0.60);
      camera.lookAt(0, 0.12, 0.24);
    } else if (preset === 'rightHand') {
      // Direct macro focus on Right Hand - every joint and nail fills the screen
      camera.position.set(0.18, 0.16, 0.44);
      camera.lookAt(0.18, 0.14, 0.24);
    } else if (preset === 'leftHand') {
      // Direct macro focus on Left Hand
      camera.position.set(-0.18, 0.16, 0.44);
      camera.lookAt(-0.18, 0.14, 0.24);
    } else if (preset === 'side') {
      camera.position.set(0.75, 0.22, 0.85);
      camera.lookAt(0, 0.14, 0.15);
    }
  }, []);

  // Toggle highlight on joint articulation rings
  useEffect(() => {
    jointMarkersRef.current.forEach((mesh) => {
      mesh.visible = highlightJoints;
    });
  }, [highlightJoints]);

  // Three.js Scene Setup & Render Loop
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    // Dimensions
    const width = mount.clientWidth || 600;
    const height = mount.clientHeight || 500;

    // 1. Scene
    const scene = new THREE.Scene();
    sceneRef.current = scene;
    scene.background = new THREE.Color(0x0a0f1d); // Deep slate high-contrast studio backdrop

    // 2. Camera with crisp focal framing
    const camera = new THREE.PerspectiveCamera(40, width / height, 0.05, 100);
    camera.position.set(0, 0.16, 1.08);
    camera.lookAt(0, 0.12, 0.12);
    cameraRef.current = camera;

    // 3. Renderer with ACESFilmic tone mapping
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.18;
    mount.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // 4. Calibrated 5-Point Studio Lighting (High Contrast for Knuckle Creases)
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.55);
    scene.add(ambientLight);

    // Warm Directional Key Light (Sculpts cylinder phalanges and creates distinct shadows)
    const keyLight = new THREE.DirectionalLight(0xfff5ea, 1.85);
    keyLight.position.set(1.4, 2.0, 1.8);
    scene.add(keyLight);

    // Soft Cyan Fill Light
    const fillLight = new THREE.DirectionalLight(0xa5f3fc, 0.75);
    fillLight.position.set(-1.4, 1.0, 1.4);
    scene.add(fillLight);

    // High-Contrast Indigo Rim Light (Accentuates finger silhouettes)
    const rimLight = new THREE.DirectionalLight(0x818cf8, 2.4);
    rimLight.position.set(0, 2.2, -1.8);
    scene.add(rimLight);

    // Dedicated Hand Macro Spotlight (Illuminates every knuckle crease and nail)
    const handSpotlight = new THREE.PointLight(0xffffff, 1.35, 2.2);
    handSpotlight.position.set(0, 0.22, 0.55);
    scene.add(handSpotlight);

    // 5. High-Definition Anatomical Materials
    const skinMaterial = new THREE.MeshStandardMaterial({
      color: 0xd58c5a, // Warm natural Indian skin tone
      roughness: 0.42,
      metalness: 0.03,
    });

    const palmMaterial = new THREE.MeshStandardMaterial({
      color: 0xf3be9a, // Palmar surface - softer and slightly lighter
      roughness: 0.50,
      metalness: 0.02,
    });

    const knuckleMaterial = new THREE.MeshStandardMaterial({
      color: 0xb86c3d, // Anatomical knuckle cartilage prominence
      roughness: 0.35,
      metalness: 0.04,
    });

    const creaseMaterial = new THREE.MeshStandardMaterial({
      color: 0x783515, // Deep joint flexion crease ring
      roughness: 0.80,
    });

    const nailMaterial = new THREE.MeshStandardMaterial({
      color: 0xffeedd, // High-specular keratin fingernail plate
      roughness: 0.16,
      metalness: 0.08,
    });

    const jointRingMaterial = new THREE.MeshStandardMaterial({
      color: 0x38bdf8, // Luminous cyan joint articulation indicator
      emissive: 0x0284c7,
      emissiveIntensity: 0.8,
      roughness: 0.2,
    });

    const shirtMaterial = new THREE.MeshStandardMaterial({
      color: 0x1e1b4b, // Deep indigo tailored suit
      roughness: 0.72,
      metalness: 0.10,
    });

    const accentMaterial = new THREE.MeshStandardMaterial({
      color: 0x6366f1, // Neon cyan/indigo collar trim
      roughness: 0.25,
      metalness: 0.4,
    });

    const hairMaterial = new THREE.MeshStandardMaterial({
      color: 0x111827, // Dark natural hair
      roughness: 0.9,
    });

    const eyeWhiteMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff });
    const irisMaterial = new THREE.MeshBasicMaterial({ color: 0x271911 });
    const lipMaterial = new THREE.MeshStandardMaterial({ color: 0xba6a5d, roughness: 0.4 });

    // 6. Build Humanoid Rig
    const humanRoot = new THREE.Group();
    scene.add(humanRoot);

    // Torso / Chest
    const torsoGroup = new THREE.Group();
    humanRoot.add(torsoGroup);

    const chestGeo = new THREE.CylinderGeometry(0.24, 0.18, 0.48, 22);
    const chestMesh = new THREE.Mesh(chestGeo, shirtMaterial);
    chestMesh.scale.set(1.0, 1.0, 0.65);
    torsoGroup.add(chestMesh);

    // Collar
    const collarGeo = new THREE.TorusGeometry(0.09, 0.015, 12, 24);
    const collarMesh = new THREE.Mesh(collarGeo, accentMaterial);
    collarMesh.rotation.x = Math.PI / 2;
    collarMesh.position.set(0, 0.23, 0);
    torsoGroup.add(collarMesh);

    // Neck
    const neckGeo = new THREE.CylinderGeometry(0.07, 0.08, 0.12, 16);
    const neckMesh = new THREE.Mesh(neckGeo, skinMaterial);
    neckMesh.position.set(0, 0.28, 0);
    torsoGroup.add(neckMesh);

    // Head
    const headGroup = new THREE.Group();
    headGroup.position.set(0, 0.42, 0.02);
    torsoGroup.add(headGroup);

    // Skull
    const headGeo = new THREE.SphereGeometry(0.12, 24, 24);
    const headMesh = new THREE.Mesh(headGeo, skinMaterial);
    headMesh.scale.set(0.95, 1.2, 1.0);
    headGroup.add(headMesh);

    // Hair
    const hairGeo = new THREE.SphereGeometry(0.125, 20, 20, 0, Math.PI * 2, 0, Math.PI * 0.55);
    const hairMesh = new THREE.Mesh(hairGeo, hairMaterial);
    hairMesh.scale.set(0.98, 1.25, 1.05);
    hairMesh.position.set(0, 0.02, -0.01);
    headGroup.add(hairMesh);

    // 3D Eyes
    const createEye = (isRight: boolean) => {
      const eyeG = new THREE.Group();
      const whiteG = new THREE.SphereGeometry(0.018, 14, 14);
      const whiteMesh = new THREE.Mesh(whiteG, eyeWhiteMaterial);
      eyeG.add(whiteMesh);

      const irisG = new THREE.SphereGeometry(0.01, 12, 12);
      const irisMesh = new THREE.Mesh(irisG, irisMaterial);
      irisMesh.position.set(0, 0, 0.012);
      eyeG.add(irisMesh);

      eyeG.position.set(isRight ? 0.042 : -0.042, 0.03, 0.105);
      return eyeG;
    };
    const rEye = createEye(true);
    const lEye = createEye(false);
    headGroup.add(rEye);
    headGroup.add(lEye);

    // Nose
    const noseGeo = new THREE.ConeGeometry(0.018, 0.045, 12);
    const noseMesh = new THREE.Mesh(noseGeo, skinMaterial);
    noseMesh.rotation.x = -Math.PI / 8;
    noseMesh.position.set(0, 0.005, 0.125);
    headGroup.add(noseMesh);

    // Mouth / Lips
    const mouthGeo = new THREE.TorusGeometry(0.025, 0.007, 10, 16, Math.PI);
    const mouthMesh = new THREE.Mesh(mouthGeo, lipMaterial);
    mouthMesh.rotation.z = Math.PI;
    mouthMesh.position.set(0, -0.045, 0.108);
    headGroup.add(mouthMesh);

    // ---------------------------------------------------------
    // Anatomical 5-Finger Articulated Hand & Arm Kinematics Builder
    // ---------------------------------------------------------
    jointMarkersRef.current = [];

    const buildArm = (isRight: boolean) => {
      const dir = isRight ? 1 : -1;

      // Shoulder joint
      const shoulder = new THREE.Group();
      shoulder.position.set(dir * 0.23, 0.20, 0);
      torsoGroup.add(shoulder);

      // Shoulder Cap Mesh
      const shoulderCap = new THREE.Mesh(new THREE.SphereGeometry(0.052, 14, 14), shirtMaterial);
      shoulder.add(shoulderCap);

      // Upper Arm Mesh (Oriented dynamically from Shoulder to Elbow)
      const upperArmMesh = new THREE.Mesh(
        new THREE.CylinderGeometry(0.044, 0.038, 1.0, 14),
        shirtMaterial
      );
      torsoGroup.add(upperArmMesh);

      // Elbow joint
      const elbow = new THREE.Group();
      torsoGroup.add(elbow);

      const elbowCap = new THREE.Mesh(new THREE.SphereGeometry(0.040, 12, 12), skinMaterial);
      elbow.add(elbowCap);

      // Forearm Mesh (Oriented dynamically from Elbow to Wrist)
      const forearmMesh = new THREE.Mesh(
        new THREE.CylinderGeometry(0.036, 0.030, 1.0, 14),
        skinMaterial
      );
      torsoGroup.add(forearmMesh);

      // Wrist Group (Independent dynamic anchor in torso space)
      const wrist = new THREE.Group();
      torsoGroup.add(wrist);

      const wristJointSphere = new THREE.Mesh(new THREE.SphereGeometry(0.032, 12, 12), skinMaterial);
      wrist.add(wristJointSphere);

      // Solid Volumetric 3D Palm (Scaled by 1.25x for clear sign language visibility)
      const palmGeo = new THREE.BoxGeometry(0.096, 0.108, 0.032);
      const palm = new THREE.Mesh(palmGeo, palmMaterial);
      palm.position.set(0, -0.054, 0);
      wrist.add(palm);

      // Palmar crease texture band
      const palmarCrease = new THREE.Mesh(
        new THREE.PlaneGeometry(0.07, 0.02),
        creaseMaterial
      );
      palmarCrease.position.set(0, 0.01, 0.017);
      palm.add(palmarCrease);

      // Build 5 Articulated 3D Phalange Chains with Knuckles & Keratin Nails
      const buildFingerPhalanges = (
        baseX: number,
        baseY: number,
        pLen: number,
        iLen: number,
        dLen: number,
        fRadius: number,
        isThumb: boolean = false
      ): THREE.Group[] => {
        // Joint 1: MCP Knuckle
        const mcpJoint = new THREE.Group();
        mcpJoint.position.set(baseX, baseY, 0);
        palm.add(mcpJoint);

        // Knuckle Prominence Sphere
        const knuckle1 = new THREE.Mesh(new THREE.SphereGeometry(fRadius * 1.22, 12, 12), knuckleMaterial);
        mcpJoint.add(knuckle1);

        // Dorsal Knuckle Crease Line
        const crease1 = new THREE.Mesh(new THREE.TorusGeometry(fRadius * 1.18, 0.0018, 8, 16), creaseMaterial);
        crease1.rotation.x = Math.PI / 2;
        mcpJoint.add(crease1);

        // High-Contrast Joint Marker (Visible when toggled)
        const marker1 = new THREE.Mesh(new THREE.TorusGeometry(fRadius * 1.35, 0.0025, 8, 16), jointRingMaterial);
        marker1.rotation.x = Math.PI / 2;
        marker1.visible = false;
        mcpJoint.add(marker1);
        jointMarkersRef.current.push(marker1);

        // Proximal Phalanx (Tapered cylinder)
        const proxPhalanx = new THREE.Mesh(
          new THREE.CylinderGeometry(fRadius * 0.98, fRadius * 0.90, pLen, 12),
          skinMaterial
        );
        proxPhalanx.position.set(0, -pLen / 2, 0);
        mcpJoint.add(proxPhalanx);

        // Joint 2: PIP Knuckle
        const pipJoint = new THREE.Group();
        pipJoint.position.set(0, -pLen, 0);
        mcpJoint.add(pipJoint);

        const knuckle2 = new THREE.Mesh(new THREE.SphereGeometry(fRadius * 1.10, 12, 12), knuckleMaterial);
        pipJoint.add(knuckle2);

        const crease2 = new THREE.Mesh(new THREE.TorusGeometry(fRadius * 1.08, 0.0016, 8, 16), creaseMaterial);
        crease2.rotation.x = Math.PI / 2;
        pipJoint.add(crease2);

        const marker2 = new THREE.Mesh(new THREE.TorusGeometry(fRadius * 1.25, 0.0022, 8, 16), jointRingMaterial);
        marker2.rotation.x = Math.PI / 2;
        marker2.visible = false;
        pipJoint.add(marker2);
        jointMarkersRef.current.push(marker2);

        // Intermediate Phalanx (or Distal for Thumb)
        const interPhalanx = new THREE.Mesh(
          new THREE.CylinderGeometry(fRadius * 0.88, fRadius * 0.80, iLen, 12),
          skinMaterial
        );
        interPhalanx.position.set(0, -iLen / 2, 0);
        pipJoint.add(interPhalanx);

        if (isThumb) {
          // Thumb Distal Tip & Keratin Nail Plate
          const thumbTip = new THREE.Mesh(new THREE.SphereGeometry(fRadius * 0.95, 12, 12), skinMaterial);
          thumbTip.position.set(0, -iLen, 0);
          pipJoint.add(thumbTip);

          // Thumb Nail on Dorsal Face (-Z)
          const thumbNailGeo = new THREE.CylinderGeometry(fRadius * 0.72, fRadius * 0.65, iLen * 0.6, 12, 1, false, 0, Math.PI);
          const thumbNail = new THREE.Mesh(thumbNailGeo, nailMaterial);
          thumbNail.rotation.y = Math.PI / 2;
          thumbNail.position.set(0, -iLen * 0.65, -fRadius * 0.55);
          pipJoint.add(thumbNail);

          return [mcpJoint, pipJoint];
        }

        // Joint 3: DIP Knuckle
        const dipJoint = new THREE.Group();
        dipJoint.position.set(0, -iLen, 0);
        pipJoint.add(dipJoint);

        const knuckle3 = new THREE.Mesh(new THREE.SphereGeometry(fRadius * 0.98, 12, 12), knuckleMaterial);
        dipJoint.add(knuckle3);

        const crease3 = new THREE.Mesh(new THREE.TorusGeometry(fRadius * 0.95, 0.0015, 8, 16), creaseMaterial);
        crease3.rotation.x = Math.PI / 2;
        dipJoint.add(crease3);

        const marker3 = new THREE.Mesh(new THREE.TorusGeometry(fRadius * 1.15, 0.002, 8, 16), jointRingMaterial);
        marker3.rotation.x = Math.PI / 2;
        marker3.visible = false;
        dipJoint.add(marker3);
        jointMarkersRef.current.push(marker3);

        // Distal Phalanx
        const distPhalanx = new THREE.Mesh(
          new THREE.CylinderGeometry(fRadius * 0.78, fRadius * 0.68, dLen, 12),
          skinMaterial
        );
        distPhalanx.position.set(0, -dLen / 2, 0);
        dipJoint.add(distPhalanx);

        // Rounded Fingertip Pad
        const tip = new THREE.Mesh(new THREE.SphereGeometry(fRadius * 0.85, 12, 12), skinMaterial);
        tip.position.set(0, -dLen, 0);
        dipJoint.add(tip);

        // Glossy Keratin Fingernail on Dorsal Face (-Z)
        const nailGeo = new THREE.CylinderGeometry(fRadius * 0.68, fRadius * 0.60, dLen * 0.65, 12, 1, false, 0, Math.PI);
        const nail = new THREE.Mesh(nailGeo, nailMaterial);
        nail.rotation.y = Math.PI / 2;
        nail.position.set(0, -dLen * 0.62, -fRadius * 0.52);
        dipJoint.add(nail);

        return [mcpJoint, pipJoint, dipJoint];
      };

      // Create 5 fingers with anatomical positions across palm (Enlarged by 1.25x for clarity)
      const thumb = buildFingerPhalanges(dir * 0.052, 0.016, 0.046, 0.038, 0, 0.0145, true);
      const index = buildFingerPhalanges(dir * 0.032, -0.054, 0.052, 0.036, 0.028, 0.0125);
      const middle = buildFingerPhalanges(dir * 0.010, -0.056, 0.056, 0.040, 0.030, 0.0130);
      const ring = buildFingerPhalanges(-dir * 0.012, -0.054, 0.050, 0.036, 0.028, 0.0120);
      const pinky = buildFingerPhalanges(-dir * 0.034, -0.050, 0.040, 0.028, 0.022, 0.0110);

      return {
        shoulder,
        upperArm: upperArmMesh,
        elbow,
        forearm: forearmMesh,
        wrist,
        palm,
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
      leftEye: lEye as any,
      rightEye: rEye as any,
      mouth: mouthMesh,
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
      targetRotX = Math.max(-0.4, Math.min(0.4, targetRotX + dy * 0.008));
    };

    const handleMouseUp = () => {
      isMouseDown = false;
    };

    const dom = renderer.domElement;
    dom.addEventListener('mousedown', handleMouseDown);
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);

    // 8. Animation Loop
    let clock = new THREE.Clock();
    let blinkTimer = 0;
    let telemetryTimer = 0;

    const animate = () => {
      reqIdRef.current = requestAnimationFrame(animate);

      const delta = Math.min(clock.getDelta(), 0.1);
      const time = clock.getElapsedTime();

      // Smooth camera rotation
      humanRoot.rotation.y += (targetRotY - humanRoot.rotation.y) * 0.1;
      humanRoot.rotation.x += (targetRotX - humanRoot.rotation.x) * 0.1;

      // Natural subtle idle breathing motion
      torsoGroup.position.y = Math.sin(time * 1.8) * 0.006;
      torsoGroup.scale.x = 1.0 + Math.sin(time * 1.8) * 0.010;

      // Natural eye blinking
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
        const S = new THREE.Vector3(dir * 0.23, 0.20, 0);
        const W = new THREE.Vector3(wristPos[0], wristPos[1], wristPos[2]);

        // Place wrist
        armRig.wrist.position.copy(W);
        armRig.wrist.rotation.set(wristRot[0], wristRot[1], wristRot[2]);

        // Solve Elbow Position
        const SW = new THREE.Vector3().subVectors(W, S);
        const d = Math.min(SW.length(), 0.44);
        const mid = new THREE.Vector3().addVectors(S, W).multiplyScalar(0.5);

        // Natural elbow sag down & out
        const naturalSag = new THREE.Vector3(dir * 0.4, -0.6, -0.2).normalize();
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

      // Apply Finger Articulation (Anatomical Flexion curls into palm on +Z)
      const applyFingerPhalanges = (joints: THREE.Group[], poseF: FingerPose) => {
        // -poseF.mcp flexes the finger forward into the palm
        joints[0].rotation.set(-poseF.mcp, 0, poseF.spread);
        joints[1].rotation.x = -poseF.pip;
        if (joints[2]) joints[2].rotation.x = -poseF.dip;
      };

      // Right Thumb (Flexes across thenar eminence)
      const rA = cur.rightArm;
      rRig.fingers.thumb[0].rotation.set(-rA.thumb.mcp * 0.75, -rA.thumb.spread, rA.thumb.mcp * 0.65);
      rRig.fingers.thumb[1].rotation.x = -rA.thumb.pip * 0.95;

      applyFingerPhalanges(rRig.fingers.index, rA.index);
      applyFingerPhalanges(rRig.fingers.middle, rA.middle);
      applyFingerPhalanges(rRig.fingers.ring, rA.ring);
      applyFingerPhalanges(rRig.fingers.pinky, rA.pinky);

      // Left Thumb & Fingers
      const lA = cur.leftArm;
      lRig.fingers.thumb[0].rotation.set(-lA.thumb.mcp * 0.75, lA.thumb.spread, -lA.thumb.mcp * 0.65);
      lRig.fingers.thumb[1].rotation.x = -lA.thumb.pip * 0.95;

      applyFingerPhalanges(lRig.fingers.index, lA.index);
      applyFingerPhalanges(lRig.fingers.middle, lA.middle);
      applyFingerPhalanges(lRig.fingers.ring, lA.ring);
      applyFingerPhalanges(lRig.fingers.pinky, lA.pinky);

      // Periodic Telemetry Update for HUD (5Hz update to avoid React re-render thrashing)
      telemetryTimer += delta;
      if (telemetryTimer > 0.2) {
        telemetryTimer = 0;
        const radToDeg = (rad: number) => Math.round((rad * 180) / Math.PI);
        const getStatus = (deg: number) => {
          if (deg < 12) return 'Straight (0°)';
          if (deg < 40) return 'Slight Bend';
          if (deg < 75) return 'Hooked / Half Curl';
          return 'Fully Curled';
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

    // Resize Observer
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
  }, []);

  return (
    <div
      className={`humanoid-avatar-wrap ${className}`}
      style={{
        position: 'relative',
        width: '100%',
        height: '100%',
        minHeight: '320px',
        background: '#070b14',
        borderRadius: '12px',
        overflow: 'hidden',
      }}
    >
      {/* 3D WebGL Canvas Mount */}
      <div ref={mountRef} style={{ width: '100%', height: '100%', minHeight: '320px', cursor: 'grab' }} />

      {/* Floating Camera Presets Bar */}
      <div style={{ position: 'absolute', top: '12px', right: '12px', display: 'flex', gap: '6px', zIndex: 10, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
        <button
          type="button"
          onClick={() => applyCameraPreset('front')}
          style={{
            background: cameraMode === 'front' ? '#4f46e5' : 'rgba(15, 23, 42, 0.8)',
            color: '#ffffff',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '6px',
            padding: '5px 9px',
            fontSize: '11px',
            fontWeight: 600,
            cursor: 'pointer',
            backdropFilter: 'blur(8px)',
          }}
          title="Upper Body Signer View"
        >
          👤 Body
        </button>
        <button
          type="button"
          onClick={() => applyCameraPreset('hands')}
          style={{
            background: cameraMode === 'hands' ? '#4f46e5' : 'rgba(15, 23, 42, 0.8)',
            color: '#ffffff',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '6px',
            padding: '5px 9px',
            fontSize: '11px',
            fontWeight: 600,
            cursor: 'pointer',
            backdropFilter: 'blur(8px)',
          }}
          title="Macro View on Both Hands"
        >
          🤲 Hands Macro
        </button>
        <button
          type="button"
          onClick={() => applyCameraPreset('rightHand')}
          style={{
            background: cameraMode === 'rightHand' ? '#0284c7' : 'rgba(15, 23, 42, 0.8)',
            color: '#ffffff',
            border: '1px solid rgba(56, 189, 248, 0.4)',
            borderRadius: '6px',
            padding: '5px 9px',
            fontSize: '11px',
            fontWeight: 600,
            cursor: 'pointer',
            backdropFilter: 'blur(8px)',
          }}
          title="Direct High-Magnification Macro on Right Hand (Fills Screen)"
        >
          👉 Right Hand HD
        </button>
        <button
          type="button"
          onClick={() => applyCameraPreset('leftHand')}
          style={{
            background: cameraMode === 'leftHand' ? '#059669' : 'rgba(15, 23, 42, 0.8)',
            color: '#ffffff',
            border: '1px solid rgba(52, 211, 153, 0.4)',
            borderRadius: '6px',
            padding: '5px 9px',
            fontSize: '11px',
            fontWeight: 600,
            cursor: 'pointer',
            backdropFilter: 'blur(8px)',
          }}
          title="Direct Macro on Left Hand"
        >
          👈 Left Hand HD
        </button>
        <button
          type="button"
          onClick={() => applyCameraPreset('side')}
          style={{
            background: cameraMode === 'side' ? '#4f46e5' : 'rgba(15, 23, 42, 0.8)',
            color: '#ffffff',
            border: '1px solid rgba(255, 255, 255, 0.2)',
            borderRadius: '6px',
            padding: '5px 9px',
            fontSize: '11px',
            fontWeight: 600,
            cursor: 'pointer',
            backdropFilter: 'blur(8px)',
          }}
          title="Side 3D Depth Angle"
        >
          📐 Depth
        </button>

        {/* Joint Indicators Toggle */}
        <button
          type="button"
          onClick={() => setHighlightJoints(!highlightJoints)}
          style={{
            background: highlightJoints ? '#10b981' : 'rgba(15, 23, 42, 0.8)',
            color: '#ffffff',
            border: '1px solid rgba(16, 185, 129, 0.5)',
            borderRadius: '6px',
            padding: '5px 9px',
            fontSize: '11px',
            fontWeight: 600,
            cursor: 'pointer',
            backdropFilter: 'blur(8px)',
          }}
          title="Toggle Luminous Skeletal Joint Rings"
        >
          ✨ Joint Rings
        </button>
      </div>

      {/* Real-time 5-Finger Articulation Telemetry HUD (Smallest Bend Verification) */}
      <div
        style={{
          position: 'absolute',
          top: '12px',
          left: '12px',
          background: 'rgba(10, 15, 29, 0.88)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(56, 189, 248, 0.35)',
          borderRadius: '8px',
          padding: '8px 12px',
          zIndex: 10,
          maxWidth: '220px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '6px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <Activity size={13} style={{ color: '#38bdf8' }} />
            <span style={{ color: '#94a3b8', fontSize: '10px', fontWeight: 700, letterSpacing: '0.05em' }}>
              FINGER BEND TELEMETRY
            </span>
          </div>
          <button
            type="button"
            onClick={() => setShowJointHUD(!showJointHUD)}
            style={{
              background: 'transparent',
              border: 'none',
              color: '#38bdf8',
              cursor: 'pointer',
              padding: 0,
              fontSize: '10px',
            }}
          >
            {showJointHUD ? 'Hide' : 'Show'}
          </button>
        </div>

        {showJointHUD && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            {[
              { label: 'Thumb', val: fingerDegrees.thumb },
              { label: 'Index', val: fingerDegrees.index },
              { label: 'Middle', val: fingerDegrees.middle },
              { label: 'Ring', val: fingerDegrees.ring },
              { label: 'Pinky', val: fingerDegrees.pinky },
            ].map(({ label, val }) => (
              <div key={label} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '10px' }}>
                <span style={{ color: '#e2e8f0', fontWeight: 600 }}>{label}:</span>
                <span style={{ color: val.angle < 30 ? '#34d399' : val.angle < 120 ? '#fbbf24' : '#f87171', fontWeight: 700 }}>
                  {val.status} ({val.angle}°)
                </span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Persistent Badge showing Current Sign & ISL Anatomical Status */}
      <div
        style={{
          position: 'absolute',
          bottom: '12px',
          left: '12px',
          right: '12px',
          background: 'rgba(10, 15, 29, 0.90)',
          backdropFilter: 'blur(12px)',
          border: '1px solid rgba(99, 102, 241, 0.35)',
          borderRadius: '8px',
          padding: '10px 14px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          zIndex: 10,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span style={{ display: 'inline-block', width: '8px', height: '8px', borderRadius: '50%', background: '#10b981', boxShadow: '0 0 8px #10b981' }} />
          <div>
            <div style={{ color: '#94a3b8', fontSize: '10px', fontWeight: 700, letterSpacing: '0.05em' }}>
              HIGH-PRECISION 3D HUMANOID SIGNER • 10 ARTICULATED FINGERS & KERATIN NAILS
            </div>
            <div style={{ color: '#ffffff', fontSize: '15px', fontWeight: 700 }}>
              {currentSign || 'GOOD MORNING'}
            </div>
          </div>
        </div>
        <div style={{ color: '#a5b4fc', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '6px' }}>
          <Sparkles size={13} />
          <span>Click & Drag to rotate 3D view</span>
        </div>
      </div>
    </div>
  );
};
