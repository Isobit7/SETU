import React, { useRef, useEffect, useState, useCallback } from 'react';
import * as THREE from 'three';
import { Sparkles } from 'lucide-react';

export interface HumanoidAvatar3DProps {
  currentSign: string;
  isTwoHanded?: boolean;
  className?: string;
}

// Finger joint angles in radians
interface FingerPose {
  mcp: number; // Metacarpophalangeal flexion (0: straight, 1.57: 90 deg curl)
  pip: number; // Proximal interphalangeal flexion
  dip: number; // Distal interphalangeal flexion
  spread: number; // Abduction / spread angle (-0.4 to 0.4 rad)
}

// 5-finger hand state
interface HandPoseState {
  wristRot: [number, number, number]; // pitch, yaw, roll
  wristPos: [number, number, number]; // x, y, z target position relative to chest
  thumb: FingerPose;
  index: FingerPose;
  middle: FingerPose;
  ring: FingerPose;
  pinky: FingerPose;
}

interface BodyPoseState {
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
const OPEN_FINGER = f(0.05, 0.05, 0.05, 0.0);
// Fully closed fist finger
const FIST_FINGER = f(1.45, 1.55, 1.2, 0.0);
// Semi-curled finger (like C or O shape)
const CURLED_FINGER = f(0.85, 0.95, 0.75, 0.0);

// Library of Authentic Indian Sign Language (ISL) Canonical 3D Poses
function getSignPose(rawSign: string): BodyPoseState {
  const s = rawSign.toUpperCase().replace(/\s+/g, '_');

  // Neutral rest pose (arms at resting chest ready position)
  const neutralLeft: HandPoseState = {
    wristRot: [0.1, -0.2, -0.1],
    wristPos: [-0.22, -0.18, 0.18],
    thumb: f(0.3, 0.4, 0.2, -0.3),
    index: f(0.4, 0.5, 0.3, -0.1),
    middle: f(0.45, 0.55, 0.3, 0.0),
    ring: f(0.45, 0.55, 0.3, 0.1),
    pinky: f(0.4, 0.5, 0.3, 0.2),
  };

  const neutralRight: HandPoseState = {
    wristRot: [0.1, 0.2, 0.1],
    wristPos: [0.22, -0.18, 0.18],
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
    pose.headRot = [-0.08, 0, 0]; // Gentle friendly nod up
    pose.rightArm = {
      wristRot: [-0.25, -0.1, 0.3],
      wristPos: [0.24, 0.28, 0.26],
      thumb: f(0.0, 0.0, 0.0, 0.4),
      index: f(0.0, 0.0, 0.0, 0.15),
      middle: f(0.0, 0.0, 0.0, 0.0),
      ring: f(0.0, 0.0, 0.0, -0.15),
      pinky: f(0.0, 0.0, 0.0, -0.3),
    };
    pose.leftArm = {
      wristRot: [-0.25, 0.1, -0.3],
      wristPos: [-0.24, 0.28, 0.26],
      thumb: f(0.0, 0.0, 0.0, -0.4),
      index: f(0.0, 0.0, 0.0, -0.15),
      middle: f(0.0, 0.0, 0.0, 0.0),
      ring: f(0.0, 0.0, 0.0, 0.15),
      pinky: f(0.0, 0.0, 0.0, 0.3),
    };
  } else if (s === 'HELLO' || s === 'HI' || s === 'WAVE') {
    // Right hand near temple in an open friendly salute/wave
    pose.headRot = [0.02, -0.05, 0.03];
    pose.rightArm = {
      wristRot: [-0.2, 0.15, 0.2],
      wristPos: [0.26, 0.36, 0.22],
      thumb: f(0.1, 0.1, 0.1, 0.3),
      index: f(0.05, 0.05, 0.05, 0.1),
      middle: f(0.05, 0.05, 0.05, 0.0),
      ring: f(0.05, 0.05, 0.05, -0.1),
      pinky: f(0.05, 0.05, 0.05, -0.2),
    };
  } else if (s === 'NAMASTE' || s === 'THANK_YOU' || s === 'THANK YOU' || s === 'THANKS') {
    // Both palms pressed flat together at center chest with head bow
    pose.isTwoHanded = true;
    pose.headRot = [0.18, 0.0, 0.0]; // Respectful head bow
    pose.rightArm = {
      wristRot: [0.15, -0.35, 0.2],
      wristPos: [0.045, 0.08, 0.25],
      thumb: f(0.1, 0.05, 0.05, 0.15),
      index: f(0.05, 0.05, 0.05, 0.05),
      middle: f(0.05, 0.05, 0.05, 0.0),
      ring: f(0.05, 0.05, 0.05, -0.05),
      pinky: f(0.05, 0.05, 0.05, -0.1),
    };
    pose.leftArm = {
      wristRot: [0.15, 0.35, -0.2],
      wristPos: [-0.045, 0.08, 0.25],
      thumb: f(0.1, 0.05, 0.05, -0.15),
      index: f(0.05, 0.05, 0.05, -0.05),
      middle: f(0.05, 0.05, 0.05, 0.0),
      ring: f(0.05, 0.05, 0.05, 0.05),
      pinky: f(0.05, 0.05, 0.05, 0.1),
    };
  } else if (s === 'HELP') {
    // ISL Two-Handed: Left palm flat facing up; right fist with thumb up placed on left palm
    pose.isTwoHanded = true;
    pose.leftArm = {
      wristRot: [0.35, 0.0, 0.0],
      wristPos: [-0.06, -0.05, 0.28],
      thumb: f(0.1, 0.1, 0.1, -0.3),
      index: f(0.05, 0.05, 0.05, -0.1),
      middle: f(0.05, 0.05, 0.05, 0.0),
      ring: f(0.05, 0.05, 0.05, 0.1),
      pinky: f(0.05, 0.05, 0.05, 0.2),
    };
    pose.rightArm = {
      wristRot: [0.1, -0.15, 0.0],
      wristPos: [0.06, 0.02, 0.28],
      thumb: f(0.0, 0.0, 0.0, 0.4), // Thumb pointing straight up
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
      wristPos: [0.18, 0.05, 0.26],
      thumb: f(0.1, 0.05, 0.05, 0.25),
      index: f(0.05, 0.05, 0.05, 0.1),
      middle: f(0.05, 0.05, 0.05, 0.0),
      ring: f(0.05, 0.05, 0.05, -0.1),
      pinky: f(0.05, 0.05, 0.05, -0.2),
    };
    pose.leftArm = {
      wristRot: [-0.1, 0.2, -0.1],
      wristPos: [-0.18, 0.05, 0.26],
      thumb: f(0.1, 0.05, 0.05, -0.25),
      index: f(0.05, 0.05, 0.05, -0.1),
      middle: f(0.05, 0.05, 0.05, 0.0),
      ring: f(0.05, 0.05, 0.05, 0.1),
      pinky: f(0.05, 0.05, 0.05, 0.2),
    };
  } else if (s === 'WATER' || s === 'DRINK') {
    // C-shaped cup brought up to mouth
    pose.rightArm = {
      wristRot: [0.35, -0.2, 0.4],
      wristPos: [0.12, 0.26, 0.2],
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
      wristPos: [0.08, 0.28, 0.18],
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
      wristRot: [0.3, 0.0, 0.1],
      wristPos: [0.22, 0.18, 0.26],
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
      wristPos: [0.2, 0.15, 0.28],
      thumb: f(0.6, 0.7, 0.5, 0.2),
      index: f(0.6, 0.8, 0.5, 0.05),
      middle: f(0.6, 0.8, 0.5, -0.05),
      ring: FIST_FINGER,
      pinky: FIST_FINGER,
    };
  } else if (s === 'YOU') {
    // Index pointing directly forward to conversational partner
    pose.rightArm = {
      wristRot: [-0.05, 0.0, 0.1],
      wristPos: [0.14, 0.08, 0.42],
      thumb: f(0.8, 1.1, 0.8, 0.2),
      index: f(0.0, 0.0, 0.0, 0.0), // Pointing straight
      middle: FIST_FINGER,
      ring: FIST_FINGER,
      pinky: FIST_FINGER,
    };
  } else if (s === 'I' || s === 'ME') {
    // Index pointing gently to center chest
    pose.rightArm = {
      wristRot: [0.65, -0.3, 0.1],
      wristPos: [0.06, 0.05, 0.12],
      thumb: f(0.8, 1.0, 0.8, 0.2),
      index: f(0.0, 0.0, 0.0, 0.0), // Pointing at chest
      middle: FIST_FINGER,
      ring: FIST_FINGER,
      pinky: FIST_FINGER,
    };
  } else if (s === 'SORRY') {
    // Fist rubbing circular chest
    pose.headRot = [0.06, 0.0, 0.0];
    pose.rightArm = {
      wristRot: [0.4, -0.2, 0.1],
      wristPos: [0.04, 0.1, 0.16],
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
      wristPos: [-0.05, -0.08, 0.26],
      thumb: f(0.1, 0.1, 0.1, -0.2),
      index: OPEN_FINGER,
      middle: OPEN_FINGER,
      ring: OPEN_FINGER,
      pinky: OPEN_FINGER,
    };
    pose.rightArm = {
      wristRot: [0.0, -1.3, 0.0], // Vertical knife hand
      wristPos: [0.03, -0.02, 0.26],
      thumb: f(0.1, 0.1, 0.1, 0.1),
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
    pose.rightArm.wristPos = [0.18, 0.12, 0.28];
    pose.rightArm.thumb = f(0.0, 0.0, 0.0, 0.4); // Upright thumb
    pose.rightArm.index = FIST_FINGER;
    pose.rightArm.middle = FIST_FINGER;
    pose.rightArm.ring = FIST_FINGER;
    pose.rightArm.pinky = FIST_FINGER;
  } else if (s === 'B') {
    // ISL Two-Handed: Left palm flat open; right palm open touching left palm
    pose.isTwoHanded = true;
    pose.leftArm = {
      wristRot: [0.1, 0.35, -0.1],
      wristPos: [-0.06, 0.12, 0.25],
      thumb: f(0.1, 0.1, 0.1, -0.2),
      index: OPEN_FINGER,
      middle: OPEN_FINGER,
      ring: OPEN_FINGER,
      pinky: OPEN_FINGER,
    };
    pose.rightArm = {
      wristRot: [0.1, -0.35, 0.1],
      wristPos: [0.06, 0.12, 0.25],
      thumb: f(0.1, 0.1, 0.1, 0.2),
      index: OPEN_FINGER,
      middle: OPEN_FINGER,
      ring: OPEN_FINGER,
      pinky: OPEN_FINGER,
    };
  } else if (s === 'C') {
    // Curved hand forming a clear C arc facing forward
    pose.rightArm.wristPos = [0.18, 0.14, 0.28];
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
      wristPos: [-0.05, 0.16, 0.26],
      thumb: FIST_FINGER,
      index: f(0.0, 0.0, 0.0, 0.0), // Upright stem
      middle: FIST_FINGER,
      ring: FIST_FINGER,
      pinky: FIST_FINGER,
    };
    pose.rightArm = {
      wristRot: [0.1, -0.4, 0.2],
      wristPos: [0.04, 0.16, 0.26],
      thumb: f(0.4, 0.5, 0.3, 0.3),
      index: f(0.6, 0.8, 0.6, 0.0),
      middle: FIST_FINGER,
      ring: FIST_FINGER,
      pinky: FIST_FINGER,
    };
  } else if (s === 'E') {
    // All fingers curled with fingertips touching thumb
    pose.rightArm.wristPos = [0.18, 0.12, 0.28];
    pose.rightArm.thumb = f(0.7, 0.8, 0.6, 0.1);
    pose.rightArm.index = f(1.1, 1.2, 0.9, 0.0);
    pose.rightArm.middle = f(1.1, 1.2, 0.9, 0.0);
    pose.rightArm.ring = f(1.1, 1.2, 0.9, 0.0);
    pose.rightArm.pinky = f(1.1, 1.2, 0.9, 0.0);
  } else if (s === 'F') {
    // Thumb and index touching in circle, middle, ring, pinky straight upright
    pose.rightArm.wristPos = [0.18, 0.16, 0.28];
    pose.rightArm.thumb = f(0.6, 0.7, 0.4, 0.2);
    pose.rightArm.index = f(0.8, 0.9, 0.6, 0.0);
    pose.rightArm.middle = OPEN_FINGER;
    pose.rightArm.ring = OPEN_FINGER;
    pose.rightArm.pinky = OPEN_FINGER;
  } else if (s === 'G') {
    // Index and thumb parallel pointing horizontally
    pose.rightArm.wristRot = [0.0, 0.5, 0.8];
    pose.rightArm.wristPos = [0.18, 0.14, 0.28];
    pose.rightArm.thumb = f(0.1, 0.0, 0.0, 0.2);
    pose.rightArm.index = f(0.0, 0.0, 0.0, 0.0);
    pose.rightArm.middle = FIST_FINGER;
    pose.rightArm.ring = FIST_FINGER;
    pose.rightArm.pinky = FIST_FINGER;
  } else if (s === 'H') {
    // Index and middle parallel pointing horizontally
    pose.rightArm.wristRot = [0.0, 0.5, 0.8];
    pose.rightArm.wristPos = [0.18, 0.14, 0.28];
    pose.rightArm.thumb = f(0.8, 1.0, 0.8, 0.1);
    pose.rightArm.index = f(0.0, 0.0, 0.0, 0.0);
    pose.rightArm.middle = f(0.0, 0.0, 0.0, 0.0);
    pose.rightArm.ring = FIST_FINGER;
    pose.rightArm.pinky = FIST_FINGER;
  } else if (s === 'I') {
    // Pinky upright, all other fingers closed
    pose.rightArm.wristPos = [0.18, 0.16, 0.28];
    pose.rightArm.thumb = f(0.9, 1.1, 0.8, 0.2);
    pose.rightArm.index = FIST_FINGER;
    pose.rightArm.middle = FIST_FINGER;
    pose.rightArm.ring = FIST_FINGER;
    pose.rightArm.pinky = f(0.0, 0.0, 0.0, 0.0); // Pinky straight up
  } else if (s === 'L') {
    // Index and thumb form sharp 90-degree L
    pose.rightArm.wristPos = [0.18, 0.16, 0.28];
    pose.rightArm.thumb = f(0.0, 0.0, 0.0, 0.45); // Extended thumb
    pose.rightArm.index = f(0.0, 0.0, 0.0, 0.0); // Upright index
    pose.rightArm.middle = FIST_FINGER;
    pose.rightArm.ring = FIST_FINGER;
    pose.rightArm.pinky = FIST_FINGER;
  } else if (s === 'M') {
    // Three fingers folded over thumb
    pose.rightArm.wristPos = [0.18, 0.12, 0.28];
    pose.rightArm.thumb = f(0.9, 1.0, 0.8, 0.0);
    pose.rightArm.index = f(1.3, 1.4, 0.9, 0.0);
    pose.rightArm.middle = f(1.3, 1.4, 0.9, 0.0);
    pose.rightArm.ring = f(1.3, 1.4, 0.9, 0.0);
    pose.rightArm.pinky = FIST_FINGER;
  } else if (s === 'N') {
    // Two fingers folded over thumb
    pose.rightArm.wristPos = [0.18, 0.12, 0.28];
    pose.rightArm.thumb = f(0.9, 1.0, 0.8, 0.0);
    pose.rightArm.index = f(1.3, 1.4, 0.9, 0.0);
    pose.rightArm.middle = f(1.3, 1.4, 0.9, 0.0);
    pose.rightArm.ring = FIST_FINGER;
    pose.rightArm.pinky = FIST_FINGER;
  } else if (s === 'O') {
    // All 5 fingers touching in closed O circle
    pose.rightArm.wristPos = [0.18, 0.14, 0.28];
    pose.rightArm.thumb = f(0.8, 0.8, 0.6, 0.2);
    pose.rightArm.index = f(0.9, 0.9, 0.8, 0.0);
    pose.rightArm.middle = f(0.9, 0.9, 0.8, 0.0);
    pose.rightArm.ring = f(0.9, 0.9, 0.8, 0.0);
    pose.rightArm.pinky = f(0.9, 0.9, 0.8, 0.0);
  } else if (s === 'P' || s === 'T') {
    // Authentic ISL Two-Handed T crossbar / P
    pose.isTwoHanded = true;
    pose.leftArm = {
      wristRot: [0.0, 0.1, 0.0],
      wristPos: [-0.05, 0.14, 0.26],
      thumb: FIST_FINGER,
      index: f(0.0, 0.0, 0.0, 0.0), // Vertical stem
      middle: FIST_FINGER,
      ring: FIST_FINGER,
      pinky: FIST_FINGER,
    };
    pose.rightArm = {
      wristRot: [0.0, 0.0, 1.57], // Horizontal crossbar
      wristPos: [0.02, 0.22, 0.26],
      thumb: FIST_FINGER,
      index: f(0.0, 0.0, 0.0, 0.0), // Horizontal crossbar finger
      middle: FIST_FINGER,
      ring: FIST_FINGER,
      pinky: FIST_FINGER,
    };
  } else if (s === 'U') {
    // Index and middle upright touching together
    pose.rightArm.wristPos = [0.18, 0.18, 0.28];
    pose.rightArm.thumb = f(0.8, 1.0, 0.8, 0.2);
    pose.rightArm.index = f(0.0, 0.0, 0.0, 0.02);
    pose.rightArm.middle = f(0.0, 0.0, 0.0, -0.02);
    pose.rightArm.ring = FIST_FINGER;
    pose.rightArm.pinky = FIST_FINGER;
  } else if (s === 'V') {
    // Index and middle extended wide in V peace sign
    pose.rightArm.wristPos = [0.18, 0.18, 0.28];
    pose.rightArm.thumb = f(0.8, 1.0, 0.8, 0.2);
    pose.rightArm.index = f(0.0, 0.0, 0.0, 0.18); // Spread left
    pose.rightArm.middle = f(0.0, 0.0, 0.0, -0.18); // Spread right
    pose.rightArm.ring = FIST_FINGER;
    pose.rightArm.pinky = FIST_FINGER;
  } else if (s === 'W') {
    // Index, middle, ring upright in W
    pose.rightArm.wristPos = [0.18, 0.18, 0.28];
    pose.rightArm.thumb = f(0.8, 1.0, 0.8, 0.2);
    pose.rightArm.index = f(0.0, 0.0, 0.0, 0.22);
    pose.rightArm.middle = f(0.0, 0.0, 0.0, 0.0);
    pose.rightArm.ring = f(0.0, 0.0, 0.0, -0.22);
    pose.rightArm.pinky = FIST_FINGER;
  } else if (s === 'Y') {
    // Thumb and pinky extended wide, middle 3 closed
    pose.rightArm.wristPos = [0.18, 0.16, 0.28];
    pose.rightArm.thumb = f(0.0, 0.0, 0.0, 0.45);
    pose.rightArm.index = FIST_FINGER;
    pose.rightArm.middle = FIST_FINGER;
    pose.rightArm.ring = FIST_FINGER;
    pose.rightArm.pinky = f(0.0, 0.0, 0.0, -0.35);
  } else if (s === '1') {
    pose.rightArm.wristPos = [0.18, 0.16, 0.28];
    pose.rightArm.thumb = FIST_FINGER;
    pose.rightArm.index = f(0.0, 0.0, 0.0, 0.0);
    pose.rightArm.middle = FIST_FINGER;
    pose.rightArm.ring = FIST_FINGER;
    pose.rightArm.pinky = FIST_FINGER;
  } else if (s === '2') {
    pose.rightArm.wristPos = [0.18, 0.16, 0.28];
    pose.rightArm.thumb = FIST_FINGER;
    pose.rightArm.index = f(0.0, 0.0, 0.0, 0.15);
    pose.rightArm.middle = f(0.0, 0.0, 0.0, -0.15);
    pose.rightArm.ring = FIST_FINGER;
    pose.rightArm.pinky = FIST_FINGER;
  } else if (s === '3') {
    pose.rightArm.wristPos = [0.18, 0.16, 0.28];
    pose.rightArm.thumb = FIST_FINGER;
    pose.rightArm.index = f(0.0, 0.0, 0.0, 0.2);
    pose.rightArm.middle = f(0.0, 0.0, 0.0, 0.0);
    pose.rightArm.ring = f(0.0, 0.0, 0.0, -0.2);
    pose.rightArm.pinky = FIST_FINGER;
  } else if (s === '4') {
    pose.rightArm.wristPos = [0.18, 0.16, 0.28];
    pose.rightArm.thumb = FIST_FINGER;
    pose.rightArm.index = f(0.0, 0.0, 0.0, 0.25);
    pose.rightArm.middle = f(0.0, 0.0, 0.0, 0.08);
    pose.rightArm.ring = f(0.0, 0.0, 0.0, -0.08);
    pose.rightArm.pinky = f(0.0, 0.0, 0.0, -0.25);
  } else if (s === '5') {
    pose.rightArm.wristPos = [0.18, 0.16, 0.28];
    pose.rightArm.thumb = f(0.0, 0.0, 0.0, 0.4);
    pose.rightArm.index = f(0.0, 0.0, 0.0, 0.25);
    pose.rightArm.middle = f(0.0, 0.0, 0.0, 0.08);
    pose.rightArm.ring = f(0.0, 0.0, 0.0, -0.08);
    pose.rightArm.pinky = f(0.0, 0.0, 0.0, -0.25);
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

  // References for scene & animation loop
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);

  // Dynamic 3D Human body components
  const bodyRef = useRef<{
    head: THREE.Group;
    torso: THREE.Group;
    leftArm: {
      shoulder: THREE.Group;
      elbow: THREE.Group;
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
      elbow: THREE.Group;
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
  const applyCameraPreset = useCallback((preset: 'front' | 'hands' | 'side') => {
    setCameraMode(preset);
    const camera = cameraRef.current;
    if (!camera) return;

    if (preset === 'front') {
      camera.position.set(0, 0.22, 1.35);
      camera.lookAt(0, 0.12, 0.1);
    } else if (preset === 'hands') {
      camera.position.set(0.08, 0.14, 0.75);
      camera.lookAt(0.05, 0.12, 0.22);
    } else if (preset === 'side') {
      camera.position.set(0.9, 0.26, 1.05);
      camera.lookAt(0, 0.15, 0.1);
    }
  }, []);

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
    scene.background = new THREE.Color(0x0f172a); // Deep slate dark navy

    // 2. Camera
    const camera = new THREE.PerspectiveCamera(42, width / height, 0.1, 100);
    camera.position.set(0, 0.22, 1.35);
    camera.lookAt(0, 0.12, 0.1);
    cameraRef.current = camera;

    // 3. Renderer with antialias
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.15;
    mount.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // 4. Studio Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.85);
    scene.add(ambientLight);

    // Warm Key Light (Front-right)
    const keyLight = new THREE.DirectionalLight(0xfff3e0, 1.6);
    keyLight.position.set(1.5, 2.0, 2.0);
    scene.add(keyLight);

    // Soft Cyan Fill Light (Front-left)
    const fillLight = new THREE.DirectionalLight(0x80d8ff, 0.9);
    fillLight.position.set(-1.5, 1.0, 1.5);
    scene.add(fillLight);

    // High-contrast Rim Light (Highlights finger silhouettes & shoulders)
    const rimLight = new THREE.DirectionalLight(0x6366f1, 2.2);
    rimLight.position.set(0, 2.5, -2.0);
    scene.add(rimLight);

    // 5. Materials
    const skinMaterial = new THREE.MeshStandardMaterial({
      color: 0xdf9b6d, // Warm natural Indian skin tone
      roughness: 0.45,
      metalness: 0.04,
    });

    const palmMaterial = new THREE.MeshStandardMaterial({
      color: 0xe8ad86, // Slightly lighter palmar surface
      roughness: 0.5,
      metalness: 0.02,
    });

    const shirtMaterial = new THREE.MeshStandardMaterial({
      color: 0x1e1b4b, // Dark indigo modern tunic/suit
      roughness: 0.7,
      metalness: 0.1,
    });

    const accentMaterial = new THREE.MeshStandardMaterial({
      color: 0x6366f1, // Indigo neon highlight trim
      roughness: 0.25,
      metalness: 0.4,
    });

    const hairMaterial = new THREE.MeshStandardMaterial({
      color: 0x111827, // Dark natural hair
      roughness: 0.9,
    });

    const eyeWhiteMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff });
    const irisMaterial = new THREE.MeshBasicMaterial({ color: 0x271911 }); // Deep brown pupil
    const lipMaterial = new THREE.MeshStandardMaterial({ color: 0xba6a5d, roughness: 0.4 });

    // 6. Build Humanoid Rig
    const humanRoot = new THREE.Group();
    scene.add(humanRoot);

    // Torso / Chest
    const torsoGroup = new THREE.Group();
    humanRoot.add(torsoGroup);

    const chestGeo = new THREE.CylinderGeometry(0.24, 0.18, 0.48, 20);
    const chestMesh = new THREE.Mesh(chestGeo, shirtMaterial);
    chestMesh.scale.set(1.0, 1.0, 0.65); // Elliptical human chest
    torsoGroup.add(chestMesh);

    // Modern collar & trim
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

    // Styled Hair
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

    // 3D Nose
    const noseGeo = new THREE.ConeGeometry(0.018, 0.045, 12);
    const noseMesh = new THREE.Mesh(noseGeo, skinMaterial);
    noseMesh.rotation.x = -Math.PI / 8;
    noseMesh.position.set(0, 0.005, 0.125);
    headGroup.add(noseMesh);

    // 3D Mouth / Lips
    const mouthGeo = new THREE.TorusGeometry(0.025, 0.007, 10, 16, Math.PI);
    const mouthMesh = new THREE.Mesh(mouthGeo, lipMaterial);
    mouthMesh.rotation.z = Math.PI; // Gentle smile curve
    mouthMesh.position.set(0, -0.045, 0.108);
    headGroup.add(mouthMesh);

    // ---------------------------------------------------------
    // Anatomical 5-Finger Articulated Hand & Arm Builder
    // ---------------------------------------------------------
    const buildArm = (isRight: boolean) => {
      const dir = isRight ? 1 : -1;

      // Shoulder joint
      const shoulder = new THREE.Group();
      shoulder.position.set(dir * 0.24, 0.21, 0);
      torsoGroup.add(shoulder);

      // Upper Arm
      const upperArmMesh = new THREE.Mesh(
        new THREE.CylinderGeometry(0.045, 0.04, 0.25, 14),
        shirtMaterial
      );
      upperArmMesh.position.set(0, -0.12, 0);
      shoulder.add(upperArmMesh);

      // Elbow joint
      const elbow = new THREE.Group();
      elbow.position.set(0, -0.24, 0);
      shoulder.add(elbow);

      // Forearm
      const forearmMesh = new THREE.Mesh(
        new THREE.CylinderGeometry(0.038, 0.03, 0.24, 14),
        skinMaterial
      );
      forearmMesh.position.set(0, -0.11, 0);
      elbow.add(forearmMesh);

      // Wrist
      const wrist = new THREE.Group();
      wrist.position.set(0, -0.23, 0);
      elbow.add(wrist);

      // Solid Volumetric 3D Palm
      const palmGeo = new THREE.BoxGeometry(0.082, 0.092, 0.026);
      const palm = new THREE.Mesh(palmGeo, palmMaterial);
      palm.position.set(0, -0.046, 0);
      wrist.add(palm);

      // Build 5 Articulated 3D Phalange Chains
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

        const knuckle1 = new THREE.Mesh(new THREE.SphereGeometry(fRadius * 1.15, 10, 10), skinMaterial);
        mcpJoint.add(knuckle1);

        // Proximal Phalanx
        const proxPhalanx = new THREE.Mesh(
          new THREE.CylinderGeometry(fRadius * 0.95, fRadius * 0.9, pLen, 10),
          skinMaterial
        );
        proxPhalanx.position.set(0, -pLen / 2, 0);
        mcpJoint.add(proxPhalanx);

        // Joint 2: PIP Knuckle
        const pipJoint = new THREE.Group();
        pipJoint.position.set(0, -pLen, 0);
        mcpJoint.add(pipJoint);

        const knuckle2 = new THREE.Mesh(new THREE.SphereGeometry(fRadius * 1.05, 10, 10), skinMaterial);
        pipJoint.add(knuckle2);

        // Intermediate Phalanx (or Distal for Thumb)
        const interPhalanx = new THREE.Mesh(
          new THREE.CylinderGeometry(fRadius * 0.85, fRadius * 0.8, iLen, 10),
          skinMaterial
        );
        interPhalanx.position.set(0, -iLen / 2, 0);
        pipJoint.add(interPhalanx);

        if (isThumb) {
          // Thumb only has 2 phalanges
          const thumbTip = new THREE.Mesh(new THREE.SphereGeometry(fRadius * 0.9, 10, 10), skinMaterial);
          thumbTip.position.set(0, -iLen, 0);
          pipJoint.add(thumbTip);
          return [mcpJoint, pipJoint];
        }

        // Joint 3: DIP Knuckle
        const dipJoint = new THREE.Group();
        dipJoint.position.set(0, -iLen, 0);
        pipJoint.add(dipJoint);

        const knuckle3 = new THREE.Mesh(new THREE.SphereGeometry(fRadius * 0.95, 10, 10), skinMaterial);
        dipJoint.add(knuckle3);

        // Distal Phalanx & Tip
        const distPhalanx = new THREE.Mesh(
          new THREE.CylinderGeometry(fRadius * 0.75, fRadius * 0.65, dLen, 10),
          skinMaterial
        );
        distPhalanx.position.set(0, -dLen / 2, 0);
        dipJoint.add(distPhalanx);

        const tip = new THREE.Mesh(new THREE.SphereGeometry(fRadius * 0.8, 10, 10), skinMaterial);
        tip.position.set(0, -dLen, 0);
        dipJoint.add(tip);

        return [mcpJoint, pipJoint, dipJoint];
      };

      // Create 5 fingers with anatomical positions across palm
      const thumb = buildFingerPhalanges(dir * 0.045, 0.015, 0.038, 0.032, 0, 0.0125, true);
      const index = buildFingerPhalanges(dir * 0.028, -0.048, 0.042, 0.03, 0.024, 0.0105);
      const middle = buildFingerPhalanges(dir * 0.009, -0.05, 0.046, 0.034, 0.026, 0.011);
      const ring = buildFingerPhalanges(-dir * 0.011, -0.048, 0.042, 0.03, 0.024, 0.01);
      const pinky = buildFingerPhalanges(-dir * 0.03, -0.044, 0.034, 0.024, 0.018, 0.009);

      return {
        shoulder,
        elbow,
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

    // 7. Mouse Orbit Controls for 3D Interaction
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

    const animate = () => {
      reqIdRef.current = requestAnimationFrame(animate);

      const delta = Math.min(clock.getDelta(), 0.1);
      const time = clock.getElapsedTime();

      // Smooth camera rotation
      humanRoot.rotation.y += (targetRotY - humanRoot.rotation.y) * 0.1;
      humanRoot.rotation.x += (targetRotX - humanRoot.rotation.x) * 0.1;

      // Natural subtle idle breathing motion
      torsoGroup.position.y = Math.sin(time * 1.8) * 0.008;
      torsoGroup.scale.x = 1.0 + Math.sin(time * 1.8) * 0.012;

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
      const lerpSpeed = delta * 9.0; // Responsive fluid gesture transition

      // Interpolate helper
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

      // Apply to Three.js Rig
      headGroup.rotation.set(cur.headRot[0], cur.headRot[1], cur.headRot[2]);

      // Apply Right Arm & 5 Fingers
      const rA = cur.rightArm;
      const rRig = bodyRef.current!.rightArm;
      rRig.wrist.position.set(rA.wristPos[0], rA.wristPos[1], rA.wristPos[2]);
      rRig.wrist.rotation.set(rA.wristRot[0], rA.wristRot[1], rA.wristRot[2]);

      // Right Thumb (flexes inward across palm)
      rRig.fingers.thumb[0].rotation.set(rA.thumb.mcp * 0.8, -rA.thumb.spread, -rA.thumb.mcp * 0.6);
      rRig.fingers.thumb[1].rotation.x = rA.thumb.pip * 0.9;

      // Right Fingers (MCP, PIP, DIP + Abduction Spread)
      const applyFingerPhalanges = (joints: THREE.Group[], poseF: FingerPose) => {
        joints[0].rotation.set(poseF.mcp, 0, poseF.spread);
        joints[1].rotation.x = poseF.pip;
        if (joints[2]) joints[2].rotation.x = poseF.dip;
      };

      applyFingerPhalanges(rRig.fingers.index, rA.index);
      applyFingerPhalanges(rRig.fingers.middle, rA.middle);
      applyFingerPhalanges(rRig.fingers.ring, rA.ring);
      applyFingerPhalanges(rRig.fingers.pinky, rA.pinky);

      // Apply Left Arm & 5 Fingers
      const lA = cur.leftArm;
      const lRig = bodyRef.current!.leftArm;
      lRig.wrist.position.set(lA.wristPos[0], lA.wristPos[1], lA.wristPos[2]);
      lRig.wrist.rotation.set(lA.wristRot[0], lA.wristRot[1], lA.wristRot[2]);

      lRig.fingers.thumb[0].rotation.set(lA.thumb.mcp * 0.8, -lA.thumb.spread, lA.thumb.mcp * 0.6);
      lRig.fingers.thumb[1].rotation.x = lA.thumb.pip * 0.9;

      applyFingerPhalanges(lRig.fingers.index, lA.index);
      applyFingerPhalanges(lRig.fingers.middle, lA.middle);
      applyFingerPhalanges(lRig.fingers.ring, lA.ring);
      applyFingerPhalanges(lRig.fingers.pinky, lA.pinky);

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
    <div className={`humanoid-avatar-wrap ${className}`} style={{ position: 'relative', width: '100%', height: '100%', minHeight: '440px', background: '#0b1329', borderRadius: '12px', overflow: 'hidden' }}>
      {/* 3D WebGL Canvas Mount */}
      <div ref={mountRef} style={{ width: '100%', height: '100%', minHeight: '440px', cursor: 'grab' }} />

      {/* Floating Camera Presets Bar */}
      <div style={{ position: 'absolute', top: '12px', right: '12px', display: 'flex', gap: '6px', zIndex: 10 }}>
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
          title="Front View"
        >
          👤 Full Body
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
          title="Close-up Hands View"
        >
          🤲 5-Finger Close-Up
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
          title="Side 3D Depth View"
        >
          📐 3D Side Angle
        </button>
      </div>

      {/* Persistent Badge showing Current Sign & ISL Anatomical Status */}
      <div
        style={{
          position: 'absolute',
          bottom: '12px',
          left: '12px',
          right: '12px',
          background: 'rgba(15, 23, 42, 0.88)',
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
              REALISTIC 3D HUMANOID SIGNER (10 ARTICULATED FINGERS)
            </div>
            <div style={{ color: '#ffffff', fontSize: '15px', fontWeight: 700 }}>
              {currentSign || 'GOOD MORNING'}
            </div>
          </div>
        </div>
        <div style={{ color: '#a5b4fc', fontSize: '11px', display: 'flex', alignItems: 'center', gap: '6px' }}>
          <Sparkles size={13} />
          <span>Drag mouse to inspect 3D posture</span>
        </div>
      </div>
    </div>
  );
};
