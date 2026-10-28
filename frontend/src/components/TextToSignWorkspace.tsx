import React, { useState, useEffect, useRef } from 'react';
import {
  Play,
  Pause,
  RotateCcw,
  Volume2,
  Mic,
  MicOff,
  Send,
  Sparkles,
  Maximize2,
  Minimize2,
  BookOpen,
} from 'lucide-react';

interface TextToSignWorkspaceProps {
  onSpeak?: (text: string) => void;
  isSpeaking?: boolean;
}

// Hand pose definition for Canvas Avatar
interface HandPose {
  x: number;
  y: number;
  fingers: number[]; // 5 finger extension states [thumb, index, middle, ring, pinky] (0: closed, 1: open)
  wristAngle: number;
  isTwoHanded?: boolean;
  leftHandX?: number;
  leftHandY?: number;
  leftFingers?: number[];
  leftWristAngle?: number;
}

// Hand gesture poses for Alphabet and Common Signs (with authentic Indian Sign Language bimanual forms)
const SIGN_DATABASE: Record<string, { label: string; desc: string; pose: HandPose }> = {
  A: {
    label: 'Letter A',
    desc: 'Closed fist with thumb upright along side of index finger.',
    pose: { x: 340, y: 260, fingers: [1, 0, 0, 0, 0], wristAngle: -0.1 },
  },
  B: {
    label: 'Letter B (ISL Two-Handed)',
    desc: 'Two-handed: Left hand held upright flat palm facing right; right hand open palm touching left palm.',
    pose: {
      x: 325,
      y: 235,
      fingers: [0.3, 1, 1, 1, 1],
      wristAngle: -0.15,
      isTwoHanded: true,
      leftHandX: 275,
      leftHandY: 235,
      leftFingers: [0.3, 1, 1, 1, 1],
      leftWristAngle: 0.15,
    },
  },
  C: {
    label: 'Letter C',
    desc: 'Curved hand forming a distinct C-shape facing forward.',
    pose: { x: 340, y: 250, fingers: [0.6, 0.6, 0.6, 0.6, 0.6], wristAngle: -0.2 },
  },
  D: {
    label: 'Letter D (ISL Two-Handed)',
    desc: 'Two-handed: Left index points straight upright (stem); right index and thumb curve to meet left index tip and base.',
    pose: {
      x: 315,
      y: 230,
      fingers: [0.8, 0.8, 0, 0, 0],
      wristAngle: -0.3,
      isTwoHanded: true,
      leftHandX: 275,
      leftHandY: 230,
      leftFingers: [0, 1, 0, 0, 0],
      leftWristAngle: 0,
    },
  },
  E: {
    label: 'Letter E',
    desc: 'Fingers curled inward with tips resting near thumb tip.',
    pose: { x: 340, y: 260, fingers: [0.2, 0.3, 0.3, 0.3, 0.3], wristAngle: 0.1 },
  },
  F: {
    label: 'Letter F',
    desc: 'Thumb and index fingertips touching in a circle, other three fingers upright.',
    pose: { x: 340, y: 230, fingers: [0.4, 0.4, 1, 1, 1], wristAngle: -0.1 },
  },
  G: {
    label: 'Letter G',
    desc: 'Index finger and thumb parallel pointing horizontally.',
    pose: { x: 330, y: 260, fingers: [0.9, 0.9, 0, 0, 0], wristAngle: 0.6 },
  },
  H: {
    label: 'Letter H',
    desc: 'Index and middle fingers extended together pointing horizontally.',
    pose: { x: 330, y: 250, fingers: [0.4, 1, 1, 0, 0], wristAngle: 0.6 },
  },
  I: {
    label: 'Letter I',
    desc: 'Pinky finger extended upright, other fingers closed in a fist.',
    pose: { x: 340, y: 240, fingers: [0, 0, 0, 0, 1], wristAngle: 0 },
  },
  J: {
    label: 'Letter J',
    desc: 'Pinky finger extended and tracing a small J curve in the air.',
    pose: { x: 340, y: 250, fingers: [0, 0, 0, 0, 1], wristAngle: -0.3 },
  },
  K: {
    label: 'Letter K',
    desc: 'Index finger pointing up, middle finger angled forward, thumb between them.',
    pose: { x: 340, y: 230, fingers: [0.8, 1, 0.8, 0, 0], wristAngle: 0 },
  },
  L: {
    label: 'Letter L',
    desc: 'Thumb and index finger form an L shape at right angles.',
    pose: { x: 340, y: 240, fingers: [1, 1, 0, 0, 0], wristAngle: 0 },
  },
  M: {
    label: 'Letter M',
    desc: 'Thumb tucked under three fingers (index, middle, ring).',
    pose: { x: 340, y: 270, fingers: [0, 0.2, 0.2, 0.2, 0], wristAngle: 0.1 },
  },
  N: {
    label: 'Letter N',
    desc: 'Thumb tucked under index and middle fingers.',
    pose: { x: 340, y: 270, fingers: [0, 0.2, 0.2, 0, 0], wristAngle: 0.1 },
  },
  O: {
    label: 'Letter O',
    desc: 'All fingertips curved to meet thumb tip, forming an O shape.',
    pose: { x: 340, y: 250, fingers: [0.5, 0.5, 0.5, 0.5, 0.5], wristAngle: 0 },
  },
  P: {
    label: 'Letter P (ISL Two-Handed)',
    desc: 'Two-handed: Left index points upright (stem); right index and thumb form an upper circular loop touching left index tip.',
    pose: {
      x: 310,
      y: 215,
      fingers: [0.8, 0.8, 0, 0, 0],
      wristAngle: -0.5,
      isTwoHanded: true,
      leftHandX: 280,
      leftHandY: 235,
      leftFingers: [0, 1, 0, 0, 0],
      leftWristAngle: 0,
    },
  },
  Q: {
    label: 'Letter Q (ISL Two-Handed)',
    desc: 'Two-handed: Left index finger points downwards; right index and thumb circle around left wrist base.',
    pose: {
      x: 320,
      y: 235,
      fingers: [0.8, 0.8, 0, 0, 0],
      wristAngle: 0.8,
      isTwoHanded: true,
      leftHandX: 280,
      leftHandY: 255,
      leftFingers: [0, 1, 0, 0, 0],
      leftWristAngle: 3.14,
    },
  },
  R: {
    label: 'Letter R',
    desc: 'Index and middle fingers crossed over each other.',
    pose: { x: 340, y: 240, fingers: [0, 1, 1, 0, 0], wristAngle: 0 },
  },
  S: {
    label: 'Letter S',
    desc: 'Tight fist with thumb folded horizontally across the front of fingers.',
    pose: { x: 340, y: 260, fingers: [0, 0, 0, 0, 0], wristAngle: 0 },
  },
  T: {
    label: 'Letter T (ISL Two-Handed)',
    desc: 'Two-handed: Left index finger points upright; right index held horizontally across top of left index forming a T crossbar.',
    pose: {
      x: 295,
      y: 205,
      fingers: [0, 1, 0, 0, 0],
      wristAngle: 1.57,
      isTwoHanded: true,
      leftHandX: 285,
      leftHandY: 235,
      leftFingers: [0, 1, 0, 0, 0],
      leftWristAngle: 0,
    },
  },
  U: {
    label: 'Letter U',
    desc: 'Index and middle fingers held upright together.',
    pose: { x: 340, y: 230, fingers: [0, 1, 1, 0, 0], wristAngle: 0 },
  },
  V: {
    label: 'Letter V',
    desc: 'Index and middle fingers held upright in a spread V shape.',
    pose: { x: 340, y: 230, fingers: [0, 1, 1, 0, 0], wristAngle: 0.1 },
  },
  W: {
    label: 'Letter W',
    desc: 'Three middle fingers held upright and spread in a W shape.',
    pose: { x: 340, y: 230, fingers: [0, 1, 1, 1, 0], wristAngle: 0 },
  },
  X: {
    label: 'Letter X',
    desc: 'Index finger hooked in a crook shape, others closed.',
    pose: { x: 340, y: 250, fingers: [0, 0.5, 0, 0, 0], wristAngle: 0.1 },
  },
  Y: {
    label: 'Letter Y',
    desc: 'Thumb and pinky extended wide, middle three fingers folded down.',
    pose: { x: 340, y: 240, fingers: [1, 0, 0, 0, 1], wristAngle: 0 },
  },
  Z: {
    label: 'Letter Z',
    desc: 'Index finger extended and tracing a Z pattern in the air.',
    pose: { x: 340, y: 240, fingers: [0, 1, 0, 0, 0], wristAngle: 0.2 },
  },
  NAMASTE: {
    label: 'Namaste (ISL Greeting)',
    desc: 'Two-handed respectful greeting: both flat palms pressed together at center chest in Anjali Mudra.',
    pose: {
      x: 314,
      y: 228,
      fingers: [1, 1, 1, 1, 1],
      wristAngle: -0.1,
      isTwoHanded: true,
      leftHandX: 286,
      leftHandY: 228,
      leftFingers: [1, 1, 1, 1, 1],
      leftWristAngle: 0.1,
    },
  },
  HELLO: {
    label: 'Hello (Greeting)',
    desc: 'Open flat hand near temple moving outward in a friendly wave/salute.',
    pose: { x: 350, y: 190, fingers: [1, 1, 1, 1, 1], wristAngle: -0.3 },
  },
  HELP: {
    label: 'Help (ISL Two-Handed)',
    desc: 'Two-handed: Left hand flat palm held open facing up; right fist with thumb up placed on left palm.',
    pose: {
      x: 315,
      y: 245,
      fingers: [1, 0, 0, 0, 0],
      wristAngle: 0,
      isTwoHanded: true,
      leftHandX: 275,
      leftHandY: 260,
      leftFingers: [1, 1, 1, 1, 1],
      leftWristAngle: 0.25,
    },
  },
  WATER: {
    label: 'Water (Drink)',
    desc: 'W-hand shape (three middle fingers upright) tapped gently against side of chin/mouth.',
    pose: { x: 325, y: 160, fingers: [0, 1, 1, 1, 0], wristAngle: -0.25, isTwoHanded: false },
  },
  FOOD: {
    label: 'Food (Eat)',
    desc: 'Fingertips clustered together with thumb touching tips and brought gently toward mouth.',
    pose: { x: 315, y: 160, fingers: [0.3, 0.3, 0.3, 0.3, 0.3], wristAngle: -0.35, isTwoHanded: false },
  },
  THANK_YOU: {
    label: 'Thank You (Gratitude)',
    desc: 'Flat hand touching chin/chest with fingertips and sweeping outward toward partner.',
    pose: { x: 300, y: 230, fingers: [0.8, 1, 1, 1, 1], wristAngle: 0.2 },
  },
  GOOD_MORNING: {
    label: 'Good Morning',
    desc: 'Compound sign: Thumb up for "good" followed by dominant hand rising like sun.',
    pose: {
      x: 350,
      y: 200,
      fingers: [1, 1, 1, 1, 1],
      wristAngle: -0.2,
      isTwoHanded: true,
      leftHandX: 250,
      leftHandY: 280,
      leftFingers: [0.8, 0.8, 0.8, 0.8, 0.8],
      leftWristAngle: 0.1,
    },
  },
  YOU: {
    label: 'You (Pronoun)',
    desc: 'Index finger pointing forward directly toward conversation partner.',
    pose: { x: 320, y: 240, fingers: [0.2, 1, 0, 0, 0], wristAngle: 0.3 },
  },
  ME: {
    label: 'I / Me (Pronoun)',
    desc: 'Index finger pointing inward to chest.',
    pose: { x: 290, y: 260, fingers: [0.2, 1, 0, 0, 0], wristAngle: -1.2 },
  },
  PLEASE: {
    label: 'Please (Polite)',
    desc: 'Flat open palm rubbing circular motion over heart/chest.',
    pose: { x: 290, y: 250, fingers: [1, 1, 1, 1, 1], wristAngle: -0.2 },
  },
  WELCOME: {
    label: 'Welcome (Courtesy)',
    desc: 'Both hands open palm-up sweeping smoothly inward toward the body.',
    pose: {
      x: 360,
      y: 260,
      fingers: [1, 1, 1, 1, 1],
      wristAngle: 0.2,
      isTwoHanded: true,
      leftHandX: 240,
      leftHandY: 260,
      leftFingers: [1, 1, 1, 1, 1],
      leftWristAngle: -0.2,
    },
  },
};

// Simplified English-to-ISL Grammar Gloss converter
// ISL typically uses Subject-Object-Verb (SOV) / Topic-Comment order with question particles at the end.
function convertEnglishToISLGloss(text: string): { glossTokens: string[]; words: string[] } {
  const clean = text.trim().toUpperCase();
  if (!clean) return { glossTokens: [], words: [] };

  const rawWords = clean.split(/\s+/).filter(Boolean);
  const glossTokens: string[] = [];

  rawWords.forEach((word) => {
    // Check known compound words/phrases
    if (word === 'NAMASTE' || word === 'VANAKKAM' || word === 'NAMASKAR') {
      glossTokens.push('NAMASTE');
    } else if (word === 'HELLO' || word === 'HI') {
      glossTokens.push('HELLO');
    } else if (word === 'HELP') {
      glossTokens.push('HELP');
    } else if (word === 'WATER' || word === 'DRINK') {
      glossTokens.push('WATER');
    } else if (word === 'FOOD' || word === 'EAT') {
      glossTokens.push('FOOD');
    } else if (word === 'THANK' || word === 'THANKS') {
      glossTokens.push('THANK_YOU');
    } else if (word === 'YOU' || word === 'YOUR') {
      glossTokens.push('YOU');
    } else if (word === 'I' || word === 'ME' || word === 'MY') {
      glossTokens.push('ME');
    } else if (word === 'PLEASE') {
      glossTokens.push('PLEASE');
    } else if (word === 'WELCOME') {
      glossTokens.push('WELCOME');
    } else {
      // Spell out letters if no whole-word sign
      for (const char of word) {
        if (/^[A-Z]$/.test(char)) {
          glossTokens.push(char);
        }
      }
    }
  });

  return { glossTokens, words: rawWords };
}

export const TextToSignWorkspace: React.FC<TextToSignWorkspaceProps> = ({ onSpeak, isSpeaking = false }) => {
  const [inputText, setInputText] = useState<string>('Hello');
  const [activeTokenIndex, setActiveTokenIndex] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(true);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1.0);
  const [isListening, setIsListening] = useState<boolean>(false);
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  const { glossTokens } = convertEnglishToISLGloss(inputText);
  const activeToken = glossTokens[activeTokenIndex] || 'HELLO';
  const activeSignData = SIGN_DATABASE[activeToken] || SIGN_DATABASE['HELLO'];

  // Current interpolated hand position (Right and Left hands for authentic bimanual rendering)
  const currentHandPosRef = useRef({
    rx: 340,
    ry: 240,
    rWristAngle: 0,
    rFingers: [1, 1, 1, 1, 1],
    lx: 230,
    ly: 310,
    lWristAngle: 0.1,
    lFingers: [0.3, 0.3, 0.3, 0.3, 0.3],
  });

  // Cycle through gloss tokens
  useEffect(() => {
    if (!isPlaying || glossTokens.length === 0) return;

    const intervalTime = Math.max(400, Math.round(1200 / playbackSpeed));
    const timer = setInterval(() => {
      setActiveTokenIndex((prev) => (prev + 1) % glossTokens.length);
    }, intervalTime);

    return () => clearInterval(timer);
  }, [isPlaying, glossTokens.length, playbackSpeed]);

  // Reset index when input changes
  useEffect(() => {
    setActiveTokenIndex(0);
  }, [inputText]);

  // Animated 3D Articulated Humanoid Avatar Canvas Loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let time = 0;

    // Helper: Draw Volumetric 3D Joint Sphere with specular highlight
    const drawJointSphere = (
      x: number,
      y: number,
      radius: number,
      baseColor: string,
      highlightColor = '#ffffff'
    ) => {
      const grad = ctx.createRadialGradient(
        x - radius * 0.35,
        y - radius * 0.35,
        radius * 0.1,
        x,
        y,
        radius
      );
      grad.addColorStop(0, highlightColor);
      grad.addColorStop(0.35, baseColor);
      grad.addColorStop(1, '#090d16');
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();

      // Subtle outer glow ring
      ctx.strokeStyle = baseColor;
      ctx.lineWidth = 1;
      ctx.stroke();
    };

    // Helper: Draw 3D Articulated Multi-Phalange Hand
    const drawArticulatedHand = (
      handX: number,
      handY: number,
      wristAngle: number,
      fingers: number[],
      isLeft: boolean
    ) => {
      ctx.save();
      ctx.translate(handX, handY);
      ctx.rotate(wristAngle);

      const themeColor = isLeft ? '#34d399' : '#38bdf8'; // Emerald Left Hand, Cyan Right Hand
      const dimColor = '#1f2937';

      // 1. Scaled Volumetric Palm Plate (46 x 42 px for high legibility)
      const pW = 46;
      const pH = 42;
      const palmGrad = ctx.createLinearGradient(0, -pH / 2, 0, pH / 2);
      palmGrad.addColorStop(0, '#334155');
      palmGrad.addColorStop(0.7, '#1e293b');
      palmGrad.addColorStop(1, '#0f172a');
      ctx.fillStyle = palmGrad;
      ctx.strokeStyle = themeColor;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.roundRect(-pW / 2, -pH / 2, pW, pH, 8);
      ctx.fill();
      ctx.stroke();

      // Bio-Luminescent Palm Core Sensor
      const palmCoreGrad = ctx.createRadialGradient(0, 0, 1, 0, 0, 10);
      palmCoreGrad.addColorStop(0, '#ffffff');
      palmCoreGrad.addColorStop(0.5, themeColor);
      palmCoreGrad.addColorStop(1, 'transparent');
      ctx.fillStyle = palmCoreGrad;
      ctx.beginPath();
      ctx.arc(0, 0, 10, 0, Math.PI * 2);
      ctx.fill();

      // 2. High-Fidelity Multi-Phalange Fingers (Enlarged lengths, pronounced curling)
      const fingerAngles = isLeft ? [0.65, 0.32, 0, -0.32, -0.65] : [-0.65, -0.32, 0, 0.32, 0.65];
      const fingerLengths = [22, 34, 38, 34, 26];
      const knuckleSpacing = isLeft ? [16, 8, 0, -8, -16] : [-16, -8, 0, 8, 16];

      for (let f = 0; f < 5; f++) {
        const ext = Math.max(0, Math.min(1, fingers[f] ?? 1));
        const baseAngle = fingerAngles[f];
        const totalLen = fingerLengths[f];
        const seg1Len = totalLen * 0.55;
        const seg2Len = totalLen * 0.52;

        // Metacarpal base knuckle
        const kX = knuckleSpacing[f];
        const kY = -pH / 2 + 2;
        drawJointSphere(kX, kY, 4.5, themeColor);

        // Proximal segment (segment 1)
        const p1X = kX + Math.sin(baseAngle) * seg1Len;
        const p1Y = kY - Math.cos(baseAngle) * seg1Len;

        ctx.strokeStyle = ext > 0.4 ? themeColor : dimColor;
        ctx.lineWidth = 4.5;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(kX, kY);
        ctx.lineTo(p1X, p1Y);
        ctx.stroke();

        // Intermediate joint sphere (shows amber highlight when curled)
        const jointHighlight = ext > 0.5 ? themeColor : '#f59e0b';
        drawJointSphere(p1X, p1Y, 3.5, jointHighlight);

        // Distal segment (curls inward into palm when closed, straightens when open)
        const curlOffset = (1 - ext) * 1.8 * (isLeft ? -1 : 1);
        const distalAngle = baseAngle + curlOffset;
        const p2X = p1X + Math.sin(distalAngle) * (seg2Len * (0.35 + ext * 0.65));
        const p2Y = p1Y - Math.cos(distalAngle) * (seg2Len * (0.35 + ext * 0.65));

        ctx.strokeStyle = ext > 0.4 ? themeColor : '#475569';
        ctx.lineWidth = 3.8;
        ctx.beginPath();
        ctx.moveTo(p1X, p1Y);
        ctx.lineTo(p2X, p2Y);
        ctx.stroke();

        // Fingertip radiant glowing node
        drawJointSphere(p2X, p2Y, 3.5, ext > 0.5 ? '#ffffff' : '#94a3b8', '#ffffff');
      }

      ctx.restore();
    };

    const render = () => {
      time += 0.05 * playbackSpeed;
      const targetPose = activeSignData.pose;
      const curr = currentHandPosRef.current;

      // 1. Right Hand Target Interpolation
      const rxTarget = targetPose.x;
      const ryTarget = targetPose.y;
      const rAngleTarget = targetPose.wristAngle;
      const rFingersTarget = targetPose.fingers;

      curr.rx += (rxTarget - curr.rx) * 0.18;
      curr.ry += (ryTarget - curr.ry) * 0.18;
      curr.rWristAngle += (rAngleTarget - curr.rWristAngle) * 0.18;
      for (let i = 0; i < 5; i++) {
        curr.rFingers[i] += ((rFingersTarget[i] ?? 1) - curr.rFingers[i]) * 0.22;
      }

      // 2. Left Hand Target Interpolation (Supports authentic dual-handed ISL signing)
      const centerX = canvas.width / 2;
      const headY = 120 + Math.sin(time * 0.8) * 1.5;

      const lxTarget = targetPose.isTwoHanded ? (targetPose.leftHandX ?? 270) : centerX - 75;
      const lyTarget = targetPose.isTwoHanded ? (targetPose.leftHandY ?? 240) : headY + 185;
      const lAngleTarget = targetPose.isTwoHanded ? (targetPose.leftWristAngle ?? 0) : 0.15;
      const lFingersTarget = targetPose.isTwoHanded ? (targetPose.leftFingers ?? [1, 1, 1, 1, 1]) : [0.3, 0.3, 0.3, 0.3, 0.3];

      curr.lx += (lxTarget - curr.lx) * 0.18;
      curr.ly += (lyTarget - curr.ly) * 0.18;
      curr.lWristAngle += (lAngleTarget - curr.lWristAngle) * 0.18;
      for (let i = 0; i < 5; i++) {
        curr.lFingers[i] += ((lFingersTarget[i] ?? 1) - curr.lFingers[i]) * 0.22;
      }

      // Clear Canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // 3. Volumetric Studio Vignette Backdrop
      const bgGrad = ctx.createRadialGradient(
        centerX,
        220,
        40,
        centerX,
        220,
        canvas.width * 0.65
      );
      bgGrad.addColorStop(0, '#151922');
      bgGrad.addColorStop(0.6, '#0d1117');
      bgGrad.addColorStop(1, '#05070a');
      ctx.fillStyle = bgGrad;
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Cyber Studio Floor Grid
      ctx.strokeStyle = 'rgba(56, 189, 248, 0.08)';
      ctx.lineWidth = 1;
      for (let y = 370; y < canvas.height; y += 22) {
        ctx.beginPath();
        ctx.moveTo(30, y);
        ctx.lineTo(canvas.width - 30, y);
        ctx.stroke();
      }

      // 4. Volumetric Head (Avatar Silhouette with 3D Shader Gradient)
      const headGrad = ctx.createRadialGradient(
        centerX - 10,
        headY - 12,
        4,
        centerX,
        headY,
        42
      );
      headGrad.addColorStop(0, '#334155');
      headGrad.addColorStop(0.5, '#1e293b');
      headGrad.addColorStop(1, '#0b0f19');
      ctx.fillStyle = headGrad;
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(centerX, headY, 38, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      // Cyber Visor / Face Illumination Band
      const visorGrad = ctx.createLinearGradient(centerX - 24, headY - 4, centerX + 24, headY - 4);
      visorGrad.addColorStop(0, 'rgba(56, 189, 248, 0.1)');
      visorGrad.addColorStop(0.5, '#38bdf8');
      visorGrad.addColorStop(1, 'rgba(56, 189, 248, 0.1)');
      ctx.fillStyle = visorGrad;
      ctx.beginPath();
      ctx.roundRect(centerX - 22, headY - 6, 44, 8, 4);
      ctx.fill();

      // 5. Neck & Anatomical Torso
      ctx.beginPath();
      ctx.moveTo(centerX - 14, headY + 36);
      ctx.lineTo(centerX - 14, headY + 52);
      ctx.lineTo(centerX - 68, headY + 70); // Left shoulder
      ctx.lineTo(centerX - 52, headY + 220); // Left hip
      ctx.lineTo(centerX + 52, headY + 220); // Right hip
      ctx.lineTo(centerX + 68, headY + 70); // Right shoulder
      ctx.lineTo(centerX + 14, headY + 52);
      ctx.lineTo(centerX + 14, headY + 36);
      ctx.closePath();

      const torsoGrad = ctx.createLinearGradient(centerX, headY + 50, centerX, headY + 220);
      torsoGrad.addColorStop(0, '#1e293b');
      torsoGrad.addColorStop(0.5, '#111827');
      torsoGrad.addColorStop(1, '#090d16');
      ctx.fillStyle = torsoGrad;
      ctx.strokeStyle = '#334155';
      ctx.lineWidth = 2;
      ctx.fill();
      ctx.stroke();

      // Center Cyber Arc Reactor / Emblem
      const coreGrad = ctx.createRadialGradient(
        centerX,
        headY + 115,
        2,
        centerX,
        headY + 115,
        20
      );
      coreGrad.addColorStop(0, '#ffffff');
      coreGrad.addColorStop(0.3, '#38bdf8');
      coreGrad.addColorStop(1, 'transparent');
      ctx.fillStyle = coreGrad;
      ctx.beginPath();
      ctx.arc(centerX, headY + 115, 20, 0, Math.PI * 2);
      ctx.fill();

      // 6. Dual-Arm Articulated Kinematics (Right Arm & Left Arm)
      const rightShoulderX = centerX + 68;
      const rightShoulderY = headY + 70;
      const leftShoulderX = centerX - 68;
      const leftShoulderY = headY + 70;

      // Draw Shoulders (Volumetric Joint Spheres)
      drawJointSphere(rightShoulderX, rightShoulderY, 8, '#38bdf8');
      drawJointSphere(leftShoulderX, leftShoulderY, 8, '#34d399');

      // Right Arm Kinematics
      const rHandX = curr.rx;
      const rHandY = curr.ry + Math.sin(time * 1.5) * 2;
      const rElbowX = (rightShoulderX + rHandX) / 2 + 26;
      const rElbowY = (rightShoulderY + rHandY) / 2 + 28;

      // Right Upper Arm & Forearm
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 5;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(rightShoulderX, rightShoulderY);
      ctx.lineTo(rElbowX, rElbowY);
      ctx.lineTo(rHandX, rHandY);
      ctx.stroke();

      // Right Elbow & Wrist Joints
      drawJointSphere(rElbowX, rElbowY, 6, '#38bdf8');
      drawJointSphere(rHandX, rHandY, 6, '#38bdf8');

      // Left Arm Kinematics
      const lHandX = curr.lx;
      const lHandY = curr.ly + Math.sin(time * 1.5 + 1) * 2;
      const lElbowX = (leftShoulderX + lHandX) / 2 - 26;
      const lElbowY = (leftShoulderY + lHandY) / 2 + 28;

      // Left Upper Arm & Forearm
      ctx.strokeStyle = '#34d399';
      ctx.lineWidth = 5;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(leftShoulderX, leftShoulderY);
      ctx.lineTo(lElbowX, lElbowY);
      ctx.lineTo(lHandX, lHandY);
      ctx.stroke();

      // Left Elbow & Wrist Joints
      drawJointSphere(lElbowX, lElbowY, 6, '#34d399');
      drawJointSphere(lHandX, lHandY, 6, '#34d399');

      // 7. Bimanual Proximity & Dynamic Contact Aura
      const dx = rHandX - lHandX;
      const dy = rHandY - lHandY;
      const handDist = Math.sqrt(dx * dx + dy * dy);

      if (targetPose.isTwoHanded && handDist < 70) {
        const midX = (rHandX + lHandX) / 2;
        const midY = (rHandY + lHandY) / 2;
        const auraGrad = ctx.createRadialGradient(midX, midY, 4, midX, midY, 45);
        auraGrad.addColorStop(0, 'rgba(56, 189, 248, 0.4)');
        auraGrad.addColorStop(0.5, 'rgba(52, 211, 153, 0.25)');
        auraGrad.addColorStop(1, 'transparent');
        ctx.fillStyle = auraGrad;
        ctx.beginPath();
        ctx.arc(midX, midY, 45, 0, Math.PI * 2);
        ctx.fill();

        // High-energy contact bridge arc
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.85)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(lHandX, lHandY);
        ctx.lineTo(rHandX, rHandY);
        ctx.stroke();
      }

      // 8. Draw Multi-Phalange Articulated Hands (Left and Right)
      drawArticulatedHand(lHandX, lHandY, curr.lWristAngle, curr.lFingers, true);
      drawArticulatedHand(rHandX, rHandY, curr.rWristAngle, curr.rFingers, false);

      animationFrameRef.current = requestAnimationFrame(render);
    };

    render();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [activeSignData, playbackSpeed]);

  // Voice Input Speech Recognition
  const toggleListening = () => {
    const SpeechRecognition =
      (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

    if (!SpeechRecognition) {
      alert('Speech recognition is not supported in this browser. Please use Google Chrome or Edge.');
      return;
    }

    if (isListening) {
      setIsListening(false);
      return;
    }

    try {
      const recognition = new SpeechRecognition();
      recognition.lang = 'en-IN';
      recognition.continuous = false;
      recognition.interimResults = false;

      recognition.onstart = () => setIsListening(true);
      recognition.onend = () => setIsListening(false);
      recognition.onerror = () => setIsListening(false);

      recognition.onresult = (event: any) => {
        const transcript = event.results[0][0].transcript;
        if (transcript) {
          setInputText(transcript);
        }
      };

      recognition.start();
    } catch {
      setIsListening(false);
    }
  };

  const toggleFullscreen = () => {
    if (!containerRef.current) return;
    if (!document.fullscreenElement) {
      containerRef.current.requestFullscreen().then(() => setIsFullscreen(true)).catch(() => {});
    } else {
      document.exitFullscreen().then(() => setIsFullscreen(false)).catch(() => {});
    }
  };

  return (
    <div
      ref={containerRef}
      className={`text-to-sign-container ${isFullscreen ? 'fullscreen-mode' : ''}`}
    >
      {/* Subheader Banner: Bidirectional Translation Mode Indicator */}
      <div className="bidirectional-subbar">
        <div className="bidirectional-badge">
          <Sparkles size={14} className="sparkle-icon" />
          <span>Bidirectional Mode: English Text → Indian Sign Language (ISL) Avatar</span>
        </div>
        <div className="vocab-count-indicator">
          <BookOpen size={13} />
          <span>26 Alphabets + Core ISL Vocabulary Gloss</span>
        </div>
      </div>

      {/* Main 2-Column Split Layout */}
      <div className="text-to-sign-grid">
        {/* Left Column: Interactive 3D Sign Language Avatar Viewport */}
        <section className="avatar-viewport-card" aria-label="ISL Sign Language Avatar">
          <div className="avatar-viewport-header">
            <div className="avatar-status-pill">
              <span className="avatar-live-dot" />
              <span>ISL Avatar Demonstrator</span>
            </div>

            <div className="avatar-tools">
              <button
                className="overlay-icon-btn"
                onClick={toggleFullscreen}
                title={isFullscreen ? 'Exit Fullscreen' : 'Fullscreen Avatar'}
              >
                {isFullscreen ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
              </button>
            </div>
          </div>

          {/* Canvas Rendering Area */}
          <div className="avatar-canvas-wrap">
            <canvas
              ref={canvasRef}
              width={600}
              height={440}
              className="avatar-canvas"
              aria-label="Indian Sign Language Humanoid Avatar"
            />

            {/* Persistent Subtitle Dock for Avatar Viewport */}
            <div className="avatar-sign-subtitle-dock">
              <div className="avatar-subtitle-panel">
                <span className="subtitle-category-label">CURRENT ISL GESTURE</span>
                <div className="avatar-subtitle-main">
                  <span className="active-sign-token">{activeSignData.label}</span>
                  <span className="sign-description-text">{activeSignData.desc}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Avatar Playback Controls Bar */}
          <div className="avatar-playback-bar">
            <div className="playback-buttons-left">
              <button
                type="button"
                className="playback-btn primary"
                onClick={() => setIsPlaying(!isPlaying)}
                title={isPlaying ? 'Pause animation' : 'Play animation'}
                aria-label={isPlaying ? 'Pause' : 'Play'}
              >
                {isPlaying ? <Pause size={15} /> : <Play size={15} />}
                <span>{isPlaying ? 'Pause' : 'Play'}</span>
              </button>

              <button
                type="button"
                className="playback-btn secondary"
                onClick={() => setActiveTokenIndex(0)}
                title="Replay from start"
                aria-label="Replay"
              >
                <RotateCcw size={14} />
                <span>Replay</span>
              </button>

              {/* Speed Switcher */}
              <div className="speed-pills">
                {[0.5, 1.0, 1.5].map((speed) => (
                  <button
                    key={speed}
                    type="button"
                    className={`speed-pill ${playbackSpeed === speed ? 'active' : ''}`}
                    onClick={() => setPlaybackSpeed(speed)}
                  >
                    {speed}x
                  </button>
                ))}
              </div>
            </div>

            {/* Token Step Progress */}
            <div className="gloss-progress-indicator">
              <span>Sign {activeTokenIndex + 1} of {Math.max(1, glossTokens.length)}</span>
            </div>
          </div>
        </section>

        {/* Right Column: Google Gemini / ChatGPT Styled Prompt Composer & Grammar Gloss */}
        <aside className="avatar-side-panel" aria-label="English Input and ISL Breakdown">
          {/* ISL Grammar Gloss Breakdown Card */}
          <div className="gloss-breakdown-card">
            <div className="card-header-clean">
              <span className="card-header-title">ISL Grammatical Gloss</span>
              <span className="grammar-tag">Topic-Comment Syntax</span>
            </div>

            <div className="gloss-tokens-container">
              {glossTokens.length > 0 ? (
                glossTokens.map((token, idx) => (
                  <button
                    key={`${token}-${idx}`}
                    type="button"
                    className={`gloss-token-pill ${idx === activeTokenIndex ? 'active' : ''}`}
                    onClick={() => setActiveTokenIndex(idx)}
                    title={`Click to show gesture for ${token}`}
                  >
                    <span className="token-index">#{idx + 1}</span>
                    <span className="token-text">{token}</span>
                  </button>
                ))
              ) : (
                <span className="gloss-empty-hint">Type English words to generate ISL signs...</span>
              )}
            </div>
          </div>

          {/* Fingerspelling Letter Rail */}
          <div className="fingerspelling-rail-card">
            <div className="card-header-clean">
              <span className="card-header-title">ISL Fingerspelling Sequence</span>
              <span className="card-sub-hint">Click any letter to demonstrate</span>
            </div>

            <div className="letters-scroll-row">
              {inputText.replace(/[^a-zA-Z]/g, '').split('').map((char, cIdx) => {
                const upper = char.toUpperCase();
                const isActive = glossTokens[activeTokenIndex] === upper;
                return (
                  <button
                    key={`${upper}-${cIdx}`}
                    type="button"
                    className={`letter-rail-box ${isActive ? 'active-letter' : ''}`}
                    onClick={() => {
                      const foundIdx = glossTokens.indexOf(upper);
                      if (foundIdx !== -1) setActiveTokenIndex(foundIdx);
                    }}
                  >
                    <span className="rail-char">{upper}</span>
                    <span className="rail-sign-type">ISL</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Quick Suggestion Chips */}
          <div className="quick-chips-card">
            <span className="chips-label">Quick Phrases:</span>
            <div className="chips-wrap">
              {[
                'Namaste',
                'Hello',
                'Help',
                'Water',
                'Food',
                'Thank you',
                'Good morning',
                'Please',
                'Welcome',
                'B D P Q T',
              ].map((phrase) => (
                <button
                  key={phrase}
                  type="button"
                  className="quick-chip-btn"
                  onClick={() => setInputText(phrase)}
                >
                  {phrase}
                </button>
              ))}
            </div>
          </div>

          {/* Gemini / ChatGPT Style Prompt Composer Box */}
          <div className="gemini-composer-container">
            <div className="composer-inner-box">
              <textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Type English words or sentences to translate to ISL signs..."
                rows={2}
                className="gemini-textarea"
                aria-label="English text input"
              />

              <div className="composer-bottom-bar">
                <div className="composer-left-actions">
                  <button
                    type="button"
                    className={`composer-icon-btn ${isListening ? 'active-listening' : ''}`}
                    onClick={toggleListening}
                    title={isListening ? 'Stop Listening' : 'Voice Input (Speak in English)'}
                    aria-label="Voice input"
                  >
                    {isListening ? <MicOff size={16} /> : <Mic size={16} />}
                  </button>

                  {onSpeak && (
                    <button
                      type="button"
                      className={`composer-icon-btn ${isSpeaking ? 'active-speaking' : ''}`}
                      onClick={() => onSpeak(inputText)}
                      title="Speak English Text"
                      aria-label="Speak text"
                      disabled={!inputText.trim()}
                    >
                      <Volume2 size={16} />
                    </button>
                  )}
                </div>

                <div className="composer-right-actions">
                  {inputText && (
                    <button
                      type="button"
                      className="composer-clear-btn"
                      onClick={() => setInputText('')}
                    >
                      Clear
                    </button>
                  )}

                  <button
                    type="button"
                    className="composer-send-btn"
                    onClick={() => {
                      // Trigger play from start
                      setActiveTokenIndex(0);
                      setIsPlaying(true);
                    }}
                    title="Translate to ISL Signs"
                    aria-label="Translate"
                  >
                    <Send size={14} />
                    <span>Translate</span>
                  </button>
                </div>
              </div>
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
};
