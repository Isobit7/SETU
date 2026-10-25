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
}

// Hand gesture poses for Alphabet and Common Signs
const SIGN_DATABASE: Record<string, { label: string; desc: string; pose: HandPose }> = {
  A: {
    label: 'Letter A',
    desc: 'Closed fist with thumb extended upright along side of index finger.',
    pose: { x: 340, y: 260, fingers: [1, 0, 0, 0, 0], wristAngle: -0.1 },
  },
  B: {
    label: 'Letter B',
    desc: 'Flat hand held vertically with four fingers together and thumb folded across palm.',
    pose: { x: 340, y: 230, fingers: [0, 1, 1, 1, 1], wristAngle: 0 },
  },
  C: {
    label: 'Letter C',
    desc: 'Curved hand forming a distinct C-shape facing forward.',
    pose: { x: 340, y: 250, fingers: [0.6, 0.6, 0.6, 0.6, 0.6], wristAngle: -0.2 },
  },
  D: {
    label: 'Letter D',
    desc: 'Index finger pointing straight up, while thumb touches middle and ring fingers.',
    pose: { x: 340, y: 230, fingers: [0.3, 1, 0, 0, 0], wristAngle: 0 },
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
    label: 'Letter P',
    desc: 'K hand shape angled downwards towards the floor.',
    pose: { x: 340, y: 280, fingers: [0.8, 1, 0.8, 0, 0], wristAngle: 1.2 },
  },
  Q: {
    label: 'Letter Q',
    desc: 'G hand shape pointing downwards.',
    pose: { x: 340, y: 280, fingers: [0.8, 0.8, 0, 0, 0], wristAngle: 1.3 },
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
    label: 'Letter T',
    desc: 'Thumb tucked between index and middle fingers.',
    pose: { x: 340, y: 260, fingers: [0.3, 0.3, 0, 0, 0], wristAngle: 0 },
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
  HELLO: {
    label: 'Hello (Greeting)',
    desc: 'Open flat hand near temple moving outward in a friendly wave/salute.',
    pose: { x: 350, y: 190, fingers: [1, 1, 1, 1, 1], wristAngle: -0.3 },
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
    if (word === 'HELLO' || word === 'HI') {
      glossTokens.push('HELLO');
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

  // Current interpolated hand position
  const currentHandPosRef = useRef<HandPose>({
    x: 340,
    y: 240,
    fingers: [1, 1, 1, 1, 1],
    wristAngle: 0,
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

  // Animated Avatar Canvas Loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let time = 0;

    const render = () => {
      time += 0.05 * playbackSpeed;
      const targetPose = activeSignData.pose;

      // Smooth interpolation towards target pose
      const curr = currentHandPosRef.current;
      curr.x += (targetPose.x - curr.x) * 0.15;
      curr.y += (targetPose.y - curr.y) * 0.15;
      curr.wristAngle += (targetPose.wristAngle - curr.wristAngle) * 0.15;

      for (let i = 0; i < 5; i++) {
        const targetFinger = targetPose.fingers[i] ?? 1;
        curr.fingers[i] += (targetFinger - curr.fingers[i]) * 0.2;
      }

      // Clear Canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // 1. Draw Subtle Architectural Grid / Floor
      ctx.strokeStyle = '#232528';
      ctx.lineWidth = 1;
      for (let y = 380; y < canvas.height; y += 25) {
        ctx.beginPath();
        ctx.moveTo(40, y);
        ctx.lineTo(canvas.width - 40, y);
        ctx.stroke();
      }

      // 2. Avatar Center Coordinates
      const centerX = canvas.width / 2;
      const headY = 120 + Math.sin(time * 0.8) * 1.5;

      // 3. Draw Stylized Head (Avatar Silhouette)
      ctx.fillStyle = '#1E1F20';
      ctx.strokeStyle = '#323538';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(centerX, headY, 38, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      // Stylized Eye & Face Line
      ctx.fillStyle = '#8AB4F8';
      ctx.beginPath();
      ctx.arc(centerX - 10, headY - 4, 3, 0, Math.PI * 2);
      ctx.arc(centerX + 10, headY - 4, 3, 0, Math.PI * 2);
      ctx.fill();

      // 4. Draw Neck & Torso
      ctx.beginPath();
      ctx.moveTo(centerX - 12, headY + 36);
      ctx.lineTo(centerX - 12, headY + 54);
      ctx.lineTo(centerX - 65, headY + 70); // Left shoulder
      ctx.lineTo(centerX - 50, headY + 220); // Left hip
      ctx.lineTo(centerX + 50, headY + 220); // Right hip
      ctx.lineTo(centerX + 65, headY + 70); // Right shoulder
      ctx.lineTo(centerX + 12, headY + 54);
      ctx.lineTo(centerX + 12, headY + 36);
      ctx.closePath();
      ctx.fillStyle = '#1A1B1D';
      ctx.strokeStyle = '#323538';
      ctx.lineWidth = 2;
      ctx.fill();
      ctx.stroke();

      // Chest Emblem / Center Bridge Arc
      ctx.strokeStyle = 'rgba(138, 180, 248, 0.4)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(centerX, headY + 110, 24, 0.2 * Math.PI, 0.8 * Math.PI, false);
      ctx.stroke();

      // 5. Draw Right Arm (Connected to Sign Hand)
      const rightShoulderX = centerX + 65;
      const rightShoulderY = headY + 70;
      const handX = curr.x;
      const handY = curr.y + Math.sin(time * 1.5) * 3;

      const elbowX = (rightShoulderX + handX) / 2 + 25;
      const elbowY = (rightShoulderY + handY) / 2 + 30;

      ctx.strokeStyle = '#8AB4F8';
      ctx.lineWidth = 4;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      // Upper arm
      ctx.beginPath();
      ctx.moveTo(rightShoulderX, rightShoulderY);
      ctx.lineTo(elbowX, elbowY);
      ctx.stroke();

      // Forearm
      ctx.beginPath();
      ctx.moveTo(elbowX, elbowY);
      ctx.lineTo(handX, handY);
      ctx.stroke();

      // Joint circle on elbow
      ctx.fillStyle = '#1E1F20';
      ctx.strokeStyle = '#8AB4F8';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(elbowX, elbowY, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      // 6. Draw Detailed ISL Hand Pose (Palm + 5 Articulated Fingers)
      ctx.save();
      ctx.translate(handX, handY);
      ctx.rotate(curr.wristAngle);

      // Palm base
      ctx.fillStyle = '#282A2C';
      ctx.strokeStyle = '#8AB4F8';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.roundRect(-14, -14, 28, 26, 6);
      ctx.fill();
      ctx.stroke();

      // 5 Fingers
      const fingerAngles = [-0.7, -0.3, 0, 0.3, 0.6];
      const fingerLengths = [18, 24, 26, 23, 19];

      for (let f = 0; f < 5; f++) {
        const ext = curr.fingers[f];
        const angle = fingerAngles[f];
        const len = fingerLengths[f] * (0.35 + ext * 0.65);

        const startX = -10 + f * 5;
        const startY = -12;
        const endX = startX + Math.sin(angle) * len;
        const endY = startY - Math.cos(angle) * len;

        // Bone
        ctx.strokeStyle = ext > 0.6 ? '#8AB4F8' : '#5E6260';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(startX, startY);
        ctx.lineTo(endX, endY);
        ctx.stroke();

        // Joint knuckle
        ctx.fillStyle = ext > 0.6 ? '#E3E3E3' : '#323538';
        ctx.beginPath();
        ctx.arc(endX, endY, 3.5, 0, Math.PI * 2);
        ctx.fill();
      }

      ctx.restore();

      // 7. If Two-Handed Sign, Draw Left Arm & Hand
      if (targetPose.isTwoHanded) {
        const leftShoulderX = centerX - 65;
        const leftShoulderY = headY + 70;
        const lHandX = targetPose.leftHandX || 250;
        const lHandY = targetPose.leftHandY || 260;
        const lElbowX = (leftShoulderX + lHandX) / 2 - 25;
        const lElbowY = (leftShoulderY + lHandY) / 2 + 30;

        ctx.strokeStyle = '#81C995'; // Secondary calm green hand
        ctx.lineWidth = 4;
        ctx.beginPath();
        ctx.moveTo(leftShoulderX, leftShoulderY);
        ctx.lineTo(lElbowX, lElbowY);
        ctx.lineTo(lHandX, lHandY);
        ctx.stroke();

        // Left palm
        ctx.fillStyle = '#282A2C';
        ctx.strokeStyle = '#81C995';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(lHandX, lHandY, 12, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }

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
                'Hello',
                'Thank you',
                'Good morning',
                'You',
                'Please help me',
                'Welcome',
                'A B C',
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
