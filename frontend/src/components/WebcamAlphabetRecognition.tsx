import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  Camera,
  Play,
  Square,
  Pause,
  Activity,
  Cpu,
  Layers,
  FlipHorizontal,
  AlertCircle,
  Sparkles,
  Type,
  CornerDownLeft,
  Delete,
  Trash2,
  Copy,
  Check,
  Clock,
} from 'lucide-react';
import {
  analyzeHandFraming,
  type HandFramingAnalysis,
} from '../utils/landmarkProcessor';
import { useTextBuffer } from '../hooks/useTextBuffer';

interface TopCandidate {
  letter: string;
  confidence: number;
}

interface PredictResponse {
  letter: string;
  confidence: number;
  top_3: TopCandidate[];
}

interface WebcamAlphabetRecognitionProps {
  apiBaseUrl?: string;
  onLetterDetected?: (letter: string, confidence: number) => void;
  className?: string;
}

// Landmark connection indices matching MediaPipe Hand Landmark Topology
const HAND_CONNECTIONS: [number, number][] = [
  // Thumb
  [0, 1], [1, 2], [2, 3], [3, 4],
  // Index finger
  [0, 5], [5, 6], [6, 7], [7, 8],
  // Middle finger
  [0, 9], [9, 10], [10, 11], [11, 12],
  // Ring finger
  [0, 13], [13, 14], [14, 15], [15, 16],
  // Pinky
  [0, 17], [17, 18], [18, 19], [19, 20],
  // Palm base
  [5, 9], [9, 13], [13, 17],
];

export const WebcamAlphabetRecognition: React.FC<WebcamAlphabetRecognitionProps> = ({
  apiBaseUrl = 'http://localhost:8000',
  onLetterDetected,
  className = '',
}) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Streaming & Camera state
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [isPaused, setIsPaused] = useState<boolean>(false);
  const [isMirrored, setIsMirrored] = useState<boolean>(true);
  const [showSkeleton, setShowSkeleton] = useState<boolean>(true);
  const [showHandGuide, setShowHandGuide] = useState<boolean>(true);
  const [dominantHand, setDominantHand] = useState<'right' | 'left'>('right');
  const [numHandsDetected, setNumHandsDetected] = useState<number>(0);
  const [framingInfo, setFramingInfo] = useState<HandFramingAnalysis | null>(null);
  const [isHandDetected, setIsHandDetected] = useState<boolean>(false);
  const [statusMessage, setStatusMessage] = useState<string>('Camera is idle. Click Start Camera to begin.');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Prediction output state
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [backendStatus, setBackendStatus] = useState<'checking' | 'connected' | 'disconnected'>('checking');

  // Text Buffer Hook for continuous sign accumulation
  const {
    text: bufferText,
    currentCandidate,
    candidateConfidence,
    holdProgress,
    isHolding,
    isLockedOut,
    processPrediction,
    resetHold,
    manualSpace,
    manualBackspace,
    clearBuffer,
  } = useTextBuffer({
    holdDurationMs: 380,
    confidenceThreshold: 0.45,
    lockoutDurationMs: 300,
  });

  const [copied, setCopied] = useState<boolean>(false);

  const handleCopyText = useCallback(() => {
    if (!bufferText) return;
    navigator.clipboard.writeText(bufferText);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }, [bufferText]);

  // Runtime references
  const streamRef = useRef<MediaStream | null>(null);
  const handsRef = useRef<any>(null);
  const animFrameIdRef = useRef<number | null>(null);
  const isRequestPendingRef = useRef<boolean>(false);
  const lastRequestTimeRef = useRef<number>(0);
  const isMountedRef = useRef<boolean>(true);

  // Check backend health
  const checkHealth = useCallback(async () => {
    try {
      const res = await fetch(`${apiBaseUrl}/health`, { signal: AbortSignal.timeout(3000) });
      if (res.ok) {
        setBackendStatus('connected');
      } else {
        setBackendStatus('disconnected');
      }
    } catch {
      setBackendStatus('disconnected');
    }
  }, [apiBaseUrl]);

  useEffect(() => {
    const timer = setTimeout(() => {
      checkHealth();
    }, 50);
    const interval = setInterval(checkHealth, 10000);
    return () => {
      clearTimeout(timer);
      clearInterval(interval);
    };
  }, [checkHealth]);

  // Draw landmarks, skeletons for both hands, and visual positioning guide
  const drawCanvas = useCallback(
    (
      allHands: Array<{ landmarks: Array<{ x: number; y: number; z?: number }>; isLeft: boolean }>,
      width: number,
      height: number,
      framing: HandFramingAnalysis | null
    ) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      ctx.clearRect(0, 0, width, height);

      // 1. Draw Visual Positioning Stencil / Silhouette Guide
      if (showHandGuide) {
        ctx.save();
        // Target hand positioning guide zone
        const gw = width * 0.44;
        const gh = height * 0.68;
        const gx = isMirrored ? width * 0.06 : width * 0.50;
        const gy = height * 0.16;

        let guideColor = 'rgba(255, 255, 255, 0.25)';
        let glowColor = 'transparent';
        let badgeText = 'Position Hand in Frame';
        let badgeBg = 'rgba(15, 23, 42, 0.75)';

        if (framing) {
          if (framing.isIdeal) {
            guideColor = '#10B981';
            glowColor = 'rgba(16, 185, 129, 0.45)';
            badgeText = '✨ Optimal Framing & Distance';
            badgeBg = 'rgba(6, 78, 59, 0.9)';
          } else if (framing.distanceStatus === 'too_far') {
            guideColor = '#F59E0B';
            glowColor = 'rgba(245, 158, 11, 0.35)';
            badgeText = '📏 Move Closer to Camera';
            badgeBg = 'rgba(120, 53, 15, 0.9)';
          } else if (framing.distanceStatus === 'too_close') {
            guideColor = '#F59E0B';
            glowColor = 'rgba(245, 158, 11, 0.35)';
            badgeText = '⬅️ Move Slightly Back';
            badgeBg = 'rgba(120, 53, 15, 0.9)';
          } else if (framing.framingStatus === 'off_edge') {
            guideColor = '#EF4444';
            glowColor = 'rgba(239, 68, 68, 0.35)';
            badgeText = '⚠️ Keep Hand Fully Visible';
            badgeBg = 'rgba(127, 29, 29, 0.9)';
          } else {
            guideColor = '#38BDF8';
            glowColor = 'rgba(56, 189, 248, 0.3)';
            badgeText = '🎯 Center Hand in Target Zone';
            badgeBg = 'rgba(12, 74, 110, 0.9)';
          }
        }

        ctx.strokeStyle = guideColor;
        ctx.lineWidth = framing?.isIdeal ? 3 : 2;
        if (glowColor !== 'transparent') {
          ctx.shadowColor = glowColor;
          ctx.shadowBlur = 12;
        }
        if (!framing) {
          ctx.setLineDash([8, 8]);
        } else {
          ctx.setLineDash([]);
        }

        const r = 16;
        ctx.beginPath();
        ctx.moveTo(gx + r, gy);
        ctx.lineTo(gx + gw - r, gy);
        ctx.quadraticCurveTo(gx + gw, gy, gx + gw, gy + r);
        ctx.lineTo(gx + gw, gy + gh - r);
        ctx.quadraticCurveTo(gx + gw, gy + gh, gx + gw - r, gy + gh);
        ctx.lineTo(gx + r, gy + gh);
        ctx.quadraticCurveTo(gx, gy + gh, gx, gy + gh - r);
        ctx.lineTo(gx, gy + r);
        ctx.quadraticCurveTo(gx, gy, gx + r, gy);
        ctx.closePath();
        ctx.stroke();

        ctx.setLineDash([]);
        ctx.shadowBlur = 0;

        // Badge pill
        ctx.fillStyle = badgeBg;
        ctx.font = '600 12px Inter, sans-serif';
        const textMetrics = ctx.measureText(badgeText);
        const textPad = 12;
        const textW = textMetrics.width + textPad * 2;
        const textH = 24;
        const textX = gx + (gw - textW) / 2;
        const textY = gy - textH - 6;

        ctx.beginPath();
        if ((ctx as any).roundRect) {
          (ctx as any).roundRect(textX, textY, textW, textH, 6);
        } else {
          ctx.rect(textX, textY, textW, textH);
        }
        ctx.fill();
        ctx.strokeStyle = guideColor;
        ctx.lineWidth = 1;
        ctx.stroke();

        ctx.fillStyle = '#FFFFFF';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(badgeText, textX + textW / 2, textY + textH / 2);

        ctx.restore();
      }

      // 2. Draw Skeletons for Detected Hands (Supports 1 or 2 hands!)
      if (showSkeleton && allHands.length > 0) {
        ctx.save();
        if (isMirrored) {
          ctx.translate(width, 0);
          ctx.scale(-1, 1);
        }

        allHands.forEach((hand, hIdx) => {
          const lms = hand.landmarks;
          if (!lms || lms.length !== 21) return;

          const isPrimary = hIdx === 0;
          const strokeCol = isPrimary ? 'rgba(56, 189, 248, 0.85)' : 'rgba(168, 85, 247, 0.85)';
          const nodeCol = isPrimary ? '#38BDF8' : '#A855F7';
          const keyCol = isPrimary ? '#34D399' : '#F472B6';

          // Connections
          ctx.lineWidth = 3;
          ctx.strokeStyle = strokeCol;
          ctx.lineCap = 'round';
          ctx.lineJoin = 'round';

          for (const [startIdx, endIdx] of HAND_CONNECTIONS) {
            const p1 = lms[startIdx];
            const p2 = lms[endIdx];
            if (p1 && p2) {
              ctx.beginPath();
              ctx.moveTo(p1.x * width, p1.y * height);
              ctx.lineTo(p2.x * width, p2.y * height);
              ctx.stroke();
            }
          }

          // Landmark nodes
          for (let i = 0; i < lms.length; i++) {
            const p = lms[i];
            const x = p.x * width;
            const y = p.y * height;
            const isKeyPoint = i === 0 || i === 9;
            const radius = isKeyPoint ? 6 : 4;

            ctx.beginPath();
            ctx.arc(x, y, radius, 0, 2 * Math.PI);
            ctx.fillStyle = isKeyPoint ? keyCol : nodeCol;
            ctx.fill();
            ctx.lineWidth = 1.0;
            ctx.strokeStyle = '#0F172A';
            ctx.stroke();
          }

          // Wrist label pill
          const wrist = lms[0];
          if (wrist) {
            const wx = wrist.x * width;
            const wy = wrist.y * height;
            ctx.save();
            ctx.font = 'bold 11px Inter, sans-serif';
            ctx.textAlign = 'center';
            const handLabel = hand.isLeft ? 'Left Hand' : 'Right Hand';
            const pillW = 72;
            const pillH = 18;
            ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
            ctx.fillRect(wx - pillW / 2, wy + 8, pillW, pillH);
            ctx.strokeStyle = strokeCol;
            ctx.lineWidth = 1;
            ctx.strokeRect(wx - pillW / 2, wy + 8, pillW, pillH);
            ctx.fillStyle = nodeCol;
            ctx.fillText(handLabel, wx, wy + 21);
            ctx.restore();
          }
        });

        ctx.restore();
      }
    },
    [isMirrored, showSkeleton, showHandGuide]
  );

  // Send landmarks and bimanual context to FastAPI POST /api/predict
  const sendPredictionRequest = useCallback(
    async (
      landmarks: Array<{ x: number; y: number; z?: number }>,
      mirrorSagittal: boolean,
      secondaryLandmarks?: Array<{ x: number; y: number; z?: number }> | null
    ) => {
      if (isRequestPendingRef.current) return;

      const now = performance.now();
      // Cap at 25 FPS (~40ms interval)
      if (now - lastRequestTimeRef.current < 40) return;

      lastRequestTimeRef.current = now;
      isRequestPendingRef.current = true;

      try {
        const payload: Record<string, any> = {
          landmarks: landmarks.map((lm) => ({ x: lm.x, y: lm.y, z: lm.z ?? 0.0 })),
          dominant_hand: mirrorSagittal ? 'left' : 'right',
        };
        if (secondaryLandmarks && secondaryLandmarks.length === 21) {
          payload.secondary_landmarks = secondaryLandmarks.map((lm) => ({
            x: lm.x,
            y: lm.y,
            z: lm.z ?? 0.0,
          }));
        }

        const res = await fetch(`${apiBaseUrl}/api/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });

        if (res.ok) {
          const data: PredictResponse = await res.json();
          if (isMountedRef.current) {
            setPrediction(data);
            setErrorMessage(null);
            if (onLetterDetected && data.letter) {
              onLetterDetected(data.letter, data.confidence);
            }
            processPrediction(data.letter, data.confidence);
          }
        } else {
          const errData = await res.json().catch(() => ({}));
          console.warn('Prediction API error:', errData);
        }
      } catch (err: any) {
        if (isMountedRef.current) {
          console.error('Failed to post prediction payload:', err);
        }
      } finally {
        isRequestPendingRef.current = false;
      }
    },
    [apiBaseUrl, onLetterDetected, processPrediction]
  );

  // Handle MediaPipe onResults callback
  const onResults = useCallback(
    (results: any) => {
      if (!isMountedRef.current || isPaused) return;

      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas) return;

      const width = video.videoWidth || 640;
      const height = video.videoHeight || 480;

      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }

      if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        const numHands = results.multiHandLandmarks.length;
        setNumHandsDetected(numHands);
        setIsHandDetected(true);

        const allHands: Array<{ landmarks: Array<{ x: number; y: number; z?: number }>; isLeft: boolean }> = [];
        for (let i = 0; i < numHands; i++) {
          const lms = results.multiHandLandmarks[i];
          let isLeft = false;
          if (results.multiHandedness && i < results.multiHandedness.length) {
            const h = results.multiHandedness[i];
            const label = h?.label || h?.[0]?.label || h?.categoryName || '';
            isLeft = label.toLowerCase() === 'left';
          }
          allHands.push({ landmarks: lms, isLeft });
        }

        // Active signing hand: match dominant hand preference if present, else primary detected hand
        let activeHand = allHands[0];
        let secondaryHand: { landmarks: Array<{ x: number; y: number; z?: number }>; isLeft: boolean } | undefined = allHands[1];
        if (numHands > 1) {
          const preferred = allHands.find((h) => (dominantHand === 'left' ? h.isLeft : !h.isLeft));
          if (preferred) {
            activeHand = preferred;
            secondaryHand = allHands.find((h) => h !== preferred);
          }
        }

        const framing = analyzeHandFraming(activeHand.landmarks);
        setFramingInfo(framing);

        if (numHands >= 2) {
          setStatusMessage(`2 hands in view — Bimanual ISL sign detected (${framing.guidanceMessage})`);
        } else {
          setStatusMessage(framing.guidanceMessage);
        }

        drawCanvas(allHands, width, height, framing);
        sendPredictionRequest(activeHand.landmarks, activeHand.isLeft, secondaryHand?.landmarks);
      } else {
        setNumHandsDetected(0);
        setIsHandDetected(false);
        setFramingInfo(null);
        setStatusMessage('Position hand clearly inside camera view');
        drawCanvas([], width, height, null);
        resetHold();
      }
    },
    [dominantHand, drawCanvas, isPaused, resetHold, sendPredictionRequest]
  );

  // Initialize MediaPipe Hands (with 2-hand ISL tracking enabled)
  const initMediaPipeHands = useCallback(async () => {
    try {
      const mpHands = (window as any).Hands;
      if (!mpHands) {
        throw new Error('MediaPipe Hands script not loaded yet. Retrying...');
      }

      const hands = new mpHands({
        locateFile: (file: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
      });

      hands.setOptions({
        maxNumHands: 2,
        modelComplexity: 0,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      hands.onResults(onResults);
      handsRef.current = hands;
      return hands;
    } catch (err: any) {
      console.warn('MediaPipe initialization warning:', err);
      return null;
    }
  }, [onResults]);

  // Frame processing loop using requestVideoFrameCallback or requestAnimationFrame
  const startFrameLoop = useCallback(
    (handsInstance: any) => {
      const process = async () => {
        if (!isMountedRef.current || !isRunning) return;

        const video = videoRef.current;
        if (video && video.readyState >= 2 && !isPaused && handsInstance) {
          try {
            await handsInstance.send({ image: video });
          } catch {
            // suppress per-frame transient render error
          }
        }

        if (video && 'requestVideoFrameCallback' in video) {
          (video as any).requestVideoFrameCallback(process);
        } else {
          animFrameIdRef.current = requestAnimationFrame(process);
        }
      };

      process();
    },
    [isPaused, isRunning]
  );

  // Start Camera
  const startCamera = async () => {
    setErrorMessage(null);
    setStatusMessage('Accessing webcam...');

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
        audio: false,
      });

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      let hands = handsRef.current;
      if (!hands) {
        hands = await initMediaPipeHands();
      }

      setIsRunning(true);
      setIsPaused(false);
      setStatusMessage('Tracking active — Show alphabet hand sign');
      startFrameLoop(hands);
    } catch (err: any) {
      console.error('Camera start error:', err);
      setErrorMessage(`Failed to open camera: ${err.message || err.name}`);
      setStatusMessage('Camera access failed.');
    }
  };

  // Stop Camera
  const stopCamera = useCallback(() => {
    if (animFrameIdRef.current) {
      cancelAnimationFrame(animFrameIdRef.current);
      animFrameIdRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      if (ctx) ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }

    setIsRunning(false);
    setIsPaused(false);
    setIsHandDetected(false);
    setPrediction(null);
    resetHold();
    setStatusMessage('Camera stopped.');
  }, [resetHold]);

  // Pause / Resume
  const togglePause = () => {
    setIsPaused((prev) => {
      const next = !prev;
      if (next) {
        resetHold();
      }
      return next;
    });
  };

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      stopCamera();
    };
  }, [stopCamera]);

  const confPercent = prediction ? Math.round(prediction.confidence * 100) : 0;

  return (
    <div className={`webcam-alphabet-root ${className}`} style={{
      display: 'flex',
      flexDirection: 'column',
      gap: '16px',
      background: 'rgba(15, 23, 42, 0.75)',
      backdropFilter: 'blur(16px)',
      border: '1px solid rgba(255, 255, 255, 0.08)',
      borderRadius: '16px',
      padding: '20px',
      boxShadow: '0 20px 40px -15px rgba(0, 0, 0, 0.5)',
      color: '#F8FAFC',
      fontFamily: 'Inter, system-ui, sans-serif',
    }}>
      {/* Top Header Strip */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '12px',
        borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
        paddingBottom: '12px',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{
            background: '#24252B',
            borderRadius: '10px',
            border: '1px solid #3A3B42',
            padding: '8px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 1px 4px rgba(0, 0, 0, 0.3)',
          }}>
            <Sparkles size={18} color="#EDEDED" />
          </div>
          <div>
            <h3 style={{ margin: 0, fontSize: '16px', fontWeight: 700, letterSpacing: '-0.3px' }}>
              RealSign 26-Class Live Recognition
            </h3>
            <span style={{ fontSize: '12px', color: '#94A3B8' }}>
              FastAPI Endpoint • MediaPipe Browser Hand Tracking (63-D Invariant Vector)
            </span>
          </div>
        </div>

        {/* Backend & Performance Pills */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            fontSize: '11px',
            fontWeight: 600,
            padding: '4px 10px',
            borderRadius: '20px',
            background: backendStatus === 'connected' ? 'rgba(78, 159, 118, 0.15)' : 'rgba(239, 68, 68, 0.15)',
            border: `1px solid ${backendStatus === 'connected' ? 'rgba(78, 159, 118, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`,
            color: backendStatus === 'connected' ? '#4E9F76' : '#EF4444',
          }}>
            <span style={{
              width: '6px',
              height: '6px',
              borderRadius: '50%',
              backgroundColor: backendStatus === 'connected' ? '#4E9F76' : '#EF4444',
            }} />
            <span>API: {apiBaseUrl.replace('http://', '')}</span>
          </div>

          {isRunning && (
            <div style={{
              fontSize: '11px',
              fontWeight: 600,
              padding: '4px 8px',
              borderRadius: '6px',
              background: isHandDetected ? 'rgba(255, 255, 255, 0.08)' : 'rgba(255, 255, 255, 0.04)',
              color: isHandDetected ? '#EDEDED' : '#94A3B8',
            }}>
              {isHandDetected ? 'Tracking hand' : 'Awaiting hand'}
            </div>
          )}
        </div>
      </div>

      {/* Main Split Layout: Camera Viewport (Left) + Recognition Output Panel (Right) */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'minmax(300px, 1.4fr) minmax(260px, 1fr)',
        gap: '20px',
      }}>
        {/* Left: Video & Canvas Container */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          <div style={{
            position: 'relative',
            width: '100%',
            aspectRatio: '4 / 3',
            background: '#0B1120',
            borderRadius: '12px',
            overflow: 'hidden',
            border: isHandDetected ? '1px solid rgba(56, 189, 248, 0.5)' : '1px solid rgba(255, 255, 255, 0.1)',
            boxShadow: isHandDetected ? '0 0 20px rgba(56, 189, 248, 0.2)' : 'none',
            transition: 'border-color 0.2s ease, box-shadow 0.2s ease',
          }}>
            <video
              ref={videoRef}
              style={{
                width: '100%',
                height: '100%',
                objectFit: 'cover',
                transform: isMirrored ? 'scaleX(-1)' : 'none',
              }}
              playsInline
              muted
              autoPlay
            />
            <canvas
              ref={canvasRef}
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                pointerEvents: 'none',
              }}
            />

            {/* Offline / Idle overlay */}
            {!isRunning && (
              <div style={{
                position: 'absolute',
                inset: 0,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '12px',
                background: 'rgba(11, 17, 32, 0.85)',
              }}>
                <Camera size={44} color="#64748B" />
                <span style={{ fontSize: '14px', color: '#94A3B8' }}>Camera is currently off</span>
                <button
                  onClick={startCamera}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    padding: '8px 18px',
                    borderRadius: '8px',
                    background: '#EDEDED',
                    color: '#121316',
                    border: '1px solid #FFFFFF',
                    fontWeight: 600,
                    cursor: 'pointer',
                    boxShadow: '0 2px 8px rgba(0, 0, 0, 0.25)',
                  }}
                >
                  <Play size={16} />
                  <span>Start Camera</span>
                </button>
              </div>
            )}

            {/* Real-time Hand Positioning & Multi-hand Guide Overlay */}
            {isRunning && (
              <div style={{
                position: 'absolute',
                top: '12px',
                left: '12px',
                right: '12px',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                pointerEvents: 'none',
                zIndex: 10,
              }}>
                {/* Hand Count & Tracking Indicator */}
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  {numHandsDetected > 0 ? (
                    <div style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px',
                      background: 'rgba(15, 23, 42, 0.85)',
                      backdropFilter: 'blur(8px)',
                      padding: '4px 10px',
                      borderRadius: '20px',
                      border: '1px solid rgba(255, 255, 255, 0.15)',
                      fontSize: '11px',
                      fontWeight: 600,
                      color: '#F8FAFC',
                    }}>
                      <span style={{
                        width: '8px',
                        height: '8px',
                        borderRadius: '50%',
                        backgroundColor: numHandsDetected === 2 ? '#A855F7' : '#38BDF8',
                        boxShadow: `0 0 8px ${numHandsDetected === 2 ? '#A855F7' : '#38BDF8'}`,
                      }} />
                      <span>{numHandsDetected === 2 ? '2 Hands Active (Dual-Hand ISL)' : `1 Hand (${dominantHand === 'right' ? 'Right' : 'Left'})`}</span>
                    </div>
                  ) : (
                    <div style={{
                      background: 'rgba(15, 23, 42, 0.75)',
                      backdropFilter: 'blur(6px)',
                      padding: '4px 10px',
                      borderRadius: '20px',
                      border: '1px solid rgba(255, 255, 255, 0.1)',
                      fontSize: '11px',
                      color: '#94A3B8',
                    }}>
                      Place hand inside stencil
                    </div>
                  )}
                </div>

                {/* Framing & Distance Guidance Pill */}
                {framingInfo && (
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    background: framingInfo.isIdeal
                      ? 'rgba(16, 185, 129, 0.2)'
                      : framingInfo.distanceStatus === 'too_close' || framingInfo.framingStatus === 'reposition'
                      ? 'rgba(245, 158, 11, 0.2)'
                      : 'rgba(56, 189, 248, 0.2)',
                    backdropFilter: 'blur(8px)',
                    border: `1px solid ${
                      framingInfo.isIdeal
                        ? 'rgba(16, 185, 129, 0.4)'
                        : framingInfo.distanceStatus === 'too_close' || framingInfo.framingStatus === 'reposition'
                        ? 'rgba(245, 158, 11, 0.4)'
                        : 'rgba(56, 189, 248, 0.4)'
                    }`,
                    borderRadius: '20px',
                    padding: '4px 10px',
                    fontSize: '11px',
                    fontWeight: 600,
                    color: framingInfo.isIdeal
                      ? '#10B981'
                      : framingInfo.distanceStatus === 'too_close' || framingInfo.framingStatus === 'reposition'
                      ? '#F59E0B'
                      : '#38BDF8',
                  }}>
                    <span>{framingInfo.guidanceMessage}</span>
                  </div>
                )}
              </div>
            )}

            {/* Live Subtitle Overlay */}
            {isRunning && prediction && isHandDetected && (
              <div style={{
                position: 'absolute',
                bottom: '12px',
                left: '50%',
                transform: 'translateX(-50%)',
                background: 'rgba(15, 23, 42, 0.85)',
                backdropFilter: 'blur(8px)',
                border: '1px solid rgba(56, 189, 248, 0.4)',
                borderRadius: '8px',
                padding: '6px 16px',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                pointerEvents: 'none',
              }}>
                <span style={{ fontSize: '18px', fontWeight: 800, color: '#EDEDED' }}>
                  LETTER {prediction.letter}
                </span>
                <span style={{ fontSize: '12px', color: '#CBD5E1', fontWeight: 600 }}>
                  {confPercent}%
                </span>
              </div>
            )}
          </div>

          {/* Under-Camera Controls Bar */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: '8px',
          }}>
            <div style={{ display: 'flex', gap: '8px' }}>
              {isRunning ? (
                <>
                  <button
                    onClick={stopCamera}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px',
                      padding: '7px 14px',
                      borderRadius: '8px',
                      background: 'rgba(239, 68, 68, 0.2)',
                      color: '#EF4444',
                      border: '1px solid rgba(239, 68, 68, 0.4)',
                      fontWeight: 600,
                      fontSize: '12px',
                      cursor: 'pointer',
                    }}
                  >
                    <Square size={14} />
                    <span>Stop</span>
                  </button>
                  <button
                    onClick={togglePause}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px',
                      padding: '7px 14px',
                      borderRadius: '8px',
                      background: 'rgba(255, 255, 255, 0.08)',
                      color: '#F8FAFC',
                      border: '1px solid rgba(255, 255, 255, 0.15)',
                      fontWeight: 600,
                      fontSize: '12px',
                      cursor: 'pointer',
                    }}
                  >
                    {isPaused ? <Play size={14} /> : <Pause size={14} />}
                    <span>{isPaused ? 'Resume' : 'Pause'}</span>
                  </button>
                </>
              ) : (
                <button
                  onClick={startCamera}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '6px',
                    padding: '7px 14px',
                    borderRadius: '8px',
                    background: '#EDEDED',
                    color: '#121316',
                    border: '1px solid #FFFFFF',
                    fontWeight: 600,
                    fontSize: '12px',
                    cursor: 'pointer',
                  }}
                >
                  <Play size={14} />
                  <span>Start Camera</span>
                </button>
              )}
            </div>

            <div style={{ display: 'flex', gap: '8px' }}>
              <button
                onClick={() => setShowHandGuide((g) => !g)}
                title="Toggle visual hand positioning stencil guide"
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                  padding: '6px 10px',
                  borderRadius: '6px',
                  background: showHandGuide ? 'rgba(56, 189, 248, 0.18)' : 'rgba(255, 255, 255, 0.05)',
                  color: showHandGuide ? '#38BDF8' : '#94A3B8',
                  border: `1px solid ${showHandGuide ? 'rgba(56, 189, 248, 0.35)' : 'rgba(255, 255, 255, 0.1)'}`,
                  fontSize: '12px',
                  cursor: 'pointer',
                }}
              >
                <Sparkles size={14} />
                <span>Guide</span>
              </button>

              <button
                onClick={() => setDominantHand((d) => (d === 'right' ? 'left' : 'right'))}
                title="Switch dominant tracking hand priority"
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                  padding: '6px 10px',
                  borderRadius: '6px',
                  background: 'rgba(255, 255, 255, 0.05)',
                  color: '#CBD5E1',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  fontSize: '12px',
                  cursor: 'pointer',
                }}
              >
                <Type size={14} />
                <span>Hand: {dominantHand === 'right' ? 'R' : 'L'}</span>
              </button>

              <button
                onClick={() => setIsMirrored((m) => !m)}
                title="Toggle camera mirror"
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                  padding: '6px 10px',
                  borderRadius: '6px',
                  background: isMirrored ? 'rgba(255, 255, 255, 0.12)' : 'rgba(255, 255, 255, 0.05)',
                  color: isMirrored ? '#EDEDED' : '#94A3B8',
                  border: '1px solid rgba(255, 255, 255, 0.12)',
                  fontSize: '12px',
                  cursor: 'pointer',
                }}
              >
                <FlipHorizontal size={14} />
                <span>Mirror</span>
              </button>

              <button
                onClick={() => setShowSkeleton((s) => !s)}
                title="Toggle 21 hand landmarks overlay"
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                  padding: '6px 10px',
                  borderRadius: '6px',
                  background: showSkeleton ? 'rgba(78, 159, 118, 0.18)' : 'rgba(255, 255, 255, 0.05)',
                  color: showSkeleton ? '#4E9F76' : '#94A3B8',
                  border: `1px solid ${showSkeleton ? 'rgba(78, 159, 118, 0.35)' : 'rgba(255, 255, 255, 0.1)'}`,
                  fontSize: '12px',
                  cursor: 'pointer',
                }}
              >
                <Layers size={14} />
                <span>Skeleton</span>
              </button>
            </div>
          </div>

          {/* Status Message */}
          <div style={{
            fontSize: '12px',
            color: errorMessage ? '#EF4444' : '#94A3B8',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
          }}>
            {errorMessage ? <AlertCircle size={14} /> : <Activity size={14} />}
            <span>{errorMessage || statusMessage}</span>
          </div>

          {/* Live Hold Progress Bar */}
          <div
            style={{
              background: 'rgba(2, 6, 23, 0.45)',
              border: isLockedOut
                ? '1px solid rgba(245, 158, 11, 0.3)'
                : isHolding
                ? '1px solid rgba(255, 255, 255, 0.25)'
                : '1px solid rgba(255, 255, 255, 0.08)',
              borderRadius: '10px',
              padding: '10px 14px',
              display: 'flex',
              flexDirection: 'column',
              gap: '8px',
              transition: 'border-color 0.2s ease',
            }}
          >
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                fontSize: '12px',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Clock
                  size={13}
                  color={isLockedOut ? '#F59E0B' : isHolding ? '#EDEDED' : '#94A3B8'}
                />
                <span
                  style={{
                    fontWeight: 600,
                    color: isLockedOut ? '#FBBF24' : isHolding ? '#EDEDED' : '#CBD5E1',
                  }}
                >
                  {isLockedOut
                    ? 'Cooldown Lockout (400ms)'
                    : isHolding && currentCandidate
                    ? `Holding "${currentCandidate}" (${Math.round(candidateConfidence * 100)}% conf)`
                    : 'Hold sign for 600ms (≥85% conf) to type'}
                </span>
              </div>
              <span
                style={{
                  fontSize: '11px',
                  fontWeight: 700,
                  color: isLockedOut ? '#F59E0B' : isHolding ? '#EDEDED' : '#64748B',
                }}
              >
                {isLockedOut ? 'LOCKED' : `${Math.round(holdProgress)}%`}
              </span>
            </div>

            {/* Progress track */}
            <div
              style={{
                width: '100%',
                height: '7px',
                background: 'rgba(255, 255, 255, 0.06)',
                borderRadius: '4px',
                overflow: 'hidden',
              }}
            >
              <div
                style={{
                  width: isLockedOut ? '100%' : `${holdProgress}%`,
                  height: '100%',
                  background: isLockedOut
                    ? 'linear-gradient(90deg, #F59E0B, #D97706)'
                    : holdProgress >= 100
                    ? '#4E9F76'
                    : '#EDEDED',
                  boxShadow: 'none',
                  borderRadius: '4px',
                  transition: isLockedOut ? 'none' : 'width 0.05s linear',
                }}
              />
            </div>
          </div>

          {/* Recognized Text Buffer Box */}
          <div
            style={{
              background: 'rgba(2, 6, 23, 0.55)',
              border: '1px solid rgba(255, 255, 255, 0.08)',
              borderRadius: '12px',
              padding: '14px',
              display: 'flex',
              flexDirection: 'column',
              gap: '10px',
            }}
          >
            {/* Buffer Header with Title and Control Buttons */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                flexWrap: 'wrap',
                gap: '8px',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Type size={16} color="#CBD5E1" />
                <span style={{ fontSize: '13px', fontWeight: 700, color: '#F1F5F9' }}>
                  Sentence Buffer
                </span>
                <span
                  style={{
                    fontSize: '11px',
                    color: '#94A3B8',
                    background: 'rgba(255, 255, 255, 0.05)',
                    padding: '2px 8px',
                    borderRadius: '12px',
                    border: '1px solid rgba(255, 255, 255, 0.08)',
                  }}
                >
                  {bufferText.length} chars
                </span>
              </div>

              {/* Action Buttons */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <button
                  type="button"
                  onClick={manualSpace}
                  title="Insert space"
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '4px',
                    padding: '5px 9px',
                    borderRadius: '6px',
                    background: 'rgba(255, 255, 255, 0.06)',
                    color: '#E2E8F0',
                    border: '1px solid rgba(255, 255, 255, 0.12)',
                    fontSize: '11px',
                    fontWeight: 600,
                    cursor: 'pointer',
                  }}
                >
                  <CornerDownLeft size={12} />
                  <span>Space</span>
                </button>

                <button
                  type="button"
                  onClick={manualBackspace}
                  disabled={bufferText.length === 0}
                  title="Delete last character"
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '4px',
                    padding: '5px 9px',
                    borderRadius: '6px',
                    background: 'rgba(255, 255, 255, 0.06)',
                    color: bufferText.length === 0 ? '#64748B' : '#E2E8F0',
                    border: '1px solid rgba(255, 255, 255, 0.12)',
                    fontSize: '11px',
                    fontWeight: 600,
                    cursor: bufferText.length === 0 ? 'not-allowed' : 'pointer',
                    opacity: bufferText.length === 0 ? 0.5 : 1,
                  }}
                >
                  <Delete size={12} />
                  <span>Backspace</span>
                </button>

                <button
                  type="button"
                  onClick={clearBuffer}
                  disabled={bufferText.length === 0}
                  title="Clear text buffer"
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '4px',
                    padding: '5px 9px',
                    borderRadius: '6px',
                    background: 'rgba(239, 68, 68, 0.1)',
                    color: bufferText.length === 0 ? '#64748B' : '#F87171',
                    border: '1px solid rgba(239, 68, 68, 0.2)',
                    fontSize: '11px',
                    fontWeight: 600,
                    cursor: bufferText.length === 0 ? 'not-allowed' : 'pointer',
                    opacity: bufferText.length === 0 ? 0.5 : 1,
                  }}
                >
                  <Trash2 size={12} />
                  <span>Clear</span>
                </button>

                <button
                  type="button"
                  onClick={handleCopyText}
                  disabled={bufferText.length === 0}
                  title="Copy text to clipboard"
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '4px',
                    padding: '5px 9px',
                    borderRadius: '6px',
                    background: copied ? 'rgba(78, 159, 118, 0.18)' : 'rgba(255, 255, 255, 0.06)',
                    color: copied ? '#4E9F76' : bufferText.length === 0 ? '#64748B' : '#EDEDED',
                    border: `1px solid ${
                      copied ? 'rgba(78, 159, 118, 0.35)' : 'rgba(255, 255, 255, 0.12)'
                    }`,
                    fontSize: '11px',
                    fontWeight: 600,
                    cursor: bufferText.length === 0 ? 'not-allowed' : 'pointer',
                    opacity: bufferText.length === 0 ? 0.5 : 1,
                  }}
                >
                  {copied ? <Check size={12} /> : <Copy size={12} />}
                  <span>{copied ? 'Copied' : 'Copy'}</span>
                </button>
              </div>
            </div>

            {/* Display Area */}
            <div
              style={{
                minHeight: '72px',
                maxHeight: '120px',
                overflowY: 'auto',
                background: 'rgba(15, 23, 42, 0.6)',
                borderRadius: '8px',
                border: '1px solid rgba(255, 255, 255, 0.08)',
                padding: '10px 12px',
                fontSize: '16px',
                fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
                letterSpacing: '0.5px',
                color: '#F8FAFC',
                wordBreak: 'break-word',
                display: 'flex',
                alignItems: 'flex-start',
              }}
            >
              {bufferText ? (
                <span>
                  {bufferText}
                  <span
                    style={{
                      display: 'inline-block',
                      width: '2px',
                      height: '18px',
                      background: '#EDEDED',
                      marginLeft: '3px',
                      verticalAlign: 'text-bottom',
                    }}
                  />
                </span>
              ) : (
                <span style={{ color: '#64748B', fontStyle: 'italic', fontSize: '13px' }}>
                  Hold letter gesture for 600ms (≥85% conf) to accumulate words and sentences...
                </span>
              )}
            </div>
          </div>
        </div>

        {/* Right: RealSign Alphabet Recognition Output Cards */}
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          gap: '16px',
          background: 'rgba(2, 6, 23, 0.4)',
          borderRadius: '12px',
          padding: '16px',
          border: '1px solid rgba(255, 255, 255, 0.06)',
        }}>
          {/* Main Detected Letter Display */}
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '24px 16px',
            background: 'linear-gradient(180deg, rgba(30, 41, 59, 0.4) 0%, rgba(15, 23, 42, 0.6) 100%)',
            border: '1px solid rgba(255, 255, 255, 0.08)',
            borderRadius: '10px',
            textAlign: 'center',
          }}>
            <span style={{ fontSize: '11px', textTransform: 'uppercase', letterSpacing: '1px', color: '#94A3B8' }}>
              Recognized Alphabet Letter
            </span>

            {prediction && isHandDetected ? (
              <>
                <div style={{
                  fontSize: '64px',
                  fontWeight: 900,
                  color: '#EDEDED',
                  lineHeight: '1.1',
                  margin: '8px 0',
                }}>
                  {prediction.letter}
                </div>

                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  marginTop: '4px',
                }}>
                  <span style={{
                    fontSize: '13px',
                    fontWeight: 700,
                    padding: '3px 10px',
                    borderRadius: '20px',
                    background: confPercent > 70 ? 'rgba(78, 159, 118, 0.2)' : 'rgba(217, 119, 6, 0.2)',
                    color: confPercent > 70 ? '#4E9F76' : '#D97706',
                    border: `1px solid ${confPercent > 70 ? 'rgba(78, 159, 118, 0.4)' : 'rgba(217, 119, 6, 0.4)'}`,
                  }}>
                    {confPercent}% Match
                  </span>
                  <span style={{ fontSize: '12px', color: '#94A3B8' }}>
                    ISL RealSign MLP
                  </span>
                </div>
              </>
            ) : (
              <div style={{ margin: '20px 0', color: '#64748B', fontSize: '14px' }}>
                {isRunning ? 'Awaiting Hand Gesture...' : 'Start Camera to Detect'}
              </div>
            )}
          </div>

          {/* Top 3 Probability Badges */}
          <div>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              marginBottom: '10px',
            }}>
              <span style={{ fontSize: '12px', fontWeight: 600, color: '#CBD5E1' }}>
                Top 3 Candidate Probabilities
              </span>
              <span style={{ fontSize: '11px', color: '#64748B' }}>
                Softmax Distribution
              </span>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {prediction?.top_3 && prediction.top_3.length > 0 ? (
                prediction.top_3.map((cand, idx) => {
                  const pct = Math.round(cand.confidence * 100);
                  const isTop = idx === 0;
                  return (
                    <div
                      key={`${cand.letter}-${idx}`}
                      style={{
                        display: 'flex',
                        flexDirection: 'column',
                        gap: '4px',
                        background: isTop ? 'rgba(255, 255, 255, 0.08)' : 'rgba(255, 255, 255, 0.02)',
                        border: isTop ? '1px solid rgba(255, 255, 255, 0.2)' : '1px solid rgba(255, 255, 255, 0.05)',
                        borderRadius: '8px',
                        padding: '8px 12px',
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <span style={{
                            fontSize: '11px',
                            fontWeight: 700,
                            color: isTop ? '#EDEDED' : '#64748B',
                          }}>
                            #{idx + 1}
                          </span>
                          <span style={{
                            fontSize: '14px',
                            fontWeight: 700,
                            color: isTop ? '#FFFFFF' : '#CBD5E1',
                          }}>
                            Letter {cand.letter}
                          </span>
                        </div>
                        <span style={{
                          fontSize: '12px',
                          fontWeight: 700,
                          color: isTop ? '#EDEDED' : '#94A3B8',
                        }}>
                          {pct}%
                        </span>
                      </div>

                      {/* Progress bar */}
                      <div style={{
                        width: '100%',
                        height: '4px',
                        background: 'rgba(255, 255, 255, 0.08)',
                        borderRadius: '2px',
                        overflow: 'hidden',
                      }}>
                        <div style={{
                          width: `${pct}%`,
                          height: '100%',
                          background: isTop ? '#EDEDED' : '#64748B',
                          borderRadius: '2px',
                          transition: 'width 0.15s ease',
                        }} />
                      </div>
                    </div>
                  );
                })
              ) : (
                <div style={{
                  padding: '16px',
                  textAlign: 'center',
                  fontSize: '12px',
                  color: '#64748B',
                  background: 'rgba(255, 255, 255, 0.02)',
                  borderRadius: '8px',
                  border: '1px dashed rgba(255, 255, 255, 0.08)',
                }}>
                  No prediction data yet
                </div>
              )}
            </div>
          </div>

          {/* Model Specification Card */}
          <div style={{
            marginTop: 'auto',
            background: 'rgba(59, 130, 246, 0.05)',
            border: '1px solid rgba(59, 130, 246, 0.15)',
            borderRadius: '8px',
            padding: '10px 12px',
            fontSize: '11px',
            color: '#94A3B8',
            display: 'flex',
            flexDirection: 'column',
            gap: '4px',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: '#60A5FA', fontWeight: 600 }}>
              <Cpu size={13} />
              <span>Model Architecture: RealSignMLP</span>
            </div>
            <span>Input: 63-D Normalized (21 × 3)</span>
            <span>Layers: Linear(63, 128) → ReLU → Linear(128, 64) → Linear(64, 26)</span>
            <span>Classes: A through Z (26 Total)</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default WebcamAlphabetRecognition;
