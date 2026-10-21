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
} from 'lucide-react';
import { normalizeHandLandmarks63D } from '../utils/landmarkProcessor';

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
  const [isHandDetected, setIsHandDetected] = useState<boolean>(false);
  const [statusMessage, setStatusMessage] = useState<string>('Camera is idle. Click Start Camera to begin.');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Prediction output state
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [fps, setFps] = useState<number>(0);
  const [latencyMs, setLatencyMs] = useState<number>(0);
  const [backendStatus, setBackendStatus] = useState<'checking' | 'connected' | 'disconnected'>('checking');

  // Runtime references
  const streamRef = useRef<MediaStream | null>(null);
  const handsRef = useRef<any>(null);
  const animFrameIdRef = useRef<number | null>(null);
  const isRequestPendingRef = useRef<boolean>(false);
  const lastRequestTimeRef = useRef<number>(0);
  const frameCountRef = useRef<number>(0);
  const lastFpsCalcTimeRef = useRef<number>(0);
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

  // Draw landmarks and skeleton on canvas
  const drawLandmarks = useCallback(
    (landmarks: Array<{ x: number; y: number; z?: number }>, width: number, height: number) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      ctx.clearRect(0, 0, width, height);

      if (!showSkeleton || !landmarks || landmarks.length !== 21) {
        return;
      }

      ctx.save();
      if (isMirrored) {
        ctx.translate(width, 0);
        ctx.scale(-1, 1);
      }

      // Draw connections
      ctx.lineWidth = 3;
      ctx.strokeStyle = 'rgba(56, 189, 248, 0.75)';
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      for (const [startIdx, endIdx] of HAND_CONNECTIONS) {
        const p1 = landmarks[startIdx];
        const p2 = landmarks[endIdx];
        if (p1 && p2) {
          ctx.beginPath();
          ctx.moveTo(p1.x * width, p1.y * height);
          ctx.lineTo(p2.x * width, p2.y * height);
          ctx.stroke();
        }
      }

      // Draw landmark nodes
      for (let i = 0; i < landmarks.length; i++) {
        const p = landmarks[i];
        const x = p.x * width;
        const y = p.y * height;

        // Wrist (0) and Middle MCP (9) highlighted in cyan/green
        const isKeyPoint = i === 0 || i === 9;
        const radius = isKeyPoint ? 6 : 4;

        ctx.beginPath();
        ctx.arc(x, y, radius, 0, 2 * Math.PI);
        ctx.fillStyle = isKeyPoint ? '#10B981' : '#38BDF8';
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = '#FFFFFF';
        ctx.stroke();
      }

      ctx.restore();
    },
    [isMirrored, showSkeleton]
  );

  // Send 63-D features to FastAPI POST /api/predict
  const sendPredictionRequest = useCallback(
    async (landmarks: Array<{ x: number; y: number; z?: number }>, isLeftHand: boolean) => {
      if (isRequestPendingRef.current) return;

      const now = performance.now();
      // Cap at 25 FPS (~40ms interval)
      if (now - lastRequestTimeRef.current < 40) return;

      lastRequestTimeRef.current = now;
      isRequestPendingRef.current = true;

      try {
        // Preprocess to 63-D normalized vector directly in browser memory
        const features63D = normalizeHandLandmarks63D(landmarks, isLeftHand);

        const startTime = performance.now();
        const res = await fetch(`${apiBaseUrl}/api/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            features: features63D,
            dominant_hand: isLeftHand ? 'left' : 'right',
          }),
        });

        const elapsed = Math.round(performance.now() - startTime);

        if (res.ok) {
          const data: PredictResponse = await res.json();
          if (isMountedRef.current) {
            setPrediction(data);
            setLatencyMs(elapsed);
            setErrorMessage(null);
            if (onLetterDetected && data.letter) {
              onLetterDetected(data.letter, data.confidence);
            }
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
    [apiBaseUrl, onLetterDetected]
  );

  // Handle MediaPipe onResults callback
  const onResults = useCallback(
    (results: any) => {
      if (!isMountedRef.current || isPaused) return;

      // Update FPS counter
      frameCountRef.current += 1;
      const now = performance.now();
      if (now - lastFpsCalcTimeRef.current >= 1000) {
        setFps(Math.round((frameCountRef.current * 1000) / (now - lastFpsCalcTimeRef.current)));
        frameCountRef.current = 0;
        lastFpsCalcTimeRef.current = now;
      }

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
        setIsHandDetected(true);
        setStatusMessage('Hand detected — Classifying letter');

        const landmarks = results.multiHandLandmarks[0];
        let isLeft = false;
        if (results.multiHandedness && results.multiHandedness.length > 0) {
          const label = results.multiHandedness[0]?.label || '';
          isLeft = label.toLowerCase() === 'left';
        }

        drawLandmarks(landmarks, width, height);
        sendPredictionRequest(landmarks, isLeft);
      } else {
        setIsHandDetected(false);
        setStatusMessage('Position hand clearly inside camera view');
        const ctx = canvas.getContext('2d');
        if (ctx) ctx.clearRect(0, 0, width, height);
      }
    },
    [drawLandmarks, isPaused, sendPredictionRequest]
  );

  // Initialize MediaPipe Hands
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
        maxNumHands: 1,
        modelComplexity: 1,
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
  const stopCamera = () => {
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
    setStatusMessage('Camera stopped.');
  };

  // Pause / Resume
  const togglePause = () => {
    setIsPaused((prev) => !prev);
  };

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      stopCamera();
    };
  }, []);

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
            background: 'linear-gradient(135deg, #0284C7 0%, #2563EB 100%)',
            borderRadius: '10px',
            padding: '8px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 4px 12px rgba(37, 99, 235, 0.35)',
          }}>
            <Sparkles size={18} color="#FFFFFF" />
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
            background: backendStatus === 'connected' ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)',
            border: `1px solid ${backendStatus === 'connected' ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`,
            color: backendStatus === 'connected' ? '#10B981' : '#EF4444',
          }}>
            <span style={{
              width: '6px',
              height: '6px',
              borderRadius: '50%',
              backgroundColor: backendStatus === 'connected' ? '#10B981' : '#EF4444',
            }} />
            <span>API: {apiBaseUrl.replace('http://', '')}</span>
          </div>

          {isRunning && (
            <>
              <div style={{
                fontSize: '11px',
                fontWeight: 600,
                padding: '4px 8px',
                borderRadius: '6px',
                background: 'rgba(255, 255, 255, 0.05)',
                color: '#38BDF8',
              }}>
                {fps} FPS
              </div>
              <div style={{
                fontSize: '11px',
                fontWeight: 600,
                padding: '4px 8px',
                borderRadius: '6px',
                background: 'rgba(255, 255, 255, 0.05)',
                color: latencyMs < 30 ? '#10B981' : '#F59E0B',
              }}>
                {latencyMs}ms
              </div>
            </>
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
                    background: '#2563EB',
                    color: '#FFF',
                    border: 'none',
                    fontWeight: 600,
                    cursor: 'pointer',
                    boxShadow: '0 4px 12px rgba(37, 99, 235, 0.4)',
                  }}
                >
                  <Play size={16} />
                  <span>Start Camera</span>
                </button>
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
                <span style={{ fontSize: '18px', fontWeight: 800, color: '#38BDF8' }}>
                  LETTER {prediction.letter}
                </span>
                <span style={{ fontSize: '12px', color: '#10B981', fontWeight: 600 }}>
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
                    background: '#2563EB',
                    color: '#FFF',
                    border: 'none',
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
                onClick={() => setIsMirrored((m) => !m)}
                title="Toggle camera mirror"
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                  padding: '6px 10px',
                  borderRadius: '6px',
                  background: isMirrored ? 'rgba(56, 189, 248, 0.2)' : 'rgba(255, 255, 255, 0.05)',
                  color: isMirrored ? '#38BDF8' : '#94A3B8',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
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
                  background: showSkeleton ? 'rgba(16, 185, 129, 0.2)' : 'rgba(255, 255, 255, 0.05)',
                  color: showSkeleton ? '#10B981' : '#94A3B8',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
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
                  color: '#38BDF8',
                  textShadow: '0 0 30px rgba(56, 189, 248, 0.4)',
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
                    background: confPercent > 70 ? 'rgba(16, 185, 129, 0.2)' : 'rgba(245, 158, 11, 0.2)',
                    color: confPercent > 70 ? '#10B981' : '#F59E0B',
                    border: `1px solid ${confPercent > 70 ? 'rgba(16, 185, 129, 0.4)' : 'rgba(245, 158, 11, 0.4)'}`,
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
                        background: isTop ? 'rgba(56, 189, 248, 0.08)' : 'rgba(255, 255, 255, 0.02)',
                        border: isTop ? '1px solid rgba(56, 189, 248, 0.25)' : '1px solid rgba(255, 255, 255, 0.05)',
                        borderRadius: '8px',
                        padding: '8px 12px',
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <span style={{
                            fontSize: '11px',
                            fontWeight: 700,
                            color: isTop ? '#38BDF8' : '#64748B',
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
                          color: isTop ? '#10B981' : '#94A3B8',
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
                          background: isTop ? '#38BDF8' : '#64748B',
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
