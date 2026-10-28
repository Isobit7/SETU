import { useEffect, useRef, useState, useCallback } from 'react';
import type { ActiveModelType, CameraStatus, HistoryItem, RecognitionPayload } from '../types';

interface UseRecognitionStreamProps {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  fpsLimit?: number;
  confidenceThreshold?: number;
  onPredictionChange?: (pred: RecognitionPayload) => void;
  isFramingGated?: boolean;
}

export function useRecognitionStream({
  videoRef,
  fpsLimit = 18,
  confidenceThreshold = 0.35,
  onPredictionChange,
  isFramingGated = false,
}: UseRecognitionStreamProps) {
  const [cameraStatus, setCameraStatus] = useState<CameraStatus>('idle');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isBackendConnected, setIsBackendConnected] = useState<boolean>(false);
  const [activeModel, setActiveModel] = useState<ActiveModelType>('realsign');
  const [currentResult, setCurrentResult] = useState<RecognitionPayload | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [selectedCameraId, setSelectedCameraId] = useState<string>('');
  const [availableCameras, setAvailableCameras] = useState<MediaDeviceInfo[]>([]);
  
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const reconnectAttemptsRef = useRef<number>(0);
  const isMountedRef = useRef<boolean>(true);
  const isStoppedManuallyRef = useRef<boolean>(false);
  const streamRef = useRef<MediaStream | null>(null);
  const frameCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const animFrameIdRef = useRef<number | null>(null);
  const lastFrameSentTimeRef = useRef<number>(0);
  const waitingForResponseRef = useRef<boolean>(false);
  const lastRecordedSignRef = useRef<string>('');
  const candidateStreakRef = useRef<{ sign: string; count: number }>({ sign: '', count: 0 });
  const lastCommittedSignRef = useRef<string>('');
  const lastCommittedTimeRef = useRef<number>(0);
  const hadIdleBaselineRef = useRef<boolean>(true);
  const isPausedRef = useRef<boolean>(false);
  const confidenceThresholdRef = useRef<number>(confidenceThreshold);
  const onPredictionChangeRef = useRef<((pred: RecognitionPayload) => void) | undefined>(onPredictionChange);
  const isFramingGatedRef = useRef<boolean>(isFramingGated);

  // Latest raw prediction ref
  const latestResultRef = useRef<RecognitionPayload | null>(null);

  useEffect(() => {
    confidenceThresholdRef.current = confidenceThreshold;
  }, [confidenceThreshold]);

  useEffect(() => {
    onPredictionChangeRef.current = onPredictionChange;
  }, [onPredictionChange]);

  useEffect(() => {
    isFramingGatedRef.current = isFramingGated;
  }, [isFramingGated]);

  // Dynamically push updated confidence threshold to active WebSocket session
  useEffect(() => {
    confidenceThresholdRef.current = confidenceThreshold;
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'set_threshold', threshold: confidenceThreshold }));
    }
  }, [confidenceThreshold]);

  // Initialize offscreen helper canvas for frame capture at full 640x480 for crisp MediaPipe landmark resolution
  useEffect(() => {
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 480;
    frameCanvasRef.current = canvas;
  }, []);

  // Fetch available cameras
  const refreshDevices = useCallback(async () => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter((d) => d.kind === 'videoinput');
      setAvailableCameras(videoDevices);
      if (videoDevices.length > 0 && !selectedCameraId) {
        setSelectedCameraId(videoDevices[0].deviceId);
      }
    } catch (err) {
      console.warn('Could not enumerate devices:', err);
    }
  }, [selectedCameraId]);

  const connectWebSocketRef = useRef<() => void>(() => {});

  // Connect WebSocket to FastAPI backend with bounded exponential backoff auto-reconnect
  const connectWebSocket = useCallback(() => {
    if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
      return;
    }

    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.hostname}:8000/api/v1/ws/recognize?model=${activeModel}&threshold=${confidenceThresholdRef.current}`;

    try {
      const ws = new WebSocket(wsUrl);
      ws.binaryType = 'arraybuffer';
      wsRef.current = ws;

      const scheduleReconnect = () => {
        if (!isMountedRef.current || isStoppedManuallyRef.current) return;
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current);
          reconnectTimeoutRef.current = null;
        }
        // Exponential backoff: starting at 1000ms, doubling up to 10000ms max
        const attempt = reconnectAttemptsRef.current;
        const delay = Math.min(1000 * Math.pow(2, attempt), 10000);
        reconnectAttemptsRef.current += 1;
        reconnectTimeoutRef.current = window.setTimeout(() => {
          if (isMountedRef.current && !isStoppedManuallyRef.current) {
            connectWebSocketRef.current();
          }
        }, delay);
      };

      ws.onopen = () => {
        if (!isMountedRef.current) {
          ws.close();
          return;
        }
        setIsBackendConnected(true);
        setErrorMessage(null);
        reconnectAttemptsRef.current = 0;
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current);
          reconnectTimeoutRef.current = null;
        }
        // Sync active model and confidence threshold
        ws.send(JSON.stringify({ action: 'set_model', model: activeModel }));
        ws.send(JSON.stringify({ action: 'set_threshold', threshold: confidenceThresholdRef.current }));
      };

      ws.onmessage = (event) => {
        waitingForResponseRef.current = false;
        try {
          const data: RecognitionPayload = JSON.parse(event.data);
          if (data.type === 'prediction') {
            // Process packet at 30 FPS inside ref and notification callback
            latestResultRef.current = data;
            if (onPredictionChangeRef.current) {
              onPredictionChangeRef.current(data);
            }

            // Immediately synchronize UI state with raw stream predictions
            setCurrentResult(data);

            // Temporal Debouncing, Confidence Hysteresis & Refractory Cooldown for Session History
            const isHand = data.hand_detected ?? false;
            const noSignProb = data.probabilities?.['no_sign'] ?? 0;
            const isKnown = data.is_known_sign ?? false;
            const label = data.label ?? '';

            // Check for idle baseline (hand dropout, no_sign label, or P(no_sign) > 0.70)
            if (!isHand || label === 'no_sign' || label === 'unknown' || label === 'Awaiting Sign Input' || noSignProb > 0.70) {
              hadIdleBaselineRef.current = true;
              candidateStreakRef.current = { sign: '', count: 0 };
            } else if (isKnown && label) {
              const requiredStreak = data.required_streak ?? (data.active_model === 'realsign' ? 2 : 2);
              const activeMinThreshold = confidenceThresholdRef.current;

              if ((data.confidence || 0) >= activeMinThreshold) {
                if (candidateStreakRef.current.sign === label) {
                  candidateStreakRef.current.count += 1;
                } else {
                  candidateStreakRef.current = { sign: label, count: 1 };
                }

                if (candidateStreakRef.current.count >= requiredStreak) {
                  const now = Date.now();
                  const timeSinceLast = now - lastCommittedTimeRef.current;
                  const isSameSign = label === lastCommittedSignRef.current;

                  // Cooldown Lockout:
                  // 1. Lock out duplicate commits of the same character for 350 ms
                  // 2. Require transition through idle baseline before allowing identical sign again
                  // 3. Allow distinct new signs to commit immediately once consecutive streak is met
                  let canCommit = false;
                  if (isSameSign) {
                    if (timeSinceLast >= 350 && hadIdleBaselineRef.current) {
                      canCommit = true;
                    }
                  } else {
                    canCommit = true;
                  }

                  if (canCommit && !isFramingGatedRef.current) {
                    lastCommittedSignRef.current = label;
                    lastCommittedTimeRef.current = now;
                    hadIdleBaselineRef.current = false;

                    const newItem: HistoryItem = {
                      id: `${Date.now()}-${Math.random().toString(36).substr(2, 4)}`,
                      timestamp: new Date(),
                      signKey: label,
                      text: data.display_name || label.toUpperCase(),
                      confidence: data.confidence || 0,
                    };
                    setHistory((prev) => [newItem, ...prev.slice(0, 49)]);
                  }
                }
              } else {
                candidateStreakRef.current = { sign: '', count: 0 };
              }
            }
          } else if (data.type === 'error') {
            setErrorMessage(data.message || 'Recognition error from backend');
          }
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e);
        }
      };

      ws.onclose = () => {
        if (!isMountedRef.current) return;
        setIsBackendConnected(false);
        waitingForResponseRef.current = false;
        wsRef.current = null;
        scheduleReconnect();
      };

      ws.onerror = () => {
        if (!isMountedRef.current) return;
        setIsBackendConnected(false);
        waitingForResponseRef.current = false;
        setErrorMessage('Cannot connect to SETU Recognition Backend at ws://localhost:8000. Reconnecting...');
        if (!wsRef.current || wsRef.current.readyState === WebSocket.CLOSED) {
          scheduleReconnect();
        }
      };
    } catch (e) {
      if (isMountedRef.current) {
        setIsBackendConnected(false);
        setErrorMessage(`WebSocket initialization error: ${e}`);
      }
    }
  }, [activeModel]);

  // Request camera and begin streaming
  const startCamera = useCallback(async () => {
    setErrorMessage(null);
    setCameraStatus('requesting');
    isStoppedManuallyRef.current = false;
    reconnectAttemptsRef.current = 0;

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setCameraStatus('no_camera');
      setErrorMessage('Your browser does not support webcam capture (navigator.mediaDevices is unavailable).');
      return;
    }

    try {
      const constraints: MediaStreamConstraints = {
        video: selectedCameraId
          ? { deviceId: { exact: selectedCameraId }, width: { ideal: 640 }, height: { ideal: 480 } }
          : { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
        audio: false,
      };

      let stream: MediaStream;
      try {
        stream = await navigator.mediaDevices.getUserMedia(constraints);
      } catch (firstErr: any) {
        if (
          selectedCameraId &&
          (firstErr.name === 'OverconstrainedError' ||
            firstErr.name === 'ConstraintNotSatisfiedError' ||
            firstErr.name === 'NotFoundError')
        ) {
          console.warn('Overconstrained camera device, falling back to default camera:', firstErr);
          stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
            audio: false,
          });
        } else {
          throw firstErr;
        }
      }
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setCameraStatus('detecting');
      isPausedRef.current = false;
      refreshDevices();

      // Ensure WebSocket connection is live
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        connectWebSocket();
      }
    } catch (err: any) {
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        setCameraStatus('permission_denied');
        setErrorMessage('Camera access was denied. Please allow camera permissions in your browser settings.');
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        setCameraStatus('no_camera');
        setErrorMessage('No camera device detected on this system.');
      } else {
        setCameraStatus('error');
        setErrorMessage(`Camera error: ${err.message || err.name}`);
      }
    }
  }, [connectWebSocket, refreshDevices, selectedCameraId, videoRef]);

  // Stop camera stream & release resources
  const stopCamera = useCallback(() => {
    isStoppedManuallyRef.current = true;
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => {
        track.stop();
        track.enabled = false;
      });
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    if (animFrameIdRef.current) {
      cancelAnimationFrame(animFrameIdRef.current);
      animFrameIdRef.current = null;
    }
    setCameraStatus('stopped');
    latestResultRef.current = null;
    setCurrentResult(null);
    waitingForResponseRef.current = false;
  }, [videoRef]);

  // Pause / Resume
  const pauseDetection = useCallback(() => {
    isPausedRef.current = true;
    setCameraStatus('paused');
  }, []);

  const resumeDetection = useCallback(() => {
    isPausedRef.current = false;
    setCameraStatus('detecting');
  }, []);

  // Reset transient buffers
  const resetSession = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'reset' }));
    }
    lastRecordedSignRef.current = '';
    candidateStreakRef.current = { sign: '', count: 0 };
    lastCommittedSignRef.current = '';
    lastCommittedTimeRef.current = 0;
    hadIdleBaselineRef.current = true;
    latestResultRef.current = null;
    setCurrentResult(null);
  }, []);

  // Clear sentence buffer on server
  const clearSentence = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'clear_sentence' }));
    }
  }, []);

  // Backspace last token/character on server
  const backspaceSentence = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'backspace' }));
    }
  }, []);

  // Clear translation history
  const clearHistory = useCallback(() => {
    setHistory([]);
  }, []);

  // Frame streaming loop
  useEffect(() => {
    const intervalMs = 1000 / fpsLimit;

    const processFrame = () => {
      const now = performance.now();
      const video = videoRef.current;
      const canvas = frameCanvasRef.current;
      const ws = wsRef.current;

      if (
        cameraStatus === 'detecting' &&
        !isPausedRef.current &&
        video &&
        video.readyState >= 2 &&
        canvas &&
        ws &&
        ws.readyState === WebSocket.OPEN
      ) {
        // Send frame if elapsed time exceeds fps interval AND (server responded or lock is stale >120ms)
        const isStaleLock = waitingForResponseRef.current && (now - lastFrameSentTimeRef.current > 120);
        if (now - lastFrameSentTimeRef.current >= intervalMs && (!waitingForResponseRef.current || isStaleLock)) {
          lastFrameSentTimeRef.current = now;
          waitingForResponseRef.current = true;

          // Preserve camera aspect ratio dynamically to prevent squishing 16:9 frames into 4:3
          if (video.videoWidth > 0 && video.videoHeight > 0) {
            const maxDim = 480;
            let targetW = 480;
            let targetH = 360;
            if (video.videoWidth >= video.videoHeight) {
              targetW = maxDim;
              targetH = Math.round((video.videoHeight / video.videoWidth) * maxDim);
            } else {
              targetH = maxDim;
              targetW = Math.round((video.videoWidth / video.videoHeight) * maxDim);
            }
            if (canvas.width !== targetW || canvas.height !== targetH) {
              canvas.width = targetW;
              canvas.height = targetH;
            }
          }

          const ctx = canvas.getContext('2d');
          if (ctx) {
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            canvas.toBlob(
              (blob) => {
                if (blob && ws.readyState === WebSocket.OPEN) {
                  blob
                    .arrayBuffer()
                    .then((buf) => {
                      if (ws.readyState === WebSocket.OPEN) {
                        ws.send(buf);
                      } else {
                        waitingForResponseRef.current = false;
                      }
                    })
                    .catch(() => {
                      waitingForResponseRef.current = false;
                    });
                } else {
                  waitingForResponseRef.current = false;
                }
              },
              'image/jpeg',
              0.60
            );
          }
        }
      }

      animFrameIdRef.current = requestAnimationFrame(processFrame);
    };

    animFrameIdRef.current = requestAnimationFrame(processFrame);

    return () => {
      if (animFrameIdRef.current) {
        cancelAnimationFrame(animFrameIdRef.current);
      }
    };
  }, [cameraStatus, fpsLimit, videoRef]);

  // Switch active classifier model
  const switchModel = useCallback((newModel: ActiveModelType) => {
    setActiveModel(newModel);
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'set_model', model: newModel, threshold: confidenceThresholdRef.current }));
    }
  }, []);

  // Resolve low-margin sign choice
  const resolveDisambiguation = useCallback((selectedLabel: string) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'resolve_disambiguation', selected: selectedLabel }));
    }
  }, []);

  // Accept predictive phrase suggestion
  const acceptPrediction = useCallback((phrase: string) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'accept_prediction', phrase }));
    }
  }, []);

  const initRef = useRef({ connectWebSocket, refreshDevices, stopCamera });
  useEffect(() => {
    initRef.current = { connectWebSocket, refreshDevices, stopCamera };
    connectWebSocketRef.current = connectWebSocket;
  }, [connectWebSocket, refreshDevices, stopCamera]);

  // Lifecycle initialization & cleanup
  useEffect(() => {
    isMountedRef.current = true;
    isStoppedManuallyRef.current = false;
    initRef.current.connectWebSocket();
    void initRef.current.refreshDevices();

    return () => {
      isMountedRef.current = false;
      isStoppedManuallyRef.current = true;
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
      initRef.current.stopCamera();
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, []);

  return {
    cameraStatus,
    errorMessage,
    isBackendConnected,
    activeModel,
    switchModel,
    currentResult,
    history,
    availableCameras,
    selectedCameraId,
    setSelectedCameraId,
    startCamera,
    stopCamera,
    pauseDetection,
    resumeDetection,
    resetSession,
    clearSentence,
    backspaceSentence,
    clearHistory,
    setErrorMessage,
    resolveDisambiguation,
    acceptPrediction,
  };
}
