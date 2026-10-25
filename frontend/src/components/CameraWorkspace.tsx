import React, { useEffect, useRef, useState } from 'react';
import {
  Camera,
  Maximize2,
  Minimize2,
  RefreshCw,
  AlertCircle,
} from 'lucide-react';
import type { CameraStatus, LandmarkStyles, RecognitionPayload, SubtitleSize } from '../types';
import { drawLandmarksOnCanvas } from '../utils/landmarksDrawer';

interface CameraWorkspaceProps {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  cameraStatus: CameraStatus;
  errorMessage: string | null;
  currentResult: RecognitionPayload | null;
  landmarkStyles: LandmarkStyles;
  subtitleVisible?: boolean;
  subtitleSize?: SubtitleSize;
  onStartCamera: () => void;
  onRetry: () => void;
}

export const CameraWorkspace: React.FC<CameraWorkspaceProps> = ({
  videoRef,
  cameraStatus,
  errorMessage,
  currentResult,
  landmarkStyles,
  subtitleVisible = true,
  subtitleSize = 'medium',
  onStartCamera,
  onRetry,
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);

  // Toggle fullscreen
  const toggleFullscreen = async () => {
    if (!containerRef.current) return;
    if (!document.fullscreenElement) {
      try {
        await containerRef.current.requestFullscreen();
        setIsFullscreen(true);
      } catch (err) {
        console.warn('Fullscreen error:', err);
      }
    } else {
      if (document.exitFullscreen) {
        await document.exitFullscreen();
        setIsFullscreen(false);
      }
    }
  };

  useEffect(() => {
    const handleFsChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };
    document.addEventListener('fullscreenchange', handleFsChange);
    return () => document.removeEventListener('fullscreenchange', handleFsChange);
  }, []);

  // Synchronize canvas size with video client dimensions and draw landmarks
  useEffect(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const width = video.videoWidth || 640;
    const height = video.videoHeight || 480;

    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }

    const ctx = canvas.getContext('2d');
    if (ctx && currentResult?.landmarks) {
      drawLandmarksOnCanvas(ctx, width, height, currentResult.landmarks, landmarkStyles);
    } else if (ctx && !currentResult?.landmarks) {
      ctx.clearRect(0, 0, width, height);
    }
  }, [currentResult, landmarkStyles, videoRef]);

  // Determine hand tracking status
  const handDetected = currentResult?.hand_detected || false;

  // Fixed movie-style subtitle calculation
  const isKnown = currentResult?.is_known_sign || false;
  const label = currentResult?.label || 'no_sign';
  const confidencePercent = currentResult?.confidence
    ? Math.round(currentResult.confidence * 100)
    : 0;

  let subtitleText = 'Waiting for sign...';
  if (cameraStatus === 'paused') {
    subtitleText = 'Detection Paused';
  } else if (cameraStatus === 'detecting') {
    if (isKnown && label !== 'no_sign' && label !== 'Awaiting Sign Input') {
      if (currentResult?.display_name) {
        subtitleText = currentResult.display_name;
      } else if (currentResult?.active_model === 'realsign' && label.length === 1) {
        subtitleText = `Letter ${label.toUpperCase()}`;
      } else {
        subtitleText = label.replace('_', ' ').toUpperCase();
      }
    } else if (handDetected) {
      subtitleText = 'Holding sign...';
    } else {
      subtitleText = 'Show hands to translate';
    }
  }

  return (
    <div
      ref={containerRef}
      className={`camera-workspace-card ${isFullscreen ? 'fullscreen' : ''} ${cameraStatus === 'requesting' ? 'camera-initializing' : ''}`}
      id="camera-workspace"
    >
      {/* Minimal Unobtrusive Viewport Tools (Zero Hardware / Mock Telemetry) */}
      <div className="camera-header-overlay">
        {cameraStatus === 'detecting' && (
          <div className="live-camera-indicator">
            <span className="live-status-dot" aria-hidden="true" />
            <span className="live-status-label">Live Camera</span>
          </div>
        )}
        <div className="right-tools">
          <button
            className="overlay-icon-btn"
            onClick={toggleFullscreen}
            title={isFullscreen ? 'Exit Fullscreen' : 'Fullscreen Camera'}
            aria-label="Toggle fullscreen"
          >
            {isFullscreen ? <Minimize2 size={15} /> : <Maximize2 size={15} />}
          </button>
        </div>
      </div>

      {/* Video & Canvas Frame Area */}
      <div className="video-viewport">
        {/* Actual Live Video */}
        <video
          ref={videoRef}
          className={`live-video ${landmarkStyles.mirrored ? 'mirrored' : ''} ${(cameraStatus === 'detecting' || cameraStatus === 'paused') ? 'active-stream' : ''}`}
          playsInline
          muted
          autoPlay
        />

        {/* Real Landmark Overlay Canvas (HANDS ONLY) */}
        <canvas
          ref={canvasRef}
          className="landmark-canvas"
          aria-label="Real MediaPipe Hand Landmark Visualization"
        />

        {/* Idle / Permission / Error Backdrop (Smooth 300ms transition, no abrupt layout shifts) */}
        <div
          className={`camera-backdrop-state ${(cameraStatus === 'detecting' || cameraStatus === 'paused') ? 'backdrop-hidden' : 'backdrop-visible'}`}
          aria-hidden={cameraStatus === 'detecting' || cameraStatus === 'paused'}
        >
          {cameraStatus === 'requesting' && (
            <div className="state-message">
              <RefreshCw className="animate-spin" size={44} />
              <h3>Starting Camera...</h3>
              <p>Please click "Allow" in your browser prompt.</p>
            </div>
          )}

          {cameraStatus === 'permission_denied' && (
            <div className="state-message error">
              <AlertCircle className="text-danger" size={44} />
              <h3>Camera Permission Denied</h3>
              <p>SETU requires webcam access to track hand signs locally.</p>
              <button className="primary-btn" onClick={onRetry}>
                Retry Permission
              </button>
            </div>
          )}

          {(cameraStatus === 'idle' || cameraStatus === 'stopped') && (
            <div className="state-message standby-state-card">
              <div className="standby-aperture-visual" aria-hidden="true">
                <Camera size={26} strokeWidth={1.8} className="standby-camera-icon" />
              </div>

              <div className="standby-text-block">
                <h3 className="standby-title">Live Sign Language Recognition</h3>
                <p className="standby-desc">
                  Position your hands inside camera view to translate Indian Sign Language into English in real time.
                </p>
              </div>

              <button
                type="button"
                className="start-camera-cta-btn"
                onClick={onStartCamera}
                aria-label="Start Camera and Begin Live Sign Translation"
              >
                <Camera size={16} />
                <span>Start Camera</span>
              </button>
            </div>
          )}

          {cameraStatus === 'no_camera' && (
            <div className="state-message error" role="alert" aria-live="assertive">
              <AlertCircle className="text-danger" size={44} />
              <h3>No Camera Found</h3>
              <p>No webcam device was detected. Please connect a USB or built-in camera.</p>
              <button className="primary-btn" onClick={onRetry} aria-label="Retry detecting camera">
                <RefreshCw size={16} />
                <span>Scan for Camera</span>
              </button>
            </div>
          )}

          {cameraStatus === 'error' && (
            <div className="state-message error" role="alert" aria-live="assertive">
              <AlertCircle size={44} />
              <h3>Camera Error</h3>
              <p>{errorMessage || 'An error occurred while accessing the camera.'}</p>
              <button className="primary-btn" onClick={onRetry} aria-label="Retry camera initialization">
                <RefreshCw size={16} />
                <span>Retry</span>
              </button>
            </div>
          )}
        </div>

        {/* PERSISTENT PREDICTED SIGN SUBTITLE BAR (Docked Permanently at Bottom during Live Detection, Zero Jitter / CLS = 0) */}
        {subtitleVisible && (cameraStatus === 'detecting' || cameraStatus === 'paused') && (
          <div className={`persistent-subtitle-dock subtitle-size-${subtitleSize}`} aria-live="polite">
            <div className={`subtitle-panel ${isKnown ? 'active-sign' : 'resting'}`}>
              <div className="subtitle-header-row">
                <span className="subtitle-category-label">CURRENT PREDICTED SIGN</span>
                {isKnown && confidencePercent > 0 && (
                  <span className="subtitle-confidence-pill">
                    {confidencePercent}% Match
                  </span>
                )}
              </div>
              <div className="subtitle-value-row">
                <span className="subtitle-main-display">
                  {cameraStatus === 'detecting'
                    ? subtitleText
                    : 'Detection Paused'}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
