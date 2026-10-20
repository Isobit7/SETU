import React, { useEffect, useRef, useState } from 'react';
import {
  Camera,
  Maximize2,
  Minimize2,
  RefreshCw,
  AlertCircle,
  Activity,
  ShieldCheck,
} from 'lucide-react';
import type { CameraStatus, LandmarkStyles, RecognitionPayload, SubtitleSize } from '../types';
import { drawLandmarksOnCanvas } from '../utils/landmarksDrawer';

interface CameraWorkspaceProps {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  cameraStatus: CameraStatus;
  errorMessage: string | null;
  currentResult: RecognitionPayload | null;
  landmarkStyles: LandmarkStyles;
  subtitleVisible: boolean;
  subtitleSize: SubtitleSize;
  onStartCamera: () => void;
  onRetry: () => void;
}

export const CameraWorkspace: React.FC<CameraWorkspaceProps> = ({
  videoRef,
  cameraStatus,
  errorMessage,
  currentResult,
  landmarkStyles,
  subtitleVisible,
  subtitleSize,
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

  // Determine dynamic subtitle placement:
  // If hands are inside lower subtitle area, move subtitle up to avoid covering hands!
  const handsInSafeZone = currentResult?.hands_in_subtitle_zone || false;
  const isKnown = currentResult?.is_known_sign || false;
  const activeLabel = currentResult?.display_name || 'Waiting for a sign...';
  const confidencePercent = currentResult?.confidence
    ? Math.round(currentResult.confidence * 100)
    : 0;

  // Determine hand tracking status text
  const handDetected = currentResult?.hand_detected || false;
  const handStatusText =
    cameraStatus === 'detecting'
      ? handDetected
        ? 'Tracking hands'
        : 'Hand not detected'
      : cameraStatus === 'paused'
      ? 'Paused'
      : cameraStatus === 'requesting'
      ? 'Camera starting'
      : 'Ready';

  return (
    <div
      ref={containerRef}
      className={`camera-workspace-card ${isFullscreen ? 'fullscreen' : ''}`}
      id="camera-workspace"
    >
      {/* Top Overlay Badge Bar */}
      <div className="camera-header-overlay">
        <div className="left-badges">
          <div className={`status-badge ${cameraStatus}`}>
            <span className="status-dot"></span>
            <span className="status-text">
              {cameraStatus === 'detecting' && 'LIVE DETECTING'}
              {cameraStatus === 'paused' && 'DETECTION PAUSED'}
              {cameraStatus === 'requesting' && 'STARTING CAMERA...'}
              {cameraStatus === 'idle' && 'READY'}
              {cameraStatus === 'stopped' && 'CAMERA OFF'}
              {cameraStatus === 'permission_denied' && 'CAMERA BLOCKED'}
              {cameraStatus === 'error' && 'ERROR'}
            </span>
          </div>

          {cameraStatus === 'detecting' && (
            <div
              className={`badge hand-detected-badge ${handDetected ? 'tracking' : 'waiting'}`}
              title={handDetected ? 'Hands tracked by MediaPipe' : 'No hands visible in camera'}
            >
              <span>{handStatusText}</span>
            </div>
          )}
        </div>

        <div className="right-tools">
          {/* Landmark Toggles quick legend - HANDS ONLY */}
          <div className="landmark-quick-legend" title="Hand tracking color legend">
            <span className="legend-tag left-hand">● Left hand</span>
            <span className="legend-tag right-hand">● Right hand</span>
          </div>

          <button
            className="overlay-icon-btn"
            onClick={toggleFullscreen}
            title={isFullscreen ? 'Exit Fullscreen' : 'Fullscreen Camera'}
            aria-label="Toggle fullscreen"
          >
            {isFullscreen ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
          </button>
        </div>
      </div>

      {/* Video & Canvas Frame Area */}
      <div className="video-viewport">
        {/* Actual Live Video */}
        <video
          ref={videoRef}
          className={`live-video ${landmarkStyles.mirrored ? 'mirrored' : ''}`}
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

        {/* Idle / Permission / Error Backdrop */}
        {cameraStatus !== 'detecting' && cameraStatus !== 'paused' && (
          <div className="camera-backdrop-state">
            {cameraStatus === 'requesting' && (
              <div className="state-message">
                <RefreshCw className="animate-spin text-primary" size={44} />
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
              <div className="state-message">
                <div className="camera-icon-wrap">
                  <Camera size={44} />
                </div>
                <h3>Camera Standby</h3>
                <p>Position your hands comfortably in camera view to translate ISL gestures.</p>
                <button className="start-large-btn" onClick={onStartCamera}>
                  <Camera size={18} />
                  <span>Start Camera & Detection</span>
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
        )}

        {/* CRITICAL REQUIREMENT: MOVIE-STYLE SUBTITLE BAND OVER CAMERA PREVIEW */}
        {subtitleVisible && (cameraStatus === 'detecting' || cameraStatus === 'paused') && (
          <div
            className={`movie-subtitle-container ${handsInSafeZone ? 'avoid-hands' : ''}`}
            aria-live="polite"
          >
            <div className={`movie-subtitle-band ${subtitleSize} ${isKnown ? 'active-sign' : 'resting'}`}>
              <div className="subtitle-content">
                <span className="subtitle-text">{activeLabel}</span>
                {isKnown && (
                  <span className="subtitle-conf-badge" title="Model ranking score (uncalibrated candidate score)">
                    Score: {confidencePercent}%
                  </span>
                )}
              </div>
              {handsInSafeZone && (
                <div className="reposition-indicator" title="Repositioned automatically because hands are in the lower frame">
                  Repositioned for hand visibility
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* 24px Micro Status Bar Directly Beneath Camera Feed */}
      <div className="camera-micro-statusbar" role="status" aria-label="Hardware and Processing Status">
        <div className="micro-status-left">
          <span className="micro-status-item">
            <Camera size={12} className="micro-status-icon" />
            <span>Hardware: {cameraStatus === 'detecting' ? 'Ready (Active)' : cameraStatus === 'paused' ? 'Paused' : 'Ready'}</span>
          </span>
          <span className="micro-status-sep">•</span>
          <span className="micro-status-item">
            <Activity size={12} className="micro-status-icon" />
            <span>158-D Holistic ({currentResult?.fps ? `${currentResult.fps} FPS` : '18 FPS'})</span>
          </span>
          <span className="micro-status-sep">•</span>
          <span className="micro-status-item">
            <ShieldCheck size={12} className="micro-status-icon" />
            <span>Local Processing</span>
          </span>
        </div>
      </div>
    </div>
  );
};
