import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  Camera,
  Maximize2,
  Minimize2,
  RefreshCw,
  AlertCircle,
  Focus,
  BookOpen,
  Sparkles,
  X,
} from 'lucide-react';
import type {
  CameraStatus,
  FramingFeedback,
  LandmarkStyles,
  RecognitionPayload,
  SubtitleSize,
} from '../types';
import { drawLandmarksOnCanvas } from '../utils/landmarksDrawer';
import { evaluateFraming } from '../utils/framingEvaluator';
import { VisualFramingGuide } from './VisualFramingGuide';
import { DisambiguationOverlay } from './DisambiguationOverlay';

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
  // Proactive Framing Guide & Gating
  showFramingGuide?: boolean;
  onToggleFramingGuide?: () => void;
  enforceFraming?: boolean;
  onToggleEnforceFraming?: () => void;
  onFramingChange?: (feedback: FramingFeedback) => void;
  onOpenVocabulary?: () => void;
  onResolveDisambiguation?: (label: string) => void;
  onAcceptPrediction?: (phrase: string) => void;
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
  showFramingGuide,
  onToggleFramingGuide,
  enforceFraming: _enforceFraming,
  onToggleEnforceFraming: _onToggleEnforceFraming,
  onFramingChange,
  onOpenVocabulary,
  onResolveDisambiguation,
  onAcceptPrediction,
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);
  const [showSignGuide, setShowSignGuide] = useState<boolean>(false);

  // Local state fallback if external handler isn't explicitly passed
  const [internalShowGuide, setInternalShowGuide] = useState<boolean>(false);

  const activeShowGuide = showFramingGuide !== undefined ? showFramingGuide : internalShowGuide;

  const handleToggleGuide = () => {
    if (onToggleFramingGuide) {
      onToggleFramingGuide();
    } else {
      setInternalShowGuide((prev) => !prev);
    }
  };

  // Real-time client-side framing feedback computation (<0.02ms, zero lag)
  const framingFeedback: FramingFeedback = useMemo(() => {
    if (cameraStatus !== 'detecting' && cameraStatus !== 'paused') {
      return {
        status: 'searching',
        message: 'Position yourself within the guide',
        isOptimal: false,
        score: 0,
        handsDetected: false,
      };
    }
    return evaluateFraming(currentResult?.landmarks, {
      mirrored: landmarkStyles.mirrored,
    });
  }, [cameraStatus, currentResult?.landmarks, landmarkStyles.mirrored]);

  // Notify parent of framing adjustments
  useEffect(() => {
    onFramingChange?.(framingFeedback);
  }, [framingFeedback, onFramingChange]);

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
    if (ctx && (currentResult?.landmarks || currentResult?.detected_objects)) {
      drawLandmarksOnCanvas(
        ctx,
        width,
        height,
        currentResult.landmarks || {},
        landmarkStyles,
        currentResult.detected_objects
      );
    } else if (ctx) {
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
      {/* Minimal Unobtrusive Viewport Tools */}
      <div className="camera-header-overlay">
        <div className="camera-header-left">
          {cameraStatus === 'requesting' && (
            <div className="camera-readiness-pill status-starting">
              <span className="readiness-dot starting" aria-hidden="true" />
              <span>Starting Camera...</span>
            </div>
          )}
          {cameraStatus === 'detecting' && (
            <div className="camera-readiness-pill status-live">
              <span className="readiness-dot live" aria-hidden="true" />
              <span>Live Recognition</span>
            </div>
          )}
          {cameraStatus === 'paused' && (
            <div className="camera-readiness-pill status-paused">
              <span className="readiness-dot paused" aria-hidden="true" />
              <span>Detection Paused</span>
            </div>
          )}
          {activeShowGuide && cameraStatus === 'detecting' && (
            <div
              className={`framing-status-chip ${
                framingFeedback.isOptimal ? 'optimal' : 'adjust'
              }`}
            >
              <span className="framing-chip-dot" />
              <span className="framing-chip-text">{framingFeedback.message}</span>
            </div>
          )}
        </div>

        <div className="right-tools">
          {/* Toggle Framing Guide Overlay */}
          <button
            className={`overlay-icon-btn ${activeShowGuide ? 'active-tool' : ''}`}
            onClick={handleToggleGuide}
            title={activeShowGuide ? 'Hide Framing Guide' : 'Show Framing Guide'}
            aria-label="Toggle visual framing guide overlay"
            aria-pressed={activeShowGuide}
          >
            <Focus size={15} />
          </button>

          {/* Toggle Reference Gesture Guide */}
          <button
            className={`overlay-icon-btn ${showSignGuide ? 'active-tool' : ''}`}
            onClick={() => setShowSignGuide((prev) => !prev)}
            title={showSignGuide ? 'Hide Sign Guide' : 'Show Sign Reference & Guidance'}
            aria-label="Toggle sign reference guide overlay"
            aria-pressed={showSignGuide}
          >
            <BookOpen size={15} />
          </button>

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
          className={`live-video ${landmarkStyles.mirrored ? 'mirrored' : ''} ${cameraStatus === 'detecting' || cameraStatus === 'paused' ? 'active-stream' : ''}`}
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

        {/* Visual Framing Guide */}
        {(cameraStatus === 'detecting' || cameraStatus === 'paused') && activeShowGuide && (
          <VisualFramingGuide
            feedback={framingFeedback}
            visible={activeShowGuide}
            mirrored={landmarkStyles.mirrored}
          />
        )}

        {/* Soft Framing Guidance Hint when hands are not yet in frame */}
        {cameraStatus === 'detecting' && !handDetected && !activeShowGuide && (
          <div className="camera-framing-hint-banner" aria-live="polite">
            <span className="framing-hint-dot" aria-hidden="true" />
            <span className="framing-hint-text">Position hands within camera frame to translate</span>
          </div>
        )}

        {/* Real-Time Reference Gesture & Sign Assistance Card */}
        {showSignGuide && (
          <div className="reference-sign-guide-overlay" role="region" aria-label="Sign Language Reference Guide">
            <div className="guide-card-header">
              <div className="guide-card-title-group">
                <Sparkles size={14} className="guide-sparkle-icon" />
                <span className="guide-card-title">
                  {currentResult?.active_model === 'include_stream'
                    ? 'AI4Bharat INCLUDE (263 Signs)'
                    : currentResult?.active_model === 'realsign'
                    ? 'RealSign ISL Alphabet (26 Letters)'
                    : 'ISL Conversational Signs'}
                </span>
              </div>
              <button
                type="button"
                className="guide-card-close"
                onClick={() => setShowSignGuide(false)}
                aria-label="Close gesture guide"
              >
                <X size={13} />
              </button>
            </div>

            <div className="guide-card-body">
              <p className="guide-card-hint">
                {currentResult?.active_model === 'include_stream'
                  ? 'Dynamic 263-class bimanual vocabulary. Articulate signs with fluid continuous motion in the frame.'
                  : currentResult?.active_model === 'realsign'
                  ? 'Authentic ISL fingerspelling. Letters B, D, P, Q, T require contact between both hands; A, C, L use one hand.'
                  : 'Core conversational gestures. Hold hand steady for recognition.'}
              </p>

              <div className="guide-sample-chips">
                {(currentResult?.active_model === 'include_stream'
                  ? ['THANK YOU', 'HELLO', 'GOOD MORNING', 'NAMASTE', 'HELP', 'WATER', 'FOOD', 'PLEASE']
                  : currentResult?.active_model === 'realsign'
                  ? ['A (Single)', 'B (Dual-Hand)', 'C', 'D (Dual-Hand)', 'L', 'P (Dual-Hand)', 'Q (Dual)', 'T (Dual)']
                  : ['HELLO', 'THANK YOU', 'GOOD MORNING', 'I', 'YOU']
                ).map((chip) => (
                  <span key={chip} className="guide-chip">
                    {chip}
                  </span>
                ))}
              </div>

              {onOpenVocabulary && (
                <button
                  type="button"
                  className="guide-card-cta"
                  onClick={onOpenVocabulary}
                >
                  <BookOpen size={13} />
                  <span>Browse All Vocabulary & Tips</span>
                </button>
              )}
            </div>
          </div>
        )}

        {/* Idle / Permission / Error Backdrop */}
        <div
          className={`camera-backdrop-state ${cameraStatus === 'detecting' || cameraStatus === 'paused' ? 'backdrop-hidden' : 'backdrop-visible'}`}
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

        {/* Show-and-Sign Multimodal Fused Text Banner */}
        {currentResult?.fused_multimodal_text && (
          <div className="show-and-sign-pill">
            <Sparkles size={14} className="sparkle-gold" />
            <span className="show-and-sign-label">Show-and-Sign:</span>
            <span className="show-and-sign-text">{currentResult.fused_multimodal_text}</span>
          </div>
        )}

        {/* Detected Physical Objects Tags */}
        {currentResult?.detected_objects && currentResult.detected_objects.length > 0 && (
          <div className="detected-objects-bar">
            {currentResult.detected_objects.slice(0, 3).map((obj) => (
              <span key={obj.label} className={`object-chip ${obj.is_held ? 'held' : ''}`}>
                <span className="object-chip-dot" />
                <span>{obj.display_name}</span>
                {obj.is_held && <span className="held-badge">Held</span>}
              </span>
            ))}
          </div>
        )}

        {/* Low-Light Enhancement Chip */}
        {currentResult?.lighting?.enhanced && (
          <div className="low-light-chip" title="Adaptive CLAHE low-light enhancement active">
            <span>🌙</span>
            <span>Low-Light Boost</span>
          </div>
        )}

        {/* Low-Margin Sign Left/Right Disambiguation HUD */}
        {currentResult?.disambiguation?.active && currentResult.disambiguation.options && (
          <DisambiguationOverlay
            active={currentResult.disambiguation.active}
            options={currentResult.disambiguation.options}
            onSelect={(label) => onResolveDisambiguation?.(label)}
          />
        )}

        {/* PERSISTENT PREDICTED SIGN SUBTITLE BAR */}
        {subtitleVisible && (cameraStatus === 'detecting' || cameraStatus === 'paused') && (
          <div className={`persistent-subtitle-dock subtitle-size-${subtitleSize}`} aria-live="polite">
            <div
              className={`subtitle-panel ${
                isKnown ? 'active-sign' : 'resting'
              }`}
            >
              <div className="subtitle-header-row">
                <span className="subtitle-category-label">
                  CURRENT PREDICTED SIGN
                </span>
                {isKnown && confidencePercent > 0 ? (
                  <span className="subtitle-confidence-pill">
                    {confidencePercent}% Match
                  </span>
                ) : null}
              </div>
              <div className="subtitle-value-row">
                <span className="subtitle-main-display">
                  {cameraStatus === 'detecting' ? subtitleText : 'Detection Paused'}
                </span>
              </div>
            </div>

            {/* Predictive Next-Word / Phrase Quick Suggestions */}
            {currentResult?.predictive_suggestions && currentResult.predictive_suggestions.length > 0 && (
              <div className="predictive-suggestions-bar">
                <span className="predictive-hint">Suggestions:</span>
                {currentResult.predictive_suggestions.map((phrase) => (
                  <button
                    key={phrase}
                    type="button"
                    className="predictive-chip-btn"
                    onClick={() => onAcceptPrediction?.(phrase)}
                    title={`Add "${phrase}" to sentence`}
                  >
                    <span>+ {phrase}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
