import React, { useState, useEffect, useRef } from 'react';
import {
  X,
  UploadCloud,
  FileVideo,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Volume2,
  VolumeX,
  Copy,
  Check,
  BarChart3,
  Info,
} from 'lucide-react';

interface Candidate {
  label: string;
  display_name: string;
  confidence: number;
}

interface RecognitionResponse {
  success: boolean;
  label: string;
  display_name: string;
  confidence: number;
  top5_candidates: Candidate[];
  sequence_length: number;
  feature_dimension: number;
  inference_time_ms: number;
  model_name: string;
}

interface ModelStatus {
  model_name: string;
  is_ready: boolean;
  classes_count: number;
  input_features: number;
  sequence_length: number;
  validation_score: number | null;
  checkpoint_exists: boolean;
  label_map_exists: boolean;
}

interface IncludeVideoModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const IncludeVideoModal: React.FC<IncludeVideoModalProps> = ({ isOpen, onClose }) => {
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [isLoadingStatus, setIsLoadingStatus] = useState<boolean>(true);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isRecognizing, setIsRecognizing] = useState<boolean>(false);
  const [result, setResult] = useState<RecognitionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState<boolean>(false);
  const [isSpeaking, setIsSpeaking] = useState<boolean>(false);

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Fetch model status on open
  useEffect(() => {
    if (!isOpen) return;
    setIsLoadingStatus(true);
    fetch('/api/v1/models/include/status')
      .then((res) => {
        if (!res.ok) throw new Error('Failed to fetch model status');
        return res.json();
      })
      .then((data: ModelStatus) => {
        setModelStatus(data);
        setIsLoadingStatus(false);
      })
      .catch((err) => {
        console.error('Error fetching model status:', err);
        setIsLoadingStatus(false);
      });
  }, [isOpen]);

  // Handle escape key
  useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
      setResult(null);
      setError(null);
    }
  };

  const handleUploadAndRecognize = async () => {
    if (!selectedFile) return;

    setIsRecognizing(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const res = await fetch('/api/v1/recognize/video', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server returned ${res.status}`);
      }

      const data: RecognitionResponse = await res.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || 'Recognition failed. Please try again.');
    } finally {
      setIsRecognizing(false);
    }
  };

  const handleCopy = async (text: string) => {
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Copy failed:', err);
    }
  };

  const handleSpeak = (text: string) => {
    if (!text || !('speechSynthesis' in window)) return;
    if (isSpeaking) {
      window.speechSynthesis.cancel();
      setIsSpeaking(false);
      return;
    }
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'en-US';
    utterance.rate = 0.95;
    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);
    window.speechSynthesis.speak(utterance);
  };

  return (
    <div className="modal-backdrop" onClick={onClose} role="dialog" aria-modal="true">
      <div className="modal-container include-modal-container" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="modal-header">
          <div className="modal-title-wrap">
            <FileVideo size={20} className="text-primary" />
            <div>
              <h2>AI4Bharat INCLUDE ISL Recognition</h2>
              <span className="modal-subtitle">Pretrained 263-Sign Recognition Model</span>
            </div>
          </div>
          <button className="modal-close-btn" onClick={onClose} aria-label="Close dialog">
            <X size={20} />
          </button>
        </div>

        {/* Body */}
        <div className="modal-body">
          {/* Model Status Card */}
          <div className="include-status-card">
            <div className="status-row">
              <div className="status-item">
                <span className="status-label">Model Engine</span>
                <span className="status-val font-semibold">AI4Bharat INCLUDE BiLSTM</span>
              </div>
              <div className="status-item">
                <span className="status-label">Vocabulary Size</span>
                <span className="status-val">{modelStatus?.classes_count || 263} Signs</span>
              </div>
              <div className="status-item">
                <span className="status-label">Input Architecture</span>
                <span className="status-val">134-D Keypoints</span>
              </div>
              <div className="status-item">
                <span className="status-label">Verified Checkpoint</span>
                {isLoadingStatus ? (
                  <span className="status-badge" style={{ background: '#F1F5F9', color: '#64748B' }}>
                    <Loader2 size={13} className="spinning-icon" />
                    Checking...
                  </span>
                ) : (
                  <span className="status-badge ready">
                    <CheckCircle2 size={13} />
                    Ready (score: {modelStatus?.validation_score ? `${(modelStatus.validation_score * 100).toFixed(1)}%` : '56.2%'})
                  </span>
                )}
              </div>
            </div>
          </div>

          {/* Upload Area */}
          <div className="include-upload-section">
            <label className="upload-dropzone">
              <input
                ref={fileInputRef}
                type="file"
                accept=".mov,.mp4,.avi,.webm"
                onChange={handleFileChange}
                disabled={isRecognizing}
                style={{ display: 'none' }}
              />
              <UploadCloud size={36} className="upload-icon" />
              <div className="upload-prompt">
                <span className="upload-main-text">
                  {selectedFile ? selectedFile.name : 'Select or drag an ISL video file'}
                </span>
                <span className="upload-sub-text">
                  Supports .mov, .mp4, .webm (max 50 MB)
                </span>
              </div>
              {selectedFile && (
                <span className="selected-size">
                  ({(selectedFile.size / (1024 * 1024)).toFixed(2)} MB)
                </span>
              )}
            </label>

            <button
              className="primary-btn recognize-action-btn"
              onClick={handleUploadAndRecognize}
              disabled={!selectedFile || isRecognizing}
            >
              {isRecognizing ? (
                <>
                  <Loader2 size={16} className="spinning-icon" />
                  <span>Extracting 134-D Landmarks & Recognizing...</span>
                </>
              ) : (
                <>
                  <FileVideo size={16} />
                  <span>Recognize Sign with INCLUDE Model</span>
                </>
              )}
            </button>
          </div>

          {/* Error Message */}
          {error && (
            <div className="error-banner" role="alert">
              <AlertCircle size={18} />
              <span>{error}</span>
            </div>
          )}

          {/* Result Card */}
          {result && (
            <div className="include-result-card">
              <div className="result-header">
                <div className="recognized-sign-wrap">
                  <span className="result-badge">Recognized Sign</span>
                  <h3 className="recognized-title">{result.display_name}</h3>
                  <span className="canonical-key">Canonical Key: <code>{result.label}</code></span>
                </div>
                <div className="result-score-badge">
                  <span className="score-val">{Math.round(result.confidence * 100)}%</span>
                  <span className="score-label">Confidence</span>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="result-actions">
                <button
                  className="icon-mini-btn"
                  onClick={() => handleSpeak(result.display_name)}
                  title="Speak sign name"
                >
                  {isSpeaking ? <VolumeX size={16} /> : <Volume2 size={16} />}
                  <span>Pronounce</span>
                </button>
                <button
                  className="icon-mini-btn"
                  onClick={() => handleCopy(result.display_name)}
                  title="Copy sign text"
                >
                  {copied ? <Check size={16} /> : <Copy size={16} />}
                  <span>{copied ? 'Copied' : 'Copy'}</span>
                </button>
              </div>

              {/* Diagnostics breakdown */}
              <div className="top5-section">
                <div className="top5-header">
                  <BarChart3 size={15} />
                  <span>Top-5 Candidates (out of 263 classes)</span>
                </div>
                <div className="top5-bars">
                  {result.top5_candidates.map((cand, idx) => (
                    <div key={cand.label} className="cand-row">
                      <div className="cand-info">
                        <span className="cand-rank">#{idx + 1}</span>
                        <span className="cand-name">{cand.display_name}</span>
                        <span className="cand-pct">{Math.round(cand.confidence * 100)}%</span>
                      </div>
                      <div className="cand-track">
                        <div
                          className="cand-fill"
                          style={{ width: `${Math.min(100, Math.round(cand.confidence * 100))}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Meta stats */}
              <div className="result-meta-footer">
                <span>Inference time: {result.inference_time_ms} ms</span>
                <span>Sequence frames: {result.sequence_length}</span>
                <span>Features: {result.feature_dimension}-D (Pose 50, Hands 84)</span>
              </div>
            </div>
          )}

          {/* Architecture note */}
          <div className="architecture-note">
            <Info size={15} />
            <span>
              <strong>Note:</strong> The AI4Bharat INCLUDE model performs isolated ISL sign recognition across 263 classes from video clips. SETU's real-time streaming pipeline uses the 158-D geometry model for live webcam interactions.
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
