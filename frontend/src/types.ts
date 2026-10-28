export interface LandmarkPoint {
  x: number;
  y: number;
  z: number;
}

export interface LandmarkGroups {
  left_hand: LandmarkPoint[];
  right_hand: LandmarkPoint[];
  pose: LandmarkPoint[];
  face: LandmarkPoint[];
}

export type ActiveModelType = 'realsign' | 'isl_mvp' | 'isl_sequence_lstm' | 'include_stream';

export interface Candidate {
  label: string;
  confidence: number;
}

export interface RecognitionPayload {
  type: 'prediction' | 'error' | 'info';
  active_model?: ActiveModelType | string;
  label?: string;
  display_name?: string;
  confidence?: number;
  candidates?: Candidate[];
  is_known_sign?: boolean;
  is_stable?: boolean;
  streak_count?: number;
  required_streak?: number;
  probabilities?: Record<string, number>;
  realsign_prediction?: {
    letter: string;
    confidence: number;
    is_known_sign: boolean;
    top_3: [string, number][];
  };
  sequence_lstm_prediction?: {
    label: string;
    confidence: number;
    is_known_sign: boolean;
    top_3: [string, number][];
  };
  include_prediction?: {
    label: string;
    display_name: string;
    confidence: number;
    is_known_sign: boolean;
    top_3: [string, number][];
    entropy?: number;
  };
  buffer_fill?: number;
  hand_detected?: boolean;
  hands_in_subtitle_zone?: boolean;
  guidance?: string;
  fps?: number;
  latency_ms?: number;
  landmarks?: LandmarkGroups;
  sentence_buffer?: string;
  translated_text?: string;
  sentence_tokens?: string[];
  message?: string;
  detected_objects?: Array<{
    label: string;
    display_name: string;
    confidence: number;
    is_held: boolean;
    is_interactive: boolean;
    bbox: [number, number, number, number];
  }>;
  fused_multimodal_text?: string | null;
  disambiguation?: {
    active: boolean;
    options: Array<{
      label: string;
      display: string;
      hand: 'left' | 'right';
    }>;
    top1?: string;
    top2?: string;
  };
  lighting?: {
    score: number;
    enhanced: boolean;
  };
  predictive_suggestions?: string[];
}

export interface HistoryItem {
  id: string;
  timestamp: Date;
  signKey: string;
  text: string;
  confidence: number;
  isBookmarked?: boolean;
}

export type CameraStatus =
  | 'idle'
  | 'requesting'
  | 'ready'
  | 'detecting'
  | 'paused'
  | 'stopped'
  | 'permission_denied'
  | 'no_camera'
  | 'error';

export type SubtitleSize = 'small' | 'medium' | 'large';

export interface LandmarkStyles {
  showLeftHand: boolean;
  showRightHand: boolean;
  mirrored: boolean;
}

export interface TTSState {
  isSpeaking: boolean;
  autoSpeakEnabled: boolean;
  availableVoices: SpeechSynthesisVoice[];
  selectedVoice: SpeechSynthesisVoice | null;
  targetLanguage?: string;
}

export interface TTSControls {
  speak: (text: string, language?: string) => void;
  stop: () => void;
  toggleAutoSpeak: () => void;
  setSelectedVoice: (voice: SpeechSynthesisVoice | null) => void;
  setTargetLanguage?: (lang: string) => void;
}

export type FramingStatus =
  | 'optimal'
  | 'step_closer'
  | 'step_back'
  | 'center_yourself'
  | 'raise_hands'
  | 'hands_too_low'
  | 'searching';

export interface FramingFeedback {
  status: FramingStatus;
  message: string;
  isOptimal: boolean;
  score: number; // 0 to 100
  distanceMetric?: number;
  centerOffset?: number;
  handsDetected: boolean;
}

