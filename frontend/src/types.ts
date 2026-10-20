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

export type ActiveModelType = 'realsign' | 'isl_mvp' | 'isl_sequence_lstm';

export interface RecognitionPayload {
  type: 'prediction' | 'error' | 'info';
  active_model?: ActiveModelType | string;
  label?: string;
  display_name?: string;
  confidence?: number;
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
  buffer_fill?: number;
  hand_detected?: boolean;
  hands_in_subtitle_zone?: boolean;
  guidance?: string;
  fps?: number;
  latency_ms?: number;
  landmarks?: LandmarkGroups;
  message?: string;
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
