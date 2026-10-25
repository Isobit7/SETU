/**
 * SETU — Frontend API Client Service
 * Centralizes REST API calls, error normalization, and timeout handling across all components.
 */

export interface ModelCatalogItem {
  id: string;
  name: string;
  type: string;
  classes_count: number;
  input_dim: number;
  is_ready: boolean;
  description: string;
  sequence_length?: number;
  device?: string;
}

export interface ModelCatalogResponse {
  models: ModelCatalogItem[];
  default_model: string;
}

export interface VocabularyItem {
  key: string;
  display: string;
  category: string;
  description: string;
}

export interface VocabularyResponse {
  vocabulary: VocabularyItem[];
  negative_class: string;
  total_classes: number;
}

export interface VideoInferenceResponse {
  success: boolean;
  predicted_label: string;
  confidence: number;
  top_candidates: Array<{ label: string; confidence: number }>;
  frames_processed?: number;
  message?: string;
}

export interface HealthStatusResponse {
  status: string;
  pipeline_ready: boolean;
  model_loaded: boolean;
  device: string;
  num_classes: number;
  supported_signs: string[];
  model_version: string;
  include_ready: boolean;
  realsign_ready: boolean;
  realsign_classes_count: number;
  sequence_lstm_ready: boolean;
  sequence_lstm_classes_count: number;
  active_classifiers: string[];
}

export class ApiService {
  private static baseUrl = 'http://localhost:8000';

  public static setBaseUrl(url: string) {
    this.baseUrl = url.replace(/\/$/, '');
  }

  public static getBaseUrl(): string {
    return this.baseUrl;
  }

  /**
   * Universal fetch helper with timeout and standardized error messages
   */
  private static async request<T>(endpoint: string, options: RequestInit = {}, timeoutMs = 12000): Promise<T> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);

    const url = `${this.baseUrl}${endpoint.startsWith('/') ? endpoint : `/${endpoint}`}`;
    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
      });

      if (!response.ok) {
        let errorDetail = `HTTP ${response.status} ${response.statusText}`;
        try {
          const errorJson = await response.json();
          if (errorJson.detail) {
            errorDetail = typeof errorJson.detail === 'string' ? errorJson.detail : JSON.stringify(errorJson.detail);
          }
        } catch {
          // fallback to text error
        }
        throw new Error(errorDetail);
      }

      return await response.json();
    } catch (err: any) {
      if (err.name === 'AbortError') {
        throw new Error(`Request timed out after ${timeoutMs}ms while connecting to ${endpoint}`);
      }
      throw err;
    } finally {
      clearTimeout(timer);
    }
  }

  /**
   * Health and readiness checks
   */
  public static async checkHealth(): Promise<HealthStatusResponse> {
    return this.request<HealthStatusResponse>('/health');
  }

  /**
   * Catalog of all swappable ML models
   */
  public static async fetchModels(): Promise<ModelCatalogResponse> {
    return this.request<ModelCatalogResponse>('/api/v1/models');
  }

  /**
   * Model status for dynamic Sequence LSTM
   */
  public static async fetchSequenceLstmStatus(): Promise<any> {
    return this.request<any>('/api/v1/models/sequence_lstm/status');
  }

  /**
   * Model status for RealSign MLP
   */
  public static async fetchRealSignStatus(): Promise<any> {
    return this.request<any>('/api/v1/models/realsign/status');
  }

  /**
   * RealSign alphabet classes list
   */
  public static async fetchRealSignClasses(): Promise<any> {
    return this.request<any>('/api/v1/models/realsign/classes');
  }

  /**
   * Supported ISL vocabulary
   */
  public static async fetchVocabulary(): Promise<VocabularyResponse> {
    return this.request<VocabularyResponse>('/api/v1/vocabulary');
  }

  /**
   * Isolated video gesture recognition (AI4Bharat INCLUDE)
   */
  public static async recognizeVideo(videoFile: File, onProgress?: (percent: number) => void): Promise<VideoInferenceResponse> {
    const formData = new FormData();
    formData.append('file', videoFile);

    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', `${this.baseUrl}/api/v1/recognize/video`);

      if (onProgress && xhr.upload) {
        xhr.upload.onprogress = (event) => {
          if (event.lengthComputable) {
            const percent = Math.round((event.loaded / event.total) * 100);
            onProgress(percent);
          }
        };
      }

      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const data = JSON.parse(xhr.responseText);
            resolve(data);
          } catch (e) {
            reject(new Error('Failed to parse server response.'));
          }
        } else {
          let errorMsg = `Server error ${xhr.status}: ${xhr.statusText}`;
          try {
            const errorJson = JSON.parse(xhr.responseText);
            if (errorJson.detail) errorMsg = errorJson.detail;
          } catch {}
          reject(new Error(errorMsg));
        }
      };

      xhr.onerror = () => reject(new Error('Network error during video upload. Please check backend connection.'));
      xhr.ontimeout = () => reject(new Error('Video upload timed out. Please try a shorter video.'));
      xhr.timeout = 60000;

      xhr.send(formData);
    });
  }

  /**
   * Custom sign recording sample upload
   */
  public static async recordCustomSample(label: string, sequence164d: number[][]): Promise<any> {
    return this.request<any>('/api/v1/dataset/record_sample', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ label, sequence_164d: sequence164d }),
    });
  }

  /**
   * List custom recorded sign samples
   */
  public static async fetchCustomSamples(): Promise<any> {
    return this.request<any>('/api/v1/dataset/custom_samples');
  }
}
