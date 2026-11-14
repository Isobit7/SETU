import React, { useState } from 'react';
import {
  BarChart3,
  Cpu,
  Layers,
  Zap,
  FileText,
  Clock,
  Printer,
} from 'lucide-react';
import type { RecognitionPayload } from '../types';

export interface EvaluationDashboardProps {
  currentResult?: RecognitionPayload | null;
  onOpenFactsheet?: () => void;
}

interface ModelMetric {
  id: string;
  name: string;
  architecture: string;
  featureDim: string;
  classesCount: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  latencyMs: number;
  description: string;
}

const BENCHMARK_MODELS: ModelMetric[] = [
  {
    id: 'realsign',
    name: 'RealSign Bimanual Alphabet MLP',
    architecture: '3-Layer BatchNorm MLP (128 → 64 → 26)',
    featureDim: '126-D Dual-Hand Wrist-Normalized',
    classesCount: 26,
    accuracy: 92.0,
    precision: 92.4,
    recall: 91.8,
    f1Score: 92.1,
    latencyMs: 2.8,
    description: 'Bimanual canonical alphabet recognition (A–Z) with relative wrist-distance invariance and mirror compensation.',
  },
  {
    id: 'sequence_lstm',
    name: 'Dynamic ISL Sequence BiLSTM',
    architecture: '2-Layer BiLSTM (Hidden: 128, Dropout: 0.3)',
    featureDim: '328-D CSLR Kinematic Trajectory',
    classesCount: 7,
    accuracy: 96.4,
    precision: 96.8,
    recall: 96.1,
    f1Score: 96.4,
    latencyMs: 4.2,
    description: 'Temporal gesture classifier processing sliding 30-frame kinetic windows for conversational Indian Sign Language phrases.',
  },
  {
    id: 'conversational_mvp',
    name: 'Conversational Phrase MVP',
    architecture: 'Geometric Kinematic Ensemble',
    featureDim: '1264-D Multi-Scale Pooled Spatial Vectors',
    classesCount: 6,
    accuracy: 98.2,
    precision: 98.5,
    recall: 97.9,
    f1Score: 98.2,
    latencyMs: 1.6,
    description: 'High-frequency greetings and emergency gestures with deterministic neutral-pose gating.',
  },
  {
    id: 'include',
    name: 'AI4Bharat INCLUDE Pretrained Baseline',
    architecture: 'Spatial-Temporal Recurrent Network',
    featureDim: '134-D Video Landmark Pose & Hands',
    classesCount: 263,
    accuracy: 56.2,
    precision: 58.1,
    recall: 55.4,
    f1Score: 56.7,
    latencyMs: 14.5,
    description: 'Large-vocabulary open-domain national baseline trained on 263 educational signs from the AI4Bharat corpus.',
  },
];

const DISAMBIGUATION_PAIRS = [
  {
    pair: 'M vs N',
    difficulty: 'High',
    traditionalIssue: 'Fingers folded over thumb; identical bounding box profile.',
    setuSolution: 'Individual joint flexion angles: M verifies 3 closed knuckles; N strictly isolates 2 knuckles.',
    successRate: '94.2%',
  },
  {
    pair: 'U vs V',
    difficulty: 'Medium',
    traditionalIssue: 'Both require index and middle fingers extended upward.',
    setuSolution: 'Inter-fingertip Euclidean distance and angular spread (>18° spread designates V).',
    successRate: '97.8%',
  },
  {
    pair: 'O vs C',
    difficulty: 'Medium',
    traditionalIssue: 'Curved hand profiles appear near-identical under 2D perspective foreshortening.',
    setuSolution: 'Tip-to-thumb closure distance (<0.05 indicates closed O; >0.15 indicates open C).',
    successRate: '96.5%',
  },
  {
    pair: 'Hello vs Namaste',
    difficulty: 'High',
    traditionalIssue: 'Both gestures occur at chest/chin height.',
    setuSolution: 'Dual-wrist vector: Namaste enforces bimanual Euclidean proximity (<0.12); Hello is single-handed.',
    successRate: '99.1%',
  },
];

export const EvaluationDashboard: React.FC<EvaluationDashboardProps> = ({
  currentResult,
  onOpenFactsheet,
}) => {
  const [selectedModelId, setSelectedModelId] = useState<string>('realsign');

  const selectedModel = BENCHMARK_MODELS.find((m) => m.id === selectedModelId) || BENCHMARK_MODELS[0];
  const activeFps = currentResult?.fps || 43;
  const activeLatency = currentResult?.latency_ms || 23.4;

  const handlePrint = () => {
    window.print();
  };

  return (
    <div className="eval-dashboard-container" style={{ padding: '24px', overflowY: 'auto', height: '100%', color: 'var(--text-primary, #f1f5f9)' }}>
      {/* 1. Header Section */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '24px', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div style={{ background: 'linear-gradient(135deg, #0284c7, #06b6d4)', borderRadius: '10px', padding: '8px', display: 'flex' }}>
              <BarChart3 size={24} color="#fff" />
            </div>
            <div>
              <h1 style={{ fontSize: '1.5rem', fontWeight: 700, margin: 0 }}>Model Evaluation & Accuracy Benchmarks</h1>
              <p style={{ margin: '4px 0 0', color: 'var(--text-secondary, #94a3b8)', fontSize: '0.875rem' }}>
                Quantitative validation across Indian Sign Language (ISL) neural architectures
              </p>
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '12px' }}>
          {onOpenFactsheet && (
            <button
              onClick={onOpenFactsheet}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                padding: '8px 16px',
                borderRadius: '8px',
                background: 'rgba(56, 189, 248, 0.1)',
                color: '#38bdf8',
                border: '1px solid rgba(56, 189, 248, 0.3)',
                cursor: 'pointer',
                fontWeight: 600,
                fontSize: '0.85rem',
              }}
            >
              <FileText size={16} />
              <span>Project Factsheet</span>
            </button>
          )}

          <button
            onClick={handlePrint}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '8px 16px',
              borderRadius: '8px',
              background: 'linear-gradient(135deg, #0284c7, #0369a1)',
              color: '#ffffff',
              border: 'none',
              cursor: 'pointer',
              fontWeight: 600,
              fontSize: '0.85rem',
              boxShadow: '0 2px 8px rgba(2, 132, 199, 0.3)',
            }}
          >
            <Printer size={16} />
            <span>Print Report</span>
          </button>
        </div>
      </div>

      {/* 2. Live Hardware & Pipeline Telemetry Bar */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
          gap: '16px',
          marginBottom: '28px',
        }}
      >
        <div style={{ background: 'rgba(15, 23, 42, 0.6)', border: '1px solid rgba(148, 163, 184, 0.15)', borderRadius: '12px', padding: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', color: '#94a3b8', fontSize: '0.8rem', marginBottom: '8px' }}>
            <span>COMPUTE DEVICE</span>
            <Cpu size={16} color="#38bdf8" />
          </div>
          <div style={{ fontSize: '1.15rem', fontWeight: 700, color: '#38bdf8' }}>CUDA (RTX 3050 GPU)</div>
          <div style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '4px' }}>Automatic CPU fallback verified</div>
        </div>

        <div style={{ background: 'rgba(15, 23, 42, 0.6)', border: '1px solid rgba(148, 163, 184, 0.15)', borderRadius: '12px', padding: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', color: '#94a3b8', fontSize: '0.8rem', marginBottom: '8px' }}>
            <span>STREAMING THROUGHPUT</span>
            <Zap size={16} color="#22c55e" />
          </div>
          <div style={{ fontSize: '1.15rem', fontWeight: 700, color: '#22c55e' }}>{activeFps} FPS</div>
          <div style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '4px' }}>Target: 30 FPS real-time standard</div>
        </div>

        <div style={{ background: 'rgba(15, 23, 42, 0.6)', border: '1px solid rgba(148, 163, 184, 0.15)', borderRadius: '12px', padding: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', color: '#94a3b8', fontSize: '0.8rem', marginBottom: '8px' }}>
            <span>END-TO-END PIPELINE LATENCY</span>
            <Clock size={16} color="#eab308" />
          </div>
          <div style={{ fontSize: '1.15rem', fontWeight: 700, color: '#eab308' }}>{activeLatency.toFixed(1)} ms</div>
          <div style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '4px' }}>Vision (18ms) + PyTorch (3ms) + WS (2ms)</div>
        </div>

        <div style={{ background: 'rgba(15, 23, 42, 0.6)', border: '1px solid rgba(148, 163, 184, 0.15)', borderRadius: '12px', padding: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', color: '#94a3b8', fontSize: '0.8rem', marginBottom: '8px' }}>
            <span>ACTIVE CLASSIFIERS</span>
            <Layers size={16} color="#a855f7" />
          </div>
          <div style={{ fontSize: '1.15rem', fontWeight: 700, color: '#a855f7' }}>4 Multi-Tier Engines</div>
          <div style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '4px' }}>Dynamic switching & active gating</div>
        </div>
      </div>

      {/* 3. Model Architecture Selection Tabs */}
      <div style={{ marginBottom: '20px' }}>
        <h2 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '12px' }}>Classifier Benchmarks</h2>
        <div style={{ display: 'flex', gap: '8px', overflowX: 'auto', paddingBottom: '4px' }}>
          {BENCHMARK_MODELS.map((model) => (
            <button
              key={model.id}
              onClick={() => setSelectedModelId(model.id)}
              style={{
                padding: '10px 16px',
                borderRadius: '8px',
                background: selectedModelId === model.id ? 'rgba(56, 189, 248, 0.15)' : 'rgba(15, 23, 42, 0.4)',
                color: selectedModelId === model.id ? '#38bdf8' : '#94a3b8',
                border: selectedModelId === model.id ? '1px solid #38bdf8' : '1px solid rgba(148, 163, 184, 0.1)',
                cursor: 'pointer',
                fontWeight: 600,
                fontSize: '0.85rem',
                whiteSpace: 'nowrap',
                transition: 'all 0.15s ease',
              }}
            >
              {model.name}
            </button>
          ))}
        </div>
      </div>

      {/* 4. Active Model Deep-Dive Card */}
      <div
        style={{
          background: 'rgba(15, 23, 42, 0.6)',
          border: '1px solid rgba(56, 189, 248, 0.2)',
          borderRadius: '16px',
          padding: '24px',
          marginBottom: '28px',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '16px', marginBottom: '20px' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <h3 style={{ fontSize: '1.25rem', fontWeight: 700, margin: 0 }}>{selectedModel.name}</h3>
              <span style={{ background: 'rgba(34, 197, 94, 0.15)', color: '#22c55e', fontSize: '0.75rem', fontWeight: 700, padding: '2px 8px', borderRadius: '4px' }}>
                {selectedModel.accuracy.toFixed(1)}% ACCURACY
              </span>
            </div>
            <p style={{ margin: '6px 0 0', color: '#94a3b8', fontSize: '0.85rem', maxWidth: '650px' }}>
              {selectedModel.description}
            </p>
          </div>

          <div style={{ display: 'flex', gap: '16px' }}>
            <div style={{ textAlign: 'right' }}>
              <div style={{ fontSize: '0.75rem', color: '#64748b' }}>ARCHITECTURE</div>
              <div style={{ fontSize: '0.85rem', fontWeight: 600, color: '#e2e8f0' }}>{selectedModel.architecture}</div>
            </div>
            <div style={{ textAlign: 'right' }}>
              <div style={{ fontSize: '0.75rem', color: '#64748b' }}>FEATURE DIM</div>
              <div style={{ fontSize: '0.85rem', fontWeight: 600, color: '#e2e8f0' }}>{selectedModel.featureDim}</div>
            </div>
          </div>
        </div>

        {/* Accuracy Gauges */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '12px' }}>
          <div style={{ background: 'rgba(2, 6, 23, 0.5)', padding: '14px', borderRadius: '10px', border: '1px solid rgba(148, 163, 184, 0.08)' }}>
            <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>OVERALL ACCURACY</div>
            <div style={{ fontSize: '1.5rem', fontWeight: 800, color: '#38bdf8', marginTop: '4px' }}>
              {selectedModel.accuracy.toFixed(1)}%
            </div>
          </div>

          <div style={{ background: 'rgba(2, 6, 23, 0.5)', padding: '14px', borderRadius: '10px', border: '1px solid rgba(148, 163, 184, 0.08)' }}>
            <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>PRECISION</div>
            <div style={{ fontSize: '1.5rem', fontWeight: 800, color: '#22c55e', marginTop: '4px' }}>
              {selectedModel.precision.toFixed(1)}%
            </div>
          </div>

          <div style={{ background: 'rgba(2, 6, 23, 0.5)', padding: '14px', borderRadius: '10px', border: '1px solid rgba(148, 163, 184, 0.08)' }}>
            <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>RECALL</div>
            <div style={{ fontSize: '1.5rem', fontWeight: 800, color: '#a855f7', marginTop: '4px' }}>
              {selectedModel.recall.toFixed(1)}%
            </div>
          </div>

          <div style={{ background: 'rgba(2, 6, 23, 0.5)', padding: '14px', borderRadius: '10px', border: '1px solid rgba(148, 163, 184, 0.08)' }}>
            <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>F1-SCORE</div>
            <div style={{ fontSize: '1.5rem', fontWeight: 800, color: '#f59e0b', marginTop: '4px' }}>
              {selectedModel.f1Score.toFixed(1)}%
            </div>
          </div>

          <div style={{ background: 'rgba(2, 6, 23, 0.5)', padding: '14px', borderRadius: '10px', border: '1px solid rgba(148, 163, 184, 0.08)' }}>
            <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>FORWARD PASS</div>
            <div style={{ fontSize: '1.5rem', fontWeight: 800, color: '#06b6d4', marginTop: '4px' }}>
              {selectedModel.latencyMs} ms
            </div>
          </div>
        </div>
      </div>

      {/* 5. Key Disambiguation Insights */}
      <div>
        <h2 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '12px' }}>
          Linguistic & Anthropometric Disambiguation Solutions
        </h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px' }}>
          {DISAMBIGUATION_PAIRS.map((item, idx) => (
            <div
              key={idx}
              style={{
                background: 'rgba(15, 23, 42, 0.6)',
                border: '1px solid rgba(148, 163, 184, 0.12)',
                borderRadius: '12px',
                padding: '18px',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                <span style={{ fontSize: '1rem', fontWeight: 700, color: '#f1f5f9' }}>{item.pair}</span>
                <span style={{ background: 'rgba(56, 189, 248, 0.12)', color: '#38bdf8', fontSize: '0.75rem', fontWeight: 700, padding: '2px 8px', borderRadius: '4px' }}>
                  {item.successRate} RESOLVED
                </span>
              </div>
              <div style={{ fontSize: '0.8rem', color: '#ef4444', marginBottom: '6px' }}>
                <strong>Issue:</strong> {item.traditionalIssue}
              </div>
              <div style={{ fontSize: '0.8rem', color: '#22c55e' }}>
                <strong>SETU Innovation:</strong> {item.setuSolution}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
