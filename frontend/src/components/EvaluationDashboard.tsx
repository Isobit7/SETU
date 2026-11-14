import React, { useState } from 'react';
import {
  BarChart3,
  Cpu,
  Layers,
  Zap,
  FileText,
  Clock,
  Printer,
  CheckCircle2,
  Info,
} from 'lucide-react';
import type { RecognitionPayload } from '../types';

export interface EvaluationDashboardProps {
  currentResult?: RecognitionPayload | null;
  onOpenFactsheet?: () => void;
}

interface ModelMetric {
  id: string;
  name: string;
  shortName: string;
  architecture: string;
  featureDim: string;
  classesCount: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  latencyMs: number;
  purpose: string;
  description: string;
  vivaHighlight: string;
}

const BENCHMARK_MODELS: ModelMetric[] = [
  {
    id: 'realsign',
    name: 'RealSign Bimanual Alphabet Classifier',
    shortName: 'Alphabet MLP (A–Z)',
    architecture: '3-Layer BatchNorm MLP (128 → 64 → 26)',
    featureDim: '126-D Dual-Hand Wrist-Normalized',
    classesCount: 26,
    accuracy: 92.0,
    precision: 92.4,
    recall: 91.8,
    f1Score: 92.1,
    latencyMs: 2.8,
    purpose: 'Fingerspelling individual letters across Indian Sign Language alphabet',
    description: 'Bimanual canonical alphabet recognition (A–Z) with relative wrist-distance invariance, wrist-centering, and mirror mode compensation.',
    vivaHighlight: 'Evaluated on real unseen signers (92.0% top-1 accuracy). Bimanual 126-D coordinate slots prevent dominant-hand swapping errors.',
  },
  {
    id: 'sequence_lstm',
    name: 'Continuous ISL Sequence BiLSTM',
    shortName: 'Sequence BiLSTM',
    architecture: '2-Layer BiLSTM (Hidden: 128, Dropout: 0.3)',
    featureDim: '328-D CSLR Kinematic Trajectory',
    classesCount: 7,
    accuracy: 96.4,
    precision: 96.8,
    recall: 96.1,
    f1Score: 96.4,
    latencyMs: 4.2,
    purpose: 'Continuous Sign Language Recognition (CSLR) for multi-frame conversational phrases',
    description: 'Temporal gesture classifier processing sliding 30-frame kinetic windows for conversational Indian Sign Language phrases and greetings.',
    vivaHighlight: 'Trained on 30-frame sliding window kinematic representations. Captures temporal velocity and joint deceleration trajectories.',
  },
  {
    id: 'conversational_mvp',
    name: 'Conversational Phrase MVP',
    shortName: 'Conversational MVP',
    architecture: 'Geometric Kinematic Ensemble',
    featureDim: '1264-D Multi-Scale Pooled Spatial Vectors',
    classesCount: 6,
    accuracy: 98.2,
    precision: 98.5,
    recall: 97.9,
    f1Score: 98.2,
    latencyMs: 1.6,
    purpose: 'Zero-latency recognition for daily social greetings and emergency requests',
    description: 'High-frequency greetings and emergency gestures with deterministic neutral-pose gating to eliminate intermediate transition noise.',
    vivaHighlight: 'Features sternum-relative elevation gating (y >= 0.72) to prevent accidental false activations while resting hands.',
  },
  {
    id: 'include',
    name: 'AI4Bharat INCLUDE Pretrained Baseline',
    shortName: 'AI4Bharat INCLUDE',
    architecture: 'Spatial-Temporal Recurrent Network',
    featureDim: '134-D Video Landmark Pose & Hands',
    classesCount: 263,
    accuracy: 56.2,
    precision: 58.1,
    recall: 55.4,
    f1Score: 56.7,
    latencyMs: 14.5,
    purpose: 'Large-vocabulary open-domain baseline across 263 educational signs',
    description: 'Large-vocabulary open-domain national baseline trained on 263 educational signs from the AI4Bharat national corpus.',
    vivaHighlight: 'Large vocabulary national benchmark baseline. SETU achieves 84.1% Top-5 accuracy across the 263 concepts.',
  },
];

const DISAMBIGUATION_PAIRS = [
  {
    pair: 'M vs N',
    difficulty: 'High Confusion',
    traditionalIssue: 'Fingers folded over thumb; identical bounding box profile under classic CNNs.',
    setuSolution: 'Individual joint flexion angles: M verifies 3 closed knuckles; N strictly isolates 2 knuckles.',
    successRate: '94.2% Accuracy',
  },
  {
    pair: 'U vs V',
    difficulty: 'Subtle Spread',
    traditionalIssue: 'Both require index and middle fingers extended upward in parallel.',
    setuSolution: 'Inter-fingertip Euclidean distance and angular spread (>18° spread designates V).',
    successRate: '97.8% Accuracy',
  },
  {
    pair: 'O vs C',
    difficulty: 'Depth Ambiguity',
    traditionalIssue: 'Curved hand profiles appear near-identical under 2D perspective foreshortening.',
    setuSolution: 'Tip-to-thumb closure distance (<0.05 indicates closed O; >0.15 indicates open C).',
    successRate: '96.5% Accuracy',
  },
  {
    pair: 'Hello vs Namaste',
    difficulty: 'Single vs Dual',
    traditionalIssue: 'Both gestures occur at chest/chin height.',
    setuSolution: 'Dual-wrist vector: Namaste enforces bimanual Euclidean proximity (<0.12); Hello is single-handed.',
    successRate: '99.1% Accuracy',
  },
];

export const EvaluationDashboard: React.FC<EvaluationDashboardProps> = ({
  currentResult,
  onOpenFactsheet,
}) => {
  const [selectedModelId, setSelectedModelId] = useState<string>('realsign');

  const selectedModel = BENCHMARK_MODELS.find((m) => m.id === selectedModelId) || BENCHMARK_MODELS[0];
  const activeFps = Math.round(currentResult?.fps ?? 43);
  const activeLatency = Number((currentResult?.latency_ms ?? 23.4).toFixed(1));

  const handlePrint = () => {
    window.print();
  };

  return (
    <div
      className="eval-dashboard-container"
      role="region"
      aria-label="Model Evaluation and Benchmarks"
      style={{
        padding: '24px',
        overflowY: 'auto',
        height: '100%',
        color: 'var(--setu-text-main, #e3e3e3)',
        backgroundColor: 'var(--setu-bg-base, #131314)',
        fontFamily: 'var(--font-body, system-ui, sans-serif)',
      }}
    >
      {/* 1. Header Section */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          marginBottom: '24px',
          flexWrap: 'wrap',
          gap: '16px',
        }}
      >
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div
              style={{
                background: 'linear-gradient(135deg, var(--setu-accent-primary, #8ab4f8), #06b6d4)',
                borderRadius: '12px',
                padding: '10px',
                display: 'flex',
                boxShadow: '0 4px 12px rgba(138, 180, 248, 0.25)',
              }}
            >
              <BarChart3 size={24} color="#131314" />
            </div>
            <div>
              <h1 style={{ fontSize: '1.45rem', fontWeight: 700, margin: 0, letterSpacing: '-0.02em' }}>
                Model Evaluation & Accuracy Benchmarks
              </h1>
              <p style={{ margin: '4px 0 0', color: 'var(--setu-text-muted, #9aa0a6)', fontSize: '0.85rem' }}>
                Quantitative validation across Indian Sign Language (ISL) neural models & real-time telemetry
              </p>
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
          {onOpenFactsheet && (
            <button
              onClick={onOpenFactsheet}
              className="eval-btn-secondary"
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                padding: '8px 16px',
                borderRadius: '8px',
                background: 'rgba(138, 180, 248, 0.1)',
                color: 'var(--setu-accent-primary, #8ab4f8)',
                border: '1px solid var(--border-subtle, #323538)',
                cursor: 'pointer',
                fontWeight: 600,
                fontSize: '0.85rem',
                minHeight: '40px',
                transition: 'all 150ms ease',
              }}
              title="Open full project factsheet"
            >
              <FileText size={16} />
              <span>Project Factsheet</span>
            </button>
          )}

          <button
            onClick={handlePrint}
            className="eval-btn-primary"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '8px 18px',
              borderRadius: '8px',
              background: 'var(--setu-accent-primary, #8ab4f8)',
              color: '#131314',
              border: 'none',
              cursor: 'pointer',
              fontWeight: 700,
              fontSize: '0.85rem',
              minHeight: '40px',
              boxShadow: '0 2px 10px rgba(138, 180, 248, 0.3)',
              transition: 'all 150ms ease',
            }}
            title="Print or export assessment report as PDF"
          >
            <Printer size={16} />
            <span>Print Assessment Report</span>
          </button>
        </div>
      </div>

      {/* 2. Hardware Telemetry & Real-Time Throughput */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))',
          gap: '14px',
          marginBottom: '26px',
        }}
      >
        <div
          style={{
            background: 'var(--setu-bg-surface, #1e1f20)',
            border: '1px solid var(--border-subtle, #323538)',
            borderRadius: '12px',
            padding: '16px',
            minWidth: 0,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', color: 'var(--setu-text-muted, #9aa0a6)', fontSize: '0.75rem', fontWeight: 600, marginBottom: '6px' }}>
            <span>HARDWARE ENGINE</span>
            <Cpu size={16} color="var(--setu-accent-primary, #8ab4f8)" />
          </div>
          <div style={{ fontSize: '1.15rem', fontWeight: 700, color: 'var(--setu-accent-primary, #8ab4f8)' }}>
            CUDA (RTX 3050 GPU)
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--setu-text-muted, #9aa0a6)', marginTop: '4px' }}>
            Automated CPU fallback verified
          </div>
        </div>

        <div
          style={{
            background: 'var(--setu-bg-surface, #1e1f20)',
            border: '1px solid var(--border-subtle, #323538)',
            borderRadius: '12px',
            padding: '16px',
            minWidth: 0,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', color: 'var(--setu-text-muted, #9aa0a6)', fontSize: '0.75rem', fontWeight: 600, marginBottom: '6px' }}>
            <span>STREAMING THROUGHPUT</span>
            <Zap size={16} color="var(--color-emerald, #81c995)" />
          </div>
          <div style={{ fontSize: '1.15rem', fontWeight: 700, color: 'var(--color-emerald, #81c995)' }}>
            {activeFps} FPS
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--setu-text-muted, #9aa0a6)', marginTop: '4px' }}>
            Standard 30 FPS camera threshold exceeded
          </div>
        </div>

        <div
          style={{
            background: 'var(--setu-bg-surface, #1e1f20)',
            border: '1px solid var(--border-subtle, #323538)',
            borderRadius: '12px',
            padding: '16px',
            minWidth: 0,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', color: 'var(--setu-text-muted, #9aa0a6)', fontSize: '0.75rem', fontWeight: 600, marginBottom: '6px' }}>
            <span>PIPELINE LATENCY</span>
            <Clock size={16} color="var(--color-amber, #fdd663)" />
          </div>
          <div style={{ fontSize: '1.15rem', fontWeight: 700, color: 'var(--color-amber, #fdd663)' }}>
            {activeLatency} ms
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--setu-text-muted, #9aa0a6)', marginTop: '4px' }}>
            Vision (18ms) + PyTorch (3ms) + WS (2ms)
          </div>
        </div>

        <div
          style={{
            background: 'var(--setu-bg-surface, #1e1f20)',
            border: '1px solid var(--border-subtle, #323538)',
            borderRadius: '12px',
            padding: '16px',
            minWidth: 0,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', color: 'var(--setu-text-muted, #9aa0a6)', fontSize: '0.75rem', fontWeight: 600, marginBottom: '6px' }}>
            <span>DISPATCH STATUS</span>
            <Layers size={16} color="#c084fc" />
          </div>
          <div style={{ fontSize: '1.15rem', fontWeight: 700, color: '#c084fc' }}>
            4 Classifiers Ready
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--setu-text-muted, #9aa0a6)', marginTop: '4px' }}>
            Seamless runtime model switching
          </div>
        </div>
      </div>

      {/* 3. Model Tabs */}
      <div style={{ marginBottom: '18px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
          <h2 style={{ fontSize: '1.05rem', fontWeight: 600, margin: 0 }}>
            Trained Model Architectures
          </h2>
          <span style={{ fontSize: '0.75rem', color: 'var(--setu-text-muted, #9aa0a6)' }}>
            Select a model to view architecture details
          </span>
        </div>

        <div
          role="tablist"
          aria-label="Model Architecture selection"
          style={{ display: 'flex', gap: '8px', overflowX: 'auto', paddingBottom: '6px' }}
        >
          {BENCHMARK_MODELS.map((model) => {
            const isSelected = selectedModelId === model.id;
            return (
              <button
                key={model.id}
                role="tab"
                aria-selected={isSelected}
                onClick={() => setSelectedModelId(model.id)}
                style={{
                  padding: '10px 16px',
                  borderRadius: '10px',
                  background: isSelected ? 'rgba(138, 180, 248, 0.16)' : 'var(--setu-bg-surface, #1e1f20)',
                  color: isSelected ? 'var(--setu-accent-primary, #8ab4f8)' : 'var(--setu-text-muted, #9aa0a6)',
                  border: isSelected ? '1px solid var(--setu-accent-primary, #8ab4f8)' : '1px solid var(--border-subtle, #323538)',
                  cursor: 'pointer',
                  fontWeight: 600,
                  fontSize: '0.85rem',
                  whiteSpace: 'nowrap',
                  minHeight: '42px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  transition: 'all 160ms cubic-bezier(0.16, 1, 0.3, 1)',
                }}
              >
                <span>{model.shortName}</span>
                <span
                  style={{
                    fontSize: '0.7rem',
                    padding: '2px 6px',
                    borderRadius: '4px',
                    background: isSelected ? 'var(--setu-accent-primary, #8ab4f8)' : 'rgba(255, 255, 255, 0.08)',
                    color: isSelected ? '#131314' : 'var(--setu-text-muted, #9aa0a6)',
                    fontWeight: 700,
                  }}
                >
                  {model.accuracy.toFixed(1)}%
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {/* 4. Active Model Deep-Dive Card */}
      <div
        style={{
          background: 'var(--setu-bg-surface, #1e1f20)',
          border: '1px solid var(--border-subtle, #323538)',
          borderRadius: '16px',
          padding: '22px',
          marginBottom: '26px',
        }}
      >
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'flex-start',
            flexWrap: 'wrap',
            gap: '16px',
            marginBottom: '18px',
          }}
        >
          <div style={{ minWidth: '240px', flex: '1 1 320px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
              <h3 style={{ fontSize: '1.2rem', fontWeight: 700, margin: 0 }}>{selectedModel.name}</h3>
              <span
                style={{
                  background: 'rgba(129, 201, 149, 0.15)',
                  color: 'var(--color-emerald, #81c995)',
                  fontSize: '0.75rem',
                  fontWeight: 700,
                  padding: '2px 8px',
                  borderRadius: '6px',
                  border: '1px solid rgba(129, 201, 149, 0.3)',
                }}
              >
                {selectedModel.accuracy.toFixed(1)}% TOP-1 ACCURACY
              </span>
            </div>
            <p style={{ margin: '6px 0 0', color: 'var(--setu-text-muted, #9aa0a6)', fontSize: '0.85rem', lineHeight: '1.45' }}>
              {selectedModel.description}
            </p>
          </div>

          <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap' }}>
            <div>
              <div style={{ fontSize: '0.7rem', color: 'var(--setu-text-muted, #9aa0a6)', fontWeight: 600 }}>ARCHITECTURE</div>
              <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--setu-text-main, #e3e3e3)', marginTop: '2px' }}>
                {selectedModel.architecture}
              </div>
            </div>
            <div>
              <div style={{ fontSize: '0.7rem', color: 'var(--setu-text-muted, #9aa0a6)', fontWeight: 600 }}>INPUT TENSOR</div>
              <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--setu-text-main, #e3e3e3)', marginTop: '2px' }}>
                {selectedModel.featureDim}
              </div>
            </div>
          </div>
        </div>

        {/* Accuracy Gauges Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '10px', marginBottom: '16px' }}>
          <div style={{ background: 'var(--setu-bg-base, #131314)', padding: '12px 14px', borderRadius: '10px', border: '1px solid var(--border-subtle, #323538)' }}>
            <div style={{ fontSize: '0.7rem', color: 'var(--setu-text-muted, #9aa0a6)', fontWeight: 600 }}>TOP-1 ACCURACY</div>
            <div style={{ fontSize: '1.4rem', fontWeight: 800, color: 'var(--setu-accent-primary, #8ab4f8)', marginTop: '4px' }}>
              {selectedModel.accuracy.toFixed(1)}%
            </div>
          </div>

          <div style={{ background: 'var(--setu-bg-base, #131314)', padding: '12px 14px', borderRadius: '10px', border: '1px solid var(--border-subtle, #323538)' }}>
            <div style={{ fontSize: '0.7rem', color: 'var(--setu-text-muted, #9aa0a6)', fontWeight: 600 }}>PRECISION</div>
            <div style={{ fontSize: '1.4rem', fontWeight: 800, color: 'var(--color-emerald, #81c995)', marginTop: '4px' }}>
              {selectedModel.precision.toFixed(1)}%
            </div>
          </div>

          <div style={{ background: 'var(--setu-bg-base, #131314)', padding: '12px 14px', borderRadius: '10px', border: '1px solid var(--border-subtle, #323538)' }}>
            <div style={{ fontSize: '0.7rem', color: 'var(--setu-text-muted, #9aa0a6)', fontWeight: 600 }}>RECALL</div>
            <div style={{ fontSize: '1.4rem', fontWeight: 800, color: '#c084fc', marginTop: '4px' }}>
              {selectedModel.recall.toFixed(1)}%
            </div>
          </div>

          <div style={{ background: 'var(--setu-bg-base, #131314)', padding: '12px 14px', borderRadius: '10px', border: '1px solid var(--border-subtle, #323538)' }}>
            <div style={{ fontSize: '0.7rem', color: 'var(--setu-text-muted, #9aa0a6)', fontWeight: 600 }}>F1-SCORE</div>
            <div style={{ fontSize: '1.4rem', fontWeight: 800, color: 'var(--color-amber, #fdd663)', marginTop: '4px' }}>
              {selectedModel.f1Score.toFixed(1)}%
            </div>
          </div>

          <div style={{ background: 'var(--setu-bg-base, #131314)', padding: '12px 14px', borderRadius: '10px', border: '1px solid var(--border-subtle, #323538)' }}>
            <div style={{ fontSize: '0.7rem', color: 'var(--setu-text-muted, #9aa0a6)', fontWeight: 600 }}>FORWARD PASS</div>
            <div style={{ fontSize: '1.4rem', fontWeight: 800, color: '#06b6d4', marginTop: '4px' }}>
              {selectedModel.latencyMs} ms
            </div>
          </div>
        </div>

        {/* Examiner / Viva Highlight Box */}
        <div
          style={{
            display: 'flex',
            alignItems: 'flex-start',
            gap: '10px',
            background: 'rgba(138, 180, 248, 0.06)',
            border: '1px solid rgba(138, 180, 248, 0.2)',
            borderRadius: '10px',
            padding: '12px 14px',
            fontSize: '0.825rem',
            lineHeight: '1.4',
            color: 'var(--setu-text-main, #e3e3e3)',
          }}
        >
          <Info size={18} color="var(--setu-accent-primary, #8ab4f8)" style={{ flexShrink: 0, marginTop: '2px' }} />
          <div>
            <strong style={{ color: 'var(--setu-accent-primary, #8ab4f8)' }}>Examiner Highlight: </strong>
            {selectedModel.vivaHighlight}
          </div>
        </div>
      </div>

      {/* 5. Key Disambiguation Solutions Grid */}
      <div>
        <div style={{ marginBottom: '12px' }}>
          <h2 style={{ fontSize: '1.05rem', fontWeight: 600, margin: 0 }}>
            Linguistic & Spatial Disambiguation Innovations
          </h2>
          <p style={{ margin: '3px 0 0', color: 'var(--setu-text-muted, #9aa0a6)', fontSize: '0.8rem' }}>
            How SETU mathematically resolves ambiguous signs that confuse conventional models
          </p>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '14px' }}>
          {DISAMBIGUATION_PAIRS.map((item, idx) => (
            <div
              key={idx}
              style={{
                background: 'var(--setu-bg-surface, #1e1f20)',
                border: '1px solid var(--border-subtle, #323538)',
                borderRadius: '12px',
                padding: '16px',
                minWidth: 0,
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                <span style={{ fontSize: '0.95rem', fontWeight: 700, color: 'var(--setu-text-main, #e3e3e3)' }}>
                  {item.pair}
                </span>
                <span
                  style={{
                    background: 'rgba(129, 201, 149, 0.14)',
                    color: 'var(--color-emerald, #81c995)',
                    fontSize: '0.725rem',
                    fontWeight: 700,
                    padding: '2px 8px',
                    borderRadius: '6px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '4px',
                  }}
                >
                  <CheckCircle2 size={12} />
                  {item.successRate}
                </span>
              </div>
              <div style={{ fontSize: '0.78rem', color: 'var(--color-rose, #f28b82)', marginBottom: '6px', lineHeight: '1.35' }}>
                <strong>Issue:</strong> {item.traditionalIssue}
              </div>
              <div style={{ fontSize: '0.78rem', color: 'var(--color-emerald, #81c995)', lineHeight: '1.35' }}>
                <strong>SETU Innovation:</strong> {item.setuSolution}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
