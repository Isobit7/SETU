import React, { useRef, useState } from 'react';
import { TopNavbar } from './components/TopNavbar';
import { CameraWorkspace } from './components/CameraWorkspace';
import { CameraControls } from './components/CameraControls';
import { TranslationPanel } from './components/TranslationPanel';
import { VocabularyModal } from './components/VocabularyModal';
import { AlphabetModal } from './components/AlphabetModal';
import { SettingsModal } from './components/SettingsModal';
import { HistoryModal } from './components/HistoryModal';
import { SavedModal } from './components/SavedModal';
import { HelpModal } from './components/HelpModal';
import { IncludeVideoModal } from './components/IncludeVideoModal';
import { useRecognitionStream } from './hooks/useRecognitionStream';
import type { LandmarkStyles, SubtitleSize } from './types';
import './index.css';

export const App: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement | null>(null);

  // Active top navigation section
  const [activeTab, setActiveTab] = useState<string>('translator');

  // Modals
  const [isVocabModalOpen, setIsVocabModalOpen] = useState<boolean>(false);
  const [isAlphabetModalOpen, setIsAlphabetModalOpen] = useState<boolean>(false);
  const [isSettingsModalOpen, setIsSettingsModalOpen] = useState<boolean>(false);
  const [isHistoryModalOpen, setIsHistoryModalOpen] = useState<boolean>(false);
  const [isSavedModalOpen, setIsSavedModalOpen] = useState<boolean>(false);
  const [isHelpModalOpen, setIsHelpModalOpen] = useState<boolean>(false);
  const [isIncludeModalOpen, setIsIncludeModalOpen] = useState<boolean>(false);

  // Subtitle & Overlay Preferences
  const [subtitleVisible, setSubtitleVisible] = useState<boolean>(true);
  const [subtitleSize, setSubtitleSize] = useState<SubtitleSize>('medium');
  const [confidenceThreshold, setConfidenceThreshold] = useState<number>(0.35);

  // Sentence Builder & Auto-Speak State
  const [sentence, setSentence] = useState<string>('');
  const [autoSpeak, setAutoSpeak] = useState<boolean>(false);

  // Stability Gate & Cooldown Refs
  const candidateStreakRef = useRef<{ sign: string; count: number }>({ sign: '', count: 0 });
  const lastAppendedSignRef = useRef<string>('');
  const lastAppendedTimeRef = useRef<number>(0);
  const hadHandDropRef = useRef<boolean>(false);

  // Callback when a new prediction arrives from WebSocket
  const handlePredictionChange = (data: any) => {
    if (!data) return;

    const isHand = data.hand_detected ?? false;
    const isKnown = data.is_known_sign ?? false;
    const conf = data.confidence ?? 0;
    const label = data.label ?? '';

    const noSignProb = data.probabilities?.['no_sign'] ?? 0;

    // If hand dropped or no sign, note the drop and reset streak
    if (!isHand || label === 'no_sign' || label === 'unknown' || label === 'Awaiting Sign Input' || noSignProb > 0.70 || !isKnown) {
      candidateStreakRef.current = { sign: '', count: 0 };
      hadHandDropRef.current = true;
      return;
    }

    // Check user confidence threshold gate
    const activeThreshold = confidenceThreshold;
    if (conf < activeThreshold) {
      candidateStreakRef.current = { sign: '', count: 0 };
      return;
    }

    // Determine the sign/character to append
    let signText = '';
    if (data.active_model === 'realsign') {
      signText = (data.realsign_prediction?.letter || label).toUpperCase();
    } else {
      signText = (data.display_name || label.replace('_', ' ')).toUpperCase();
    }

    if (!signText || signText === 'NO_SIGN' || signText === 'UNKNOWN' || signText === 'AWAITING SIGN INPUT') return;

    // Temporal Streak Counter: 2 consecutive frames for Alphabet, 3 for Sequence
    const requiredStreak = data.active_model === 'realsign' ? 2 : 3;

    if (candidateStreakRef.current.sign === signText) {
      candidateStreakRef.current.count += 1;
    } else {
      candidateStreakRef.current = { sign: signText, count: 1 };
      return;
    }

    if (candidateStreakRef.current.count < requiredStreak) {
      return;
    }

    // Refractory Cooldown & Duplicate Prevention
    const now = Date.now();
    const timeSinceLast = now - lastAppendedTimeRef.current;
    const isSameSign = signText === lastAppendedSignRef.current;

    // Duplicate lockout: 350ms lockout AND requires idle baseline transition
    if (isSameSign && (timeSinceLast < 350 || !hadHandDropRef.current)) {
      return;
    }

    // Confirmed append! (Distinct new signs commit immediately once consecutive streak is met)
    lastAppendedSignRef.current = signText;
    lastAppendedTimeRef.current = now;
    hadHandDropRef.current = false;

    setSentence((prev) => {
      // If single-character alphabet letter (A-Z)
      if (signText.length === 1) {
        return prev + signText;
      }
      // If phrase (e.g., "HELLO", "THANK YOU")
      const trimmed = prev.trim();
      if (!trimmed) {
        return signText + ' ';
      }
      return prev.endsWith(' ') ? `${prev}${signText} ` : `${prev} ${signText} `;
    });

    // Auto-Speak if enabled
    if (autoSpeak && 'speechSynthesis' in window) {
      try {
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(signText);
        utterance.lang = 'en-US';
        utterance.rate = 1.0;
        window.speechSynthesis.speak(utterance);
      } catch (e) {
        console.warn('Auto-speak failed:', e);
      }
    }
  };

  // Hand-only landmark layer styles (Pose and Face permanently excluded)
  const [landmarkStyles, setLandmarkStyles] = useState<LandmarkStyles>({
    showLeftHand: true,
    showRightHand: true,
    mirrored: true,
  });

  // Connect to Live Recognition Pipeline Hook
  const {
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
    clearHistory,
  } = useRecognitionStream({
    videoRef,
    confidenceThreshold,
    onPredictionChange: handlePredictionChange,
  });

  // Toggle hand landmarks on/off together
  const handleToggleLandmarks = () => {
    const anyActive = landmarkStyles.showLeftHand || landmarkStyles.showRightHand;
    setLandmarkStyles((prev: LandmarkStyles) => ({
      ...prev,
      showLeftHand: !anyActive,
      showRightHand: !anyActive,
    }));
  };

  const handleToggleMirror = () => {
    setLandmarkStyles((prev: LandmarkStyles) => ({
      ...prev,
      mirrored: !prev.mirrored,
    }));
  };

  const handleNavSelect = (tab: string) => {
    if (tab === 'history') {
      setIsHistoryModalOpen(true);
    } else if (tab === 'saved') {
      setIsSavedModalOpen(true);
    } else if (tab === 'vocabulary') {
      setIsVocabModalOpen(true);
    } else if (tab === 'alphabet') {
      setIsAlphabetModalOpen(true);
    } else if (tab === 'settings') {
      setIsSettingsModalOpen(true);
    } else if (tab === 'help') {
      setIsHelpModalOpen(true);
    } else if (tab === 'include') {
      setIsIncludeModalOpen(true);
    } else {
      setActiveTab(tab);
    }
  };

  const handleClearSentence = () => {
    setSentence('');
    lastAppendedSignRef.current = '';
    candidateStreakRef.current = { sign: '', count: 0 };
  };

  const handleBackspaceSentence = () => {
    setSentence((prev) => (prev.length > 0 ? prev.slice(0, -1) : ''));
  };

  const handleAddSpaceSentence = () => {
    setSentence((prev) => (prev.length > 0 && !prev.endsWith(' ') ? prev + ' ' : prev));
  };

  return (
    <div className="app-container">
      {/* Responsive Top Navigation */}
      <TopNavbar
        activeTab={activeTab}
        setActiveTab={handleNavSelect}
        isBackendConnected={isBackendConnected}
        fps={currentResult?.fps || 0}
        latencyMs={currentResult?.latency_ms || 0}
        onOpenSettings={() => setIsSettingsModalOpen(true)}
        onOpenVocabulary={() => setIsVocabModalOpen(true)}
        onOpenAlphabetModal={() => setIsAlphabetModalOpen(true)}
        onOpenHistory={() => setIsHistoryModalOpen(true)}
        onOpenSaved={() => setIsSavedModalOpen(true)}
        onOpenHelp={() => setIsHelpModalOpen(true)}
        onOpenIncludeModal={() => setIsIncludeModalOpen(true)}
        historyCount={history.length}
      />

      {/* Main Workspace Layout (Zero-Scroll 100vh Locked Grid) */}
      <main className="workspace-grid" role="main">
        {/* Left Column: Live Camera, Micro Status & Controls */}
        <section className="workspace-camera-section" aria-label="Camera and Sign Recognition View">
          <CameraWorkspace
            videoRef={videoRef}
            cameraStatus={cameraStatus}
            errorMessage={errorMessage}
            currentResult={currentResult}
            landmarkStyles={landmarkStyles}
            subtitleVisible={subtitleVisible}
            subtitleSize={subtitleSize}
            onStartCamera={startCamera}
            onRetry={startCamera}
          />

          {/* Under-Camera Controls */}
          <CameraControls
            cameraStatus={cameraStatus}
            availableCameras={availableCameras}
            selectedCameraId={selectedCameraId}
            activeModel={activeModel}
            onSwitchModel={switchModel}
            confidenceThreshold={confidenceThreshold}
            onThresholdChange={setConfidenceThreshold}
            autoSpeak={autoSpeak}
            onToggleAutoSpeak={() => setAutoSpeak((prev) => !prev)}
            landmarkStyles={landmarkStyles}
            subtitleVisible={subtitleVisible}
            onStart={startCamera}
            onPause={pauseDetection}
            onResume={resumeDetection}
            onStop={stopCamera}
            onReset={resetSession}
            onSwitchCamera={(id) => setSelectedCameraId(id)}
            onToggleLandmarks={handleToggleLandmarks}
            onToggleMirror={handleToggleMirror}
            onToggleSubtitles={() => setSubtitleVisible(!subtitleVisible)}
            onOpenIncludeModal={() => setIsIncludeModalOpen(true)}
          />
        </section>

        {/* Right Column: Active Output, Candidate Scores & Sentence Buffer */}
        <aside className="workspace-side-section" aria-label="Translation Output and Candidates">
          <TranslationPanel
            currentResult={currentResult}
            onClearCurrent={resetSession}
            onOpenHistory={() => setIsHistoryModalOpen(true)}
            historyCount={history.length}
            sentence={sentence}
            onClearSentence={handleClearSentence}
            onBackspaceSentence={handleBackspaceSentence}
            onAddSpaceSentence={handleAddSpaceSentence}
          />
        </aside>
      </main>

      {/* Modals */}
      <AlphabetModal
        isOpen={isAlphabetModalOpen}
        onClose={() => setIsAlphabetModalOpen(false)}
      />

      <VocabularyModal
        isOpen={isVocabModalOpen}
        onClose={() => setIsVocabModalOpen(false)}
      />

      <SettingsModal
        isOpen={isSettingsModalOpen}
        onClose={() => setIsSettingsModalOpen(false)}
        landmarkStyles={landmarkStyles}
        setLandmarkStyles={setLandmarkStyles}
        subtitleSize={subtitleSize}
        setSubtitleSize={setSubtitleSize}
        confidenceThreshold={confidenceThreshold}
        setConfidenceThreshold={setConfidenceThreshold}
      />

      <HistoryModal
        isOpen={isHistoryModalOpen}
        onClose={() => setIsHistoryModalOpen(false)}
        history={history}
        onClearHistory={clearHistory}
      />

      <SavedModal
        isOpen={isSavedModalOpen}
        onClose={() => setIsSavedModalOpen(false)}
        history={history}
      />

      <HelpModal
        isOpen={isHelpModalOpen}
        onClose={() => setIsHelpModalOpen(false)}
      />

      <IncludeVideoModal
        isOpen={isIncludeModalOpen}
        onClose={() => setIsIncludeModalOpen(false)}
      />
    </div>
  );
};

export default App;
