import React, { useRef, useState, useCallback, useEffect } from 'react';
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
import { RecordSignModal } from './components/RecordSignModal';
import { TextToSignWorkspace } from './components/TextToSignWorkspace';
import { PracticeStudio } from './components/PracticeStudio';
import { ConversationStudio } from './components/ConversationStudio';
import { EvaluationDashboard } from './components/EvaluationDashboard';
import { useRecognitionStream } from './hooks/useRecognitionStream';
import { useTTS } from './hooks/useTTS';
import { useSentenceBuilder } from './hooks/useSentenceBuilder';
import type { BufferInsertionMode, FramingFeedback, LandmarkStyles, SubtitleSize } from './types';
import './index.css';

export const App: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement | null>(null);

  // URL query parameter helper for deep-linking
  const urlParams = typeof window !== 'undefined' ? new URLSearchParams(window.location.search) : null;
  const initialTab = urlParams?.get('tab') || 'translator';
  const initialModal = urlParams?.get('modal') || '';
  const initialDir = urlParams?.get('dir') === 'en-to-isl' ? 'en-to-isl' : 'isl-to-en';

  // 2-Way Translation Direction ('isl-to-en' or 'en-to-isl')
  const [translationDirection, setTranslationDirection] = useState<'isl-to-en' | 'en-to-isl'>(initialDir);

  // Active top navigation section
  const [activeTab, setActiveTab] = useState<string>(initialTab);

  // Modals
  const [isVocabModalOpen, setIsVocabModalOpen] = useState<boolean>(initialModal === 'vocab');
  const [isAlphabetModalOpen, setIsAlphabetModalOpen] = useState<boolean>(initialModal === 'alphabet');
  const [isSettingsModalOpen, setIsSettingsModalOpen] = useState<boolean>(initialModal === 'settings');
  const [isHistoryModalOpen, setIsHistoryModalOpen] = useState<boolean>(initialModal === 'history');
  const [isSavedModalOpen, setIsSavedModalOpen] = useState<boolean>(initialModal === 'saved');
  const [isHelpModalOpen, setIsHelpModalOpen] = useState<boolean>(initialModal === 'help');
  const [isIncludeModalOpen, setIsIncludeModalOpen] = useState<boolean>(initialModal === 'include');
  const [isRecordModalOpen, setIsRecordModalOpen] = useState<boolean>(initialModal === 'record');

  // Backend Health Poll
  const [backendReady, setBackendReady] = useState<boolean>(false);
  useEffect(() => {
    let isMounted = true;
    const checkHealth = async () => {
      try {
        const res = await fetch(`http://${window.location.hostname}:8000/health`);
        if (isMounted) setBackendReady(res.ok);
      } catch {
        if (isMounted) setBackendReady(false);
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 8000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  // Subtitle & Overlay Preferences
  const [subtitleVisible, setSubtitleVisible] = useState<boolean>(true);
  const [subtitleSize, setSubtitleSize] = useState<SubtitleSize>('medium');
  const [confidenceThreshold, setConfidenceThreshold] = useState<number>(0.35);

  // Proactive Framing Guide & Recognition Gating State
  const [showFramingGuide, setShowFramingGuide] = useState<boolean>(false);
  const [enforceFraming, setEnforceFraming] = useState<boolean>(false);
  const [framingFeedback, setFramingFeedback] = useState<FramingFeedback>({
    status: 'searching',
    message: 'Ready to sign',
    isOptimal: true,
    score: 0,
    handsDetected: false,
  });

  const isFramingGated = enforceFraming && !framingFeedback.isOptimal;

  // Browser-Native Text-to-Speech (TTS) Hook
  const {
    isSpeaking,
    autoSpeakEnabled,
    availableVoices,
    selectedVoice,
    targetLanguage,
    setTargetLanguage,
    speak,
    stop,
    toggleAutoSpeak,
    setSelectedVoice,
  } = useTTS();

  // Decoupled Sentence Accumulator & Temporal Debouncer Hook
  const {
    sentence,
    setSentence,
    handlePredictionChange,
    handleClear: handleClearSentence,
    handleBackspace: handleBackspaceSentence,
  } = useSentenceBuilder({
    confidenceThreshold,
    autoSpeakEnabled,
    speak,
    stop,
    targetLanguage,
    onClearBackend: () => clearSentence(),
    onBackspaceBackend: () => {
      backspaceSentence();
      undoLastToken();
    },
  });

  // Filtered prediction callback: passes clean predictions to sentence builder
  const handleFilteredPredictionChange = useCallback(
    (pred: any) => {
      handlePredictionChange(pred);
    },
    [handlePredictionChange]
  );

  // Hand-only landmark layer styles (Pose and Face permanently excluded)
  const [landmarkStyles, setLandmarkStyles] = useState<LandmarkStyles>({
    showLeftHand: true,
    showRightHand: true,
    showFace: false,
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
    clearSentence,
    backspaceSentence,
    clearHistory,
    resolveDisambiguation,
    acceptPrediction,
    setBufferMode,
    undoLastToken,
  } = useRecognitionStream({
    videoRef,
    confidenceThreshold,
    onPredictionChange: handleFilteredPredictionChange,
    isFramingGated,
  });

  // Sentence Buffer Insertion Mode ('auto' stream vs 'head_nod' confirmed)
  const [bufferInsertionMode, setBufferInsertionMode] = useState<BufferInsertionMode>('head_nod');

  const handleSetBufferInsertionMode = useCallback(
    (mode: BufferInsertionMode) => {
      setBufferInsertionMode(mode);
      setBufferMode(mode);
    },
    [setBufferMode]
  );

  // Toggle hand landmarks on/off together
  const handleToggleLandmarks = () => {
    const anyActive = landmarkStyles.showLeftHand || landmarkStyles.showRightHand;
    setLandmarkStyles((prev: LandmarkStyles) => ({
      ...prev,
      showLeftHand: !anyActive,
      showRightHand: !anyActive,
      showFace: false,
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

  const handleResetSession = () => {
    stop();
    handleClearSentence();
    resetSession();
  };

  const handleAddSpaceSentence = useCallback(() => {
    setSentence((prev) => (prev.length > 0 && !prev.endsWith(' ') ? prev + ' ' : prev));
  }, [setSentence]);

  // Global assistive keyboard shortcuts (Space to separate words, Backspace to delete, S to speak)
  React.useEffect(() => {
    const handleGlobalKeyDown = (e: KeyboardEvent) => {
      const activeTag = (document.activeElement?.tagName || '').toLowerCase();
      if (activeTag === 'input' || activeTag === 'textarea' || activeTag === 'select') return;

      if (
        isVocabModalOpen ||
        isAlphabetModalOpen ||
        isSettingsModalOpen ||
        isHistoryModalOpen ||
        isSavedModalOpen ||
        isHelpModalOpen ||
        isIncludeModalOpen ||
        isRecordModalOpen
      ) {
        return;
      }

      if (translationDirection !== 'isl-to-en') return;

      if (e.key === ' ' || e.code === 'Space') {
        e.preventDefault();
        handleAddSpaceSentence();
      } else if (e.key === 'Backspace') {
        e.preventDefault();
        handleBackspaceSentence();
      } else if ((e.key === 's' || e.key === 'S') && !e.ctrlKey && !e.metaKey && !e.altKey) {
        e.preventDefault();
        const textToSpeak = (currentResult?.translated_text || sentence || '').trim();
        if (textToSpeak) {
          if (isSpeaking) {
            stop();
          } else {
            speak(textToSpeak);
          }
        }
      }
    };

    window.addEventListener('keydown', handleGlobalKeyDown);
    return () => window.removeEventListener('keydown', handleGlobalKeyDown);
  }, [
    sentence,
    currentResult?.translated_text,
    isSpeaking,
    speak,
    stop,
    handleAddSpaceSentence,
    handleBackspaceSentence,
    translationDirection,
    isVocabModalOpen,
    isAlphabetModalOpen,
    isSettingsModalOpen,
    isHistoryModalOpen,
    isSavedModalOpen,
    isHelpModalOpen,
    isIncludeModalOpen,
    isRecordModalOpen,
  ]);

  return (
    <div className="app-container">
      {/* Responsive Top Navigation */}
      <TopNavbar
        activeTab={activeTab}
        setActiveTab={handleNavSelect}
        isBackendConnected={isBackendConnected}
        backendReady={backendReady}
        fps={currentResult?.fps || 0}
        latencyMs={currentResult?.latency_ms || 0}
        onOpenSettings={() => setIsSettingsModalOpen(true)}
        onOpenVocabulary={() => setIsVocabModalOpen(true)}
        onOpenAlphabetModal={() => setIsAlphabetModalOpen(true)}
        onOpenHistory={() => setIsHistoryModalOpen(true)}
        onOpenSaved={() => setIsSavedModalOpen(true)}
        onOpenHelp={() => setIsHelpModalOpen(true)}
        onOpenIncludeModal={() => setIsIncludeModalOpen(true)}
        onOpenRecordModal={() => setIsRecordModalOpen(true)}
        historyCount={history.length}
        autoSpeakEnabled={autoSpeakEnabled}
        onToggleAutoSpeak={toggleAutoSpeak}
        isSpeaking={isSpeaking}
        translationDirection={translationDirection}
        onToggleDirection={() =>
          setTranslationDirection((prev) => (prev === 'isl-to-en' ? 'en-to-isl' : 'isl-to-en'))
        }
      />

      {/* Main Workspace Layout (Zero-Scroll 100vh Locked Grid) */}
      <main className={`workspace-grid ${translationDirection === 'en-to-isl' ? 'bidirectional-active' : ''} ${activeTab === 'practice' || activeTab === 'evaluation' ? 'practice-mode-active' : ''}`} role="main">
        {activeTab === 'evaluation' ? (
          <EvaluationDashboard
            currentResult={currentResult}
            onOpenFactsheet={() => window.open('docs/showcase/index.html', '_blank')}
          />
        ) : activeTab === 'conversation' ? (
          <ConversationStudio
            currentResult={currentResult}
            onSpeak={speak}
            videoRef={videoRef}
            isCameraActive={cameraStatus === 'detecting' || cameraStatus === 'ready'}
            onStartCamera={startCamera}
            onStopCamera={stopCamera}
          />
        ) : translationDirection === 'isl-to-en' ? (
          <>
            {/* Left Column: Live Camera, Persistent Prediction Bar & Controls */}
            <section className="workspace-camera-section" aria-label="Camera and Sign Recognition View">
              <CameraWorkspace
                videoRef={videoRef}
                cameraStatus={cameraStatus}
                errorMessage={errorMessage}
                currentResult={currentResult}
                landmarkStyles={landmarkStyles}
                subtitleVisible={subtitleVisible}
                subtitleSize={subtitleSize}
                showFramingGuide={showFramingGuide}
                onToggleFramingGuide={() => setShowFramingGuide((prev) => !prev)}
                enforceFraming={enforceFraming}
                onToggleEnforceFraming={() => setEnforceFraming((prev) => !prev)}
                onFramingChange={setFramingFeedback}
                onStartCamera={startCamera}
                onRetry={startCamera}
                onOpenVocabulary={() => setIsVocabModalOpen(true)}
                onResolveDisambiguation={resolveDisambiguation}
                onAcceptPrediction={acceptPrediction}
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
                autoSpeak={autoSpeakEnabled}
                onToggleAutoSpeak={toggleAutoSpeak}
                landmarkStyles={landmarkStyles}
                subtitleVisible={subtitleVisible}
                showFramingGuide={showFramingGuide}
                onToggleFramingGuide={() => setShowFramingGuide((prev) => !prev)}
                enforceFraming={enforceFraming}
                onToggleEnforceFraming={() => setEnforceFraming((prev) => !prev)}
                onStart={startCamera}
                onPause={pauseDetection}
                onResume={resumeDetection}
                onStop={stopCamera}
                onReset={handleResetSession}
                onSwitchCamera={(id) => setSelectedCameraId(id)}
                onToggleLandmarks={handleToggleLandmarks}
                onToggleMirror={handleToggleMirror}
                onToggleSubtitles={() => setSubtitleVisible(!subtitleVisible)}
                onOpenIncludeModal={() => setIsIncludeModalOpen(true)}
                onOpenRecordModal={() => setIsRecordModalOpen(true)}
              />
            </section>

            {/* Right Column: Translation Panel OR Interactive Practice Studio */}
            <aside className="workspace-side-section" aria-label="Translation Output and Candidates">
              {activeTab === 'practice' ? (
                <PracticeStudio
                  currentResult={currentResult}
                  onSpeak={speak}
                  onClose={() => setActiveTab('translator')}
                />
              ) : (
                <TranslationPanel
                  currentResult={currentResult}
                  onClearCurrent={handleResetSession}
                  onOpenHistory={() => setIsHistoryModalOpen(true)}
                  historyCount={history.length}
                  sentence={sentence}
                  translatedText={currentResult?.translated_text}
                  onClearSentence={handleClearSentence}
                  onBackspaceSentence={handleBackspaceSentence}
                  onAddSpaceSentence={handleAddSpaceSentence}
                  isSpeaking={isSpeaking}
                  onSpeak={speak}
                  onStopSpeak={stop}
                  bufferInsertionMode={bufferInsertionMode}
                  onSetBufferInsertionMode={handleSetBufferInsertionMode}
                />
              )}
            </aside>
          </>
        ) : (
          <TextToSignWorkspace onSpeak={speak} isSpeaking={isSpeaking} />
        )}
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
        activeModel={activeModel}
        onSwitchModel={switchModel}
        confidenceThreshold={confidenceThreshold}
        setConfidenceThreshold={setConfidenceThreshold}
        bufferInsertionMode={bufferInsertionMode}
        onSetBufferInsertionMode={handleSetBufferInsertionMode}
        availableCameras={availableCameras}
        selectedCameraId={selectedCameraId}
        onSelectCamera={setSelectedCameraId}
        landmarkStyles={landmarkStyles}
        setLandmarkStyles={setLandmarkStyles}
        subtitleSize={subtitleSize}
        setSubtitleSize={setSubtitleSize}
        autoSpeakEnabled={autoSpeakEnabled}
        onToggleAutoSpeak={toggleAutoSpeak}
        targetLanguage={targetLanguage}
        onSelectLanguage={setTargetLanguage}
        availableVoices={availableVoices}
        selectedVoice={selectedVoice}
        onSelectVoice={setSelectedVoice}
        onTestSpeak={speak}
        isSpeaking={isSpeaking}
        enforceFraming={enforceFraming}
        onToggleEnforceFraming={() => setEnforceFraming((prev) => !prev)}
        showFramingGuide={showFramingGuide}
        onToggleFramingGuide={() => setShowFramingGuide((prev) => !prev)}
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

      <RecordSignModal
        isOpen={isRecordModalOpen}
        onClose={() => setIsRecordModalOpen(false)}
        currentLandmarks={currentResult?.landmarks}
        isCameraActive={cameraStatus === 'detecting'}
      />
    </div>
  );
};

export default App;
