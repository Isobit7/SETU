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
import { TextToSignWorkspace } from './components/TextToSignWorkspace';
import { useRecognitionStream } from './hooks/useRecognitionStream';
import { useTTS } from './hooks/useTTS';
import { useSentenceBuilder } from './hooks/useSentenceBuilder';
import type { LandmarkStyles, SubtitleSize } from './types';
import './index.css';

export const App: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement | null>(null);

  // 2-Way Translation Direction ('isl-to-en' or 'en-to-isl')
  const [translationDirection, setTranslationDirection] = useState<'isl-to-en' | 'en-to-isl'>('isl-to-en');

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

  // Browser-Native Text-to-Speech (TTS) Hook
  const {
    isSpeaking,
    autoSpeakEnabled,
    availableVoices,
    selectedVoice,
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
    onClearBackend: () => clearSentence(),
    onBackspaceBackend: () => backspaceSentence(),
  });

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
    clearSentence,
    backspaceSentence,
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

  const handleResetSession = () => {
    stop();
    handleClearSentence();
    resetSession();
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
        autoSpeakEnabled={autoSpeakEnabled}
        onToggleAutoSpeak={toggleAutoSpeak}
        isSpeaking={isSpeaking}
        translationDirection={translationDirection}
        onToggleDirection={() =>
          setTranslationDirection((prev) => (prev === 'isl-to-en' ? 'en-to-isl' : 'isl-to-en'))
        }
      />

      {/* Main Workspace Layout (Zero-Scroll 100vh Locked Grid) */}
      <main className={`workspace-grid ${translationDirection === 'en-to-isl' ? 'bidirectional-active' : ''}`} role="main">
        {translationDirection === 'isl-to-en' ? (
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
                autoSpeak={autoSpeakEnabled}
                onToggleAutoSpeak={toggleAutoSpeak}
                landmarkStyles={landmarkStyles}
                subtitleVisible={subtitleVisible}
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
              />
            </section>

            {/* Right Column: Active Output, Candidate Scores & Sentence Buffer */}
            <aside className="workspace-side-section" aria-label="Translation Output and Candidates">
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
              />
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
        landmarkStyles={landmarkStyles}
        setLandmarkStyles={setLandmarkStyles}
        subtitleSize={subtitleSize}
        setSubtitleSize={setSubtitleSize}
        confidenceThreshold={confidenceThreshold}
        setConfidenceThreshold={setConfidenceThreshold}
        autoSpeakEnabled={autoSpeakEnabled}
        onToggleAutoSpeak={toggleAutoSpeak}
        availableVoices={availableVoices}
        selectedVoice={selectedVoice}
        onSelectVoice={setSelectedVoice}
        isSpeaking={isSpeaking}
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
