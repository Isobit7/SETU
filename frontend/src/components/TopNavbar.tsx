import React, { useState, useRef, useEffect } from 'react';
import {
  Video,
  History,
  Bookmark,
  BookOpen,
  Settings,
  HelpCircle,
  Menu,
  X,
  ShieldCheck,
  Film,
  Type,
  ChevronDown,
  ArrowRight,
  ArrowLeftRight,
  Volume2,
  VolumeX,
} from 'lucide-react';

interface TopNavbarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  isBackendConnected: boolean;
  fps?: number;
  latencyMs?: number;
  onOpenSettings: () => void;
  onOpenVocabulary: () => void;
  onOpenAlphabetModal?: () => void;
  onOpenHistory?: () => void;
  onOpenSaved?: () => void;
  onOpenHelp: () => void;
  onOpenIncludeModal?: () => void;
  historyCount?: number;
  autoSpeakEnabled?: boolean;
  onToggleAutoSpeak?: () => void;
  isSpeaking?: boolean;
  translationDirection?: 'isl-to-en' | 'en-to-isl';
  onToggleDirection?: () => void;
}

export const TopNavbar: React.FC<TopNavbarProps> = ({
  activeTab,
  setActiveTab,
  isBackendConnected,
  fps: _fps = 0,
  latencyMs: _latencyMs = 0,
  onOpenSettings,
  onOpenVocabulary,
  onOpenAlphabetModal,
  onOpenHistory,
  onOpenSaved,
  onOpenHelp,
  onOpenIncludeModal,
  historyCount = 0,
  autoSpeakEnabled = true,
  onToggleAutoSpeak,
  isSpeaking = false,
  translationDirection = 'isl-to-en',
  onToggleDirection,
}) => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [resourcesOpen, setResourcesOpen] = useState(false);
  const [historyMenuOpen, setHistoryMenuOpen] = useState(false);

  const resourcesRef = useRef<HTMLDivElement | null>(null);
  const historyRef = useRef<HTMLDivElement | null>(null);

  // Close dropdowns when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (resourcesRef.current && !resourcesRef.current.contains(e.target as Node)) {
        setResourcesOpen(false);
      }
      if (historyRef.current && !historyRef.current.contains(e.target as Node)) {
        setHistoryMenuOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <header className="top-navbar" role="banner">
      {/* 1. Brand Logo & Mark */}
      <div className="navbar-brand">
        <div className="brand-logo-wrap" title="SETU: Indian Sign Language Translator">
          <div className="brand-symbol" aria-hidden="true">
            <img
              src="/logo-square.png"
              alt="SETU Logo"
              className="brand-logo-img"
            />
          </div>
          <div className="brand-text">
            <span className="brand-name">SETU</span>
            <span className="brand-tag">ISL Translator</span>
          </div>
        </div>

        {/* 2. Interactive 2-Way Communication Toggle */}
        <button
          type="button"
          className="navbar-direction-toggle"
          onClick={onToggleDirection}
          title={`Click to switch mode: ${
            translationDirection === 'isl-to-en'
              ? 'ISL → English (Sign Recognition)'
              : 'English → ISL (Sign Avatar Translation)'
          }`}
          aria-label="Toggle translation direction"
        >
          <span className={`dir-lang ${translationDirection === 'isl-to-en' ? 'active-source' : ''}`}>
            {translationDirection === 'isl-to-en' ? 'ISL' : 'English'}
          </span>
          <div className={`dir-arrow-wrap ${translationDirection === 'en-to-isl' ? 'rotated' : ''}`}>
            <ArrowRight size={13} className={`dir-arrow-icon ${translationDirection === 'en-to-isl' ? 'rotated' : ''}`} />
          </div>
          <span className={`dir-lang ${translationDirection === 'en-to-isl' ? 'active-source' : ''}`}>
            {translationDirection === 'isl-to-en' ? 'English' : 'ISL'}
          </span>
          <ArrowLeftRight size={11} className="dir-swap-hint" />
        </button>
      </div>

      {/* Center Navigation Links (Streamlined) */}
      <nav className="navbar-links" aria-label="Main Navigation">
        {/* 3. Translator Active Tab */}
        <button
          className={`nav-link-btn ${activeTab === 'translator' ? 'active' : ''}`}
          onClick={() => {
            setActiveTab('translator');
            setMobileMenuOpen(false);
          }}
          aria-current={activeTab === 'translator' ? 'page' : undefined}
        >
          <Video size={14} className="nav-icon" />
          <span>Translator</span>
        </button>

        {/* 4. Resources Dropdown: 26 Letters, ISL Vocabulary, 263 Signs, Help */}
        <div className="nav-dropdown-wrapper" ref={resourcesRef}>
          <button
            className={`nav-link-btn dropdown-trigger ${resourcesOpen ? 'dropdown-active' : ''}`}
            onClick={() => {
              setResourcesOpen(!resourcesOpen);
              setHistoryMenuOpen(false);
            }}
            aria-expanded={resourcesOpen}
            aria-haspopup="true"
          >
            <BookOpen size={14} className="nav-icon" />
            <span>Resources</span>
            <ChevronDown size={12} className={`chevron-icon ${resourcesOpen ? 'open' : ''}`} />
          </button>

          {resourcesOpen && (
            <div className="navbar-dropdown-menu" role="menu">
              {onOpenAlphabetModal && (
                <button
                  className="dropdown-item"
                  role="menuitem"
                  onClick={() => {
                    onOpenAlphabetModal();
                    setResourcesOpen(false);
                  }}
                >
                  <Type size={14} className="dropdown-item-icon" />
                  <div className="dropdown-item-text">
                    <span className="item-title">26 Letters (RealSign)</span>
                    <span className="item-desc">A–Z Alphabet classifier</span>
                  </div>
                </button>
              )}

              <button
                className="dropdown-item"
                role="menuitem"
                onClick={() => {
                  onOpenVocabulary();
                  setResourcesOpen(false);
                }}
              >
                <BookOpen size={14} className="dropdown-item-icon" />
                <div className="dropdown-item-text">
                  <span className="item-title">ISL Vocabulary</span>
                  <span className="item-desc">5 Core conversational phrases</span>
                </div>
              </button>

              {onOpenIncludeModal && (
                <button
                  className="dropdown-item"
                  role="menuitem"
                  onClick={() => {
                    onOpenIncludeModal();
                    setResourcesOpen(false);
                  }}
                >
                  <Film size={14} className="dropdown-item-icon" />
                  <div className="dropdown-item-text">
                    <span className="item-title">263 Signs (INCLUDE)</span>
                    <span className="item-desc">Video file recognition</span>
                  </div>
                </button>
              )}

              <button
                className="dropdown-item"
                role="menuitem"
                onClick={() => {
                  onOpenHelp();
                  setResourcesOpen(false);
                }}
              >
                <HelpCircle size={14} className="dropdown-item-icon" />
                <div className="dropdown-item-text">
                  <span className="item-title">Help & Guide</span>
                  <span className="item-desc">Framing and hand tips</span>
                </div>
              </button>
            </div>
          )}
        </div>

        {/* 5. History & Saved Dropdown */}
        <div className="nav-dropdown-wrapper" ref={historyRef}>
          <button
            className={`nav-link-btn dropdown-trigger ${historyMenuOpen ? 'dropdown-active' : ''}`}
            onClick={() => {
              setHistoryMenuOpen(!historyMenuOpen);
              setResourcesOpen(false);
            }}
            aria-expanded={historyMenuOpen}
            aria-haspopup="true"
          >
            <History size={14} className="nav-icon" />
            <span>History & Saved</span>
            {historyCount > 0 && (
              <span className="navbar-item-badge">{historyCount}</span>
            )}
            <ChevronDown size={12} className={`chevron-icon ${historyMenuOpen ? 'open' : ''}`} />
          </button>

          {historyMenuOpen && (
            <div className="navbar-dropdown-menu" role="menu">
              {onOpenHistory && (
                <button
                  className="dropdown-item"
                  role="menuitem"
                  onClick={() => {
                    onOpenHistory();
                    setHistoryMenuOpen(false);
                  }}
                >
                  <History size={14} className="dropdown-item-icon" />
                  <div className="dropdown-item-text">
                    <span className="item-title">Session History</span>
                    <span className="item-desc">
                      {historyCount > 0 ? `${historyCount} signs logged` : 'No signs logged yet'}
                    </span>
                  </div>
                  {historyCount > 0 && (
                    <span className="navbar-item-badge">{historyCount}</span>
                  )}
                </button>
              )}

              {onOpenSaved && (
                <button
                  className="dropdown-item"
                  role="menuitem"
                  onClick={() => {
                    onOpenSaved();
                    setHistoryMenuOpen(false);
                  }}
                >
                  <Bookmark size={14} className="dropdown-item-icon" />
                  <div className="dropdown-item-text">
                    <span className="item-title">Saved Signs</span>
                    <span className="item-desc">Bookmarked translations</span>
                  </div>
                </button>
              )}
            </div>
          )}
        </div>

        {/* 6. Settings Link */}
        <button
          className="nav-link-btn"
          onClick={onOpenSettings}
          title="Settings & Calibration"
        >
          <Settings size={14} className="nav-icon" />
          <span>Settings</span>
        </button>
      </nav>

      {/* 7. Right: Status Badges & Quick Toggles */}
      <div className="navbar-actions">
        {/* Auto-Speak TTS Toggle */}
        {onToggleAutoSpeak && (
          <button
            className={`nav-tts-toggle ${autoSpeakEnabled ? 'active' : 'muted'} ${isSpeaking ? 'speaking' : ''}`}
            onClick={onToggleAutoSpeak}
            title={
              autoSpeakEnabled
                ? 'Auto-Speak (TTS) is Active. Click to mute.'
                : 'Auto-Speak (TTS) is Muted. Click to enable.'
            }
            aria-label="Toggle Auto-Speak TTS"
            aria-pressed={autoSpeakEnabled}
          >
            {isSpeaking ? (
              <div className="soundwave-bars" aria-hidden="true">
                <span className="soundwave-bar" />
                <span className="soundwave-bar" />
                <span className="soundwave-bar" />
              </div>
            ) : autoSpeakEnabled ? (
              <Volume2 size={13} className="tts-active-icon" />
            ) : (
              <VolumeX size={13} className="tts-muted-icon" />
            )}
            <span className="tts-pill-label">Auto-TTS</span>
          </button>
        )}

        {/* Connection status pill - Clean Single-Indicator Minimalist Badge */}
        <div
          className={`connection-pill ${isBackendConnected ? 'connected' : 'disconnected'}`}
          title={
            isBackendConnected
              ? 'Connected to SETU Engine (Active WebSocket)'
              : 'Backend disconnected (ws://localhost:8000)'
          }
        >
          <span className="connection-status-dot" aria-hidden="true" />
          <span className="status-label">{isBackendConnected ? 'Online' : 'Offline'}</span>
        </div>

        {/* Privacy Pill */}
        <div className="privacy-pill" title="Local frame processing. No external cloud recording.">
          <ShieldCheck size={13} className="privacy-icon" />
          <span className="privacy-text">Local</span>
        </div>

        {/* Mobile Menu Toggle */}
        <button
          className="mobile-menu-toggle"
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          aria-label={mobileMenuOpen ? 'Close menu' : 'Open menu'}
        >
          {mobileMenuOpen ? <X size={18} /> : <Menu size={18} />}
        </button>
      </div>

      {/* Mobile Nav Drawer */}
      {mobileMenuOpen && (
        <div className="mobile-nav-drawer" role="navigation">
          {onToggleAutoSpeak && (
            <button
              className={`mobile-nav-item ${autoSpeakEnabled ? 'active' : ''}`}
              onClick={() => {
                onToggleAutoSpeak();
                setMobileMenuOpen(false);
              }}
            >
              {autoSpeakEnabled ? <Volume2 size={16} /> : <VolumeX size={16} />}
              <span>Auto-Speak: {autoSpeakEnabled ? 'Enabled' : 'Muted'}</span>
            </button>
          )}
          <button
            className={`mobile-nav-item ${activeTab === 'translator' ? 'active' : ''}`}
            onClick={() => {
              setActiveTab('translator');
              setMobileMenuOpen(false);
            }}
          >
            <Video size={16} />
            <span>Translator</span>
          </button>
          {onOpenAlphabetModal && (
            <button
              className="mobile-nav-item"
              onClick={() => {
                onOpenAlphabetModal();
                setMobileMenuOpen(false);
              }}
            >
              <Type size={16} />
              <span>26 Letters (RealSign)</span>
            </button>
          )}
          <button
            className="mobile-nav-item"
            onClick={() => {
              onOpenVocabulary();
              setMobileMenuOpen(false);
            }}
          >
            <BookOpen size={16} />
            <span>ISL Vocabulary</span>
          </button>
          {onOpenIncludeModal && (
            <button
              className="mobile-nav-item"
              onClick={() => {
                onOpenIncludeModal();
                setMobileMenuOpen(false);
              }}
            >
              <Film size={16} />
              <span>263 Signs (INCLUDE)</span>
            </button>
          )}
          {onOpenHistory && (
            <button
              className="mobile-nav-item"
              onClick={() => {
                onOpenHistory();
                setMobileMenuOpen(false);
              }}
            >
              <History size={16} />
              <span>History {historyCount > 0 ? `(${historyCount})` : ''}</span>
            </button>
          )}
          {onOpenSaved && (
            <button
              className="mobile-nav-item"
              onClick={() => {
                onOpenSaved();
                setMobileMenuOpen(false);
              }}
            >
              <Bookmark size={16} />
              <span>Saved Signs</span>
            </button>
          )}
          <button
            className="mobile-nav-item"
            onClick={() => {
              onOpenSettings();
              setMobileMenuOpen(false);
            }}
          >
            <Settings size={16} />
            <span>Settings</span>
          </button>
        </div>
      )}
    </header>
  );
};
