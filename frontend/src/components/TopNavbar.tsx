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
  Film,
  Type,
  ChevronDown,
  ArrowRight,
  ArrowLeftRight,
  Volume2,
  VolumeX,
  Target,
  Users,
  Sun,
  Moon,
  Contrast,
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
  onOpenRecordModal?: () => void;
  historyCount?: number;
  autoSpeakEnabled?: boolean;
  onToggleAutoSpeak?: () => void;
  isSpeaking?: boolean;
  translationDirection?: 'isl-to-en' | 'en-to-isl';
  onToggleDirection?: () => void;
  theme?: 'dark' | 'light' | 'high-contrast';
  onCycleTheme?: () => void;
}

export const TopNavbar: React.FC<TopNavbarProps> = ({
  activeTab,
  setActiveTab,
  isBackendConnected,
  fps = 0,
  latencyMs = 0,
  onOpenSettings,
  onOpenVocabulary,
  onOpenAlphabetModal,
  onOpenHistory,
  onOpenSaved,
  onOpenHelp,
  onOpenIncludeModal,
  onOpenRecordModal,
  historyCount = 0,
  autoSpeakEnabled = true,
  onToggleAutoSpeak,
  isSpeaking = false,
  translationDirection = 'isl-to-en',
  onToggleDirection,
  theme: propTheme,
  onCycleTheme,
}) => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [libraryOpen, setLibraryOpen] = useState(false);

  const libraryRef = useRef<HTMLDivElement | null>(null);

  // Theme & High-Contrast A11y State (Prefers lifted prop, falls back to local)
  const [localTheme, setLocalTheme] = useState<'dark' | 'light' | 'high-contrast'>(() => {
    if (typeof window === 'undefined') return 'dark';
    try {
      return (localStorage.getItem('setu_theme') as 'dark' | 'light' | 'high-contrast') || 'dark';
    } catch {
      return 'dark';
    }
  });

  const activeTheme = propTheme || localTheme;

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', activeTheme);
    try {
      localStorage.setItem('setu_theme', activeTheme);
    } catch {}
  }, [activeTheme]);

  const cycleTheme = () => {
    if (onCycleTheme) {
      onCycleTheme();
      return;
    }
    setLocalTheme((prev) => {
      if (prev === 'dark') return 'light';
      if (prev === 'light') return 'high-contrast';
      return 'dark';
    });
  };

  // Close dropdowns when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (libraryRef.current && !libraryRef.current.contains(e.target as Node)) {
        setLibraryOpen(false);
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

        {/* Practice Studio Active Tab */}
        <button
          className={`nav-link-btn ${activeTab === 'practice' ? 'active' : ''}`}
          onClick={() => {
            setActiveTab('practice');
            setMobileMenuOpen(false);
          }}
          aria-current={activeTab === 'practice' ? 'page' : undefined}
        >
          <Target size={14} className="nav-icon" />
          <span>Practice Studio</span>
        </button>

        {/* Conversation Studio Tab */}
        <button
          className={`nav-link-btn ${activeTab === 'conversation' ? 'active' : ''}`}
          onClick={() => {
            setActiveTab('conversation');
            setMobileMenuOpen(false);
          }}
          aria-current={activeTab === 'conversation' ? 'page' : undefined}
        >
          <Users size={14} className="nav-icon" />
          <span>Conversation</span>
        </button>

        {/* 4. Consolidated Library Dropdown: Sign Guides, Activity & History */}
        <div className="nav-dropdown-wrapper" ref={libraryRef}>
          <button
            className={`nav-link-btn dropdown-trigger ${libraryOpen ? 'dropdown-active' : ''}`}
            onClick={() => setLibraryOpen(!libraryOpen)}
            aria-expanded={libraryOpen}
            aria-haspopup="true"
            title="Sign References, Guides, and Activity"
          >
            <BookOpen size={14} className="nav-icon" />
            <span>Library</span>
            {historyCount > 0 && (
              <span className="navbar-item-badge">{historyCount}</span>
            )}
            <ChevronDown size={12} className={`chevron-icon ${libraryOpen ? 'open' : ''}`} />
          </button>

          {libraryOpen && (
            <div className="navbar-dropdown-menu library-dropdown-menu" role="menu">
              <div className="dropdown-section-title">Sign Guides & References</div>

              {onOpenAlphabetModal && (
                <button
                  className="dropdown-item"
                  role="menuitem"
                  onClick={() => {
                    onOpenAlphabetModal();
                    setLibraryOpen(false);
                  }}
                >
                  <Type size={14} className="dropdown-item-icon" />
                  <div className="dropdown-item-text">
                    <span className="item-title">Alphabet Guide (A–Z)</span>
                    <span className="item-desc">Fingerspelling postures & cheatsheet</span>
                  </div>
                </button>
              )}

              <button
                className="dropdown-item"
                role="menuitem"
                onClick={() => {
                  onOpenVocabulary();
                  setLibraryOpen(false);
                }}
              >
                <BookOpen size={14} className="dropdown-item-icon" />
                <div className="dropdown-item-text">
                  <span className="item-title">Common Signs (5 Phrases)</span>
                  <span className="item-desc">Everyday conversational expressions</span>
                </div>
              </button>

              {onOpenIncludeModal && (
                <button
                  className="dropdown-item"
                  role="menuitem"
                  onClick={() => {
                    onOpenIncludeModal();
                    setLibraryOpen(false);
                  }}
                >
                  <Film size={14} className="dropdown-item-icon" />
                  <div className="dropdown-item-text">
                    <span className="item-title">Sign Dictionary (263 Signs)</span>
                    <span className="item-desc">INCLUDE video recognition & search</span>
                  </div>
                </button>
              )}

              {onOpenRecordModal && (
                <button
                  className="dropdown-item"
                  role="menuitem"
                  onClick={() => {
                    onOpenRecordModal();
                    setLibraryOpen(false);
                  }}
                >
                  <Video size={14} className="dropdown-item-icon text-danger" />
                  <div className="dropdown-item-text">
                    <span className="item-title">Custom Sign Studio</span>
                    <span className="item-desc">Record & train personal gestures</span>
                  </div>
                </button>
              )}

              <div className="dropdown-divider" />
              <div className="dropdown-section-title">Your Activity</div>

              {onOpenHistory && (
                <button
                  className="dropdown-item"
                  role="menuitem"
                  onClick={() => {
                    onOpenHistory();
                    setLibraryOpen(false);
                  }}
                >
                  <History size={14} className="dropdown-item-icon" />
                  <div className="dropdown-item-text">
                    <span className="item-title">Session History</span>
                    <span className="item-desc">
                      {historyCount > 0 ? `${historyCount} signs logged this session` : 'No signs logged yet'}
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
                    setLibraryOpen(false);
                  }}
                >
                  <Bookmark size={14} className="dropdown-item-icon" />
                  <div className="dropdown-item-text">
                    <span className="item-title">Saved Bookmarks</span>
                    <span className="item-desc">Saved signs and translations</span>
                  </div>
                </button>
              )}

              <div className="dropdown-divider" />
              <div className="dropdown-section-title">Support</div>

              <button
                className="dropdown-item"
                role="menuitem"
                onClick={() => {
                  onOpenHelp();
                  setLibraryOpen(false);
                }}
              >
                <HelpCircle size={14} className="dropdown-item-icon" />
                <div className="dropdown-item-text">
                  <span className="item-title">Camera Tips & Framing</span>
                  <span className="item-desc">Optimal lighting and hand distance</span>
                </div>
              </button>
            </div>
          )}
        </div>

      </nav>

      {/* 6. Right Utilities: Clean Minimalist Icon Bar & Status */}
      <div className="navbar-actions">
        {/* Auto-Speak TTS Icon Toggle */}
        {onToggleAutoSpeak && (
          <button
            className={`nav-icon-action-btn ${autoSpeakEnabled ? 'active' : 'muted'} ${isSpeaking ? 'speaking' : ''}`}
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
              <Volume2 size={15} className="tts-active-icon" />
            ) : (
              <VolumeX size={15} className="tts-muted-icon" />
            )}
          </button>
        )}

        {/* Theme Switcher Icon (Dark / Light / High-Contrast) */}
        <button
          type="button"
          className={`nav-icon-action-btn ${activeTheme}`}
          onClick={cycleTheme}
          title={`Theme: ${
            activeTheme === 'dark' ? 'Dark Mode' : activeTheme === 'light' ? 'Light Mode' : 'High-Contrast A11y (WCAG AAA)'
          }. Click to switch.`}
          aria-label="Switch Theme Mode"
        >
          {activeTheme === 'dark' && <Moon size={15} className="theme-toggle-icon" />}
          {activeTheme === 'light' && <Sun size={15} className="theme-toggle-icon" />}
          {activeTheme === 'high-contrast' && <Contrast size={15} className="theme-toggle-icon a11y-gold" />}
        </button>

        {/* Settings Icon Button */}
        <button
          className="nav-icon-action-btn"
          onClick={onOpenSettings}
          title="Settings, Calibration & Developer Options"
          aria-label="Open Settings"
        >
          <Settings size={15} />
        </button>

        {/* Clean System Status Pill (Telemetry on hover tooltip) */}
        <div
          className={`connection-pill-clean ${isBackendConnected ? 'connected' : 'disconnected'}`}
          title={`SETU Engine: ${
            isBackendConnected ? 'Connected (ws://localhost:8000)' : 'Disconnected'
          } · Local Processing · Zero Cloud Recording${fps > 0 ? ` · ${fps} FPS` : ''}${latencyMs > 0 ? ` · ${latencyMs}ms` : ''}`}
        >
          <span className="connection-status-dot" aria-hidden="true" />
          <span className="status-label-clean">{isBackendConnected ? 'Live' : 'Offline'}</span>
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
          <button
            type="button"
            className={`mobile-nav-item theme-mobile-item ${activeTheme}`}
            onClick={cycleTheme}
          >
            {activeTheme === 'dark' && <Moon size={16} />}
            {activeTheme === 'light' && <Sun size={16} />}
            {activeTheme === 'high-contrast' && <Contrast size={16} className="a11y-gold" />}
            <span>
              Theme: {activeTheme === 'dark' ? 'Dark Mode' : activeTheme === 'light' ? 'Light Mode' : 'High-Contrast A11y'}
            </span>
          </button>
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
              <span>Alphabet Guide (A–Z)</span>
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
            <span>Common Signs (5 Phrases)</span>
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
              <span>Sign Dictionary (263 Signs)</span>
            </button>
          )}
          {onOpenRecordModal && (
            <button
              className="mobile-nav-item"
              onClick={() => {
                onOpenRecordModal();
                setMobileMenuOpen(false);
              }}
            >
              <Video size={16} />
              <span>Custom Sign Studio</span>
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
              <span>Session History {historyCount > 0 ? `(${historyCount})` : ''}</span>
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
              <span>Saved Bookmarks</span>
            </button>
          )}
          <button
            className="mobile-nav-item"
            onClick={() => {
              onOpenHelp();
              setMobileMenuOpen(false);
            }}
          >
            <HelpCircle size={16} />
            <span>Camera Tips & Guide</span>
          </button>
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
