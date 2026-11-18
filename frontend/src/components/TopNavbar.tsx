import React, { useState, useRef, useEffect } from 'react';
import {
  Video,
  History,
  Bookmark,
  Settings,
  HelpCircle,
  Menu,
  X,
  ChevronDown,
  ArrowRight,
  ArrowLeftRight,
  Target,
  Users,
  Sun,
  Moon,
  Contrast,
  BarChart3,
  Sparkles,
} from 'lucide-react';

interface TopNavbarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  isBackendConnected: boolean;
  backendReady?: boolean;
  fps?: number;
  latencyMs?: number;
  onOpenSettings: () => void;
  onOpenVocabulary?: () => void;
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
}

export const TopNavbar: React.FC<TopNavbarProps> = ({
  activeTab,
  setActiveTab,
  isBackendConnected,
  backendReady = false,
  fps = 0,
  latencyMs = 0,
  onOpenSettings,
  onOpenHistory,
  onOpenSaved,
  onOpenHelp,
  onOpenRecordModal,
  historyCount = 0,
  translationDirection = 'isl-to-en',
  onToggleDirection,
}) => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [resourcesOpen, setResourcesOpen] = useState(false);

  const resourcesRef = useRef<HTMLDivElement | null>(null);

  // Theme & High-Contrast A11y State
  const [theme, setTheme] = useState<'dark' | 'light' | 'high-contrast'>(() => {
    if (typeof window === 'undefined') return 'dark';
    try {
      return (localStorage.getItem('setu_theme') as 'dark' | 'light' | 'high-contrast') || 'dark';
    } catch {
      return 'dark';
    }
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    try {
      localStorage.setItem('setu_theme', theme);
    } catch {}
  }, [theme]);

  const cycleTheme = () => {
    setTheme((prev) => {
      if (prev === 'dark') return 'light';
      if (prev === 'light') return 'high-contrast';
      return 'dark';
    });
  };

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (resourcesRef.current && !resourcesRef.current.contains(e.target as Node)) {
        setResourcesOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <header className="top-navbar" role="banner">
      {/* 1. Left: Brand & Compact 2-Way Translation Switcher */}
      <div className="navbar-brand">
        <div className="brand-logo-wrap" title="SETU: Indian Sign Language Platform">
          <div className="brand-symbol" aria-hidden="true">
            <img src="/logo-square.png" alt="SETU Logo" className="brand-logo-img" />
          </div>
          <div className="brand-text">
            <span className="brand-name">SETU</span>
            <span className="brand-tag">ISL</span>
          </div>
        </div>

        {/* 2-Way Direction Pill */}
        <button
          type="button"
          className="navbar-direction-toggle compact"
          onClick={onToggleDirection}
          title={`Click to switch: ${
            translationDirection === 'isl-to-en'
              ? 'ISL → English (Sign Recognition)'
              : 'English → ISL (Sign Avatar Translation)'
          }`}
          aria-label="Toggle translation direction"
        >
          <span className={`dir-lang ${translationDirection === 'isl-to-en' ? 'active-source' : ''}`}>
            {translationDirection === 'isl-to-en' ? 'ISL' : 'EN'}
          </span>
          <div className={`dir-arrow-wrap ${translationDirection === 'en-to-isl' ? 'rotated' : ''}`}>
            <ArrowRight size={12} className={`dir-arrow-icon ${translationDirection === 'en-to-isl' ? 'rotated' : ''}`} />
          </div>
          <span className={`dir-lang ${translationDirection === 'en-to-isl' ? 'active-source' : ''}`}>
            {translationDirection === 'isl-to-en' ? 'EN' : 'ISL'}
          </span>
          <ArrowLeftRight size={10} className="dir-swap-hint" />
        </button>
      </div>

      {/* 2. Center: Core Navigation Links (Only 3 Primary Workspace Tabs) */}
      <nav className="navbar-links" aria-label="Main Navigation">
        <button
          className={`nav-link-btn ${activeTab === 'translator' && translationDirection === 'isl-to-en' ? 'active' : ''}`}
          onClick={() => {
            setActiveTab('translator');
            if (translationDirection !== 'isl-to-en' && onToggleDirection) {
              onToggleDirection();
            }
            setMobileMenuOpen(false);
          }}
          aria-current={activeTab === 'translator' && translationDirection === 'isl-to-en' ? 'page' : undefined}
          title="Sign Language Recognition (Camera to Text)"
        >
          <Video size={14} className="nav-icon" />
          <span>Sign to Text</span>
        </button>

        <button
          className={`nav-link-btn ${activeTab === 'translator' && translationDirection === 'en-to-isl' ? 'active' : ''}`}
          onClick={() => {
            setActiveTab('translator');
            if (translationDirection !== 'en-to-isl' && onToggleDirection) {
              onToggleDirection();
            }
            setMobileMenuOpen(false);
          }}
          aria-current={activeTab === 'translator' && translationDirection === 'en-to-isl' ? 'page' : undefined}
          title="Sign Language Generation (Text to 3D Avatar)"
        >
          <Sparkles size={14} className="nav-icon" />
          <span>Text to Sign</span>
        </button>

        <button
          className={`nav-link-btn ${activeTab === 'practice' ? 'active' : ''}`}
          onClick={() => {
            setActiveTab('practice');
            setMobileMenuOpen(false);
          }}
          aria-current={activeTab === 'practice' ? 'page' : undefined}
        >
          <Target size={14} className="nav-icon" />
          <span>Practice</span>
        </button>

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

        <button
          className={`nav-link-btn ${activeTab === 'evaluation' ? 'active' : ''}`}
          onClick={() => {
            setActiveTab('evaluation');
            setMobileMenuOpen(false);
          }}
          aria-current={activeTab === 'evaluation' ? 'page' : undefined}
          title="Model Accuracy & System Evaluation Metrics"
        >
          <BarChart3 size={14} className="nav-icon" />
          <span>Evaluation</span>
        </button>
      </nav>

      {/* 3. Right: Single Consolidated Dropdown + Theme + Status + Settings */}
      <div className="navbar-actions">
        {/* Consolidated "Records & Tools ▾" Dropdown */}
        <div className="nav-dropdown-wrapper" ref={resourcesRef}>
          <button
            className={`nav-link-btn dropdown-trigger ${resourcesOpen ? 'dropdown-active' : ''}`}
            onClick={() => setResourcesOpen(!resourcesOpen)}
            aria-expanded={resourcesOpen}
            aria-haspopup="true"
            title="Explore session history, saved signs, and tools"
          >
            <History size={14} className="nav-icon" />
            <span>Records & Tools</span>
            {historyCount > 0 && <span className="navbar-item-badge">{historyCount}</span>}
            <ChevronDown size={11} className={`chevron-icon ${resourcesOpen ? 'open' : ''}`} />
          </button>

          {resourcesOpen && (
            <div className="navbar-dropdown-menu" role="menu">
              {/* Session History & Saved */}
              <div className="dropdown-section-title">Session & Records</div>

              {onOpenHistory && (
                <button
                  className="dropdown-item"
                  role="menuitem"
                  onClick={() => {
                    onOpenHistory();
                    setResourcesOpen(false);
                  }}
                >
                  <History size={14} className="dropdown-item-icon text-amber" />
                  <div className="dropdown-item-text">
                    <span className="item-title">Session History</span>
                    <span className="item-desc">
                      {historyCount > 0 ? `${historyCount} signs logged` : 'View recorded translations'}
                    </span>
                  </div>
                  {historyCount > 0 && <span className="navbar-item-badge">{historyCount}</span>}
                </button>
              )}

              {onOpenSaved && (
                <button
                  className="dropdown-item"
                  role="menuitem"
                  onClick={() => {
                    onOpenSaved();
                    setResourcesOpen(false);
                  }}
                >
                  <Bookmark size={14} className="dropdown-item-icon text-emerald" />
                  <div className="dropdown-item-text">
                    <span className="item-title">Saved Signs</span>
                    <span className="item-desc">Bookmarked translations</span>
                  </div>
                </button>
              )}

              {/* Creator Tools & Help */}
              <div className="dropdown-section-title">Tools & Support</div>

              {onOpenRecordModal && (
                <button
                  className="dropdown-item"
                  role="menuitem"
                  onClick={() => {
                    onOpenRecordModal();
                    setResourcesOpen(false);
                  }}
                >
                  <Video size={14} className="dropdown-item-icon text-danger" />
                  <div className="dropdown-item-text">
                    <span className="item-title">Record Custom Sign</span>
                    <span className="item-desc">Capture custom gestures & poses</span>
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
                <HelpCircle size={14} className="dropdown-item-icon text-muted" />
                <div className="dropdown-item-text">
                  <span className="item-title">Help & Signer Guide</span>
                  <span className="item-desc">Framing, lighting, and gestures</span>
                </div>
              </button>
            </div>
          )}
        </div>

        {/* Theme Icon Button (Compact) */}
        <button
          type="button"
          className="nav-icon-btn"
          onClick={cycleTheme}
          title={`Theme: ${
            theme === 'dark' ? 'Dark' : theme === 'light' ? 'Light' : 'High-Contrast A11y'
          }. Click to switch.`}
          aria-label="Toggle Theme"
        >
          {theme === 'dark' && <Moon size={14} />}
          {theme === 'light' && <Sun size={14} />}
          {theme === 'high-contrast' && <Contrast size={14} className="a11y-gold" />}
        </button>

        {/* Minimal Live Status Indicator (Clean Dot with Tooltip) */}
        <div
          className={`connection-pill-minimal ${
            isBackendConnected ? 'connected' : backendReady ? 'ready' : 'disconnected'
          }`}
          title={`SETU AI Engine: ${
            isBackendConnected
              ? 'Active Recognition Stream (CUDA Ready)'
              : backendReady
              ? 'Backend Online & Model Loaded (CUDA Ready)'
              : 'Backend Offline'
          }${fps > 0 ? ` · ${fps} FPS` : ''}${latencyMs > 0 ? ` · ${latencyMs}ms` : ''}`}
        >
          <span className="connection-status-dot" aria-hidden="true" />
          <span className="status-label-minimal">
            {isBackendConnected ? (fps > 0 ? `${fps} FPS` : 'Live') : backendReady ? 'Ready' : 'Offline'}
          </span>
        </div>

        {/* Settings Icon Button */}
        <button
          type="button"
          className="nav-icon-btn settings-trigger-btn"
          onClick={onOpenSettings}
          title="Open Settings & Calibration"
          aria-label="Settings"
        >
          <Settings size={15} />
        </button>

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
            className={`mobile-nav-item theme-mobile-item ${theme}`}
            onClick={cycleTheme}
          >
            {theme === 'dark' && <Moon size={16} />}
            {theme === 'light' && <Sun size={16} />}
            {theme === 'high-contrast' && <Contrast size={16} className="a11y-gold" />}
            <span>
              Theme: {theme === 'dark' ? 'Dark Mode' : theme === 'light' ? 'Light Mode' : 'High-Contrast A11y'}
            </span>
          </button>
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
          <button
            className={`mobile-nav-item ${activeTab === 'practice' ? 'active' : ''}`}
            onClick={() => {
              setActiveTab('practice');
              setMobileMenuOpen(false);
            }}
          >
            <Target size={16} />
            <span>Practice</span>
          </button>
          <button
            className={`mobile-nav-item ${activeTab === 'conversation' ? 'active' : ''}`}
            onClick={() => {
              setActiveTab('conversation');
              setMobileMenuOpen(false);
            }}
          >
            <Users size={16} />
            <span>Conversation</span>
          </button>
          <button
            className={`mobile-nav-item ${activeTab === 'evaluation' ? 'active' : ''}`}
            onClick={() => {
              setActiveTab('evaluation');
              setMobileMenuOpen(false);
            }}
          >
            <BarChart3 size={16} />
            <span>Evaluation</span>
          </button>
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
          {onOpenRecordModal && (
            <button
              className="mobile-nav-item"
              onClick={() => {
                onOpenRecordModal();
                setMobileMenuOpen(false);
              }}
            >
              <Video size={16} className="text-danger" />
              <span>Record Custom Sign</span>
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
            <span>Help & Guide</span>
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

export default TopNavbar;
