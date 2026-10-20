import React from 'react';
import { Video, History, Bookmark, Settings, HelpCircle, Menu, Globe } from 'lucide-react';

interface SidebarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  collapsed: boolean;
  setCollapsed: (collapsed: boolean) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  setActiveTab,
  collapsed,
  setCollapsed,
}) => {
  const navItems = [
    { id: 'translator', label: 'Translator', icon: Video, active: true },
    { id: 'history', label: 'History', icon: History, badge: 'Session' },
    { id: 'saved', label: 'Saved Signs', icon: Bookmark },
    { id: 'vocabulary', label: 'ISL Vocabulary', icon: Globe, badge: '5 Signs' },
    { id: 'settings', label: 'Settings', icon: Settings },
    { id: 'help', label: 'Help & Guide', icon: HelpCircle },
  ];

  return (
    <aside
      className={`sidebar ${collapsed ? 'collapsed' : ''}`}
      aria-label="Main Navigation"
    >
      {/* Brand Header */}
      <div className="sidebar-brand">
        <div className="brand-logo-wrap">
          <div className="brand-symbol">
            <span className="brand-letter">S</span>
          </div>
          {!collapsed && (
            <div className="brand-text">
              <span className="brand-name">SETU</span>
              <span className="brand-tag">ISL Translator</span>
            </div>
          )}
        </div>
        <button
          className="collapse-btn"
          onClick={() => setCollapsed(!collapsed)}
          aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          <Menu size={18} />
        </button>
      </div>

      {/* Navigation Links */}
      <nav className="sidebar-nav">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isSelected = activeTab === item.id;
          return (
            <button
              key={item.id}
              className={`nav-item ${isSelected ? 'active' : ''}`}
              onClick={() => setActiveTab(item.id)}
              aria-current={isSelected ? 'page' : undefined}
            >
              <Icon size={20} className="nav-icon" />
              {!collapsed && (
                <span className="nav-label">
                  {item.label}
                  {item.badge && <span className="nav-badge">{item.badge}</span>}
                </span>
              )}
            </button>
          );
        })}
      </nav>

      {/* Footer Info */}
      {!collapsed && (
        <div className="sidebar-footer">
          <div className="pipeline-indicator">
            <div className="pulse-dot"></div>
            <span>ISL MVP 5-Sign Engine</span>
          </div>
          <p className="footer-subtext">MediaPipe 3D • 158-D Feature Extraction</p>
        </div>
      )}
    </aside>
  );
};
