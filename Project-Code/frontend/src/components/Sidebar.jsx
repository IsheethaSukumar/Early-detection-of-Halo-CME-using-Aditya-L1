import React from 'react';
import { Activity, Calendar, Bell, User, LogOut, Settings, HelpCircle } from 'lucide-react';

export default function Sidebar({ activeTab, setActiveTab, currentUser, openLoginModal, openGuideModal, onLogout }) {
  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: Activity },
    { id: 'calendar', label: 'Flare Calendar', icon: Calendar },
    { id: 'alerts', label: 'Alerts & Notifications', icon: Bell },
    { id: 'telemetry', label: 'Telemetry & Reports', icon: Activity },
  ];

  return (
    <aside className="sidebar-container glass-panel">
      {/* Brand Header */}
      <div className="sidebar-brand flex items-center gap-3">
        <div className="sun-icon-box">
          <Activity size={24} color="#38bdf8" />
        </div>
        <div>
          <div className="sidebar-title">SolarFlare Watch</div>
          <div className="sidebar-tagline">Predict · Monitor · Stay Prepared</div>
        </div>
      </div>

      {/* Nav List */}
      <nav className="sidebar-nav">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              className={`sidebar-nav-item ${isActive ? 'active' : ''}`}
              onClick={() => setActiveTab(item.id)}
            >
              <Icon size={18} />
              <span>{item.label}</span>
            </button>
          );
        })}

        <button className="sidebar-nav-item" onClick={openGuideModal}>
          <HelpCircle size={18} color="#38bdf8" />
          <span>Scientific Guide</span>
        </button>
      </nav>

      {/* Footer Profile */}
      <div className="sidebar-footer">
        <div className="user-profile-card" onClick={openLoginModal}>
          <div className="avatar-circle">
            <User size={18} color="#38bdf8" />
          </div>
          <div className="user-details">
            <div className="user-name">{currentUser?.name || 'Isheetha S'}</div>
            <div className="user-role-sub">{currentUser?.roleTitle || 'Researcher'}</div>
          </div>
        </div>

        <button className="logout-btn" onClick={onLogout}>
          <LogOut size={16} />
          <span>Logout / Exit</span>
        </button>
      </div>
    </aside>
  );
}
