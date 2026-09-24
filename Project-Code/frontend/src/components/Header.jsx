import React, { useState, useEffect } from 'react';
import { Activity, Calendar, Bell, User, HelpCircle, ChevronDown, LogOut } from 'lucide-react';

export default function Header({ activeTab, setActiveTab, userRole, openLoginModal, openGuideModal, isOnline, onLogout }) {
  const [timeString, setTimeString] = useState('');

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      const options = {
        weekday: 'short',
        day: '2-digit',
        month: 'short',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: true,
        timeZone: 'Asia/Kolkata'
      };
      setTimeString(new Intl.DateTimeFormat('en-IN', options).format(now) + ' (IST)');
    };
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  const tabs = [
    { id: 'dashboard', label: 'Dashboard', icon: Activity },
    { id: 'calendar', label: 'Flare Calendar', icon: Calendar },
    { id: 'alerts', label: 'Alerts & Notifications', icon: Bell },
    { id: 'telemetry', label: 'Live Telemetry', icon: Activity },
  ];

  return (
    <header className="top-nav-bar">
      {/* Brand & Mission Title */}
      <div className="brand-group">
        <div className="sun-pulse-logo">
          <Activity className="sun-icon-animated" size={28} color="#38bdf8" />
        </div>
        <div>
          <div className="brand-title">
            <span>Aditya-L1</span>
            <span className="isro-badge">ISSDC / ISRO</span>
          </div>
          <div className="brand-subtitle">Solar Activity Monitoring & Prediction System</div>
        </div>
      </div>

      {/* Primary Navigation Tabs (Map Removed) */}
      <nav className="nav-tabs">
        {tabs.map((t) => {
          const Icon = t.icon;
          const isActive = activeTab === t.id;
          return (
            <button
              key={t.id}
              className={`nav-btn ${isActive ? 'active' : ''}`}
              onClick={() => setActiveTab(t.id)}
            >
              <Icon size={16} />
              <span>{t.label}</span>
            </button>
          );
        })}

        <button className="nav-btn guide-nav-btn" onClick={openGuideModal}>
          <HelpCircle size={16} color="#38bdf8" />
          <span>Scientific Guide</span>
        </button>
      </nav>

      {/* System Status & User Actions */}
      <div className="header-actions">
        <div className="live-clock">{timeString || '23 Sep 2026, 05:45 (IST)'}</div>

        <div
          className={`status-pill ${isOnline ? 'online' : 'fallback'}`}
          title={isOnline ? 'Connected to FastAPI Backend (localhost:8000)' : 'Backend offline. Running on ISSDC Autonomous Telemetry Engine.'}
        >
          <span className="dot"></span>
          <span>{isOnline ? 'STREAM ACTIVE' : 'L1 AUTONOMOUS MODE'}</span>
        </div>

        <button className="user-profile-btn" onClick={openLoginModal}>
          <User size={16} className="text-cyan-400" />
          <div className="user-info-text">
            <span className="user-name">Isheetha S</span>
            <span className="user-role">{userRole || 'Lead Researcher'}</span>
          </div>
        </button>

        <button className="logout-icon-btn" onClick={onLogout} title="Logout to Login Screen">
          <LogOut size={16} color="#ef4444" />
        </button>
      </div>
    </header>
  );
}
