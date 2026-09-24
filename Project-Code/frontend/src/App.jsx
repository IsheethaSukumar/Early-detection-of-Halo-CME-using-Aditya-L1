import React, { useState, useEffect } from 'react';
import BackgroundSpace from './components/BackgroundSpace';
import LoginPage from './components/LoginPage';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import MetricsSummary from './components/MetricsSummary';
import PredictedFlaresTable from './components/PredictedFlaresTable';
import SolarActivityGauge from './components/SolarActivityGauge';
import SolarFlareCalendar from './components/SolarFlareCalendar';
import StakeholderAlerts from './components/StakeholderAlerts';

import ScientificGuide from './components/ScientificGuide';

import TelemetryGauges from './components/TelemetryGauges';
import TimeSeriesChart from './components/TimeSeriesChart';
import HistoryTable from './components/HistoryTable';
import AnomalyStatus from './components/AnomalyStatus';

import './App.css';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [activeTab, setActiveTab] = useState('dashboard');
  const [isLoginModalOpen, setIsLoginModalOpen] = useState(false);
  const [isGuideModalOpen, setIsGuideModalOpen] = useState(false);
  const [isOnline, setIsOnline] = useState(false);
  const [lastUpdated, setLastUpdated] = useState('');

  const [currentUser, setCurrentUser] = useState({
    name: 'Isheetha S',
    email: 'isheetha@isro.gov.in',
    organization: 'ISSDC / ISRO Space Data Centre',
    purpose: 'researcher',
    roleTitle: 'Scientific Researcher'
  });

  // API telemetry & predictions state
  const [latestPrediction, setLatestPrediction] = useState(null);
  const [latestTelemetry, setLatestTelemetry] = useState(null);
  const [history, setHistory] = useState([]);

  const fetchData = async () => {
    try {
      const resPred = await fetch(`${API_BASE}/latest`);
      if (resPred.ok) {
        const predData = await resPred.json();
        setLatestPrediction(predData);
      }

      const resTel = await fetch(`${API_BASE}/telemetry/latest`);
      if (resTel.ok) {
        const telData = await resTel.json();
        setLatestTelemetry(telData);
      }

      const resHist = await fetch(`${API_BASE}/predictions?limit=50`);
      if (resHist.ok) {
        const histData = await resHist.json();
        setHistory(histData.predictions || []);
      }

      setIsOnline(true);
      const now = new Date();
      setLastUpdated(now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) + ' IST');
    } catch (err) {
      setIsOnline(false);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 2000); // 2s live polling cadence
    return () => clearInterval(interval);
  }, []);

  const handleLogin = (userInfo) => {
    setCurrentUser(userInfo);
    setIsLoggedIn(true);
  };

  const handleLogout = () => {
    setIsLoggedIn(false);
  };

  const handleQuickAction = (actionId) => {
    if (actionId === 'calendar') setActiveTab('calendar');
    else if (actionId === 'alerts') setActiveTab('alerts');
    else if (actionId === 'export') {
      alert('Exporting backend prediction history (JSON)...');
    }
  };

  // Compile real alert items from API backend
  const realAlerts = [];
  if (latestPrediction && latestPrediction.risk_level && latestPrediction.risk_level !== 'NORMAL') {
    realAlerts.push({
      id: `alert-latest`,
      severity: latestPrediction.risk_level.toLowerCase(),
      risk_level: latestPrediction.risk_level,
      title: `Aditya-L1 Risk Alert: ${latestPrediction.risk_level}`,
      body: `CME probability computed at ${(latestPrediction.cme_probability * 100).toFixed(1)}%. Model version ${latestPrediction.model_version || 'v1.0.0'}.`,
      timestamp: latestPrediction.timestamp ? new Date(latestPrediction.timestamp).toLocaleString() : 'Just now',
      target: 'ISSDC Alert Pipeline'
    });
  }

  return (
    <div className="app-root-container">
      {/* Deep Space Background Layer */}
      <BackgroundSpace />

      {/* 1. LOGIN SCREEN REQUIRED FIRST */}
      {!isLoggedIn ? (
        <LoginPage onLogin={handleLogin} />
      ) : (
        /* 2. DASHBOARD VIEW (AUTHENTICATED) */
        <div className="layout-wrapper">
          {/* Scientific Parameter Descriptions & ML Guide Modal */}
          <ScientificGuide
            isOpen={isGuideModalOpen}
            onClose={() => setIsGuideModalOpen(false)}
          />

          {/* User Settings Modal */}
          <UserOnboardingModal
            isOpen={isLoginModalOpen}
            onClose={() => setIsLoginModalOpen(false)}
            currentUser={currentUser}
            onSaveUser={(updatedUser) => setCurrentUser(updatedUser)}
          />

          {/* Left Sidebar Navigation */}
          <Sidebar
            activeTab={activeTab}
            setActiveTab={setActiveTab}
            currentUser={currentUser}
            openLoginModal={() => setIsLoginModalOpen(true)}
            openGuideModal={() => setIsGuideModalOpen(true)}
            onLogout={handleLogout}
          />

          {/* Main Content Viewport */}
          <main className="main-content-area">
            {/* Top Navigation Header */}
            <Header
              activeTab={activeTab}
              setActiveTab={setActiveTab}
              userRole={currentUser.roleTitle}
              openLoginModal={() => setIsLoginModalOpen(true)}
              openGuideModal={() => setIsGuideModalOpen(true)}
              isOnline={isOnline}
              onLogout={handleLogout}
            />

            {/* Welcome Banner */}
            <div className="welcome-banner">
              <div>
                <h1 className="welcome-title">Welcome, {currentUser.name}</h1>
                <p className="welcome-subtitle">
                  Aditya-L1 Halo CME Near-Real-Time Forecasting & Space Weather Telemetry Stream
                </p>
              </div>
              <div className="user-purpose-badge">
                Role: <span className="text-cyan-300 font-semibold">{currentUser.roleTitle}</span>
              </div>
            </div>

            {/* Top Metrics Cards Summary (Derived strictly from backend history) */}
            <MetricsSummary
              history={history}
              isOnline={isOnline}
              onCardClick={(id) => {
                if (id === 'upcoming') setActiveTab('dashboard');
                if (id === 'alerts') setActiveTab('alerts');
                if (id === 'recent') setActiveTab('calendar');
              }}
            />

            {/* Main Tabs */}
            {activeTab === 'dashboard' && (
              <div className="dashboard-view-stack">
                {/* Row 1: Model CME Risk Predictions & Solar Activity Gauge */}
                <div className="dashboard-grid-2col">
                  <PredictedFlaresTable predictions={history} onViewAll={() => setActiveTab('calendar')} />
                  <SolarActivityGauge telemetryData={latestTelemetry} lastUpdated={lastUpdated} />
                </div>

                {/* Row 2: Solar Flare Calendar & Stakeholder Alerts */}
                <div className="dashboard-grid-2col">
                  <SolarFlareCalendar history={history} />
                  <StakeholderAlerts alerts={realAlerts} onViewAll={() => setActiveTab('alerts')} />
                </div>

                {/* Row 3: Quick Actions */}
                <div className="w-full">
                  <QuickActions onAction={handleQuickAction} openGuideModal={() => setIsGuideModalOpen(true)} />
                </div>
              </div>
            )}

            {activeTab === 'calendar' && (
              <div className="tab-view-container flex flex-col gap-6">
                <SolarFlareCalendar history={history} />
                <div className="w-full">
                  <PredictedFlaresTable predictions={history} />
                </div>
              </div>
            )}

            {activeTab === 'alerts' && (
              <div className="tab-view-container flex flex-col gap-6">
                <StakeholderAlerts alerts={realAlerts} />
                <QuickActions onAction={handleQuickAction} openGuideModal={() => setIsGuideModalOpen(true)} />
              </div>
            )}

            {activeTab === 'telemetry' && (
              <div className="tab-view-container flex flex-col gap-6">
                <div className="top-grid">
                  <TelemetryGauges latestTelemetry={latestTelemetry} />
                  <SolarActivityGauge telemetryData={latestTelemetry} lastUpdated={lastUpdated} />
                </div>
                <TimeSeriesChart history={history} />
                <AnomalyStatus latestPrediction={latestPrediction} />
                <HistoryTable history={history} />
              </div>
            )}
          </main>
        </div>
      )}
    </div>
  );
}
