import React, { useState } from 'react';
import { Sun, Lock, User, Mail, Shield, Zap, Satellite, Globe, GraduationCap, ArrowRight, Sparkles, Building, KeyRound, CheckCircle2 } from 'lucide-react';

export default function LoginPage({ onLogin }) {
  const [activeTab, setActiveTab] = useState('login'); // 'login' or 'register'

  // Form Fields
  const [email, setEmail] = useState('isheetha@isro.gov.in');
  const [password, setPassword] = useState('••••••••••••');
  const [confirmPassword, setConfirmPassword] = useState('••••••••••••');
  const [name, setName] = useState('Isheetha S');
  const [organization, setOrganization] = useState('ISSDC / ISRO Space Data Centre');
  const [selectedPurpose, setSelectedPurpose] = useState('researcher');

  const purposes = [
    {
      id: 'researcher',
      title: 'Scientific Researcher',
      org: 'ISSDC / ISRO Solar Physics',
      icon: GraduationCap,
      color: '#38bdf8',
      desc: 'Access magnetograms, X-ray spectra & solar flare ML predictions.'
    },
    {
      id: 'defense_gov',
      title: 'Government & Defense',
      org: 'Space Weather Warning Center',
      icon: Shield,
      color: '#ef4444',
      desc: 'High-priority HF blackout alerts & strategic satellite advisories.'
    },
    {
      id: 'grid_operator',
      title: 'Power Grid Operator',
      org: 'National Electrical Transmission',
      icon: Zap,
      color: '#f59e0b',
      desc: 'Geomagnetic Induced Current (GIC) transformer overload warnings.'
    },
    {
      id: 'satellite_aviation',
      title: 'Satellite & Aviation Operator',
      org: 'ISTRAC / Flight Operations',
      icon: Satellite,
      color: '#a855f7',
      desc: 'Ionospheric absorption (D-RAP) & radiation particle flux tracking.'
    },
    {
      id: 'public',
      title: 'Public / Citizen Science',
      org: 'Space Weather Enthusiast',
      icon: Globe,
      color: '#10b981',
      desc: 'Real-time Kp index, aurora oval forecasts & solar flare updates.'
    }
  ];

  const handleSubmit = (e) => {
    e.preventDefault();
    if (activeTab === 'register' && password !== confirmPassword) {
      alert('Passwords do not match. Please verify your password.');
      return;
    }

    const chosenObj = purposes.find((p) => p.id === selectedPurpose);
    onLogin({
      name: name || 'Isheetha S',
      email: email || 'isheetha@isro.gov.in',
      organization: organization || 'ISSDC / ISRO',
      purpose: selectedPurpose,
      roleTitle: chosenObj ? chosenObj.title : 'Scientific Researcher'
    });
  };

  return (
    <div className="login-screen-container">
      <div className="login-glass-card">
        {/* Header Branding */}
        <div className="login-header text-center">
          <div className="login-logo-ring">
            <Sun size={36} color="#f97316" className="sun-icon-animated" />
          </div>
          <h1 className="login-brand-title">Aditya-L1</h1>
          <div className="login-isro-tag font-mono">ISSDC / ISRO SPACE DATA CENTRE</div>
          <p className="login-tagline">Solar Activity Monitoring & Prediction System</p>
          <div className="login-sub-motto font-italic text-slate-400 text-xs mt-1">
            "Observing the Sun to Protect Our World"
          </div>
        </div>

        {/* Tab Selector: Login vs Create Account (Matching Reference Image 2) */}
        <div className="auth-tab-switch">
          <button
            type="button"
            className={`auth-tab-btn ${activeTab === 'login' ? 'active' : ''}`}
            onClick={() => setActiveTab('login')}
          >
            Login to Account
          </button>
          <button
            type="button"
            className={`auth-tab-btn ${activeTab === 'register' ? 'active' : ''}`}
            onClick={() => setActiveTab('register')}
          >
            Create an Account
          </button>
        </div>

        {/* Form Body */}
        <form onSubmit={handleSubmit} className="login-form">
          {activeTab === 'register' && (
            <div className="form-group-row">
              <div className="input-box">
                <label>Full Name / Station ID</label>
                <div className="input-wrapper">
                  <User size={16} className="text-cyan-400" />
                  <input
                    type="text"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder="e.g. Isheetha S"
                    required
                  />
                </div>
              </div>

              <div className="input-box">
                <label>Organization / Institution</label>
                <div className="input-wrapper">
                  <Building size={16} className="text-cyan-400" />
                  <input
                    type="text"
                    value={organization}
                    onChange={(e) => setOrganization(e.target.value)}
                    placeholder="e.g. ISSDC / ISRO / University"
                    required
                  />
                </div>
              </div>
            </div>
          )}

          <div className="input-box mb-4">
            <label>Username / Official Email Address</label>
            <div className="input-wrapper">
              <Mail size={16} className="text-cyan-400" />
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="user@isro.gov.in"
                required
              />
            </div>
          </div>

          <div className="form-group-row mb-4">
            <div className="input-box">
              <label>Password</label>
              <div className="input-wrapper">
                <Lock size={16} className="text-cyan-400" />
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••••••"
                  required
                />
              </div>
            </div>

            {activeTab === 'register' && (
              <div className="input-box">
                <label>Confirm Password</label>
                <div className="input-wrapper">
                  <KeyRound size={16} className="text-cyan-400" />
                  <input
                    type="password"
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    placeholder="••••••••••••"
                    required
                  />
                </div>
              </div>
            )}
          </div>

          {/* User Role & Purpose Dropdown Selection */}
          <div className="input-box mb-6">
            <label className="section-label flex items-center justify-between mb-2">
              <span>Select Your Operational Purpose & Role</span>
              <span className="text-xs text-cyan-400 flex items-center gap-1">
                <Sparkles size={12} /> Configures Dashboard Insights
              </span>
            </label>
            <div className="input-wrapper">
              <User size={16} className="text-cyan-400" />
              <select
                value={selectedPurpose}
                onChange={(e) => setSelectedPurpose(e.target.value)}
                className="login-select-dropdown"
                style={{
                  width: '100%',
                  background: 'transparent',
                  border: 'none',
                  color: '#f8fafc',
                  padding: '8px 4px',
                  fontSize: '0.9rem',
                  outline: 'none',
                  cursor: 'pointer'
                }}
              >
                {purposes.map((p) => (
                  <option key={p.id} value={p.id} style={{ background: '#0f172a', color: '#f8fafc' }}>
                    {p.title} — {p.desc}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Submit Action */}
          <button type="submit" className="login-submit-btn">
            <span>{activeTab === 'login' ? 'Login to Dashboard' : 'Create Account & Access System'}</span>
            <ArrowRight size={18} />
          </button>
        </form>
      </div>
    </div>
  );
}
