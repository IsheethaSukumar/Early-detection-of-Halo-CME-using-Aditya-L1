import React, { useState } from 'react';
import { Shield, Zap, Satellite, Globe, Check, Lock, User, Mail, X, Radio, GraduationCap } from 'lucide-react';

export default function UserOnboardingModal({ isOpen, onClose, currentUser, onSaveUser }) {
  if (!isOpen) return null;

  const [name, setName] = useState(currentUser?.name || 'Isheetha S');
  const [email, setEmail] = useState(currentUser?.email || 'isheetha@isro.gov.in');
  const [selectedPurpose, setSelectedPurpose] = useState(currentUser?.purpose || 'researcher');
  const [mode, setMode] = useState('login'); // 'login' or 'purpose'

  const purposes = [
    {
      id: 'researcher',
      title: 'Scientific Researcher / Astronomer',
      org: 'ISSDC / ISRO / Academic Institutions',
      icon: GraduationCap,
      color: '#38bdf8',
      desc: 'Access high-resolution vector magnetograms, solar X-ray spectra, and flare prediction models.'
    },
    {
      id: 'defense_gov',
      title: 'Government & National Defense',
      org: 'Space Weather Threat Monitoring & Disaster Mgmt',
      icon: Shield,
      color: '#ef4444',
      desc: 'High-priority blackout warnings, infrastructure safety advisories, and strategic communications.'
    },
    {
      id: 'grid_operator',
      title: 'Power Grid Infrastructure Operator',
      org: 'Electrical Power Transmissions / High Latitudes',
      icon: Zap,
      color: '#f59e0b',
      desc: 'Geomagnetic Induced Current (GIC) hazard forecasts and transformer saturation warnings.'
    },
    {
      id: 'satellite_aviation',
      title: 'Satellite & Trans-Polar Aviation Operator',
      org: 'ISRO Telemetry (ISTRAC) / Commercial Satellites',
      icon: Satellite,
      color: '#a855f7',
      desc: 'HF radio propagation blackout alerts (D-RAP) and high-altitude solar proton particle fluxes.'
    },
    {
      id: 'public',
      title: 'Public & Aurora Enthusiast',
      org: 'General Interest / Citizen Science',
      icon: Globe,
      color: '#10b981',
      desc: 'Real-time geomagnetic Kp index tracking, aurora oval forecasts, and educational solar news.'
    }
  ];

  const handleSubmit = (e) => {
    e.preventDefault();
    const chosenObj = purposes.find((p) => p.id === selectedPurpose);
    onSaveUser({
      name,
      email,
      purpose: selectedPurpose,
      roleTitle: chosenObj ? chosenObj.title : 'Scientific Researcher'
    });
    onClose();
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-container dark-glass-panel" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="flex items-center gap-3">
            <div className="modal-icon-badge">
              <User size={20} color="#38bdf8" />
            </div>
            <div>
              <h2 className="modal-title">Space Weather User Access</h2>
              <p className="modal-subtitle">Configure your organization purpose and stakeholder alert preferences</p>
            </div>
          </div>
          <button className="close-btn" onClick={onClose}><X size={18} /></button>
        </div>

        <form onSubmit={handleSubmit} className="modal-body">
          {/* User Details */}
          <div className="form-group-row">
            <div className="input-box">
              <label>Full Name / Station ID</label>
              <div className="input-wrapper">
                <User size={16} className="input-icon" />
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
              <label>Official Email Address</label>
              <div className="input-wrapper">
                <Mail size={16} className="input-icon" />
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="e.g. user@isro.gov.in"
                  required
                />
              </div>
            </div>
          </div>

          {/* User Purpose Selection */}
          <div className="purpose-section">
            <label className="section-label">Select Your Operational Purpose & Role</label>
            <div className="purpose-grid">
              {purposes.map((p) => {
                const Icon = p.icon;
                const isSelected = selectedPurpose === p.id;
                return (
                  <div
                    key={p.id}
                    className={`purpose-card ${isSelected ? 'selected' : ''}`}
                    onClick={() => setSelectedPurpose(p.id)}
                    style={{ borderColor: isSelected ? p.color : 'rgba(255,255,255,0.08)' }}
                  >
                    <div className="purpose-card-header">
                      <div className="purpose-icon" style={{ backgroundColor: `${p.color}20`, color: p.color }}>
                        <Icon size={18} />
                      </div>
                      <div className="purpose-title">{p.title}</div>
                      {isSelected && <div className="selected-badge"><Check size={12} /></div>}
                    </div>
                    <div className="purpose-org">{p.org}</div>
                    <div className="purpose-desc">{p.desc}</div>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="modal-footer">
            <button type="button" className="btn-secondary" onClick={onClose}>
              Cancel
            </button>
            <button type="submit" className="btn-primary">
              <Lock size={14} />
              Save Access Profile
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
