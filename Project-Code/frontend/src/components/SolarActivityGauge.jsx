import React from 'react';
import { Gauge, Activity, Wind, Zap, Compass, RefreshCw, Info } from 'lucide-react';

export default function SolarActivityGauge({ telemetryData, lastUpdated }) {
  if (!telemetryData) {
    return (
      <div className="card glass-panel flex-1 flex flex-col justify-between">
        <div className="card-header-flex">
          <div className="flex items-center gap-2">
            <Activity size={18} color="#38bdf8" />
            <h3 className="card-title-text">Solar Telemetry Status</h3>
          </div>
          <span className="live-pulse-badge">AWAITING FEED</span>
        </div>

        <div className="empty-state-box text-center py-10 text-slate-400">
          <Info size={28} className="mx-auto mb-2 text-slate-500" />
          <p className="font-semibold text-sm">No Telemetry Stream Connected</p>
          <p className="text-xs text-slate-500 mt-1 max-w-xs mx-auto">
            Run backend microservice (`uvicorn Project-Code.ML_Model.app:app`) or `simulate_telemetry.py` to stream live solar wind measurements.
          </p>
        </div>
      </div>
    );
  }

  const speed = telemetryData.plasma_speed || telemetryData.plasma_speed_kmps || 0;
  const density = telemetryData.density || telemetryData.proton_density || 0;
  const bz = telemetryData.Bz || telemetryData.bz_gse || 0;
  const bmag = telemetryData.Bmag || telemetryData.B_magnitude || 0;

  // Calculate dynamic Kp estimate based on plasma speed & southward Bz field
  let kp = 1;
  if (bz < -10 || speed > 700) kp = 8;
  else if (bz < -5 || speed > 550) kp = 5;
  else if (bz < 0 || speed > 450) kp = 3;

  const needleAngle = -90 + (kp / 9) * 180;

  const getKpLabel = (val) => {
    if (val <= 3) return { text: 'Low (Quiet)', color: '#10b981' };
    if (val <= 6) return { text: 'Moderate Storm', color: '#f59e0b' };
    return { text: 'Severe Storm', color: '#ef4444' };
  };

  const statusObj = getKpLabel(kp);

  return (
    <div className="card glass-panel flex-1 flex flex-col justify-between">
      <div className="card-header-flex">
        <div className="flex items-center gap-2">
          <Activity size={18} color="#38bdf8" />
          <h3 className="card-title-text">Solar Telemetry Status</h3>
        </div>
        <span className="live-pulse-badge">LIVE L1 FEED</span>
      </div>

      {/* Speedometer Gauge */}
      <div className="gauge-container">
        <svg viewBox="0 0 200 120" className="gauge-svg">
          <path
            d="M 20 100 A 80 80 0 0 1 180 100"
            fill="none"
            stroke="rgba(255, 255, 255, 0.08)"
            strokeWidth="16"
            strokeLinecap="round"
          />
          <path
            d="M 20 100 A 80 80 0 0 1 67.5 35"
            fill="none"
            stroke="#10b981"
            strokeWidth="14"
            strokeLinecap="round"
          />
          <path
            d="M 67.5 35 A 80 80 0 0 1 132.5 35"
            fill="none"
            stroke="#f59e0b"
            strokeWidth="14"
          />
          <path
            d="M 132.5 35 A 80 80 0 0 1 180 100"
            fill="none"
            stroke="#ef4444"
            strokeWidth="14"
            strokeLinecap="round"
          />
          <g transform={`rotate(${needleAngle}, 100, 100)`}>
            <line x1="100" y1="100" x2="100" y2="30" stroke="#ffffff" strokeWidth="3" strokeLinecap="round" />
            <circle cx="100" cy="100" r="7" fill="#38bdf8" stroke="#ffffff" strokeWidth="2" />
          </g>
        </svg>

        <div className="gauge-status-box">
          <div className="gauge-status-value" style={{ color: statusObj.color }}>
            {statusObj.text}
          </div>
          <div className="gauge-kp-tag">Estimated Kp: {kp} / 9</div>
        </div>
      </div>

      {/* Metric List */}
      <div className="activity-metrics-list">
        <div className="metric-row">
          <div className="flex items-center gap-2 text-slate-400">
            <Wind size={14} />
            <span>Plasma Speed</span>
          </div>
          <span className="font-mono text-cyan-300 font-bold">{speed.toFixed(1)} km/s</span>
        </div>

        <div className="metric-row">
          <div className="flex items-center gap-2 text-slate-400">
            <Zap size={14} />
            <span>Proton Density</span>
          </div>
          <span className="font-mono text-amber-400 font-bold">{density.toFixed(1)} N/cm³</span>
        </div>

        <div className="metric-row">
          <div className="flex items-center gap-2 text-slate-400">
            <Compass size={14} />
            <span>Bz Magnetic Field</span>
          </div>
          <span className="font-mono text-rose-400 font-bold">{bz.toFixed(1)} nT</span>
        </div>

        <div className="metric-row">
          <div className="flex items-center gap-2 text-slate-400">
            <Gauge size={14} />
            <span>B Magnitude</span>
          </div>
          <span className="font-mono text-purple-400 font-bold">{bmag.toFixed(1)} nT</span>
        </div>
      </div>

      <div className="gauge-footer-timestamp">
        <RefreshCw size={12} className="spin-icon" />
        <span>Last Updated: {lastUpdated || 'Stream Active'}</span>
      </div>
    </div>
  );
}
