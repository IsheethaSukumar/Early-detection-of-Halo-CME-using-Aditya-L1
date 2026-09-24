import React from 'react';

export default function TelemetryGauges({ latestTelemetry }) {
  const t = latestTelemetry || {};

  const metrics = [
    { label: 'Plasma Speed', val: t.plasma_speed || t.plasma_speed_kmps || 0, unit: 'km/s', color: '#38bdf8' },
    { label: 'Proton Density', val: t.density || t.proton_density || 0, unit: 'N/cm³', color: '#818cf8' },
    { label: 'Temperature', val: t.temperature || t.plasma_temperature_k || 0, unit: 'K', color: '#c084fc' },
    { label: 'Flow Pressure', val: t.flow_pressure || t.flow_pressure_npa || 0, unit: 'nPa', color: '#f472b6' },
    { label: 'Bz (GSE)', val: t.Bz || t.bz_gse || 0, unit: 'nT', color: t.Bz < 0 ? '#ef4444' : '#10b981' },
    { label: 'B Magnitude', val: t.Bmag || t.B_magnitude || 0, unit: 'nT', color: '#f59e0b' },
  ];

  return (
    <div className="card">
      <div className="card-title">
        <span>Aditya-L1 SWIS Plasma Telemetry</span>
        <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>5-Min Cadence</span>
      </div>

      <div className="telemetry-grid">
        {metrics.map((m, idx) => (
          <div key={idx} className="telemetry-card">
            <div className="telemetry-label">{m.label}</div>
            <div className="telemetry-value" style={{ color: m.color }}>
              {typeof m.val === 'number' ? m.val.toFixed(1) : m.val}
              <span className="telemetry-unit">{m.unit}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
