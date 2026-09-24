import React from 'react';

export default function AnomalyStatus({ latestPrediction }) {
  const isAnomaly = latestPrediction ? latestPrediction.is_anomaly : false;
  const score = latestPrediction ? latestPrediction.anomaly_score : 0.0;

  return (
    <div className="card section-margin">
      <div className="card-title">
        <span>Isolation Forest Sensor Anomaly Detector</span>
        <span style={{
          padding: '0.2rem 0.6rem',
          borderRadius: '12px',
          fontSize: '0.75rem',
          fontWeight: '600',
          backgroundColor: isAnomaly ? 'rgba(239, 68, 68, 0.2)' : 'rgba(16, 185, 129, 0.2)',
          color: isAnomaly ? '#ef4444' : '#10b981',
          border: `1px solid ${isAnomaly ? '#ef4444' : '#10b981'}`
        }}>
          {isAnomaly ? 'ANOMALY DETECTED' : 'NORMAL PATTERN'}
        </span>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: '0.5rem' }}>
        <div>
          <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Isolation Forest Score</div>
          <div style={{ fontSize: '1.5rem', fontWeight: '700', fontFamily: 'var(--font-mono)' }}>
            {score.toFixed(4)}
          </div>
        </div>
        <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', maxWidth: '400px' }}>
          Parallel Isolation Forest monitors raw SWIS telemetry for instrument anomalies, missing sensors, or abrupt space plasma spikes.
        </div>
      </div>
    </div>
  );
}
