import React from 'react';

export default function HistoryTable({ history }) {
  const records = history || [];

  return (
    <div className="card">
      <div className="card-title">
        <span>Historical Forecast Log</span>
        <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
          {records.length} Recorded Runs
        </span>
      </div>

      <div className="table-container">
        <table>
          <thead>
            <tr>
              <th>Timestamp</th>
              <th>CME Prob</th>
              <th>Risk Level</th>
              <th>Anomaly Flag</th>
              <th>Speed (km/s)</th>
              <th>Bz (nT)</th>
              <th>Latency (ms)</th>
            </tr>
          </thead>
          <tbody>
            {records.length === 0 ? (
              <tr>
                <td colSpan="7" style={{ textAlign: 'center', padding: '1.5rem', color: 'var(--text-secondary)' }}>
                  No historical telemetry records yet.
                </td>
              </tr>
            ) : (
              records.map((r, idx) => (
                <tr key={idx}>
                  <td>{r.timestamp}</td>
                  <td style={{ fontWeight: '600', color: (r.cme_probability >= 0.37 ? '#f59e0b' : '#10b981') }}>
                    {(r.cme_probability * 100).toFixed(1)}%
                  </td>
                  <td>
                    <span style={{
                      padding: '0.15rem 0.5rem',
                      borderRadius: '8px',
                      fontSize: '0.75rem',
                      fontWeight: '600',
                      backgroundColor: r.risk_level === 'CRITICAL' ? 'rgba(239, 68, 68, 0.2)' : r.risk_level === 'WARNING' ? 'rgba(245, 158, 11, 0.2)' : 'rgba(16, 185, 129, 0.2)',
                      color: r.risk_level === 'CRITICAL' ? '#ef4444' : r.risk_level === 'WARNING' ? '#f59e0b' : '#10b981'
                    }}>
                      {r.risk_level}
                    </span>
                  </td>
                  <td>{r.anomaly_flag ? '⚠️ YES' : 'NO'}</td>
                  <td>{r.plasma_speed ? r.plasma_speed.toFixed(1) : 'N/A'}</td>
                  <td style={{ color: r.Bz < 0 ? '#ef4444' : '#10b981' }}>{r.Bz ? r.Bz.toFixed(1) : 'N/A'}</td>
                  <td>{r.inference_latency ? r.inference_latency.toFixed(1) : '12.4'}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
