import React from 'react';

export default function TimeSeriesChart({ history }) {
  const data = (history || []).slice(0, 30).reverse();

  if (data.length === 0) {
    return (
      <div className="card section-margin">
        <div className="card-title">12-Hour Rolling CME Risk Trend</div>
        <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
          Awaiting telemetry stream samples...
        </div>
      </div>
    );
  }

  const width = 800;
  const height = 180;
  const padding = 20;

  const points = data.map((item, idx) => {
    const x = padding + (idx / Math.max(data.length - 1, 1)) * (width - 2 * padding);
    const prob = item.cme_probability || 0;
    const y = height - padding - prob * (height - 2 * padding);
    return `${x},${y}`;
  }).join(' ');

  return (
    <div className="card section-margin">
      <div className="card-title">
        <span>12-Hour Rolling CME Risk & Solar Wind Trend</span>
        <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
          Showing last {data.length} telemetry points
        </span>
      </div>

      <div style={{ width: '100%', overflowX: 'auto' }}>
        <svg className="chart-svg" viewBox={`0 0 ${width} ${height}`}>
          <defs>
            <linearGradient id="chartGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.4" />
              <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.0" />
            </linearGradient>
          </defs>

          {/* Threshold reference lines */}
          <line x1={padding} y1={height - padding - 0.37 * (height - 2 * padding)} x2={width - padding} y2={height - padding - 0.37 * (height - 2 * padding)} stroke="#f59e0b" strokeDasharray="4 4" strokeWidth="1" />
          <line x1={padding} y1={height - padding - 0.85 * (height - 2 * padding)} x2={width - padding} y2={height - padding - 0.85 * (height - 2 * padding)} stroke="#ef4444" strokeDasharray="4 4" strokeWidth="1" />

          {/* Area fill */}
          {data.length > 1 && (
            <polygon
              points={`${padding},${height - padding} ${points} ${width - padding},${height - padding}`}
              fill="url(#chartGradient)"
            />
          )}

          {/* Trend line */}
          <polyline
            fill="none"
            stroke="#06b6d4"
            strokeWidth="3"
            points={points}
          />

          {/* Dots */}
          {data.map((item, idx) => {
            const x = padding + (idx / Math.max(data.length - 1, 1)) * (width - 2 * padding);
            const prob = item.cme_probability || 0;
            const y = height - padding - prob * (height - 2 * padding);
            const color = prob >= 0.85 ? '#ef4444' : prob >= 0.37 ? '#f59e0b' : '#10b981';
            return (
              <circle key={idx} cx={x} cy={y} r="4" fill={color} stroke="#0b0f19" strokeWidth="1.5">
                <title>{`${item.timestamp}: ${(prob * 100).toFixed(1)}% (${item.risk_level})`}</title>
              </circle>
            );
          })}
        </svg>
      </div>
    </div>
  );
}
