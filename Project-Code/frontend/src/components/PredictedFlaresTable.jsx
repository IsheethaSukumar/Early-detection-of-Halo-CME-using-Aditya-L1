import React, { useState } from 'react';
import { Sparkles, Info, ShieldAlert } from 'lucide-react';

export default function PredictedFlaresTable({ predictions = [], onViewAll }) {
  const [filterLevel, setFilterLevel] = useState('ALL');

  // Sort predictions so the latest timestamp appears at the top (latest first)
  const sortedPredictions = [...predictions].sort((a, b) => {
    if (!a.timestamp) return 1;
    if (!b.timestamp) return -1;
    return new Date(b.timestamp) - new Date(a.timestamp);
  });

  const filteredList = sortedPredictions.filter((item) => {
    if (filterLevel === 'ALL') return true;
    return item.risk_level === filterLevel;
  });

  const getImpactBadge = (level) => {
    if (level === 'CRITICAL' || level === 'High') {
      return (
        <span className="impact-tag tag-high">
          <span className="dot-indicator dot-red"></span> CRITICAL
        </span>
      );
    }
    if (level === 'WARNING' || level === 'Moderate') {
      return (
        <span className="impact-tag tag-yellow">
          <span className="dot-indicator dot-yellow"></span> WARNING
        </span>
      );
    }
    return (
      <span className="impact-tag tag-low">
        <span className="dot-indicator dot-green"></span> NORMAL
      </span>
    );
  };

  return (
    <div className="card glass-panel flex-1">
      <div className="card-header-flex">
        <div className="flex items-center gap-2">
          <Sparkles size={18} color="#38bdf8" />
          <h3 className="card-title-text">Model CME Risk Predictions & SWIS Telemetry</h3>
        </div>

        <div className="flex items-center gap-3">
          <div className="filter-tab-group">
            {['ALL', 'CRITICAL', 'WARNING', 'NORMAL'].map((lvl) => (
              <button
                key={lvl}
                className={`filter-btn ${filterLevel === lvl ? 'active' : ''}`}
                onClick={() => setFilterLevel(lvl)}
              >
                {lvl}
              </button>
            ))}
          </div>

          {onViewAll && (
            <button className="view-all-link" onClick={onViewAll}>
              View All &rarr;
            </button>
          )}
        </div>
      </div>

      <div className="table-responsive">
        {filteredList.length === 0 ? (
          <div className="empty-state-box text-center py-8 text-slate-400">
            <Info size={24} className="mx-auto mb-2 text-slate-500" />
            <p className="font-semibold text-sm">No Inference Predictions Available</p>
            <p className="text-xs text-slate-500 mt-1">
              Start the FastAPI backend server (`uvicorn Project-Code.ML_Model.app:app`) to stream real ML model inferences.
            </p>
          </div>
        ) : (
          <table className="flare-table">
            <thead>
              <tr>
                <th>Timestamp (IST)</th>
                <th>CME Prob</th>
                <th>Risk Level</th>
                <th>Anomaly Flag</th>
                <th>Speed (km/s)</th>
                <th>Bz (nT)</th>
                <th>Latency</th>
              </tr>
            </thead>
            <tbody>
              {filteredList.map((item, idx) => (
                <tr key={item.id || idx}>
                  <td className="font-mono text-cyan-300">
                    {item.timestamp ? new Date(item.timestamp).toLocaleString('en-IN', { timeZone: 'Asia/Kolkata' }) : 'N/A'}
                  </td>
                  <td className="font-mono font-bold text-slate-100">
                    {item.cme_probability !== undefined ? `${(item.cme_probability * 100).toFixed(1)}%` : '0.0%'}
                  </td>
                  <td>{getImpactBadge(item.risk_level)}</td>
                  <td>
                    {item.is_anomaly ? (
                      <span className="text-rose-400 font-semibold text-xs flex items-center gap-1">
                        <ShieldAlert size={12} /> ANOMALY
                      </span>
                    ) : (
                      <span className="text-slate-400 text-xs">Nominal</span>
                    )}
                  </td>
                  <td className="font-mono text-cyan-400 text-xs">
                    {item.plasma_speed !== undefined ? item.plasma_speed.toFixed(1) : (item.telemetry?.plasma_speed ? item.telemetry.plasma_speed.toFixed(1) : 'N/A')}
                  </td>
                  <td className="font-mono text-amber-400 text-xs">
                    {item.bz_gse !== undefined ? item.bz_gse.toFixed(1) : (item.telemetry?.bz_gse ? item.telemetry.bz_gse.toFixed(1) : 'N/A')}
                  </td>
                  <td className="font-mono text-slate-400 text-xs">
                    {item.inference_latency !== undefined ? `${item.inference_latency} ms` : 'N/A'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
