import React, { useState } from 'react';
import { Bell, AlertTriangle, AlertCircle, CheckCircle2, Radio, Info } from 'lucide-react';

export default function StakeholderAlerts({ alerts = [], onViewAll }) {
  const [filter, setFilter] = useState('ALL');

  const filteredAlerts = alerts.filter((a) => {
    if (filter === 'ALL') return true;
    return a.severity?.toLowerCase() === filter.toLowerCase() || a.risk_level?.toLowerCase() === filter.toLowerCase();
  });

  const getSeverityIcon = (sev) => {
    if (sev === 'critical' || sev === 'CRITICAL') return <AlertTriangle size={20} color="#ef4444" />;
    if (sev === 'warning' || sev === 'WARNING') return <AlertCircle size={20} color="#f59e0b" />;
    return <CheckCircle2 size={20} color="#10b981" />;
  };

  return (
    <div className="card glass-panel flex-1">
      <div className="card-header-flex">
        <div className="flex items-center gap-2">
          <Bell size={18} color="#38bdf8" />
          <h3 className="card-title-text">Stakeholder Alerts & Notifications</h3>
        </div>

        <div className="flex items-center gap-3">
          <div className="filter-tab-group">
            {['ALL', 'CRITICAL', 'WARNING', 'INFO'].map((lvl) => (
              <button
                key={lvl}
                className={`filter-btn ${filter === lvl ? 'active' : ''}`}
                onClick={() => setFilter(lvl)}
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

      <div className="alerts-feed-list">
        {filteredAlerts.length === 0 ? (
          <div className="empty-state-box text-center py-8 text-slate-400">
            <Info size={24} className="mx-auto mb-2 text-slate-500" />
            <p className="font-semibold text-sm">No Active Alerts Logged</p>
            <p className="text-xs text-slate-500 mt-1">
              System is monitoring Aditya-L1 telemetry. Alerts are automatically logged when CME risk exceeds thresholds.
            </p>
          </div>
        ) : (
          filteredAlerts.map((alt, idx) => (
            <div key={alt.id || idx} className={`alert-feed-item severity-${(alt.severity || alt.risk_level || 'info').toLowerCase()}`}>
              <div className="alert-icon-column">
                {getSeverityIcon(alt.severity || alt.risk_level)}
              </div>

              <div className="alert-content-column">
                <div className="alert-title-row">
                  <span className="alert-item-title">{alt.title || `CME Risk Alert: ${alt.risk_level}`}</span>
                  <span className="alert-timestamp">{alt.timestamp || alt.created_at}</span>
                </div>
                <p className="alert-item-body">{alt.body || alt.message || `CME probability calculated at ${(alt.cme_probability * 100).toFixed(1)}%.`}</p>

                <div className="alert-tag-row">
                  <span className="stakeholder-tag">
                    <Radio size={12} /> {alt.target || 'ISSDC Alert Channel'}
                  </span>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
