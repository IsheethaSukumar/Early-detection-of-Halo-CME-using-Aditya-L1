import React from 'react';
import { Sun, Clock, AlertTriangle, ShieldCheck, ChevronRight } from 'lucide-react';

export default function MetricsSummary({ history = [], isOnline = true, onCardClick }) {
  const upcomingCount = history.filter((p) => (p.cme_probability || 0) >= 0.37 || p.risk_level === 'WARNING' || p.risk_level === 'CRITICAL').length;
  const recentCount = history.length;
  const activeAlertsCount = history.filter((p) => p.risk_level === 'CRITICAL').length;
  const systemStatus = isOnline ? 'Normal' : 'Standby';

  const cards = [
    {
      id: 'upcoming',
      title: 'Elevated Risk Flares',
      value: upcomingCount,
      subtext: 'in historical database',
      icon: Sun,
      iconColor: '#f97316',
      badgeBg: 'rgba(249, 115, 22, 0.15)',
    },
    {
      id: 'recent',
      title: 'Inferences Evaluated',
      value: recentCount,
      subtext: 'total backend records',
      icon: Clock,
      iconColor: '#38bdf8',
      badgeBg: 'rgba(56, 189, 248, 0.15)',
    },
    {
      id: 'alerts',
      title: 'Active Critical Alerts',
      value: activeAlertsCount,
      subtext: 'requiring attention',
      icon: AlertTriangle,
      iconColor: '#ef4444',
      badgeBg: 'rgba(239, 68, 68, 0.15)',
      isAlert: activeAlertsCount > 0
    },
    {
      id: 'status',
      title: 'System Status',
      value: systemStatus,
      subtext: isOnline ? 'API Stream Operational' : 'Telemetry Standby Mode',
      icon: ShieldCheck,
      iconColor: isOnline ? '#10b981' : '#f59e0b',
      badgeBg: isOnline ? 'rgba(16, 185, 129, 0.15)' : 'rgba(245, 158, 11, 0.15)',
    },
  ];

  return (
    <div className="metrics-summary-grid">
      {cards.map((c) => {
        const Icon = c.icon;
        return (
          <div
            key={c.id}
            className={`metric-card-item ${c.isAlert ? 'alert-border' : ''}`}
            onClick={() => onCardClick && onCardClick(c.id)}
          >
            <div className="metric-card-top">
              <div className="metric-icon-box" style={{ backgroundColor: c.badgeBg, color: c.iconColor }}>
                <Icon size={22} />
              </div>
              <ChevronRight size={18} className="arrow-icon" />
            </div>

            <div className="metric-card-content">
              <span className="metric-title-label">{c.title}</span>
              <div className="metric-main-value" style={{ color: typeof c.value === 'number' ? '#ffffff' : c.iconColor }}>
                {c.value}
              </div>
              <span className="metric-subtext">{c.subtext}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}
