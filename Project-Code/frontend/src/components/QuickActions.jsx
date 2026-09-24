import React from 'react';
import { Calendar, Bell, Download, ChevronRight, Zap, HelpCircle } from 'lucide-react';

export default function QuickActions({ onAction, openGuideModal }) {
  const actions = [
    {
      id: 'calendar',
      title: 'View Full Flare Calendar',
      desc: 'Check past and upcoming solar events',
      icon: Calendar,
      color: '#38bdf8'
    },
    {
      id: 'guide',
      title: 'Scientific Guide & Parameter Definitions',
      desc: 'Learn how flare ML models predict active regions',
      icon: HelpCircle,
      color: '#a855f7'
    },
    {
      id: 'alerts',
      title: 'Manage Stakeholder Alerts',
      desc: 'Set up notifications for power grid & aviation',
      icon: Bell,
      color: '#ef4444'
    },
    {
      id: 'export',
      title: 'Export Data Report',
      desc: 'Download ISSDC solar flare CSV/JSON dataset',
      icon: Download,
      color: '#10b981'
    }
  ];

  const handleClick = (actId) => {
    if (actId === 'guide') {
      openGuideModal();
    } else if (onAction) {
      onAction(actId);
    }
  };

  return (
    <div className="card glass-panel flex-1">
      <div className="card-header-flex">
        <div className="flex items-center gap-2">
          <Zap size={18} color="#38bdf8" />
          <h3 className="card-title-text">Quick Actions</h3>
        </div>
      </div>

      <div className="quick-actions-list">
        {actions.map((act) => {
          const Icon = act.icon;
          return (
            <div
              key={act.id}
              className="quick-action-item"
              onClick={() => handleClick(act.id)}
            >
              <div className="flex items-center gap-3">
                <div className="quick-action-icon-box" style={{ backgroundColor: `${act.color}18`, color: act.color }}>
                  <Icon size={18} />
                </div>
                <div>
                  <div className="quick-action-title">{act.title}</div>
                  <div className="quick-action-desc">{act.desc}</div>
                </div>
              </div>
              <ChevronRight size={18} className="arrow-icon" />
            </div>
          );
        })}
      </div>
    </div>
  );
}
