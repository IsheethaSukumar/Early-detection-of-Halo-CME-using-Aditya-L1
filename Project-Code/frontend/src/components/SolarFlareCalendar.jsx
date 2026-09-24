import React, { useState } from 'react';
import { Calendar as CalendarIcon, ChevronLeft, ChevronRight, Info } from 'lucide-react';

export default function SolarFlareCalendar({ history = [] }) {
  const [selectedDay, setSelectedDay] = useState(new Date().getDate());
  const [currentMonth, setCurrentMonth] = useState('September 2026');

  // Days in month
  const daysInMonth = Array.from({ length: 30 }, (_, i) => i + 1);
  const paddingDays = [null]; // 1 empty slot for layout

  // Map real prediction history records by day of month
  const eventsByDay = {};
  history.forEach((pred) => {
    if (pred.timestamp) {
      const d = new Date(pred.timestamp);
      const dayNum = d.getDate();
      if (!eventsByDay[dayNum]) eventsByDay[dayNum] = [];
      eventsByDay[dayNum].push({
        prob: (pred.cme_probability * 100).toFixed(1) + '%',
        risk: pred.risk_level || 'NORMAL',
        time: d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        type: pred.risk_level === 'CRITICAL' ? 'major' : pred.risk_level === 'WARNING' ? 'moderate' : 'minor'
      });
    }
  });

  const getDotClass = (type) => {
    if (type === 'major') return 'dot-red';
    if (type === 'moderate') return 'dot-yellow';
    return 'dot-green';
  };

  const selectedEvents = eventsByDay[selectedDay] || [];

  return (
    <div className="card glass-panel flex-1">
      <div className="card-header-flex">
        <div className="flex items-center gap-2">
          <CalendarIcon size={18} color="#38bdf8" />
          <h3 className="card-title-text">Solar Flare Calendar</h3>
        </div>

        <div className="flex items-center gap-2">
          <button className="icon-btn-sm"><ChevronLeft size={16} /></button>
          <span className="font-semibold text-slate-200 text-sm">{currentMonth}</span>
          <button className="icon-btn-sm"><ChevronRight size={16} /></button>
        </div>
      </div>

      {/* Calendar Grid Header Days */}
      <div className="calendar-grid-header">
        {['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'].map((d) => (
          <div key={d} className="calendar-header-day">{d}</div>
        ))}
      </div>

      {/* Calendar Days */}
      <div className="calendar-grid">
        {paddingDays.map((_, idx) => (
          <div key={`pad-${idx}`} className="calendar-day empty"></div>
        ))}

        {daysInMonth.map((day) => {
          const dayEvents = eventsByDay[day] || [];
          const isSelected = selectedDay === day;
          const isToday = day === new Date().getDate();

          return (
            <div
              key={day}
              className={`calendar-day ${isSelected ? 'selected' : ''} ${isToday ? 'today' : ''}`}
              onClick={() => setSelectedDay(day)}
            >
              <span className="day-number">{day}</span>
              {dayEvents.length > 0 && (
                <div className="event-dots">
                  {dayEvents.slice(0, 3).map((ev, i) => (
                    <span key={i} className={`calendar-dot ${getDotClass(ev.type)}`}></span>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Legend Footer */}
      <div className="calendar-legend">
        <div className="legend-item"><span className="calendar-dot dot-green"></span> Normal Risk</div>
        <div className="legend-item"><span className="calendar-dot dot-yellow"></span> Warning Risk</div>
        <div className="legend-item"><span className="calendar-dot dot-red"></span> Critical Risk</div>
      </div>

      {/* Selected Day Event Details */}
      <div className="selected-day-details">
        <div className="selected-day-title">
          <span>Events for Day {selectedDay}</span>
          {selectedEvents.length === 0 && <span className="text-slate-500 text-xs">No recorded flare predictions</span>}
        </div>

        {selectedEvents.map((ev, i) => (
          <div key={i} className="day-event-row">
            <div className="flex items-center gap-2">
              <span className={`calendar-dot ${getDotClass(ev.type)}`}></span>
              <span className="font-mono font-bold text-slate-100">{ev.risk} Risk</span>
              <span className="text-xs text-amber-400 font-mono">({ev.prob})</span>
            </div>
            <span className="text-xs text-slate-400">{ev.time}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
