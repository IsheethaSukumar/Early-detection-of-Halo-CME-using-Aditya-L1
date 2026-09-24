import React from 'react';
import { Info, HelpCircle, Sun, Activity, Zap, Compass, ShieldAlert, Sparkles, BookOpen } from 'lucide-react';

export default function ScientificGuide({ isOpen, onClose }) {
  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-container dark-glass-panel max-w-3xl" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="flex items-center gap-3">
            <div className="modal-icon-badge">
              <BookOpen size={20} color="#38bdf8" />
            </div>
            <div>
              <h2 className="modal-title">Space Weather Parameters & Forecast Methodology</h2>
              <p className="modal-subtitle">Scientific Definitions & Aditya-L1 ML Prediction Model Guide</p>
            </div>
          </div>
          <button className="close-btn" onClick={onClose}>&times;</button>
        </div>

        <div className="modal-body overflow-y-auto max-h-[70vh] flex flex-col gap-4">
          {/* How Predictions Work */}
          <div className="guide-card">
            <h4 className="guide-card-title flex items-center gap-2 text-amber-400">
              <Sparkles size={16} /> How Are Solar Flares Predicted for Future Days?
            </h4>
            <p className="guide-card-text">
              Solar flares are <strong>NOT predicted randomly</strong>. Aditya-L1 instruments (SUIT & VELC) monitor Active Regions (sunspot clusters) on the solar photosphere and corona.
            </p>
            <ul className="guide-list">
              <li>
                <strong>Vector Magnetograms:</strong> Measures magnetic field strength, shear angle, and free magnetic energy stored in active region sunspots.
              </li>
              <li>
                <strong>Machine Learning Pipeline:</strong> Deep Learning models (BiLSTM, XGBoost, Random Forest) analyze real-time magnetic flux growth rates (d&Phi;/dt) to calculate flare emission probabilities over 24-hour, 48-hour, and 72-hour windows.
              </li>
              <li>
                <strong>Active Region Tracking:</strong> Predictions are indexed by target solar Active Regions (e.g. <code>AR 3429</code>, <code>AR 4189</code>) based on magnetic complexity (Hale class &beta;&gamma;&delta;).
              </li>
            </ul>
          </div>

          {/* Solar Flare Classes */}
          <div className="guide-card">
            <h4 className="guide-card-title flex items-center gap-2 text-cyan-400">
              <Sun size={16} /> Solar Flare Intensity Classification (GOES X-Ray Scale)
            </h4>
            <div className="guide-grid">
              <div className="guide-item border-red">
                <span className="font-bold text-rose-400">X-Class (Major)</span>
                <p>Peak flux &gt; 10⁻⁴ W/m². Causes global high-frequency (HF) radio blackouts and long-lasting radiation storms.</p>
              </div>
              <div className="guide-item border-yellow">
                <span className="font-bold text-amber-400">M-Class (Moderate)</span>
                <p>Peak flux 10⁻⁵ to 10⁻⁴ W/m². Causes brief HF radio blackouts over Earth's polar regions.</p>
              </div>
              <div className="guide-item border-green">
                <span className="font-bold text-emerald-400">C-Class (Minor)</span>
                <p>Peak flux 10⁻⁶ to 10⁻⁵ W/m². Small flares with minimal noticeable impact on Earth.</p>
              </div>
            </div>
          </div>

          {/* Telemetry Metrics */}
          <div className="guide-card">
            <h4 className="guide-card-title flex items-center gap-2 text-purple-400">
              <Activity size={16} /> Telemetry Metrics & Units Explained
            </h4>
            <ul className="guide-list">
              <li>
                <strong>Planetary Kp Index (0 to 9):</strong> Quantifies geomagnetic storm intensity caused by solar wind interacting with Earth's magnetosphere (0-3: Quiet, 4-6: Moderate Storm, 7-9: Severe Storm).
              </li>
              <li>
                <strong>Solar Wind Speed (km/s):</strong> Speed of charged plasma particles flowing from Sun to Earth (Nominal: 300-400 km/s, CME arrival: &gt; 700 km/s).
              </li>
              <li>
                <strong>Interplanetary Magnetic Field B<sub>z</sub> (nT):</strong> Southward turning (B<sub>z</sub> &lt; 0) couples solar wind energy directly into Earth's magnetosphere, triggering geomagnetic storms.
              </li>
            </ul>
          </div>
        </div>

        <div className="modal-footer">
          <button className="btn-primary" onClick={onClose}>
            Got It! Return to Dashboard
          </button>
        </div>
      </div>
    </div>
  );
}
