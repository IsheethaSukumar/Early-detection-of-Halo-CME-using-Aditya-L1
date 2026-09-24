import React from 'react';

export default function BackgroundSpace() {
  return (
    <div className="background-space-container">
      {/* Real Earth Space Background Image */}
      <div className="background-image-layer" style={{ backgroundImage: "url('/earth_space_bg.jpg')" }} />

      {/* Dark Navy Atmospheric Gradient Overlay for Glass Visibility */}
      <div className="background-overlay-layer" />
    </div>
  );
}
