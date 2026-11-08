import React, { useState, useRef, useEffect } from 'react';

export default function C1SpatialConfig({ videoFile, onStartDetection }) {
  const [crowdLimit, setCrowdLimit] = useState(3);
  const [polygonPoints, setPolygonPoints] = useState([]); // [[normX, normY], ...]
  const [canvasDimensions, setCanvasDimensions] = useState({ width: 640, height: 360 });

  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!videoFile) return;

    const videoUrl = URL.createObjectURL(videoFile);
    const video = document.createElement('video');
    video.src = videoUrl;
    video.crossOrigin = 'anonymous';

    video.onloadedmetadata = () => {
      const aspect = video.videoHeight / video.videoWidth || 0.5625;
      const displayWidth = 640;
      const displayHeight = Math.round(displayWidth * aspect);

      setCanvasDimensions({ width: displayWidth, height: displayHeight });
      video.currentTime = 0.5;
    };

    video.onseeked = () => {
      videoRef.current = video;
      drawCanvas();
    };
  }, [videoFile]);

  const drawCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas || !videoRef.current) return;
    const ctx = canvas.getContext('2d');

    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);

    if (polygonPoints.length > 0) {
      ctx.strokeStyle = '#ef4444';
      ctx.fillStyle = 'rgba(239, 68, 68, 0.25)';
      ctx.lineWidth = 3;

      ctx.beginPath();
      polygonPoints.forEach(([normX, normY], idx) => {
        const px = normX * canvas.width;
        const py = normY * canvas.height;
        if (idx === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });

      if (polygonPoints.length > 2) {
        ctx.closePath();
        ctx.fill();
      }
      ctx.stroke();

      polygonPoints.forEach(([normX, normY]) => {
        ctx.fillStyle = '#ef4444';
        ctx.beginPath();
        ctx.arc(normX * canvas.width, normY * canvas.height, 5, 0, 2 * Math.PI);
        ctx.fill();
      });
    }
  };

  useEffect(() => {
    drawCanvas();
  }, [polygonPoints]);

  const handleCanvasClick = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const normX = Number(((e.clientX - rect.left) / rect.width).toFixed(4));
    const normY = Number(((e.clientY - rect.top) / rect.height).toFixed(4));

    setPolygonPoints((prev) => [...prev, [normX, normY]]);
  };

  const handleClear = () => {
    setPolygonPoints([]);
  };

  // Inside src/components/C1SpatialConfig.jsx
const handleStartDetection = async ({ crowdLimit, polygon }) => {
  if (!videoFile) return;

  const params = new URLSearchParams({
    source_id: `upload-${videoFile.name}`,
    filename: videoFile.name,
    crowd_limit: crowdLimit,
    polygon: JSON.stringify(polygon)
  });

  try {
    const res = await fetch(`http://localhost:8000/api/v1/detect/video?${params.toString()}`, {
      method: 'POST'
    });

    const data = await res.json();
    
    // Pass detected events to the parent CheckScreen -> handleFound callback
    if (onFound && data.events) {
      onFound(data.events, { id: `upload-${videoFile.name}`, label: videoFile.name });
    }
  } catch (err) {
    console.error("Detection error:", err);
  }
};

  const handleRun = () => {
    if (polygonPoints.length > 0 && polygonPoints.length < 3) {
      alert('Please select at least 3 points to form a polygon zone.');
      return;
    }

    onStartDetection({
      crowdLimit: Number(crowdLimit),
      polygon: polygonPoints
    });
  };

  return (
    <div style={{ background: '#1e293b', padding: '16px', borderRadius: '8px', color: '#fff', marginBottom: '16px' }}>
      <h3 style={{ marginTop: 0, marginBottom: '12px' }}>C1 Dynamic Spatial Settings</h3>

      {/* 1. Crowd Capacity Input */}
      <div style={{ marginBottom: '16px' }}>
        <label style={{ display: 'block', marginBottom: '4px', fontWeight: '600' }}>
          Crowd Capacity Threshold (Head Count):
        </label>
        <input
          type="number"
          min="1"
          max="50"
          value={crowdLimit}
          onChange={(e) => setCrowdLimit(e.target.value)}
          style={{
            padding: '6px 10px',
            borderRadius: '4px',
            border: '1px solid #475569',
            background: '#0f172a',
            color: '#fff',
            width: '100px'
          }}
        />
        <span style={{ marginLeft: '10px', fontSize: '0.85rem', color: '#94a3b8' }}>
          Alerts if headcount &gt; {crowdLimit}
        </span>
      </div>

      {/* 2. Dynamic Polygon Drawing */}
      <div style={{ marginBottom: '16px' }}>
        <label style={{ display: 'block', marginBottom: '4px', fontWeight: '600' }}>
          Click on video frame below to draw Restricted Area:
        </label>
        <canvas
          ref={canvasRef}
          width={canvasDimensions.width}
          height={canvasDimensions.height}
          onClick={handleCanvasClick}
          style={{
            border: '2px dashed #64748b',
            borderRadius: '6px',
            cursor: 'crosshair',
            display: 'block'
          }}
        />
        <div style={{ marginTop: '8px', display: 'flex', gap: '10px', alignItems: 'center' }}>
          <button
            onClick={handleClear}
            style={{
              padding: '4px 10px',
              background: '#475569',
              color: '#fff',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            Clear Zone ({polygonPoints.length} points)
          </button>
        </div>
      </div>

      {/* 3. Execute Detection */}
      <button
        onClick={handleRun}
        style={{
          padding: '8px 16px',
          background: '#2563eb',
          color: '#fff',
          fontWeight: 'bold',
          border: 'none',
          borderRadius: '4px',
          cursor: 'pointer'
        }}
      >
        Run Detection Engine
      </button>
    </div>
  );
}