/**
 * The "Check" section: the home choice, recorded video, and live camera.
 *
 * Recorded and live stay mounted once opened, so switching between them — or
 * going to Alerts and back — does not lose a finished check or stop a camera
 * that is watching.
 */
import React, { useState } from 'react';
import type { IncidentEvent } from '../../types/events';
// @ts-expect-error C1SpatialConfig is a JavaScript component without a declaration file.
import C1SpatialConfig from '../C1SpatialConfig';


export type CheckMode = 'home' | 'file' | 'live';

interface CheckScreenProps {
  mode: CheckMode;
  onModeChange: (mode: CheckMode) => void;
  onFound: (events: IncidentEvent[], source: { id: string; label: string }) => void;
  onOpenAlert: (eventId: string) => void;
  onLiveChange: (live: boolean) => void;
  needLook: number;
  onOpenAlerts: () => void;
}

export function CheckScreen({
  onFound,
  onOpenAlerts,
}: CheckScreenProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
      setError(null);
    }
  };

  const handleResetFile = () => {
    setSelectedFile(null);
    setError(null);
  };

  const handleStartDetection = async ({
    crowdLimit,
    polygon,
  }: {
    crowdLimit: number;
    polygon: number[][];
  }) => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);

    const cleanName = selectedFile.name.replace(/[^a-zA-Z0-9._-]/g, '_');
    const sourceId = `upload-${cleanName}`;

    const params = new URLSearchParams({
      source_id: sourceId,
      filename: selectedFile.name,
      crowd_limit: crowdLimit.toString(),
      polygon: JSON.stringify(polygon),
    });

    try {
      const response = await fetch(
        `http://localhost:8000/api/v1/detect/video?${params.toString()}`,
        {
          method: 'POST',
          body: selectedFile,
        }
      );

      if (!response.ok) {
        throw new Error(`Detection request failed with status ${response.status}`);
      }

      const data = await response.json();
      const events: IncidentEvent[] = data.events || [];

      // Pass detected events to top-level App state
      onFound(events, { id: sourceId, label: selectedFile.name });

      if (events.length > 0) {
        onOpenAlerts();
      }
    } catch (err: any) {
      setError(err.message || 'Failed to analyze video.');
    } finally {
      setLoading(false);
    }

    try {
      const response = await fetch(
        `http://localhost:8000/api/v1/detect/video?${params.toString()}`,
        {
          method: 'POST',
          body: selectedFile,
        }
      );

      if (!response.ok) {
        throw new Error(`Detection request failed with status ${response.status}`);
      }

      const data = await response.json();
      const events: IncidentEvent[] = data.events || [];

      // Create a direct video URL for local browser playback
      const localVideoUrl = URL.createObjectURL(selectedFile);

      // Pass events and media_ref to dashboard
      interface CheckScreenProps {
        mode: CheckMode;
        onModeChange: (mode: CheckMode) => void;
        onFound: (
          events: IncidentEvent[],
          source: { id: string; label: string; media_ref?: string | null }
        ) => void;
        onOpenAlert: (eventId: string) => void;
        onLiveChange: (live: boolean) => void;
        needLook: number;
        onOpenAlerts: () => void;
      }

      // Navigate to alerts screen
      onOpenAlerts();
    } catch (err: any) {
      setError(err.message || 'Failed to analyze video.');
    } finally {
      setLoading(false);
    }

  };


  return (
    <div style={{ padding: '24px', maxWidth: '850px', margin: '0 auto' }}>
      {!selectedFile ? (
        <div
          style={{
            border: '2px dashed #475569',
            borderRadius: '12px',
            padding: '48px 24px',
            textAlign: 'center',
            background: '#0f172a',
            color: '#94a3b8',
          }}
        >
          <h2 style={{ color: '#fff', marginTop: 0, marginBottom: '8px' }}>
            Check CCTV Video
          </h2>
          <p style={{ marginBottom: '20px' }}>
            Upload a recorded video to set custom restricted zones and headcount capacity limits.
          </p>
          <input
            type="file"
            accept="video/*"
            onChange={handleFileChange}
            style={{ display: 'none' }}
            id="video-file-input"
          />
          <label
            htmlFor="video-file-input"
            style={{
              display: 'inline-block',
              padding: '12px 24px',
              background: '#2563eb',
              color: '#fff',
              fontWeight: 'bold',
              borderRadius: '6px',
              cursor: 'pointer',
            }}
          >
            Select Recorded Video
          </label>
        </div>
      ) : (
        <div>
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '16px',
              background: '#1e293b',
              padding: '12px 20px',
              borderRadius: '8px',
            }}
          >
            <span style={{ color: '#fff', fontWeight: 'bold' }}>
              File: {selectedFile.name}
            </span>
            <button
              onClick={handleResetFile}
              disabled={loading}
              style={{
                background: '#475569',
                color: '#fff',
                border: 'none',
                padding: '6px 14px',
                borderRadius: '4px',
                cursor: 'pointer',
              }}
            >
              Choose Different Video
            </button>
          </div>

          {loading ? (
            <div
              style={{
                padding: '40px',
                textAlign: 'center',
                background: '#1e293b',
                borderRadius: '8px',
                color: '#60a5fa',
              }}
            >
              <h3 style={{ margin: 0 }}>Running YOLOv8 & Spatial Anomaly Rules...</h3>
              <p style={{ color: '#94a3b8', marginTop: '8px' }}>
                Evaluating headcount and restricted polygon intrusion.
              </p>
            </div>
          ) : (
            <C1SpatialConfig
              videoFile={selectedFile}
              onStartDetection={handleStartDetection}
            />
          )}

          {error && (
            <div
              style={{
                marginTop: '16px',
                padding: '14px',
                background: '#7f1d1d',
                color: '#fecaca',
                borderRadius: '6px',
              }}
            >
              {error}
            </div>
          )}
        </div>
      )}
    </div>
  );
}