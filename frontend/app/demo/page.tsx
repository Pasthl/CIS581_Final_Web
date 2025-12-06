"use client";
import React, { useRef, useState } from "react";
import Link from "next/link";

const API_BASE_URL = "http://localhost:5000";

export default function DemoPage() {
  const fileRef = useRef<HTMLInputElement>(null);
  const [origUrl, setOrigUrl] = useState<string | null>(null);
  const [groundTruthUrl, setGroundTruthUrl] = useState<string | null>(null);
  const [degradedUrl, setDegradedUrl] = useState<string | null>(null);
  const [preprocessedUrl, setPreprocessedUrl] = useState<string | null>(null);
  const [deblurredUrl, setDeblurredUrl] = useState<string | null>(null);
  const [edsrUrl, setEdsrUrl] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string>("");
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [processingTime, setProcessingTime] = useState<string | null>(null);
  const [lightboxImage, setLightboxImage] = useState<{ url: string; title: string } | null>(null);
  const [isZoomed, setIsZoomed] = useState<boolean>(false);

  // Pipeline step toggles
  const [enablePreprocess, setEnablePreprocess] = useState<boolean>(true);
  const [enableDeblur, setEnableDeblur] = useState<boolean>(true);
  const [enableEDSR, setEnableEDSR] = useState<boolean>(true);
  const [enableFaceEnhance, setEnableFaceEnhance] = useState<boolean>(false);
  const [evaluationMode, setEvaluationMode] = useState<boolean>(false);
  const [degradationType, setDegradationType] = useState<string>('light');

  // Metrics data
  type MetricsData = {
    psnr: number;
    ssim: number;
    mse: number;
    mae: number;
  };
  const [metrics, setMetrics] = useState<{
    preprocessed?: MetricsData;
    deblurred?: MetricsData;
    edsr?: MetricsData;
  } | null>(null);

  async function onSelectFile() {
    const f = fileRef.current?.files?.[0];
    if (!f) return;

    setFileName(f.name);
    setError(null);
    setProcessingTime(null);
    setMetrics(null);

    const url = URL.createObjectURL(f);
    setOrigUrl(url);

    await processImage(f);
  }

  async function processImage(file: File) {
    setIsProcessing(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("image", file);
      formData.append("enable_preprocess", enablePreprocess.toString());
      formData.append("enable_deblur", enableDeblur.toString());
      formData.append("enable_edsr", enableEDSR.toString());
      formData.append("enable_face_enhance", enableFaceEnhance.toString());
      formData.append("evaluation_mode", evaluationMode.toString());
      formData.append("degradation_type", degradationType);

      const response = await fetch(`${API_BASE_URL}/api/pipeline`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();

      if (data.success) {
        setGroundTruthUrl(data.ground_truth || null);
        setDegradedUrl(data.degraded || null);
        setPreprocessedUrl(data.preprocessed || null);
        setDeblurredUrl(data.deblurred || null);
        setEdsrUrl(data.edsr || null);
        setProcessingTime(data.processing_time);
        setMetrics(data.metrics || null);
      } else {
        throw new Error(data.error || "Processing failed");
      }
    } catch (err) {
      console.error("Error processing image:", err);
      setError(err instanceof Error ? err.message : "Failed to process image. Make sure the backend server is running.");
      setGroundTruthUrl(null);
      setDegradedUrl(null);
      setPreprocessedUrl(null);
      setDeblurredUrl(null);
      setEdsrUrl(null);
      setMetrics(null);
    } finally {
      setIsProcessing(false);
    }
  }

  function clearSelection() {
    if (origUrl) URL.revokeObjectURL(origUrl);
    setOrigUrl(null);
    setGroundTruthUrl(null);
    setDegradedUrl(null);
    setPreprocessedUrl(null);
    setDeblurredUrl(null);
    setEdsrUrl(null);
    setFileName("");
    setError(null);
    setProcessingTime(null);
    setMetrics(null);
    if (fileRef.current) fileRef.current.value = "";
  }

  function openLightbox(url: string, title: string) {
    setLightboxImage({ url, title });
    setIsZoomed(false);
  }

  function closeLightbox() {
    setLightboxImage(null);
    setIsZoomed(false);
  }

  function toggleZoom(e: React.MouseEvent) {
    e.stopPropagation();
    setIsZoomed(!isZoomed);
  }

  const ImageCard = ({ url, title, badge }: { url: string | null; title: string; badge: string }) => (
    <div className="preview-card">
      <div className="preview-header">
        <span className="preview-badge">{badge}</span>
      </div>
      {url ? (
        <div className="image-container">
          <img
            src={url}
            alt={title}
            className="preview-image"
            onClick={() => openLightbox(url, title)}
            style={{ cursor: 'zoom-in' }}
            title="Click to enlarge"
          />
        </div>
      ) : (
        <div className="empty-state">
          {isProcessing ? (
            <>
              <div className="empty-icon loading">⚙️</div>
              <div className="empty-text">Processing...</div>
            </>
          ) : (
            <>
              <div className="empty-icon">✨</div>
              <div className="empty-text">{title}</div>
            </>
          )}
        </div>
      )}
    </div>
  );

  return (
    <>
      <nav className="nav">
        <div className="nav-inner">
          <div className="brand">Pixel Revival</div>
          <Link className="link" href="/">Home</Link>
        </div>
      </nav>

      <main className="container">
        <div className="demo-header-card">
          <h1 className="demo-title">Image Processing Pipeline</h1>
          <p className="demo-description">
            Upload an image to see the full processing pipeline: Preprocessing → Deblur → Super-Resolution
          </p>

          <div className="demo-controls">
            <label className="file-upload-label">
              <input
                ref={fileRef}
                type="file"
                accept="image/*"
                onChange={onSelectFile}
                className="file-input"
                disabled={isProcessing}
              />
              <span className="upload-icon">📁</span>
              <span>{isProcessing ? "Processing..." : "Choose Image"}</span>
            </label>
            <button
              className="clear-btn"
              onClick={clearSelection}
              disabled={!origUrl || isProcessing}
            >
              <span className="clear-icon">✕</span>
              Clear
            </button>
            {fileName && (
              <div className="file-info">
                <span className="file-icon">🖼️</span>
                <span className="file-name">{fileName}</span>
              </div>
            )}
          </div>

          <div className="pipeline-toggles">
            <label className="toggle-label">
              <input
                type="checkbox"
                checked={enablePreprocess}
                onChange={(e) => setEnablePreprocess(e.target.checked)}
                disabled={isProcessing}
              />
              <span>Preprocessing</span>
            </label>
            <label className="toggle-label">
              <input
                type="checkbox"
                checked={enableDeblur}
                onChange={(e) => setEnableDeblur(e.target.checked)}
                disabled={isProcessing}
              />
              <span>Real-ESRGAN (4x)</span>
            </label>
            <label className="toggle-label">
              <input
                type="checkbox"
                checked={enableFaceEnhance}
                onChange={(e) => setEnableFaceEnhance(e.target.checked)}
                disabled={isProcessing || !enableDeblur}
              />
              <span>Face Enhance (GFPGAN)</span>
            </label>
            <label className="toggle-label">
              <input
                type="checkbox"
                checked={enableEDSR}
                onChange={(e) => setEnableEDSR(e.target.checked)}
                disabled={isProcessing}
              />
              <span>EDSR (4x)</span>
            </label>
            <label className="toggle-label">
              <input
                type="checkbox"
                checked={evaluationMode}
                onChange={(e) => setEvaluationMode(e.target.checked)}
                disabled={isProcessing}
              />
              <span>🔬 Evaluation Mode</span>
            </label>
          </div>

          {evaluationMode && (
            <>
              <div className="evaluation-notice">
                <span className="notice-icon">ℹ️</span>
                <div className="notice-text">
                  <strong>Evaluation Mode:</strong> Your high-quality image will be automatically degraded,
                  then processed through the pipeline. Quality metrics will compare the restored image against your original.
                </div>
              </div>

              <div className="degradation-selector">
                <label className="selector-label">
                  <span className="selector-title">Degradation Level:</span>
                  <select
                    value={degradationType}
                    onChange={(e) => setDegradationType(e.target.value)}
                    disabled={isProcessing}
                    className="degradation-select"
                  >
                    <option value="light">Light (Blur + Noise, Same Resolution)</option>
                    <option value="medium">Medium (Blur + Noise + Compression + 2x Downscale)</option>
                    <option value="heavy">Heavy (Blur + Noise + Compression + 4x Downscale)</option>
                  </select>
                </label>
              </div>
            </>
          )}

          {isProcessing && (
            <div className="status-message processing">
              <span className="status-icon">⚙️</span>
              Processing pipeline...
            </div>
          )}

          {error && (
            <div className="status-message error">
              <span className="status-icon">⚠️</span>
              {error}
            </div>
          )}

          {processingTime && !isProcessing && (
            <div className="status-message success">
              <span className="status-icon">✓</span>
              Pipeline completed in {processingTime}
            </div>
          )}
        </div>

        {evaluationMode && groundTruthUrl && (
          <div className="evaluation-images">
            <h3 className="section-title">📸 Evaluation Images</h3>
            <div className="eval-grid">
              <ImageCard url={groundTruthUrl} title="Ground Truth (Original)" badge="Ground Truth" />
              <ImageCard url={degradedUrl} title="Degraded (Input)" badge="Degraded Input" />
            </div>
          </div>
        )}

        <div className="preview-grid">
          {!evaluationMode && <ImageCard url={origUrl} title="Original" badge="1. Original" />}
          <ImageCard url={preprocessedUrl} title="Preprocessed" badge={evaluationMode ? "1. Preprocessed" : "2. Preprocessed"} />
          <ImageCard url={deblurredUrl} title="Real-ESRGAN (4x)" badge={evaluationMode ? "2. Real-ESRGAN" : "3. Real-ESRGAN"} />
          <ImageCard url={edsrUrl} title="Super-Resolution (4x)" badge={evaluationMode ? "3. EDSR (4x)" : "4. EDSR (4x)"} />
        </div>

        {/* Quality Metrics Display */}
        {evaluationMode && metrics && (
          <div className="metrics-container">
            <h2 className="metrics-title">📊 Quality Metrics (vs. Ground Truth)</h2>
            <p className="metrics-description">
              Metrics compare restored images against the original high-quality image. Higher PSNR and SSIM values indicate better quality.
            </p>
            <div className="metrics-grid">
              {metrics.preprocessed && (
                <div className="metric-card">
                  <h3 className="metric-step">Preprocessing</h3>
                  <div className="metric-values">
                    <div className="metric-item">
                      <span className="metric-label">PSNR</span>
                      <span className="metric-value">{metrics.preprocessed.psnr.toFixed(2)} dB</span>
                    </div>
                    <div className="metric-item">
                      <span className="metric-label">SSIM</span>
                      <span className="metric-value">{metrics.preprocessed.ssim.toFixed(4)}</span>
                    </div>
                    <div className="metric-item">
                      <span className="metric-label">MSE</span>
                      <span className="metric-value">{metrics.preprocessed.mse.toFixed(2)}</span>
                    </div>
                    <div className="metric-item">
                      <span className="metric-label">MAE</span>
                      <span className="metric-value">{metrics.preprocessed.mae.toFixed(2)}</span>
                    </div>
                  </div>
                </div>
              )}

              {metrics.deblurred && (
                <div className="metric-card">
                  <h3 className="metric-step">Real-ESRGAN</h3>
                  <div className="metric-values">
                    <div className="metric-item">
                      <span className="metric-label">PSNR</span>
                      <span className="metric-value">{metrics.deblurred.psnr.toFixed(2)} dB</span>
                    </div>
                    <div className="metric-item">
                      <span className="metric-label">SSIM</span>
                      <span className="metric-value">{metrics.deblurred.ssim.toFixed(4)}</span>
                    </div>
                    <div className="metric-item">
                      <span className="metric-label">MSE</span>
                      <span className="metric-value">{metrics.deblurred.mse.toFixed(2)}</span>
                    </div>
                    <div className="metric-item">
                      <span className="metric-label">MAE</span>
                      <span className="metric-value">{metrics.deblurred.mae.toFixed(2)}</span>
                    </div>
                  </div>
                </div>
              )}

              {metrics.edsr && (
                <div className="metric-card">
                  <h3 className="metric-step">EDSR</h3>
                  <div className="metric-values">
                    <div className="metric-item">
                      <span className="metric-label">PSNR</span>
                      <span className="metric-value">{metrics.edsr.psnr.toFixed(2)} dB</span>
                    </div>
                    <div className="metric-item">
                      <span className="metric-label">SSIM</span>
                      <span className="metric-value">{metrics.edsr.ssim.toFixed(4)}</span>
                    </div>
                    <div className="metric-item">
                      <span className="metric-label">MSE</span>
                      <span className="metric-value">{metrics.edsr.mse.toFixed(2)}</span>
                    </div>
                    <div className="metric-item">
                      <span className="metric-label">MAE</span>
                      <span className="metric-value">{metrics.edsr.mae.toFixed(2)}</span>
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div className="metrics-legend">
              <h4>Metrics Guide:</h4>
              <ul>
                <li><strong>PSNR</strong>: Peak Signal-to-Noise Ratio</li>
                <li><strong>SSIM</strong>: Structural Similarity Index</li>
                <li><strong>MSE</strong>: Mean Squared Error</li>
                <li><strong>MAE</strong>: Mean Absolute Error</li>
              </ul>
            </div>
          </div>
        )}
      </main>

      {lightboxImage && (
        <div
          onClick={closeLightbox}
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            width: '100vw',
            height: '100vh',
            backgroundColor: 'rgba(0, 0, 0, 0.95)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 9999,
            cursor: 'zoom-out',
            overflow: isZoomed ? 'auto' : 'hidden'
          }}
        >
          <button
            onClick={closeLightbox}
            style={{
              position: 'fixed',
              top: '20px',
              right: '20px',
              background: 'rgba(255, 255, 255, 0.15)',
              border: 'none',
              color: 'white',
              fontSize: '32px',
              width: '50px',
              height: '50px',
              borderRadius: '50%',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transition: 'background 0.2s',
              zIndex: 10001,
              fontWeight: 'normal',
              lineHeight: '1'
            }}
            onMouseEnter={(e) => (e.currentTarget.style.background = 'rgba(255, 255, 255, 0.3)')}
            onMouseLeave={(e) => (e.currentTarget.style.background = 'rgba(255, 255, 255, 0.15)')}
          >
            ×
          </button>
          <div
            style={{
              maxWidth: isZoomed ? 'none' : '90vw',
              maxHeight: isZoomed ? 'none' : '90vh',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: '20px',
              padding: '20px'
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <img
              src={lightboxImage.url}
              alt={lightboxImage.title}
              onClick={toggleZoom}
              style={{
                maxWidth: isZoomed ? 'none' : '100%',
                maxHeight: isZoomed ? 'none' : '80vh',
                objectFit: 'contain',
                borderRadius: '8px',
                boxShadow: '0 0 50px rgba(0, 0, 0, 0.8)',
                cursor: isZoomed ? 'zoom-out' : 'zoom-in',
                transition: 'transform 0.2s ease'
              }}
              title={isZoomed ? 'Click to zoom out' : 'Click to zoom in'}
            />
            <div
              style={{
                color: 'white',
                fontSize: '18px',
                fontWeight: '600',
                textAlign: 'center',
                textShadow: '0 2px 8px rgba(0, 0, 0, 0.8)'
              }}
            >
              {lightboxImage.title} {isZoomed && '(100% size)'}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
