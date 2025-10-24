"use client";
import React, { useRef, useState } from "react";
import Link from "next/link";

const API_BASE_URL = "http://localhost:5000";

export default function DemoPage() {
  const fileRef = useRef<HTMLInputElement>(null);
  const [origUrl, setOrigUrl] = useState<string | null>(null);
  const [denoisedUrl, setDenoisedUrl] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string>("");
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [processingTime, setProcessingTime] = useState<string | null>(null);
  const [lightboxImage, setLightboxImage] = useState<{ url: string; title: string } | null>(null);
  const [isZoomed, setIsZoomed] = useState<boolean>(false);

  async function onSelectFile() {
    const f = fileRef.current?.files?.[0];
    if (!f) return;

    setFileName(f.name);
    setError(null);
    setProcessingTime(null);

    // Show local preview
    const url = URL.createObjectURL(f);
    setOrigUrl(url);

    // Upload and process with backend
    await processImage(f);
  }

  async function processImage(file: File) {
    setIsProcessing(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("image", file);

      const response = await fetch(`${API_BASE_URL}/api/denoise`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();

      if (data.success) {
        // Use base64 image data directly (no file storage needed)
        setDenoisedUrl(data.output.image_data);
        setProcessingTime(data.processing_time);
      } else {
        throw new Error(data.error || "Processing failed");
      }
    } catch (err) {
      console.error("Error processing image:", err);
      setError(err instanceof Error ? err.message : "Failed to process image. Make sure the backend server is running.");
      setDenoisedUrl(null);
    } finally {
      setIsProcessing(false);
    }
  }

  function clearSelection() {
    if (origUrl) URL.revokeObjectURL(origUrl);
    setOrigUrl(null);
    setDenoisedUrl(null);
    setFileName("");
    setError(null);
    setProcessingTime(null);
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
          <h1 className="demo-title">Image Denoising Demo</h1>
          <p className="demo-description">
            Upload an image to experience super-resolution with EDSR.
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

          {/* Status messages */}
          {isProcessing && (
            <div className="status-message processing">
              <span className="status-icon">⚙️</span>
              Processing image with EDSR model...
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
              Image processed in {processingTime}
            </div>
          )}
        </div>

        <div className="preview">
          <div className="preview-card">
            <div className="preview-header">
              <span className="preview-badge">Original</span>
            </div>
            {origUrl ? (
              <div className="image-container">
                <img
                  src={origUrl}
                  alt="original preview"
                  className="preview-image"
                  onClick={() => openLightbox(origUrl, "Original Image")}
                  style={{ cursor: 'zoom-in' }}
                  title="Click to enlarge"
                />
              </div>
            ) : (
              <div className="empty-state">
                <div className="empty-icon">🖼️</div>
                <div className="empty-text">No image selected</div>
                <div className="empty-hint">Upload an image to get started</div>
              </div>
            )}
          </div>

          <div className="preview-card">
            <div className="preview-header">
              <span className="preview-badge denoised">Super-Resolution (4x)</span>
              {isProcessing && <span className="processing-badge">Processing...</span>}
            </div>
            {denoisedUrl ? (
              <div className="image-container">
                <img
                  src={denoisedUrl}
                  alt="denoised result"
                  className="preview-image"
                  onClick={() => openLightbox(denoisedUrl, "Super-Resolution Image (4x)")}
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
                    <div className="empty-hint">Running EDSR model</div>
                  </>
                ) : (
                  <>
                    <div className="empty-icon">✨</div>
                    <div className="empty-text">Result</div>
                    <div className="empty-hint">Upload an image to see the magic</div>
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Lightbox Modal */}
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
                width: isZoomed ? 'auto' : 'auto',
                height: isZoomed ? 'auto' : 'auto',
                objectFit: 'contain',
                borderRadius: '8px',
                boxShadow: '0 0 50px rgba(0, 0, 0, 0.8)',
                cursor: isZoomed ? 'zoom-out' : 'zoom-in',
                transition: 'transform 0.2s ease'
              }}
              title={isZoomed ? 'Click to zoom out' : 'Click to zoom in to 100%'}
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
              {lightboxImage.title} {isZoomed && '(100% size - scroll to view)'}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
