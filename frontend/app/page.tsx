import Link from "next/link";

export default function Home() {
  return (
    <>
      <nav className="nav">
        <div className="nav-inner">
          <div className="brand">Pixel Revival</div>
          <Link className="link" href="/demo">Demo</Link>
        </div>
      </nav>

      <main className="container">
        <div className="hero-card">
          <h1 className="hero-title">Pixel Revival</h1>
          <p className="hero-subtitle">
            Transform noisy images into crystal-clear masterpieces with integrated denoising technology
          </p>
          <Link className="demo-button" href="/demo">
            <span className="button-text">Experience the Magic</span>
            <span className="button-arrow">→</span>
          </Link>
        </div>

        <div className="pipeline-card">
          <h2 className="pipeline-title">How It Works</h2>
          <div className="pipeline-steps">
            <div className="pipeline-step step-1">
              <div className="step-number">1</div>
              <div className="step-content">
                <h3>Upload</h3>
                <p>Select your noisy or low-resolution image</p>
              </div>
            </div>
            <div className="pipeline-step step-2">
              <div className="step-number">2</div>
              <div className="step-content">
                <h3>Denoise</h3>
                <p>Advanced algorithms remove noise while preserving details</p>
              </div>
            </div>
            <div className="pipeline-step step-3">
              <div className="step-number">3</div>
              <div className="step-content">
                <h3>Enhance</h3>
                <p>Super-resolution models upscale and sharpen your image</p>
              </div>
            </div>
          </div>
        </div>

        <div className="author-card">
          <div className="author-label">Created by</div>
          <div className="author-names">
            <span className="author-name">Ann Hua</span>
            <span className="author-separator">·</span>
            <span className="author-name">Livia Yuan</span>
          </div>
        </div>
      </main>
    </>
  );
}
