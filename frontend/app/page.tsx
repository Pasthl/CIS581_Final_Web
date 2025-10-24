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
