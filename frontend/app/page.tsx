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
            Transform noisy images into crystal-clear masterpieces with AI-powered denoising technology
          </p>
          <Link className="demo-button" href="/demo">
            <span className="button-text">Experience the Magic</span>
            <span className="button-arrow">→</span>
          </Link>
        </div>
      </main>
    </>
  );
}
