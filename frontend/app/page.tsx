import Link from "next/link";

export default function Home() {
  return (
    <>
      <nav className="nav">
        <div className="nav-inner">
          <div className="brand">Denoise Web</div>
          <Link className="link" href="/demo">Demo</Link>
        </div>
      </nav>

      <main className="container">
        <div className="card">
          <h1>图像去噪 · 最小演示</h1>
          <h2>前端测试页（仅上传并预览原图，不做保存与推理）</h2>
          <p style={{marginTop: 6}}>
            点击下方按钮进入 Demo 页面，上传一张图片，马上在本地预览。之后再逐步接入后端与模型推理。
          </p>
          <div style={{marginTop: 14}}>
            <Link className="btn" href="/demo">进入 Demo</Link>
          </div>
        </div>
      </main>
    </>
  );
}
