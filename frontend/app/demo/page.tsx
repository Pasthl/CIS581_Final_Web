"use client";
import React, { useRef, useState } from "react";
import Link from "next/link";

export default function DemoPage() {
  const fileRef = useRef<HTMLInputElement>(null);
  const [origUrl, setOrigUrl] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string>("");

  function onSelectFile() {
    const f = fileRef.current?.files?.[0];
    if (!f) return;
    setFileName(f.name);
    // 仅本地预览，不上传、不保存
    const url = URL.createObjectURL(f);
    setOrigUrl(url);
  }

  function clearSelection() {
    if (origUrl) URL.revokeObjectURL(origUrl);
    setOrigUrl(null);
    setFileName("");
    if (fileRef.current) fileRef.current.value = "";
  }

  return (
    <>
      <nav className="nav">
        <div className="nav-inner">
          <div className="brand">Denoise Web</div>
          <Link className="link" href="/">Home</Link>
        </div>
      </nav>

      <main className="container">
        <div className="card" style={{marginBottom: 16}}>
          <h1>Demo：上传并预览原图</h1>
          <h2>当前仅做前端本地预览，不进行任何保存与推理</h2>

          <div style={{display: "flex", gap: 12, alignItems: "center", marginTop: 10, flexWrap: "wrap"}}>
            <label className="input">
              <input
                ref={fileRef}
                type="file"
                accept="image/*"
                onChange={onSelectFile}
                style={{ display: "block" }}
              />
            </label>
            <button className="btn" onClick={clearSelection} disabled={!origUrl} style={{opacity: origUrl ? 1 : 0.6}}>
              清空
            </button>
            {fileName && <span className="label">已选择：{fileName}</span>}
          </div>
        </div>

        <div className="preview">
          <div className="card imgbox">
            <div className="label">原始图片预览</div>
            {origUrl ? (
              <img src={origUrl} alt="original preview" />
            ) : (
              <div className="label">尚未选择图片</div>
            )}
          </div>

          <div className="card imgbox">
            <div className="label">（留空位：未来用于展示去噪结果）</div>
            <div style={{height: 120, border: "1px dashed var(--border)", borderRadius: 10, background: "#fbfffd"}} />
          </div>
        </div>
      </main>
    </>
  );
}
