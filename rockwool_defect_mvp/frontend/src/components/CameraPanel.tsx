"use client";
import { useEffect, useRef, useState } from "react";

export function CameraPanel({ onCapture, busy }: { onCapture: (blob: Blob, filename: string) => Promise<void>; busy: boolean }) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [err, setErr] = useState<string | null>(null);

  async function start() {
    setErr(null);
    try {
      const s = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", width: { ideal: 1920 } }, audio: false });
      setStream(s);
      if (videoRef.current) { videoRef.current.srcObject = s; await videoRef.current.play(); }
    } catch (e: unknown) {
      setErr(e instanceof Error ? e.message : "kamera erişilemiyor");
    }
  }
  function stop() { stream?.getTracks().forEach((t) => t.stop()); setStream(null); }
  useEffect(() => () => stop(), []); // eslint-disable-line react-hooks/exhaustive-deps

  async function snap() {
    const v = videoRef.current; if (!v || !stream) return;
    const c = document.createElement("canvas");
    c.width = v.videoWidth; c.height = v.videoHeight;
    c.getContext("2d")!.drawImage(v, 0, 0);
    const blob = await new Promise<Blob | null>((res) => c.toBlob(res, "image/jpeg", 0.9));
    if (!blob) return;
    await onCapture(blob, "kamera.jpg");
  }

  return (
    <div className="p-3">
      <div className="aspect-video bg-black rounded-lg overflow-hidden mb-3 relative">
        <video ref={videoRef} className="w-full h-full object-contain" muted playsInline />
        {!stream && <div className="absolute inset-0 flex items-center justify-center text-white/70 text-sm">Kamera kapalı</div>}
      </div>
      <div className="flex flex-wrap gap-2 items-center">
        {!stream ? (
          <button onClick={start} className="btn btn-primary">Kamerayı Aç</button>
        ) : (
          <>
            <button onClick={snap} className="btn btn-primary" disabled={busy}>Çek ve Analiz Et</button>
            <button onClick={stop} className="btn btn-outline">Kapat</button>
          </>
        )}
        {err && <span className="text-sm text-[var(--mega-red)]">Hata: {err}</span>}
      </div>
    </div>
  );
}
