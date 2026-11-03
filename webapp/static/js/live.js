// Prediksi real-time: kirim 1 frame/detik ke /api/predict.
(() => {
  const video = $("video"), live = $("live"), startBtn = $("start"), stopBtn = $("stop"), off = $("camoff");
  const saveBtn = $("save");
  let stream = null, timer = null, busy = false;
  const canvas = document.createElement("canvas");

  const show = (html, color) => { live.style.setProperty("--cc", color || "var(--line)"); live.innerHTML = html; };

  async function tick(save) {
    if (busy || !video.videoWidth) return;
    busy = true;
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    try {
      const d = await postJSON("/api/predict", {image: canvas.toDataURL("image/jpeg", 0.8), save: save === true});
      if (d.saved) toast("Hasil disimpan ke riwayat.");
      if (!d.face_found) show('<div class="muted">Wajah tidak terdeteksi.</div>');
      else show(`<div class="big">${CLASS_NAMES[d.label]}</div><div class="muted">Keyakinan ${(d.confidence * 100).toFixed(1)}%</div>`,
                `var(--c-${d.label})`);
    } catch (e) { show(`<div class="muted">${esc(e.message)}</div>`); }
    busy = false;
  }

  if (saveBtn) saveBtn.onclick = () => (busy ? toast("Tunggu sebentar, lalu coba lagi.") : tick(true));

  startBtn.onclick = async () => {
    try { stream = await navigator.mediaDevices.getUserMedia({video: {width: 640, height: 480}, audio: false}); }
    catch (e) { toast("Kamera tidak dapat diakses: " + e.message); return; }
    video.srcObject = stream; off.hidden = true;
    timer = setInterval(tick, 1000);
    if (saveBtn) saveBtn.disabled = false;
    startBtn.disabled = true; stopBtn.disabled = false;
    show('<div class="muted">Menganalisis…</div>');
  };
  stopBtn.onclick = () => {
    clearInterval(timer);
    if (stream) stream.getTracks().forEach((t) => t.stop());
    video.srcObject = null; off.hidden = false;
    startBtn.disabled = false; stopBtn.disabled = true;
    if (saveBtn) saveBtn.disabled = true;
    show('<div class="muted">Kamera dihentikan.</div>');
  };
})();
