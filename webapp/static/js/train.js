// Halaman Latih: memulai job dan memantau status.
(() => {
  const ORDER = ["preprocess", "train", "evaluate"];
  const start = $("start"), panel = $("panel"), logEl = $("log"), bar = $("bar"), done = $("done");
  let timer = null, lastLen = -1;

  function paint(s) {
    if (s.status === "idle") return;
    panel.hidden = false;
    bar.style.width = s.progress + "%";
    const idx = s.stage === "done" ? ORDER.length : ORDER.indexOf(s.stage);
    document.querySelectorAll(".stage").forEach((el) => {
      const i = ORDER.indexOf(el.dataset.s);
      el.className = "stage" + (i < idx ? " done" : i === idx && s.status === "running" ? " on" : "");
    });
    if (s.log.length !== lastLen) {
      lastLen = s.log.length;
      logEl.innerHTML = s.log.map((l) => `<div class="${/^Gagal|^Kesalahan/.test(l) ? "err" : ""}">${esc(l)}</div>`).join("");
      logEl.scrollTop = logEl.scrollHeight;
    }
    const running = s.status === "running";
    start.disabled = running; $("state").textContent = running ? "Sedang berjalan…" : "";
    if (s.status === "done") finish();
    if (s.status === "error") { stop(); done.hidden = true; toast("Pelatihan gagal. Lihat log."); }
  }

  async function finish() {
    stop();
    try {
      const m = await (await fetch("/reports/metrics.json")).json();
      $("done-title").textContent = `Selesai: akurasi ${(m.accuracy * 100).toFixed(1)}% pada ${m.n_test} citra uji`;
      $("done-sub").textContent = `F1 macro ${(m.f1_macro * 100).toFixed(1)}%, C=${m.best_params.C}, gamma=${m.best_params.gamma}. ` +
        (m.test_source === "upload" ? "Memakai data uji yang diunggah." : "Memakai pembagian 80/20 dari data latih.");
      done.hidden = false;
    } catch (e) { done.hidden = true; }
  }

  async function poll() {
    try { paint(await (await fetch("/api/train/status")).json()); } catch (e) { /* coba lagi */ }
  }
  function begin() { stop(); poll(); timer = setInterval(poll, 1000); }
  function stop() { if (timer) clearInterval(timer); timer = null; }

  start.onclick = async () => {
    done.hidden = true; lastLen = -1;
    try { await postJSON("/api/train", {already_cropped: $("cropped").checked}); begin(); }
    catch (e) { toast(e.message); }
  };

  if (window.JOB && window.JOB.status === "running") begin();
  else if (window.JOB && window.JOB.status !== "idle") paint(window.JOB);
})();
