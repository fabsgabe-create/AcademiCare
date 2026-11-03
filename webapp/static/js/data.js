// Halaman Data: unggah (dengan progres), galeri, hapus.
(() => {
  const picker = $("picker"), drop = $("drop"), filesEl = $("files"), send = $("send"), reset = $("reset");
  const prog = $("prog"), report = $("report"), gal = $("gal");
  const radio = (name) => document.querySelector(`input[name="${name}"]:checked`).value;
  const isZip = (f) => /\.zip$/i.test(f.name);
  const PAGE = 24;
  let selected = [], counts = window.INIT_COUNTS, shown = 0, total = 0;

  // ---------- ringkasan jumlah ----------
  function paintCounts() {
    ["train", "test"].forEach((sp) => {
      const c = counts[sp], n = Object.values(c).reduce((a, b) => a + b, 0);
      $("n-" + sp).textContent = n;
      const parts = Object.keys(CLASS_NAMES).map((k) => c[k] && n ?
        `<i class="bg-${k}" style="width:${c[k] / n * 100}%"></i>` : "").join("");
      const legend = Object.keys(CLASS_NAMES).map((k) =>
        `<span style="--cc:var(--c-${k})">${k[0].toUpperCase() + k.slice(1)} <b>${c[k]}</b></span>`).join("");
      $("stack-" + sp).innerHTML = `<div class="stack">${parts}</div><div class="legend">${legend}</div>`;
    });
    const sp = radio("gsplit");
    Object.keys(CLASS_NAMES).forEach((k) => { $("gc-" + k).textContent = counts[sp][k]; });
  }

  // ---------- pilih berkas ----------
  function setFiles(list) {
    selected = [...list].filter((f) => /\.(jpe?g|png|zip)$/i.test(f.name) || /^image\/(jpeg|png)$/.test(f.type));
    const skipped = list.length - selected.length;
    filesEl.innerHTML = selected.slice(0, 12).map((f) => `<span>${esc(f.name)}</span>`).join("") +
      (selected.length > 12 ? `<span>+${selected.length - 12} lainnya</span>` : "");
    if (skipped) toast(`${skipped} berkas dilewati (hanya JPG/PNG/ZIP).`);
    send.disabled = !selected.length; reset.hidden = !selected.length;
    send.textContent = selected.length ? `Unggah ${selected.length} berkas` : "Unggah";
  }
  picker.onchange = () => setFiles(picker.files);
  reset.onclick = () => { picker.value = ""; setFiles([]); };
  ["dragenter", "dragover"].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (ev) => setFiles(ev.dataTransfer.files));

  // ---------- unggah ----------
  send.onclick = () => {
    const split = radio("split"), label = radio("label");
    if (split === "auto" && selected.some((f) => !isZip(f))) {
      toast("Mode 'Ikuti folder ZIP' hanya untuk berkas ZIP. Pilih Data latih atau Data uji untuk citra lepas.");
      return;
    }
    const fd = new FormData();
    fd.append("split", split); fd.append("label", label);
    selected.forEach((f) => fd.append("files", f, f.name));
    send.disabled = true; prog.hidden = false; report.hidden = true;
    const bar = prog.firstElementChild; bar.style.width = "0%";
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/upload");
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) bar.style.width = (e.loaded / e.total * 95) + "%"; };
    xhr.onerror = () => { toast("Gagal menghubungi server."); send.disabled = false; prog.hidden = true; };
    xhr.onload = () => {
      bar.style.width = "100%";
      let d = {};
      try { d = JSON.parse(xhr.responseText); } catch (e) { /* kosong */ }
      send.disabled = !selected.length;
      if (xhr.status !== 200) { toast(d.error || `Gagal (${xhr.status}). Berkas mungkin terlalu besar.`); prog.hidden = true; return; }
      counts = d.counts; paintCounts();
      const per = Object.entries(d.per).map(([k, v]) => `${k.replace("train", "latih").replace("test", "uji")}: ${v}`).join(", ");
      let html = `<b>${d.saved} citra tersimpan</b>${per ? ` <span class="muted">(${esc(per)})</span>` : ""}`;
      if (d.rejected_total) {
        html += `<br><span style="color:var(--bad)">${d.rejected_total} ditolak</span><ul>` +
          d.rejected.map((r) => `<li>${esc(r.name)}: ${esc(r.reason)}</li>`).join("") +
          (d.rejected_total > d.rejected.length ? `<li>…dan ${d.rejected_total - d.rejected.length} lainnya</li>` : "") + "</ul>";
      }
      report.innerHTML = html; report.hidden = false;
      toast(`${d.saved} citra tersimpan.`);
      picker.value = ""; setFiles([]); loadGallery(true);
    };
    xhr.send(fd);
  };

  // ---------- galeri ----------
  async function loadGallery(fresh) {
    const split = radio("gsplit"), label = radio("glabel");
    if (fresh) { gal.innerHTML = ""; shown = 0; }
    const r = await fetch(`/api/dataset/list?split=${split}&label=${label}&offset=${shown}&limit=${PAGE}`);
    const d = await r.json();
    total = d.total;
    d.names.forEach((n) => {
      const it = document.createElement("div"); it.className = "it";
      it.innerHTML = `<img loading="lazy" alt="" src="/thumb/${split}/${label}/${encodeURIComponent(n)}">` +
        `<button type="button" title="Hapus citra" aria-label="Hapus citra"><svg class="i" viewBox="0 0 24 24" style="width:14px;height:14px"><path d="M6 6l12 12M18 6L6 18"/></svg></button>`;
      it.querySelector("button").onclick = async () => {
        try {
          const res = await postJSON("/api/dataset/delete", {split, label, name: n});
          counts = res.counts; paintCounts(); it.remove(); total--; shown--;
          $("gal-empty").hidden = total > 0; $("more").hidden = shown >= total;
        } catch (e) { toast(e.message); }
      };
      gal.appendChild(it);
    });
    shown += d.names.length;
    $("gal-empty").hidden = total > 0; $("more").hidden = shown >= total;
  }
  $("more").onclick = () => loadGallery(false);
  document.querySelectorAll('input[name="gsplit"],input[name="glabel"]').forEach((el) =>
    el.addEventListener("change", () => { paintCounts(); loadGallery(true); }));
  $("clear").onclick = async () => {
    const split = radio("gsplit"), label = radio("glabel");
    const n = counts[split][label];
    if (!n) return toast("Tidak ada citra untuk dihapus.");
    if (!confirm(`Hapus ${n} citra kelas "${label}" dari ${split === "train" ? "data latih" : "data uji"}? Tindakan ini tidak dapat dibatalkan.`)) return;
    try { const res = await postJSON("/api/dataset/clear", {split, label}); counts = res.counts; paintCounts(); loadGallery(true); toast(`${res.removed} citra dihapus.`); }
    catch (e) { toast(e.message); }
  };

  paintCounts(); loadGallery(true);
})();
