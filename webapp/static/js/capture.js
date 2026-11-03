// Akuisisi citra dari kamera -> /api/capture (data latih atau data uji).
(() => {
  const video = $("video"), msg = $("msg"), snap = $("snap"), off = $("camoff");
  const canvas = document.createElement("canvas");
  let counts = window.INIT_COUNTS;
  const radio = (name) => document.querySelector(`input[name="${name}"]:checked`).value;

  function renderCounts() {
    const rows = Object.keys(CLASS_NAMES).map((k) =>
      `<tr><td>${CLASS_NAMES[k]}</td><td class="num">${counts.train[k]}</td><td class="num">${counts.test[k]}</td></tr>`).join("");
    $("counts").innerHTML = `<tr><th>Kelas</th><th class="num">Latih</th><th class="num">Uji</th></tr>${rows}`;
  }
  renderCounts();

  $("start").onclick = async () => {
    try {
      video.srcObject = await navigator.mediaDevices.getUserMedia({video: {width: 640, height: 480}, audio: false});
      snap.disabled = false; off.hidden = true; msg.textContent = "Kamera aktif.";
    } catch (e) { msg.textContent = "Kamera tidak dapat diakses: " + e.message; }
  };

  snap.onclick = async () => {
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    snap.disabled = true;
    try {
      const d = await postJSON("/api/capture", {
        image: canvas.toDataURL("image/jpeg", 0.92), subject_id: $("subject").value.trim(),
        label: radio("label"), split: radio("split"), consent: $("consent").checked,
      });
      counts = d.counts; renderCounts(); msg.textContent = "Tersimpan: " + d.saved; toast("Citra tersimpan.");
    } catch (e) { msg.textContent = e.message; toast(e.message); }
    snap.disabled = false;
  };
})();
