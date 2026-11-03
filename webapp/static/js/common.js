// Utilitas bersama
window.$ = (id) => document.getElementById(id);
window.CLASS_NAMES = {normal: "Tidak Stres (Normal)", ringan: "Stres Ringan", sedang: "Stres Sedang", berat: "Stres Berat"};

window.toast = (msg, ms = 3600) => {
  const el = document.createElement("div");
  el.className = "toast"; el.textContent = msg;
  $("toasts").appendChild(el);
  setTimeout(() => el.remove(), ms);
};

window.postJSON = async (url, body) => {
  const r = await fetch(url, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
  let d = {};
  try { d = await r.json(); } catch (e) { /* abaikan */ }
  if (!r.ok) throw new Error(d.error || `Permintaan gagal (${r.status})`);
  return d;
};

window.esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
