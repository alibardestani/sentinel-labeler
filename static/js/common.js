(function () {
  // ---------- Modal: Prelabel ----------
  function openPrelabel() {
    const m = document.getElementById('modal');
    if (!m) { console.warn('modal element not found'); return; }
    m.classList.remove('hidden');
    m.style.display = 'flex';
    m.style.zIndex = 9999;

    // toggle threshold input
    const sel = document.getElementById('prelabelMethod');
    const wrap = document.getElementById('ndviThreshWrap');
    if (sel && wrap) {
      const toggle = () => { wrap.style.display = (sel.value === 'ndvi_thresh') ? 'flex' : 'none'; };
      sel.onchange = toggle;
      toggle();
    }
    console.log('openPrelabel: modal opened');
  }

  function closePrelabel() {
    const m = document.getElementById('modal');
    if (!m) { console.warn('modal element not found'); return; }
    m.classList.add('hidden');
    m.style.display = 'none';
    console.log('closePrelabel: modal closed');
  }

  // ---------- Progress Modal (shared) ----------
  let running = false;
  let pollTimer = null;

  function openProgress() {
    const el = document.getElementById('progressModal');
    if (!el) return;
    el.classList.remove('hidden');
    el.style.display = 'flex';
    startPollingProgress();
  }

  function closeProgress() {
    const el = document.getElementById('progressModal');
    if (!el) return;
    el.classList.add('hidden');
    el.style.display = 'none';
    stopPollingProgress();
  }

  function startPollingProgress() {
    stopPollingProgress();
    pollTimer = setInterval(async () => {
      try {
        const r = await fetch('/api/progress', { cache: 'no-store' });
        if (!r.ok) return;
        const { percent = 0, phase = '', note = '' } = await r.json();

        const bar   = document.querySelector('#progressModal .progress .bar');
        const title = document.querySelector('#progressModal .modal-title');

        if (bar)   bar.style.width = Math.max(0, Math.min(100, percent)) + '%';
        if (title) title.textContent = `در حال پردازش… (${Math.round(percent)}%)${note ? ' - ' + note : ''}`;

        // وقتی رسید 100%، polling رو قطع کن و مودال رو بعد یه مکث ببند
        if (percent >= 100) {
          stopPollingProgress();
          setTimeout(() => { try { closeProgress(); } catch {} }, 500);
        }
      } catch {
        // نادیده بگیر؛ تلاش بعدی
      }
    }, 500);
  }

  function stopPollingProgress() {
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  }

  // ---------- Run Prelabel ----------
  async function runPrelabel() {
    if (running) return;
    running = true;

    openProgress();

    const btn = document.getElementById('prelabelRunBtn');
    if (btn) btn.disabled = true;

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 300_000); // 300s

    try {
      const methodEl = document.getElementById('prelabelMethod');
      if (!methodEl) { alert('Prelabel control missing'); return; }
      const method = methodEl.value;

      const payload = { method };
      if (method === 'ndvi_thresh') {
        const v = parseFloat(document.getElementById('ndviThreshold')?.value || '0.2');
        payload.ndvi_threshold = Number.isFinite(v) ? v : 0.2;
      }

      console.log('runPrelabel: sending', payload);

      const r = await fetch('/api/prelabel', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: controller.signal
      });

      if (r.ok) {
        // اجازه بده نوار به 100% برسه، بعد برو صفحه ماسک
        setTimeout(() => { window.location.href = '/mask'; }, 350);
      } else {
        const err = await r.json().catch(() => ({ error: 'Pre-label failed' }));
        alert(err.error || 'Pre-label failed');
        closeProgress();
      }
    } catch (e) {
      alert(e?.name === 'AbortError' ? 'Request timed out' : ('Network error: ' + (e?.message || e)));
      closeProgress();
    } finally {
      clearTimeout(timer);
      running = false;
      if (btn) btn.disabled = false;
    }
  }

  // ---------- Model: info / upload / run ----------
  async function refreshModelInfo() {
    const box = document.getElementById('modelInfo');
    if (!box) return;
    try {
      const r = await fetch('/api/model_info', { cache: 'no-store' });
      const info = await r.json();
      if (!info.loaded) {
        box.textContent = 'مدلی لود نشده است.';
      } else {
        const providers = (info.providers || []).join(', ');
        const bands = (info.bands || []).join(', ');
        box.textContent = `Loaded • providers: ${providers} • tile=${info.tile_size} • bands=${bands}`;
      }
    } catch {
      box.textContent = 'خطا در گرفتن اطلاعات مدل';
    }
  }

  const uploadBtn = document.getElementById('uploadModelBtn');
  if (uploadBtn) {
    uploadBtn.addEventListener('click', async () => {
      const inp = document.getElementById('modelFile');
      const f = inp?.files?.[0];
      if (!f) { alert('فایل مدل را انتخاب کنید.'); return; }
      const fd = new FormData();
      fd.append('file', f);
      try {
        const r = await fetch('/api/model_upload', { method: 'POST', body: fd });
        const j = await r.json().catch(() => ({}));
        if (r.ok) {
          alert('مدل بارگذاری شد.');
          refreshModelInfo();
        } else {
          alert('خطا: ' + (j.error || 'upload failed'));
        }
      } catch (e) {
        alert('Network error: ' + (e?.message || e));
      }
    });
  }

  const runModelBtn = document.getElementById('runModelBtn');
  if (runModelBtn) {
    runModelBtn.addEventListener('click', async () => {
      // استفاده از همان Progress
      openProgress();
      try {
        const r = await fetch('/api/run_model', { method: 'POST' });
        if (r.ok) {
          setTimeout(() => location.href = '/mask', 450);
        } else {
          const j = await r.json().catch(() => ({}));
          alert('خطا: ' + (j.error || 'run failed'));
          closeProgress();
        }
      } catch (e) {
        alert('Network error: ' + (e?.message || e));
        closeProgress();
      }
    });
  }

  // ---------- Expose ----------
  window.openPrelabel  = openPrelabel;
  window.closePrelabel = closePrelabel;
  window.runPrelabel   = runPrelabel;

  // ---------- Boot ----------
  window.addEventListener('DOMContentLoaded', () => {
    // فقط اگر کنترل‌های مدل روی این صفحه هست
    refreshModelInfo();
  });

  console.log('common.js ready');
})();
