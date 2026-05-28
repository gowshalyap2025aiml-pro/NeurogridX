function updateClock() {
  const el = document.getElementById('topbarTime');
  if (!el) return;
  const now = new Date();
  el.textContent = now.toLocaleTimeString('en-US', {hour12: false}) +
                   ' | ' + now.toLocaleDateString('en-US', {month:'short', day:'numeric', year:'numeric'});
}
setInterval(updateClock, 1000);
updateClock();

const sidebarToggle = document.getElementById('sidebarToggle');
const sidebar       = document.getElementById('sidebar');
if (sidebarToggle && sidebar) {
  sidebarToggle.addEventListener('click', () => {
    sidebar.classList.toggle('open');
  });
  document.addEventListener('click', e => {
    if (!sidebar.contains(e.target) && !sidebarToggle.contains(e.target)) {
      sidebar.classList.remove('open');
    }
  });
}

function animateCounter(el, target, duration = 1200, decimals = 0) {
  const start = performance.now();
  const from = 0;
  function step(ts) {
    const progress = Math.min((ts - start) / duration, 1);
    const ease = 1 - Math.pow(1 - progress, 3);
    const val = from + (target - from) * ease;
    el.textContent = val.toFixed(decimals);
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

document.querySelectorAll('[data-counter]').forEach(el => {
  const target   = parseFloat(el.getAttribute('data-counter'));
  const decimals = parseInt(el.getAttribute('data-decimals') || '0');
  animateCounter(el, target, 1400, decimals);
});

if (typeof Chart !== 'undefined') {
  Chart.defaults.color          = '#94A3B8';
  Chart.defaults.borderColor    = '#1E293B';
  Chart.defaults.font.family    = "'Segoe UI', system-ui, sans-serif";
  Chart.defaults.font.size      = 12;
  Chart.defaults.plugins.legend.labels.boxWidth  = 12;
  Chart.defaults.plugins.legend.labels.padding   = 16;
  Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(11,17,32,.95)';
  Chart.defaults.plugins.tooltip.borderColor     = '#1E293B';
  Chart.defaults.plugins.tooltip.borderWidth     = 1;
  Chart.defaults.plugins.tooltip.padding         = 10;
  Chart.defaults.plugins.tooltip.titleColor      = '#00E5FF';
  Chart.defaults.plugins.tooltip.bodyColor       = '#E2E8F0';
}

setTimeout(() => {
  document.querySelectorAll('.alert.fade.show').forEach(el => {
    el.classList.remove('show');
    setTimeout(() => el.remove(), 300);
  });
}, 4000);

const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.classList.add('animate-fade-up');
      observer.unobserve(entry.target);
    }
  });
}, { threshold: 0.1 });
document.querySelectorAll('.observe-fade').forEach(el => observer.observe(el));

function drawHealthRing(score) {
  const svg = document.getElementById('healthRingSvg');
  if (!svg) return;
  const r = 60, cx = 70, cy = 70;
  const circ = 2 * Math.PI * r;
  const offset = circ - (score / 100) * circ;

  let color;
  if (score >= 90) color = '#10B981';
  else if (score >= 75) color = '#00E5FF';
  else if (score >= 60) color = '#F59E0B';
  else if (score >= 40) color = '#EF4444';
  else color = '#DC2626';

  const fg = svg.querySelector('.health-ring-fg');
  if (fg) {
    fg.setAttribute('stroke', color);
    fg.setAttribute('stroke-dasharray', circ);
    fg.setAttribute('stroke-dashoffset', circ);
    setTimeout(() => {
      fg.style.transition = 'stroke-dashoffset 1.4s cubic-bezier(.4,0,.2,1)';
      fg.setAttribute('stroke-dashoffset', offset);
    }, 100);
  }

  const scoreEl = document.getElementById('healthScoreVal');
  if (scoreEl) {
    scoreEl.style.fill = color;
    animateCounterSVG(scoreEl, score, 1400);
  }
}

function animateCounterSVG(el, target, duration) {
  const start = performance.now();
  function step(ts) {
    const p = Math.min((ts - start) / duration, 1);
    el.textContent = Math.round(p * target);
    if (p < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

let liveInterval = null;
function startLiveRefresh() {
  if (document.querySelector('[data-live]')) {
    liveInterval = setInterval(fetchLiveStats, 8000);
  }
}

function fetchLiveStats() {
  fetch('/api/live_stats')
    .then(r => r.json())
    .then(data => {
      const setEl = (id, val) => {
        const el = document.getElementById(id);
        if (el) { el.textContent = val; el.classList.add('counter-anim');
                  setTimeout(() => el.classList.remove('counter-anim'), 500); }
      };
      setEl('liveDemand', data.demand.toLocaleString());
      setEl('liveSolar',  data.solar.toLocaleString());
      setEl('liveScore',  data.grid_score);
      setEl('liveAnomalies', data.anomalies);
    })
    .catch(() => {});
}

document.addEventListener('DOMContentLoaded', startLiveRefresh);
