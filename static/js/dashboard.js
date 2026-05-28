

// Colors
const C = {
  cyan:   '#00E5FF', purple: '#8B5CF6',
  green:  '#10B981', orange: '#F59E0B',
  red:    '#EF4444', muted:  '#94A3B8',
};

function initTrendChart(hours, demand, solar) {
  const ctx = document.getElementById('trendChart');
  if (!ctx) return;
  new Chart(ctx, {
    type: 'line',
    data: {
      labels: hours.map(h => h + ':00'),
      datasets: [
        {
          label: 'Demand (MW)',
          data: demand,
          borderColor: C.cyan,
          backgroundColor: 'rgba(0,229,255,.08)',
          borderWidth: 2.5,
          pointRadius: 0,
          fill: true,
          tension: 0.4,
          yAxisID: 'y',
        },
        {
          label: 'Solar (kW)',
          data: solar,
          borderColor: C.orange,
          backgroundColor: 'rgba(245,158,11,.06)',
          borderWidth: 2,
          pointRadius: 0,
          fill: true,
          tension: 0.4,
          yAxisID: 'y1',
          borderDash: [5, 3],
        }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { intersect: false, mode: 'index' },
      scales: {
        x: { grid: { color: '#1E293B' }, ticks: { maxTicksLimit: 8 } },
        y: {
          grid: { color: '#1E293B' },
          title: { display: true, text: 'Demand (MW)', color: C.muted },
        },
        y1: {
          position: 'right',
          grid: { drawOnChartArea: false },
          title: { display: true, text: 'Solar (kW)', color: C.orange },
        },
      },
      plugins: { legend: { position: 'top' } }
    }
  });
}

function initMixDonut(solar, demand) {
  const ctx = document.getElementById('mixDonut');
  if (!ctx) return;
  const renewable = Math.min(100, (solar / (demand / 1000 + 1e-6)) * 100);
  const fossil = Math.max(0, 100 - renewable);
  new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Renewable', 'Conventional'],
      datasets: [{
        data: [renewable, fossil],
        backgroundColor: ['rgba(16,185,129,.7)', 'rgba(30,41,59,.8)'],
        borderColor: ['#10B981', '#1E293B'],
        borderWidth: 2,
        hoverOffset: 4,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false, cutout: '70%',
      plugins: {
        legend: { position: 'bottom' },
        tooltip: { callbacks: {
          label: ctx => ` ${ctx.parsed.toFixed(1)}%`
        }}
      }
    }
  });
}

function initAnomalyBar(count) {
  const ctx = document.getElementById('anomalyBar');
  if (!ctx) return;
  const labels = ['00','03','06','09','12','15','18','21'];
  const data   = labels.map(() => Math.floor(Math.random() * count + 1));
  data[Math.floor(Math.random() * labels.length)] = count;
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Anomalies',
        data,
        backgroundColor: data.map(v =>
          v >= count ? 'rgba(239,68,68,.7)' : 'rgba(0,229,255,.3)'),
        borderColor: data.map(v =>
          v >= count ? '#EF4444' : '#00E5FF'),
        borderWidth: 1, borderRadius: 4,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { grid: { display: false } },
        y: { grid: { color: '#1E293B' }, beginAtZero: true }
      },
      plugins: { legend: { display: false } }
    }
  });
}
