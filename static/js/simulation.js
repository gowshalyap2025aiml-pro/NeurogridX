

function updateSliderDisplay(id, displayId, unit) {
  const slider  = document.getElementById(id);
  const display = document.getElementById(displayId);
  if (!slider || !display) return;
  display.textContent = slider.value + (unit || '');
  slider.addEventListener('input', () => {
    display.textContent = slider.value + (unit || '');
  });
}

document.addEventListener('DOMContentLoaded', () => {
  updateSliderDisplay('demandGrowth',  'demandGrowthVal', '%');
  updateSliderDisplay('temperature',   'temperatureVal',  '°C');
  updateSliderDisplay('renewablePct',  'renewablePctVal', '%');
  updateSliderDisplay('simHour',       'simHourVal',      ':00');
});

function drawSimGauge(score) {
  const ctx = document.getElementById('simGauge');
  if (!ctx || typeof Chart === 'undefined') return;
  let color;
  if (score >= 90) color = '#10B981';
  else if (score >= 75) color = '#00E5FF';
  else if (score >= 60) color = '#F59E0B';
  else color = '#EF4444';

  new Chart(ctx, {
    type: 'doughnut',
    data: {
      datasets: [{
        data: [score, 100 - score],
        backgroundColor: [color + 'CC', 'rgba(30,41,59,.4)'],
        borderColor: [color, 'transparent'],
        borderWidth: [2, 0],
      }]
    },
    options: {
      cutout: '75%', responsive: true, maintainAspectRatio: false,
      circumference: 270, rotation: -135,
      plugins: { legend: { display: false }, tooltip: { enabled: false } }
    }
  });
}

document.addEventListener('DOMContentLoaded', () => {
  const scoreEl = document.getElementById('simScoreVal');
  if (scoreEl) {
    const score = parseFloat(scoreEl.getAttribute('data-score') || '0');
    drawSimGauge(score);
  }
});
