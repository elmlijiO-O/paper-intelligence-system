function renderChart(tfidfTerms, transformerTerms) {
  const ctx = document.getElementById("terms-chart").getContext("2d");

  if (window._chartInstance) {
    window._chartInstance.destroy();
  }

  const allTerms = [...new Set([...tfidfTerms, ...transformerTerms])].slice(0, 8);

  const tfidfScores = allTerms.map(t => tfidfTerms.includes(t) ? 1 : 0);
  const transformerScores = allTerms.map(t => transformerTerms.includes(t) ? 1 : 0);

  window._chartInstance = new Chart(ctx, {
    type: "bar",
    data: {
      labels: allTerms,
      datasets: [
        {
          label: "TF-IDF",
          data: tfidfScores,
          backgroundColor: "rgba(240, 192, 64, 0.7)",
          borderColor: "rgba(240, 192, 64, 1)",
          borderWidth: 1,
        },
        {
          label: "Transformer",
          data: transformerScores,
          backgroundColor: "rgba(64, 144, 240, 0.7)",
          borderColor: "rgba(64, 144, 240, 1)",
          borderWidth: 1,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { position: "top" },
        title: {
          display: true,
          text: "Top terms selected by each method",
        },
      },
      scales: {
        y: {
          beginAtZero: true,
          max: 1.2,
          ticks: { stepSize: 1 },
          title: { display: true, text: "Term present (1 = yes)" },
        },
      },
    },
  });
}