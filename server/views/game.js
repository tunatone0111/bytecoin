window.onload = function () {
  let ctx = document.getElementById("priceChart").getContext("2d");
  let priceChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets
    },
    options: {
      responsive: true,
      legend: {
        display: false,
      },
      title: {
        display: false,
        text: "Stock Info"
      },
    },
  });
};
