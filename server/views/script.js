function focusOnStockGame() {
  $(".card.stock-list > *:not(.card-title)").fadeToggle();
}

$(window).ready(function () {
  $(".card.stock-game").on("click", () => {
    // $(".card.stock-game > *:not(.card-title)").fadeToggle();
  });

  $(".card.stock-list").on("click", () => {
    // $(".card.stock-list > *:not(.card-title)").fadeToggle();
  });
});

window.onload = function () {
  google.charts.load("current", { packages: ["corechart"] });
  google.charts.setOnLoadCallback(drawChart);

  function drawChart() {
    var data = google.visualization.arrayToDataTable([
      ["Stock", "Price"],
      [0, 5755],
      [1, 6878],
      [2, 8201],
      [3, 8911],
      [4, 7373],
      [5, 6767],
      [6, 8911],
    ]);

    // Optional; add a title and set the width and height of the chart
    var options = {
      width: 300,
      vAxis: { format: "decimal", gridlines: { interval: 1 } },
      legend: { position: "none" },
    };

    // Display the chart inside the <div> element with id="piechart"
    var chart = new google.visualization.LineChart(
      document.getElementById("piechart")
    );
    chart.draw(data, options);
  }
};
