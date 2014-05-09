// Word clouds

var width = 600;
var height = 400;
var fill = d3.scale.category20();

var cuisine_words = null

var valOf = function (d) {
  return parseFloat(d.coefficient);
};

d3.json("htmldata/word_cloud.json", function(error, data) {
  console.log(data)
  cuisine_words = data
  generate_clouds();
});

function generate_clouds() {
  for (var i = 0; i < cuisine_words.length; i++) {
    var cuisine = cuisine_words[

    // define font scale
    var fontSize = d3.scale.log()
        .domain([d3.min(topic, valOf), d3.max(topic, valOf)])
        .range([14, 80]);

    d3.layout.cloud()
        .size([width, height])
        .timeInterval(10)
        .words(topic)
        .padding(5)
        //.rotate(function() { return ~~(Math.random() * 5) * 30 - 30; })
        .rotate(function() { return 0; })
        .fontSize(function(d) { return fontSize(parseFloat(d.value)); })
        .on("end", draw)
        .start();
  }
}

function draw(words) {
  var svg = d3.select("#clouds").append("svg")
      .attr("width", width)
      .attr("height", height);

  // the words
  svg.append("g")
      .attr("transform", "translate(" + width / 2 + ", " + height / 2 + ")")
    .selectAll("text")
      .data(words)
    .enter().append("text")
      .style("font-size", function(d) { return d.size + "px"; })
      .style("font-family", "Sans-Serif")
      .style("fill", function(d, i) { return fill(i); })
      .attr("text-anchor", "middle")
      .attr("transform", function(d) {
        return "translate(" + [d.x, d.y] + ")rotate(" + d.rotate + ")";
      })
      .text(function(d) { return d.text; });

  // the border
  var borderPath = svg.append("rect")
    .attr("x", 0)
    .attr("y", 0)
    .attr("height", height)
    .attr("width", width)
    .style("stroke", "black")
    .style("fill", "none")
    .style("stroke-width", "3");
}

