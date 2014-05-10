// Full Word clouds

var width = 450;
var height = 400;
var fill = d3.scale.category20();

var valOf = function (d) {
  return parseFloat(d.coefficient);
};

d3.json("htmldata/full_word_cloud.json", function(error, data) {
  generate_clouds(data);
});

function generate_clouds(cuisine_words) {
  Object.keys(cuisine_words).forEach(function(label) {
    cuisine = cuisine_words[label]

    // define font scale
    var fontSize = d3.scale.linear()
        .domain([d3.min(cuisine, valOf), d3.max(cuisine, valOf)])
        .range([14, 80]);

    d3.layout.cloud()
        .size([width, height])
        .words(cuisine)
        .padding(5)
        .font("Sans-Serif")
        .rotate(function() { return 0; })
        .fontSize(function(d) { return fontSize(parseFloat(d.coefficient)); })
        .on("end", function(obj) {
          draw(cuisine, label);
        })
        .start();
  });
}

function draw(words, label) {
  var svg = d3.select("#clouds").append("svg")
      .attr("width", width)
      .attr("height", height);

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

  // the label
  var catLabel = svg.append("text")
    .attr("x", width/2)
    .attr("y", height - 15)
    .style("font-size", '20px')
    .style("stroke-width", "6")
    .style("font-family", "Sans-Serif")
    .style("fill", "black")
    .attr("text-anchor", "middle")
    .text(label)
}

