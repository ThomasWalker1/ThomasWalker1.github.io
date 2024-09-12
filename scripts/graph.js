var nodes = new vis.DataSet([]);
var edges = new vis.DataSet([]);

var container = document.getElementById('graph-container');
var data = {
    nodes: nodes,
    edges: edges
};
var options = {
    physics: {
      enabled: true,
      solver: 'forceAtlas2Based',
      forceAtlas2Based: {
        gravitationalConstant: -50,
        centralGravity: 0.01,
        springLength: 100,
        springConstant: 0.08
      },
      maxVelocity: 50,
      timestep: 0.5,
      stabilization: {
        enabled: true,
        iterations: 1000,
        fit: true
      }
    }
  };
var network = new vis.Network(container, data, options);

fetch('assets/graph_data.json')
.then(response => response.json())
.then(jsonData => {
  nodes.add(jsonData.nodes);
  edges.add(jsonData.edges);
})
.catch(error => console.error('Error loading JSON data:', error));

var table = document.getElementById('edgeTable');
var thead = table.createTHead();
var headerRow = thead.insertRow();
function addCell(tr, text) {
    var td = tr.insertCell();
    td.textContent = text;
    return td;
}
addCell(headerRow, 'From Node');
addCell(headerRow, 'To Node');
addCell(headerRow, 'Description');
edges.forEach(function(edge) {
    var row = table.insertRow();
    addCell(row, edge.from);
    addCell(row, edge.to);
    addCell(row, edge.title)

    row.addEventListener('mouseenter', function() {
    network.selectEdges([edge.id]);
    });
    row.addEventListener('mouseleave', function() {
    network.unselectAll();
    });
});