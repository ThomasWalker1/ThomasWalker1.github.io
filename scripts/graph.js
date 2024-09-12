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

function populateEdgeTable() {
    var tbody = document.getElementById('edgeTable').getElementsByTagName('tbody')[0];
    edges.forEach(function(edge) {
        var row = document.createElement('tr');
        row.dataset.edgeId = edge.id;

        var cell1 = document.createElement('td');
        var cell2 = document.createElement('td');
        var cell3 = document.createElement('td');

        cell1.textContent = edge.from;
        cell2.textContent = edge.to;
        cell3.textContent = edge.title;

        row.appendChild(cell1);
        row.appendChild(cell2);
        row.appendChild(cell3);

        tbody.appendChild(row);

        row.addEventListener('mouseenter', function() {
        network.selectEdges([edge.id]);
        });
        row.addEventListener('mouseleave', function() {
        network.unselectAll();
        });
    });
}

populateEdgeTable();