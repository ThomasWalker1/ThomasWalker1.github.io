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



var tbody = document.getElementById('edgeTable').getElementsByTagName('tbody')[0];
edges.forEach(edge => {
    const row = document.createElement("tr");

    const fromCell = document.createElement("td");
    fromCell.textContent = edge.from;

    const toCell = document.createElement("td");
    toCell.textContent = edge.to;

    const descriptionCell = document.createElement("td");
    descriptionCell.textContent = edge.title;

    row.appendChild(fromCell);
    row.appendChild(toCell);
    row.appendChild(descriptionCell);

    tbody.appendChild(row);

    row.addEventListener('mouseenter', function() {
    network.selectEdges([edge.id]);
    });
    row.addEventListener('mouseleave', function() {
    network.unselectAll();
    });
});