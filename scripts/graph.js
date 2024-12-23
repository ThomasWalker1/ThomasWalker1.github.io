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

function updateGraph() {
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
    var tbody = table.createTBody()
    edges.forEach(function (edge) {
        var row = tbody.insertRow();
        addCell(row, edge.from);
        addCell(row, edge.to);
        addCell(row, edge.title)
        
        row.addEventListener('click', function() {
            if (this.classList.contains('selected'))
                this.classList.remove('selected')
            else
                this.classList.add('selected');
            if (network.getSelectedEdges().includes(edge.id))
                network.unselectAll();
            else
                network.selectEdges([edge.id]);
        });
    });
}

setTimeout(updateGraph,500);