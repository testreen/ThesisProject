from neo4j import GraphDatabase

class_names = ['inflammatory', 'lymphocyte', 'fibroblast and endothelial',
               'epithelial', 'apoptosis / civiatte body']


def create_cell(tx, type, xCord, yCord, image):
    tx.run("CREATE (a:Cell {type: $type, x:$x, y:$y, image:$image}) RETURN a", type=type, x=xCord, y=yCord, image=image)


def save_results(boxes, labels, image_name):
    assert len(boxes) == len(labels)

    uri = "bolt://localhost:7687"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "123Broccoli456"))

    query = "CREATE "

    with driver.session() as session:
        tx = session.begin_transaction()
        for i in range(len(boxes)):
            xCen = (boxes[i][0] + boxes[i][2])//2
            yCen = (boxes[i][1] + boxes[i][3])//2
            label = labels[i]
            create_cell(tx, class_names[int(label)], xCen, yCen, image_name)
        tx.commit()

    driver.close()

'''
Get all cells within an area plus all their neighbors up to n hops away
Return list of cells and adjacency matrix
'''
def all_cells_with_n_hops_in_area(image_name, coords, hops=1):
    uri = "bolt://localhost:7687"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "123Broccoli456"))

    with driver.session() as session:
        result = session.run('''
            MATCH (p:Cell {image: $image})
            WHERE p.x > $xmin AND p.x < $xmax AND p.y > $ymin AND p.y < $ymax
            CALL apoc.neighbors.byhop(p, "CLOSE_TO", $hops)
            YIELD nodes as Nodes
            UNWIND Nodes as node
            WITH collect(node)+collect(p) as NodeList
            WITH apoc.coll.toSet(NodeList) as NodeList
            // For each vertices combination...
            WITH NodeList, [n IN NodeList |
                [m IN NodeList |
            		// ...Check for edge existence.
                	CASE size((n)-[:CLOSE_TO]-(m))
            			WHEN 0 THEN 0
            			ELSE 1
            		END
                ]
            ] AS AdjacencyMatrix
            // Unroll rows.
            with AdjacencyMatrix, NodeList, range(0,size(NodeList)-1,1) AS coll_size WHERE size(AdjacencyMatrix) = size(NodeList)
            UNWIND coll_size AS idx
            RETURN NodeList[idx] as nodes, AdjacencyMatrix[idx] as AdjacencyRows
        ''', image=image_name, xmin=coords[0], xmax=coords[1], ymin=coords[2], ymax=coords[3], hops=hops)

        cells = []
        adj = []
        for record in result:
            cells.append(record['nodes'])
            adj.append(record['AdjacencyRows'])

    driver.close()
    return cells, adj

if __name__ == '__main__':
    all_cells_with_n_hops_in_area("P7_HE_Default_Extended_1_1", (700, 730, 630, 680), 2)
