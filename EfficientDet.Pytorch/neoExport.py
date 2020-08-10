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
