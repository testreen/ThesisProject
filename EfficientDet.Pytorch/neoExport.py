from neo4j import GraphDatabase

label_paths = [
    #'KI-Dataset/For KTH/Rachael/Rach_P9/P9_1_1',
    #'KI-Dataset/For KTH/Rachael/Rach_P9/P9_2_1',
    #'KI-Dataset/For KTH/Rachael/Rach_P9/P9_2_2',
    'P9_3_1',
    #'KI-Dataset/For KTH/Rachael/Rach_P9/P9_3_2',
    'P9_4_1',
    #'KI-Dataset/For KTH/Rachael/Rach_P9/P9_4_2',
    'P13_1_1',
    'P13_1_2',
    #'KI-Dataset/For KTH/Rachael/Rach_P13/P13_2_1',
    'P13_2_2',
    #'KI-Dataset/For KTH/Rachael/Rach_P19/P19_1_1',
    #'KI-Dataset/For KTH/Rachael/Rach_P19/P19_1_2',
    #'KI-Dataset/For KTH/Rachael/Rach_P19/P19_2_1',
    #'KI-Dataset/For KTH/Rachael/Rach_P19/P19_2_2',
    #'KI-Dataset/For KTH/Rachael/Rach_P19/P19_3_1',
    #'KI-Dataset/For KTH/Rachael/Rach_P19/P19_3_2',
    #'KI-Dataset/For KTH/Rachael/Rach_P20/P20_1_3',
    #'KI-Dataset/For KTH/Rachael/Rach_P20/P20_1_4',
    #'KI-Dataset/For KTH/Rachael/Rach_P20/P20_2_2',
    #'KI-Dataset/For KTH/Rachael/Rach_P20/P20_2_3',
    #'KI-Dataset/For KTH/Rachael/Rach_P20/P20_2_4',
    #'KI-Dataset/For KTH/Rachael/Rach_P20/P20_3_1',
    #'KI-Dataset/For KTH/Rachael/Rach_P20/P20_3_2',
    #'KI-Dataset/For KTH/Rachael/Rach_P20/P20_3_3',
    #'KI-Dataset/For KTH/Rachael/Rach_P20/P20_4_1',
    #'KI-Dataset/For KTH/Rachael/Rach_P20/P20_4_2',
    #'KI-Dataset/For KTH/Rachael/Rach_P20/P20_4_3',
    #'KI-Dataset/For KTH/Rachael/Rach_P20/P20_5_1',
    #'KI-Dataset/For KTH/Rachael/Rach_P20/P20_5_2',
    #'KI-Dataset/For KTH/Rachael/Rach_P20/P20_6_1',
    #'KI-Dataset/For KTH/Rachael/Rach_P20/P20_6_2',
    #'KI-Dataset/For KTH/Rachael/Rach_P20/P20_7_1',
    #'KI-Dataset/For KTH/Rachael/Rach_P20/P20_7_2',
    #'KI-Dataset/For KTH/Rachael/Rach_P20/P20_8_1',
    #'KI-Dataset/For KTH/Rachael/Rach_P20/P20_8_2',
    #'KI-Dataset/For KTH/Rachael/Rach_P20/P20_9_1',
    #'KI-Dataset/For KTH/Rachael/Rach_P20/P20_9_2',
    #'KI-Dataset/For KTH/Rachael/Rach_P25/P25_2_1',
    #'KI-Dataset/For KTH/Rachael/Rach_P25/P25_3_1',
    #'KI-Dataset/For KTH/Rachael/Rach_P25/P25_3_2',
    #'KI-Dataset/For KTH/Rachael/Rach_P25/P25_4_2',
    #'KI-Dataset/For KTH/Rachael/Rach_P25/P25_5_1',
    #'KI-Dataset/For KTH/Rachael/Rach_P25/P25_8_2',
    #'KI-Dataset/For KTH/Rachael/Rach_P28/P28_10_4',
    #'KI-Dataset/For KTH/Rachael/Rach_P28/P28_10_5',
    #'KI-Dataset/For KTH/Helena/Helena_P7/P7_HE_Default_Extended_1_1',
    'P7_HE_Default_Extended_2_1',
    'P7_HE_Default_Extended_2_2',
    'P7_HE_Default_Extended_3_1',
    'N10_1_1',
    'N10_1_2',
    #'KI-Dataset/For KTH/Helena/N10/N10_1_3',
    'N10_2_1',
    'N10_2_2',
    #'KI-Dataset/For KTH/Nikolce/N10_1_1',
    #'KI-Dataset/For KTH/Nikolce/N10_1_2',
] # Len 58

class_names = ['inflammatory', 'lymphocyte', 'fibroblast and endothelial',
               'epithelial', 'apoptosis / civiatte body']


def create_cell(tx, type, xCord, yCord, image, label_scores):
    tx.run("CREATE (a:Cell {type: $type, x:$x, y:$y, image:$image, c0: $c0, c1: $c1, c2:$c2, c3:$c3}) RETURN a", type=type, x=xCord, y=yCord, image=image, c0=label_scores[0], c1=label_scores[1], c2=label_scores[2], c3=label_scores[3])


def save_results(boxes, labels, image_name, label_scores):
    assert len(boxes) == len(labels)
    assert len(boxes) == len(label_scores)

    uri = "bolt://localhost:7687"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "123Broccoli456"))

    query = "CREATE "

    with driver.session() as session:
        tx = session.begin_transaction()
        for i in range(len(boxes)):
            xCen = (boxes[i][0] + boxes[i][2])//2
            yCen = (boxes[i][1] + boxes[i][3])//2
            label = labels[i]
            create_cell(tx, class_names[int(label)], xCen, yCen, image_name, label_scores[i])
        tx.commit()

    driver.close()

def generate_graph(image_name):
    uri = "bolt://localhost:7687"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "123Broccoli456"))

    with driver.session() as session:
        result = session.run('''
            MATCH (p:Cell {image: $image})
            WITH {item:id(p), weights: [p.x, p.y]} as userData
            WITH collect(userData) as data
            CALL gds.alpha.similarity.euclidean.write({
            	nodeProjection: '*',
                relationshipProjection: '*',
                data: data,
                topK: 10,
                similarityCutoff: 2000,
                showComputations: true,
                writeRelationshipType: "CLOSE_TO"
            })
             YIELD nodes, similarityPairs, writeRelationshipType, writeProperty, min, max, mean, stdDev, p25, p50, p75, p90, p95, p99, p999, p100
             RETURN nodes, similarityPairs, writeRelationshipType, writeProperty, min, max, mean, p95
        ''', image=image_name)
    driver.close()

if __name__ == '__main__':
    for image in label_paths:
        generate_graph(image)
