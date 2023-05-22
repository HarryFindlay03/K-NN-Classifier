##################
# kNN classifier #
##################

import random
import math

### LOAD DATASET AND SPLIT TRAINING AND TEST SETS
def loadDataset(filename, split):
    res = []
    with open(filename, "r") as f:
        lines = [x.strip() for x in f]
        for i in range(len(lines)):
            res.append(lines[i].split(","))

    random.shuffle(res)

    splitAt = int(split * len(res))
    return (res[:splitAt], res[splitAt:])


### SIMILARITY MEASURE
def euclideanDistance(instance1, instance2):
    # MINOWSKI DISTANCE where p = 2 for euclidean distance
    p = 2
    ans = 0

    n = len(instance1)
    for i in range(0, n - 1):
        ans += math.pow(abs(float(instance1[i]) - float(instance2[i])), p)

    ans = math.pow(ans, (1 / p))
    return ans


### FIND NEIGHBOURS
def getNeighbors(trainingSet, testInstance, k):
    # work out the euclidean distance between all nodes, order the nodes, return the k best nodes

    # map will the distance as a key and as a value store the category
    map = {}

    for i in range(0, len(trainingSet)):
        distance = euclideanDistance(trainingSet[i], testInstance)
        current_category = trainingSet[i][len(trainingSet[i]) - 1]
        map[distance] = current_category

    sorted_map = sorted(map.items())

    return sorted_map[1 : k + 1]


### GENERATE RESPONSE
def getResponse(neighbors):
    # take a plurality vote from NN(k, X_q)

    # map of category and score for each neighbour returned
    map = {}
    for i in range(0, len(neighbors)):
        if neighbors[i][0] in map.keys():
            map[neighbors[i][1]] += 1
        else:
            map[neighbors[i][1]] = 1

    # get the largest value from the map
    max = 0
    category = ""
    for key in map:
        if map[key] > max:
            category = key

    return category


### DETERMINE PREDICTION ACCURACY
def getAccuracy(testSet, predictions):
    # for each item in the predictions check against actual value
    # add 1 to the count if the value is correct then devide count value by the total number of items in validation set

    count = 0
    for i in range(0, len(predictions)):
        if predictions[i] == testSet[i][-1]:
            count += 1

    return count / len(predictions) * 100


### MAIN
def main():
    # prepare data
    trainingSet = []
    testSet = []
    split = 4.0 / 5  # proportion of traning data vs test data
    trainingSet, testSet = loadDataset("files/iris.data.txt", split)

    # generate predictions
    predictions = []
    k = 3  # number of neighbours
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print(f"PREDICTED -> {result}\t ACTUAL -> {testSet[x][-1]}")

    # compute prediction accuracy
    accuracy = getAccuracy(testSet, predictions)
    print("\n\n\nACCURACY: " + repr(accuracy) + "% 🕺🕺\n\n\n")


main()
