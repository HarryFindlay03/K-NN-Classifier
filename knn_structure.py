
##################
# kNN classifier #
##################

import csv
import random
import math
import operator


### LOAD DATASET AND SPLIT TRAINING AND TEST SETS 
 
def loadDataset(filename, split):
    #this function should load the dataset contained in the file with name
    #passed as input, and should then split the data in a training and
    #test set according to a ratio given by the split parameters

    res = []
    with open(filename, 'r') as f:
        lines = [x.strip() for x in f]
        for i in range(len(lines)):
            res[i] = lines[i].split(',')
    
    random.shuffle(res)
    
    return (res[:split], res[split:])
    
### SIMILARITY MEASURE
 
def euclideanDistance(instance1, instance2, length):
    #this function should return the euclidean distance between instance1
    #and instance2
    pass


### FIND NEIGHBOURS
 
def getNeighbors(trainingSet, testInstance, k):
	#this function should return the k nearest neighbours of testInstance
    #in trainingSet


### GENERATE RESPONSE
 
def getResponse(neighbors):
	#this function should generate the prediction based on the classes of
    #neighbours
    pass

### DETERMINE PREDICTION ACCURACY
 
def getAccuracy(testSet, predictions):
	#this function should calculate the accuracy based on the correct classes
    #of testSet and the predicted ones in predictions
    pass

### MAIN
	
def main():

	# prepare data
	trainingSet=[]
	testSet=[]
	split = 2.0/3 # proportion of traning data vs test data  
	trainingSet, testSet = loadDataset('iris.data.txt', split)
	print('Train set: ' + repr(len(trainingSet)))
	print('Test set: ' + repr(len(testSet)))

	# # generate predictions
	# predictions=[]
	# k = 3 # number of neighbours
	# for x in range(len(testSet)):
	# 	neighbors = getNeighbors(trainingSet, testSet[x], k)
	# 	result = getResponse(neighbors)
	# 	predictions.append(result)
	# 	print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
	#
 #        # compute prediction accuracy
	# accuracy = getAccuracy(testSet, predictions)
	# print('Accuracy: ' + repr(accuracy) + '%')
	
main()
