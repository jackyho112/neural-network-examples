
# coding: utf-8

# In[162]:

import numpy
import scipy.special
import matplotlib.pyplot
get_ipython().magic('matplotlib inline')

# three layer
class NeuralNetwork:
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        self.inputNodes = inputNodes
        self.hiddenNodes = hiddenNodes
        self.outputNodes = outputNodes

        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.normal.html
        # to achieve an even distribution
        self.wih = numpy.random.normal(
            0.0,
            pow(self.hiddenNodes, -0.5),
            (self.hiddenNodes, self.inputNodes)
        ) # input to hidden input
        self.who = numpy.random.normal(
            0.0,
            pow(self.outputNodes, -0.5), # power
            (self.outputNodes, self.hiddenNodes)
        ) # hidden input to final output
        self.activation_function = lambda x: scipy.special.expit(x)
        self.learningRate = learningRate

        pass

    def train(self, inputList, targetList):
        # convert input list to 2d array
        inputs = numpy.array(inputList, ndmin = 2).T
        targets = numpy.array(targetList, ndmin = 2).T

        hiddenInputs = numpy.dot(self.wih, inputs)
        hiddenOutputs = self.activation_function(hiddenInputs)
        finalInputs = numpy.dot(self.who, hiddenOutputs)
        finalOutputs = self.activation_function(finalInputs)

        # error is the (target - actual)
        outputErrors = targets - finalOutputs
        hiddenErrors = numpy.dot(self.who.T, outputErrors)

        self.who += self.learningRate * numpy.dot(
            (outputErrors * finalOutputs * (1 - finalOutputs)),
            numpy.transpose(hiddenOutputs)
        )

        self.wih += self.learningRate * numpy.dot(
            (hiddenErrors * hiddenOutputs * (1 - hiddenOutputs)),
            numpy.transpose(inputs)
        )

        pass

    def query(self, inputList):
        inputs = numpy.array(inputList, ndmin = 2).T

        hiddenInputs = numpy.dot(self.wih, inputs)
        hiddenOutputs = self.activation_function(hiddenInputs)
        finalInputs = numpy.dot(self.who, hiddenOutputs)
        finalOutputs = self.activation_function(finalInputs)

        return finalOutputs

        pass


# In[163]:

inputNodes = 784
hiddenNodes = 100
outputNodes = 10

learningRate = 0.3

n = NeuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)


# In[164]:

dataFile = open("mnist_train.csv", "r")


# In[165]:

dataList = dataFile.readlines()


# In[166]:

dataFile.close()


# In[167]:

for record in dataList:
    allValues = record.split(',')
    inputs = numpy.asfarray(allValues[1:]) / 255.0 * 0.99 + 0.01
    targets = numpy.zeros(outputNodes) + 0.01
    targets[int(allValues[0])] = 0.99 # the first item is the label for the number
    n.train(inputs, targets)
    pass


# In[168]:

# test
testDataFile = open("mnist_test.csv", 'r')
testDataList = testDataFile.readlines()
testDataFile.close()
allValues = testDataList[0].split(',')
print(allValues[0])


# In[169]:

imageArray = numpy.asfarray(allValues[1:]).reshape((28, 28))
matplotlib.pyplot.imshow(imageArray, cmap='Greys', interpolation='None')


# In[170]:

n.query(
    (numpy.asfarray(allValues[1:]))/255.0 * 0.99 + 0.01
)


# In[177]:

# it worked!

# More testing
scorecard = []

# go through all the records in the test data set for record in testDataList:
for record in testDataList:
    allValues = record.split(',')
    correctLabel = int(allValues[0])
    print(correctLabel, 'correct label')
    inputs = (numpy.asfarray(allValues[1:]) / 255.0 * 0.99) + 0.01
    outputs = n.query(inputs)
    label = numpy.argmax(outputs)
    print(label, "network's answer")
    if (label == correctLabel):
        scorecard.append(1)
    else:
        scorecard.append(0)
    pass

scorecardArray = numpy.asarray(scorecard)

print(scorecardArray.sum() / scorecardArray.size)
