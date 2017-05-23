
# coding: utf-8

# In[ ]:

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
        ) # input to hidden inputs of hidden nodes  
        self.who = numpy.random.normal(
            0.0,
            pow(self.outputNodes, -0.5), # power
            (self.outputNodes, self.hiddenNodes)
        ) # hidden outputs to final output
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


# In[ ]:

inputNodes = 784
hiddenNodes = 200
outputNodes = 10

learningRate = 0.1

n = NeuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)


# In[ ]:

dataFile = open("mnist_train.csv", "r")


# In[ ]:

dataList = dataFile.readlines()


# In[ ]:

dataFile.close()


# In[ ]:

for record in dataList:
    allValues = record.split(',')
    inputs = numpy.asfarray(allValues[1:]) / 255.0 * 0.99 + 0.01
    targets = numpy.zeros(outputNodes) + 0.01
    targets[int(allValues[0])] = 0.99 # the first item is the label for the number
    n.train(inputs, targets)
    pass


# In[ ]:

# test
testDataFile = open("mnist_test.csv", 'r')
testDataList = testDataFile.readlines()
testDataFile.close()
allValues = testDataList[0].split(',')
print(allValues[0])


# In[ ]:

imageArray = numpy.asfarray(allValues[1:]).reshape((28, 28))
matplotlib.pyplot.imshow(imageArray, cmap='Greys', interpolation='None')


# In[ ]:

n.query(
    (numpy.asfarray(allValues[1:]))/255.0 * 0.99 + 0.01
)


# In[ ]:

# train the neutral network
# epochs is the number of times the training data set is used for training
epochs = 6

for e in range(epochs):
    # go through all records in the training data set
    for record in dataList:
        # split the record by ',' commas
        allValues = record.split(',')
        # scale and shift the inputs
        inputs = (numpy.asfarray(allValues[1:]) / 255.0 * 0.99) + 0.01
        
        # create the target output values (all 0.01, except the desired label which is 0.99) 
        targets = numpy.zeros(outputNodes) + 0.01
        # allValues[0] is the target label for this record
        targets[int(allValues[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass


# In[ ]:

# it worked!

# More testing
scorecard = []

# go through all the records in the test data set for record in testDataList:
for record in testDataList:
    allValues = record.split(',')
    correctLabel = int(allValues[0])

    inputs = (numpy.asfarray(allValues[1:]) / 255.0 * 0.99) + 0.01
    outputs = n.query(inputs)
    label = numpy.argmax(outputs)
    if (label == correctLabel):
        scorecard.append(1)
    else:
        scorecard.append(0)
    pass

scorecardArray = numpy.asarray(scorecard)

print(scorecardArray.sum() / scorecardArray.size)


# In[64]:

# import scipy.misc
# imgArray = scipy.misc.imread(imageFileName, flatten = True)
# imgData = 255.0 - imgArray.reshape(784)
# imgData = imgData/255.0 * 0.99 + 0.01


# In[65]:

# create rotated variations
# rotated anticlockwise by 10 degrees
# inputsPlus10Img = scipy.ndimage.interpolation.rotate(scaledInput.reshape(28, 28), 10, cval=0.01, reshape = False)
# rotated clockwise by 10 degrees
# inputMinus10Img = scipy.ndimage.interpolation.rotate(scaledInput.reshape(28,28), -10, cval=0.01, reshape = False)


# In[ ]:



