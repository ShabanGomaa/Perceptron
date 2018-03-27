from scipy import misc
import numpy as np
#for retrieving string related objects
import string as str
import glob, os
import plotly.plotly as py
import matplotlib.pyplot as plt

trainingData = []
labels = []

'''
Perceptron class
'''
class Perceptron(object):
    # n_iter can set to 20 for enhance performance
    def __init__(self, eta=0.05, n_iter=30):
        self.eta = eta
        self.n_iter = n_iter

    '''
    charData are vectors contain all char data
    charLabels are alphabetic labels
    character is character itself
    '''
    def trainPerceptron(self, charData, charLabels, character):
        self.w_ = np.zeros(np.shape(trainingData)[1])
        self.w_[0] = 1 # bias
        self.errors_ = []

        localError = 0
        labels = self.switchLabels(character, charLabels)
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(charData, labels):
                localError = self.predict(xi)
                update = self.eta * (target - localError)
                self.w_ += update * xi
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self.w_

    '''
    Convert alphabitic labels to numeric labels,
    for ex. when the char is a, the charlabels array will contain 1 at positions that represent a character
    '''
    def switchLabels(self, character, charLabels):
        res = []
        i = 0
        for char in charLabels:
            if(character == char):
                res.append(1)
            else:
                res.append(-1)
            i = i + 1
        return  res

    '''
    Multiply X vector by weight vector
    '''
    def multiply(self, X):
        return np.dot(X, self.w_)

    def predict(self, X):
        if (self.multiply(X) >= 0.0):
            return 1
        else:
            return -1

    '''
    Apply perceptron algorithm on test data
    '''
    def testPerceptron(self, allWs, labels, charTest):
        res = []
        for item in allWs:
            res.append((np.dot(np.transpose(item), charTest)))
        import operator
        index, value = max(enumerate(res), key=operator.itemgetter(1))
        return labels[index]

'''
Utilities class
'''
class Utilities:
    def __init__(self, allWs = []):
        self.allWs = allWs

    '''
    Load training data
    '''
    def loadTrainingData(self):
        for infile in glob.glob("Images/Train/*.jpg"):
            img = misc.imread(infile)
            img = img.reshape(144,)
            img = np.append(img, 1)
            trainingData.append(img)
            file, ext = os.path.splitext(infile)
            labels.append(file[-2:-1])

    def execute(self):
        Utilities().loadTrainingData()
        for char in str.ascii_lowercase:
            WofCharacter = Perceptron().trainPerceptron(trainingData, labels, char)
            self.allWs.append((char, WofCharacter))
        return self.allWs

    '''
    Load test data and apply perceptron on the loaded data
    '''
    def applyTest(self, allWs):
        decisons = []
        for infile in glob.glob("Images/Test/*.jpg"):
            img = misc.imread(infile)
            img = img.reshape(144, )
            img = np.append(img, 1)
            file, ext = os.path.splitext(infile)
            expected = Perceptron().testPerceptron(allWs[:, 1], allWs[:, 0], img)
            decisons.append((file[-2:-1], expected))
        return decisons


allWs = Utilities().execute()
allWs = np.array(allWs, dtype=object)
# decisions contains array
# @index 0, the image from test data
# @index 1, the result of perceptron data
decisions = Utilities().applyTest(allWs)
decisions = np.array(decisions, dtype=object)

occurrenceArray = []
for char in str.ascii_lowercase:
    occurence = len(decisions[:,1][decisions[:,1] == char])
    occurrenceArray.append((char, occurence))
occurrenceArray = np.array(occurrenceArray, dtype=object)

print("decisions are: ")
print(decisions)

# draw graph
xAxis = list(str.ascii_lowercase)
yAxis = occurrenceArray[:,1]
y_pos = np.arange(len(xAxis))
plt.bar(y_pos, yAxis, align='center', alpha=0.5)
plt.xticks(y_pos, xAxis)
plt.ylabel('Occurrence from test data')
plt.title('Perceptron Classifier')
plt.savefig('./Accuracy.jpg', dpi=300)
plt.show()