import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
import seaborn as sns


# KNNClassifier
class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.k_Neighbors = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
        return True

    def predict(self, x: np.ndarray):
        self.predictions = []
        for test in x:
            distance_matrix = []
            for neighbors in range(len(self.X)):
                euclidian_distance = np.sqrt(
                    np.sum(self.X[neighbors, :] - test)**2)
                distance_matrix.append(euclidian_distance)
            distance_matrix = np.array(distance_matrix)
            self.k_Neighbors = np.argsort(distance_matrix)[:self.k]
            votes = self.y[self.k_Neighbors]
            vote_counts = {}
            for label in votes:
                if label[0] in vote_counts.keys():
                    vote_counts[label[0]] += 1
                else:
                    vote_counts[label[0]] = 1
            majority = sorted(vote_counts.items(),
                              key=lambda byVote: byVote[1])[-1][0]
            self.predictions.append(majority)
        return np.array(self.predictions).reshape(len(self.predictions), 1)


# Classification Model Evaluators
def Jaccard_index(Y, yhat):
    total = len(Y)
    if not isinstance(Y, np.ndarray):
        Y = Y.values
    if not isinstance(yhat, np.ndarray):
        yhat = yhat.values
    correct_prediction = sum(
        [1 if Y[index][0] == yhat[index][0] else 0 for index in range(total)])
    jaccard_index_score = (
        correct_prediction/((total+total) - correct_prediction))
    return round(jaccard_index_score, 4)


def F1_Score(Y, yhat):
    total = len(Y)
    f1_score = []
    if not isinstance(Y, np.ndarray):
        Y = Y.values
    if not isinstance(yhat, np.ndarray):
        yhat = yhat.values
    TP = 0; FN = 0; FP = 0; TN = 0
    for index in range(total):
        if Y[index][0] == yhat[index][0]:
            if Y[index][0] == 1:
                TP += 1
            else:
                TN += 1
        else:
            if yhat[index][0] == 1:
                FN += 1
            else:
                FP += 1
    precision_1 = TP/(TP+FP)
    recall_1 = TP/(TP+FN)
    f1_score.append((2*(precision_1*recall_1))/(precision_1+recall_1))
    precision_0 = TN/(TN+FN)
    recall_0 = TN/(TN+FP)
    f1_score.append((2*(precision_0*recall_0))/(precision_0+recall_0))
    return round((sum(f1_score)/len(f1_score)), 4)


# Data Pre-processing
train_data = pd.read_csv(
    r'C:\PYTHON\machineLearning-from-scratch\classification\train.csv')
test_data = pd.read_csv(
    r'C:\PYTHON\machineLearning-from-scratch\classification\test.csv')

split = int(train_data.shape[0] * 0.8)

train_X = train_data[['glucose_concentration', 'blood_pressure']]
trainX = train_X[:split]
testX = train_X[split:]

train_Y = train_data[['diabetes']]
trainY = train_Y[:split]
testY = train_Y[split:]

# Model Building
clf = KNNClassifier(k=5)
clf.fit(trainX.values, trainY.values)
yhat = clf.predict(testX.values)

print(Jaccard_index(testY, yhat))
print(F1_Score(testY, yhat))
