
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import mode



# KNN Regressor From Scratch
class KNNRegressor:
    def __init__(self, k=5):
        self.k = k
        self.neighbors = None
    

    def fit(self, X,y):
        self.X = X
        self.y = y
        return True
    
    def predict(self, X_test):
        self.predictions = []
        for test in X_test:
            distance_matrix = []
            for point in range(len(self.X)):
                try:
                    distance = np.sqrt(np.sum((self.X[point, :] - test)**2 ))
                    distance_matrix.append(distance)
                except Exception as e:
                    return e
            
            distance_matrix = np.array(distance_matrix)
            self.neighbors = np.argsort(distance_matrix)[:self.k]
            votes = self.y[self.neighbors]
            self.predictions.append(int(sum(votes)/len(votes)))
        
        return np.array(self.predictions).reshape(len(self.predictions), 1)


# Data Preprocessing
data = pd.read_csv(r"C:\PYTHON\machineLearning-with-sklearn\regression\FuelConsumption.csv")
print(data.info())


#Co-relation
corr_ = data.corr()
sns.heatmap(corr_, annot=True)

dataFrame = data[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
corr_ = dataFrame.corr()
sns.heatmap(corr_, annot=True)

split = int(data.shape[0]*0.8)
# Spliting Data into Train Test Set
x_train = data[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY']][:split].values
y_train = data[["CO2EMISSIONS"]][:split].values
x_test = data[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY']][split:].values
y_test = data[["CO2EMISSIONS"]][split:].values


# Model Evaluators
def mean_absolute_error(actual_y, yhat):
    mae = (1/len(yhat)) * np.sum(np.abs(actual_y - yhat))
    return mae


def root_mean_squared_error(actual_y, yhat):
    mae = (1/len(yhat)) * np.sum(np.square(np.abs(actual_y - yhat)))
    return np.sqrt(mae)


def r2_score(actual_y, yhat):
    r2 = 1 - (np.sum(np.square(actual_y - yhat)) /
              np.sum(np.square(actual_y - np.mean(actual_y))))
    return r2


# Initializing Model and Training MOdel
model = KNNRegressor(k=3)
model.fit(x_train, y_train)

#Prediction
ypred = model.predict(x_test)


# Model Evaluation
print("Mean Absolute Error : {}".format(mean_absolute_error(y_test, ypred)))
print("Root Mean Squared Error : {}".format(root_mean_squared_error(y_test, ypred)))
print("R2 Score : {}".format(r2_score(y_test, ypred)))




# K-Fold Cross Validation
MAe = [] 
MSe = []
R2 =[]
step = int(data.shape[0] * 0.2)
low, up = 0,step
for Fold in range(5):
    if low == 0:
        nx_train = data[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY']][up:].values
        ny_train = data[["CO2EMISSIONS"]][up:].values
    elif up == data.shape[0]:
        nx_train = data[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY']][:low].values
        ny_train = data[["CO2EMISSIONS"]][:low].values
    else:
        train_indexe1 = list(range(0,low))
        train_indexe2 = list(range(up,data.shape[0]-1))
        train_indexes = train_indexe1+train_indexe2

    nx_test = data[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY']].iloc[list(range(low,up))].values
    ny_test = data[["CO2EMISSIONS"]].iloc[list(range(low,up))].values

    model = KNNRegressor(k=3)
    model.fit(nx_train, ny_train)
    ny_pred = model.predict(nx_test)

    mae = mean_absolute_error(ny_test, ny_pred)
    mse = root_mean_squared_error(ny_test, ny_pred)
    r2 = r2_score(ny_test, ny_pred)
    MAe.append(mae); MSe.append(mse); R2.append(r2)
    print(">>>>>>>\n MAE: ", mae)
    print("MSE: ",mse )
    print("(Higher R2 Value Means Better Fit) R2: ", r2)
    low , up = up, up+step
    print(nx_test[0][0])
    # # Plotting Model
    # plt.scatter(nx_test[0], ny_test)
    # plt.plot(nx_test[0,:], ny_pred)
    # plt.show()


print(f">>>>>>>>>>>>>>>\n Average MAE {sum(MAe)/len(MAe)} \n Averagr MSE {sum(MSe)/len(MSe)} \n Average R2 {sum(R2)/len(R2)} ")
