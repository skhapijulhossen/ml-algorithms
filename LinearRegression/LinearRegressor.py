import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Loading Fuel Consumption Data
data = pd.read_csv(r"C:\PYTHON\machineLearning-from-scratch\regression\FuelConsumption.csv")
print(data.head(10))
print(data.info())
split = int(data.shape[0]*0.8)
# Spliting Data into Train Test Set
x_train = data[["ENGINESIZE"]][:split].values
y_train = data[["CO2EMISSIONS"]][:split].values
x_test = data[["ENGINESIZE"]][split:].values
y_test = data[["CO2EMISSIONS"]][split:].values


# Regression Model From Scratch
class LinearRegressor:
    def __init__(self):
        self.intercept = 0
        self.coefficient = 0
        print("Linear Regressor Initialized!")

    def fit(self, x, y):
        xbar = np.mean(x)
        ybar = np.mean(y)
        self.coefficient = int(
            np.sum(((x-xbar) * (y - ybar)))/np.sum(((x-xbar)**2)))
        self.intercept = int(ybar - (xbar * self.coefficient))

    def predict(self, x):
        yhat = np.array(self.intercept +
                        (x * self.coefficient), dtype=np.int64)
        return yhat


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
model = LinearRegressor()
model.fit(x_train, y_train)


# K-Fold Cross Validation
MAe = [] 
MSe = []
R2 =[]
step = int(data.shape[0] * 0.2)
low, up = 0,step
for Fold in range(5):
    if low == 0:
        nx_train = data[["ENGINESIZE"]][up:].values
        ny_train = data[["CO2EMISSIONS"]][up:].values
    elif up == data.shape[0]:
        nx_train = data[["ENGINESIZE"]][:low].values
        ny_train = data[["CO2EMISSIONS"]][:low].values
    else:
        train_indexe1 = list(range(0,low))
        train_indexe2 = list(range(up,data.shape[0]-1))
        train_indexes = train_indexe1+train_indexe2

    nx_test = data[["ENGINESIZE"]].iloc[list(range(low,up))].values
    ny_test = data[["CO2EMISSIONS"]].iloc[list(range(low,up))].values

    model = LinearRegressor()
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

    # Plotting Model
    plt.scatter(nx_test, ny_test)
    plt.plot(nx_test, ny_pred)
    plt.show()


print(f">>>>>>>>>>>>>>>\n Average MAE {sum(MAe)/len(MAe)} \n Averagr MSE {sum(MSe)/len(MSe)} \n Average R2 {sum(R2)/len(R2)} ")
