import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

data = pd.read_csv("ML\Salary_dataset.csv")

print(data.head())

data = data[['YearsExperience','Salary']]

plt.scatter(data['YearsExperience'],data['Salary'])
plt.show()

X = data[['YearsExperience']]
y = data['Salary']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

X_train = X_train.to_numpy().reshape(-1,1)
X_test = X_test.to_numpy().reshape(-1,1)

model = LinearRegression()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

# 6. Evaluate the model
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("Coefficient of determination (R²): %.2f" % r2_score(y_test, y_pred))