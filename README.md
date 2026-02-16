# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Collect the dataset
Obtain the employee dataset containing independent variables (features such as age, experience, education, job role, department, years at company, etc.) and the dependent variable Y (Employee Salary).

2.Identify variables
Let X be the set of independent variables (employee attributes).
Let Y be the dependent variable representing the salary of the employee.

3.Preprocess the data
Handle missing values if any.
Encode categorical variables into numerical form.
Split the dataset into training data and testing data.

4.Select the splitting criterion
Choose a measure to evaluate splits at each node such as Mean Squared Error (MSE) or Mean Absolute Error (MAE).
Mean Squared Error formula:

<img width="359" height="99" alt="image" src="https://github.com/user-attachments/assets/1bc34c22-bc12-4be6-9efa-979e1a7548d6" />


5.Find the best attribute for splitting
Calculate the reduction in error (variance reduction) for each feature.
Select the attribute that provides the minimum MSE.

6.Create the decision tree
Make the selected attribute the decision node.
Split the dataset into subsets based on attribute values.
Repeat steps 4–6 recursively for each subset.

7.Apply stopping conditions
Stop splitting when:
Maximum tree depth is reached, or
Minimum number of samples per node is achieved, or
No further reduction in error is possible.

8.Assign predicted values
Assign the mean salary value to the leaf nodes as the predicted output.

9.Train the model
Construct the decision tree regressor using the training dataset.

10.Test the model
Use the testing dataset to predict the salary of the employee.

11.Evaluate model performance
Measure performance using metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-Squared score.

## Program:
```python
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: SUDARSAN.A
RegisterNumber:  212224220111
*/

import pandas as pd
# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
# Import plot_tree from sklearn.tree
from sklearn.tree import plot_tree
data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
x.head()
y=data[["Salary"]]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
plt.figure(figsize=(20, 8))
plot_tree(dt, feature_names=x.columns, filled=True)
plt.show()
```

## Output:

### Data.Head():
<img width="1239" height="520" alt="Screenshot 2026-02-16 084528" src="https://github.com/user-attachments/assets/843b0bde-c8f8-4419-bd26-0e3b02641270" />

### Data.info():
<img width="290" height="57" alt="Screenshot 2026-02-16 084406" src="https://github.com/user-attachments/assets/f27c379c-2de4-4b91-be4a-446aa24faecd" />

### isnull() and sum():
<img width="355" height="118" alt="Screenshot 2026-02-16 084353" src="https://github.com/user-attachments/assets/0ee47d72-ffc0-4ec0-a76c-111f231831c8" />

### Data.Head() for salary:
<img width="420" height="295" alt="image" src="https://github.com/user-attachments/assets/522d4484-e613-4d57-b348-4a4ed9cca7c6" />

### MSE Value:
<img width="164" height="47" alt="image" src="https://github.com/user-attachments/assets/68ce3125-c9d9-4fed-b554-8d3d73a244d5" />

### R2 Value:
<img width="270" height="44" alt="image" src="https://github.com/user-attachments/assets/ff936337-09a6-4645-bfef-7c3687158a56" />

### Data Prediction:
<img width="537" height="269" alt="Screenshot 2026-02-16 084331" src="https://github.com/user-attachments/assets/59a92d1d-3ec7-4bd3-b1b9-12a3c3897e39" />

### Decision Tree:
<img width="622" height="315" alt="Screenshot 2026-02-16 084310" src="https://github.com/user-attachments/assets/90363cc8-efad-48da-b72c-05d340f9bb60" />


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
