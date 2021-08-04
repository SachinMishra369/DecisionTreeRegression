#IMPORTING LIBRARIES
import pandas as pd
import numpy as np


#READING DATASET
dataset=pd.read_csv("salary.csv")

#read data of dependent variable from dataset
X=dataset.iloc[:,1:2].values
#read the data of independent variable from dataset
Y=dataset.iloc[:,-1].values

#feature scaling putting the data into a range for better visualzation
# from sklearn.preprocessing import StandardScaler 
# sc=StandardScaler()
# X=sc.fit_transform(X)
# sc_y=StandardScaler()
# Y=sc_y.fit_transform(np.reshape(Y,(10,1)))

# Y=np.reshape(len(Y),(1,1))

#Performing Linear Regression

#Performing SVR model
from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor(random_state=0)
dtr.fit(X,Y)
print("Salary of employee having 6.5 year of experience is {}".format(dtr.predict([[7]])))

#visualizing data
import matplotlib.pyplot  as plt
# plt.xlabel("EXPERIENCE")
# plt.ylabel("SALARY")
# plt.scatter(X,Y,color='green')
# plt.plot(X,dtr.predict(X),color='red')
# plt.show()

#visualizing data in highier density
x_grid=np.arange(min(X),max(X),0.1)
x_grid=x_grid.reshape(len(x_grid),1)
plt.xlabel("EXPERIENCE")
plt.ylabel("SALARY")
plt.scatter(X,Y,color='green')
plt.plot(x_grid,dtr.predict(x_grid),color='red')
plt.show()




