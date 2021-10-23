import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

dataset = pd.read_csv('D:\\Ankush Files\\Visual Studio Code\\python\\HousePriceDataset.csv')



x = dataset[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms','Avg. Area Number of Bedrooms', 'Area Population']]
y = dataset['Price']

from sklearn.model_selection import train_test_split
X_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 101)

from sklearn.linear_model import LinearRegression 
ml = LinearRegression()
ml.fit(X_train, y_train)


y_pred = ml.predict(x_test)





#from sklearn.metrics import r2_score
#perc = r2_score(y_test, y_pred)



a = ml.predict([[65510.581803666224,5.992305307333977,6.792336104424982,4.07,46501.28380314165]])



pickle.dump(ml, open('HousePricePrediction.pkl', 'wb'))