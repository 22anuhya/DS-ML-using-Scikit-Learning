"""
Created on Tue May 18 10:15:17 2021

@author: Anuhya
"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
game_data=pd.read_csv("video_game_data.csv")
#split data into i/p and o/p objects
X=game_data.drop(["completion_time"],axis=1)#input
y=game_data["completion_time"]#expected o/p
#split data into training and test sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#intantiate model obj
regressor=RandomForestRegressor(random_state=42)
#train model
regressor.fit(X_train,y_train)
#acess accuracy
y_pred=regressor.predict(X_test)
prediction_comparision=pd.DataFrame({"actual":y_test,
                                     "prediction": y_pred})
#prediction score
print("SCORE FOR PREDICTION IS",r2_score(y_test,y_pred))