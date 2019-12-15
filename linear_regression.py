import pandas as pd
import numpy as np
from sklearn.model_selection  import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
#(1)Load Data
data_path="Advertising.csv"
data = pd.read_csv(data_path,index_col=0)

feature_names = ["TV","radio","newspaper"]

X=data[feature_names]
y=data.sales

#(2)Split Data for Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=1)

linreg = LinearRegression()

#(3)Train Model
linreg.fit(X_train,y_train)

#(4)Test
y_pred=linreg.predict(X_test)

#RSME of prediction (Evaluation metric)
print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
