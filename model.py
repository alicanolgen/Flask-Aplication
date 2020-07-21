
import pandas as pd
import numpy as np

from sklearn import preprocessing
import pickle

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

data = pd.read_csv('Social_Network_Ads.csv')

data = data.drop(['User ID'],axis=1)
# print(data.dtypes)

data["Gender"] = data["Gender"].astype('category')
labelencoder= LabelEncoder()
data['Gender'] = labelencoder.fit_transform(data['Gender'])

def normalize(dataset):
    dataNorm=((dataset-dataset.min())/(dataset.max()-dataset.min()))
    dataNorm["Gender"]=dataset["Gender"]
    return dataNorm



X = data.values[:,0:3] 
Y = data.values[:, 3] 

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)


data_clf_lr = DecisionTreeClassifier()

data_clf_lr.fit(X_train,y_train)
predicted = data_clf_lr.predict(X_test)


filename = 'finalized_model.sav'
pickle.dump(data_clf_lr, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(loaded_model.predict((X[3]).reshape(1,3)))


print("accuracy score:{}".format(accuracy_score(y_test,predicted)*100))



