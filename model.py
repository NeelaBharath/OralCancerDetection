import cv2
from matplotlib import pyplot as plt
import numpy as np
import pickle
import os
def get_data():
    directory = 'C:\\Users\\Ranjith\\CIT\\Adv ML\\Project\\datasets\\OralCancer'
    non_cancer,cancer = [],[]
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if filename == 'non-cancer':
            for i in os.listdir(f):
                temp = os.path.join(f, i)
                img = cv2.imread(temp,0)
                non_cancer.append(cv2.resize(img,(256,256)))
        if filename == 'cancer':
            for i in os.listdir(f):
                temp = os.path.join(f, i)
                img = cv2.imread(temp,0)           
                cancer.append(cv2.resize(img,(256,256)))
    return non_cancer,cancer

def data_for_model():
    non_cancer,cancer = get_data()
    y = np.append(np.zeros(len(non_cancer)),np.ones(len(cancer)))
    y = y.reshape(-1,1)
    X = non_cancer + cancer
    temp = X[0]
    for i in X[1:]:
        temp = np.concatenate((temp,i))
    temp = temp.reshape(len(X),256,256)
    from sklearn import utils
    temp, y = utils.shuffle(temp, y)
    return temp,y

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
temp,y = data_for_model()
X = temp.reshape(131,-1)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
svm = SVC(decision_function_shape = 'ovo' , kernel = 'poly' , degree=10 , max_iter = 100)
svm.fit(X_train,y_train)
print(accuracy_score(svm.predict(X_test),y_test))
pickle.dump(svm,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))

