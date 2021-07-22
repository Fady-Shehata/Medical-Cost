import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score



data= pd.read_csv('Medical_Cost.csv')

def bmi_category(bmi):
    if bmi < 18.5:
        return 'under-weight'
    elif bmi >= 18.5 and bmi <= 24.9:
        return 'normal-weight'
    elif bmi >= 24 and bmi <= 29.9:
        return 'over-weight'
    elif bmi > 30.0:
        return "obese"


def age_category(age):
    age_dict = {
        0: '0-9',
        1: '10-19',
        2: '20-29',
        3: '30-39',
        4: '40-49',
        5: '50-59',
        6: '60-69',
        7: '70-79',
        8: '80-89',
        9: '90-99',
        10: '100-200'
    }
    return age_dict[age // 10]


data['cbmi'] = data['bmi'].apply(lambda x: "none")
data['cage'] = data['age'].apply(lambda x: "none")

for idx, row in data.iterrows():
    data.at[idx, 'cage'] = age_category(row['age'])
    data.at[idx, 'cbmi'] = bmi_category(row['bmi'])

target = data['charges']
features = data.drop(['age', 'bmi', 'charges'], axis=1)

output = pd.DataFrame(index=features.index)

for col, col_data in features.iteritems():
    if object == col_data.dtype:
        col_data = col_data.replace(['yes', 'no'], [1, 0])

    if object == col_data.dtype:
        col_data = pd.get_dummies(col_data, prefix=col)
    output = output.join(col_data)

features = output
X_train, X_test, y_train, y_test = train_test_split(features,target, test_size=0.20)

def train_predict_model(clf, X_train, y_train, X_test, y_test):

    print ("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train)))
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    print ("R^2 score for training set: {:4f}".format(r2_score(y_train.values, y_pred_train)))
    y_pred_Test = clf.predict(X_test)
    print("R^2 score for testing set: {:4f}".format(r2_score(y_test.values, y_pred_Test)))
    print("")
    print("Mean square error for training set: {:4f}".format(mean_squared_error(y_train.values, y_pred_train)))
    print("Mean square error for testing set: {:4f}".format(mean_squared_error(y_test.values, y_pred_Test)))
    print("")
    print("Mean absolute error for training set: {:4f}".format(mean_absolute_error(y_train.values, y_pred_train)))
    print("Mean absolute error for testing set: {:4f}".format(mean_absolute_error(y_test.values, y_pred_Test)))
   


clf_a =DecisionTreeRegressor(random_state=1)
clf_b = SVR()
clf_c = KNeighborsRegressor()

for clf in (clf_a, clf_b, clf_c):
    for size in (400,800):
        train_predict_model(clf, X_train[:size], y_train[:size], X_test, y_test)
        print('-'* 80)
    print ('+'*80)


