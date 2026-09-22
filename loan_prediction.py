import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('loan.csv')

data.drop(["Loan_ID"], axis= 1, inplace= True)

plt.figure(figsize=(8, 4))
ax = sns.countplot(x="Gender", data=data)
for container in ax.containers:
    ax.bar_label(container)
plt.show()

label_encoder = preprocessing.LabelEncoder()
obj = (data.dtypes == "str")
for col in list(obj[obj].index):
    data[col] = label_encoder.fit_transform(data[col])

plt.figure(figsize=(11,12))
sns.heatmap(data.corr(), annot=True)
plt.show()

for col in data.columns:
    data[col] = data[col].fillna(data[col].mean())

x = data.drop(['Loan_Status'], axis=1)
y = data['Loan_Status']

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)

lc_model = LogisticRegression(max_iter= 10000)
knn_model = KNeighborsClassifier(n_neighbors=3)

for clf in (lc_model, knn_model):
    clf.fit(X_train, Y_train)

    y_pred = clf.predict(X_test)
    print('Acuuracy score of', clf, '=', metrics.accuracy_score(Y_test,y_pred) * 100, '%')
