

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('/content/test.csv')

df.describe()
# print(df.info())

df.head()

df.tail()

df.sample(10)

df.info()

df['hours-per-week'].value_counts()

df

df.isnull().sum()

df.bfill(inplace=True)



df.isnull().sum()

categorical_columns = ['workclass', 'education', 'marital-status', 'occupation',
                       'relationship', 'race', 'gender', 'native-country']

df_encoded = pd.get_dummies(df, columns=categorical_columns)
df_encoded= df_encoded.astype(int)
print(df_encoded.head())



sns.countplot(df['age'])

df['gender'].value_counts().plot(kind='pie',autopct='%.2f')

import matplotlib.pyplot as plt
plt.hist(df['age'],bins=5)

sns.distplot(df['age'])

sns.boxplot(df['age'])

df['age'].min()

df['age'].max()

df['age'].mean()

df['age'].skew()

df.corr(numeric_only = True)
sns.heatmap(df.corr(numeric_only = True), annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True)
plt.show()

target_name='hours-per-week'


y = df_encoded[target_name]


x = df_encoded.drop(target_name, axis=1)

x.head()

y.head()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

x_train

y_train

x_test

y_test

from sklearn. ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train, y_train)

print("Train Accuracy of Random Forest Algorithm ", rf.score(x_train,y_train)*100)
print("Test Accuracy score of Random Forest Algorithm ", rf.score(x_test,y_test)*100)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)

print("Train Accuracy of Decision Tree Algorithm ", dt.score(x_train,y_train)*100)
print("Test Accuracy score of Decision Tree Algorithm ", dt.score(x_test,y_test)*100)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='liblinear', multi_class='ovr')
lr.fit(x_train, y_train)

print("Train Accuracy of Logistic Regression ", lr.score(x_train,y_train)*100)
print("Test Accuracy score of Logistic Regression ", lr.score(x_test,y_test)*100)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)

print("Train Accuracy of KNN Algorithm ", knn.score(x_train,y_train)*100)
print("Test Accuracy score of KNN Algorithm ", knn.score(x_test,y_test)*100)

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train, y_train)

print("Train Accuracy of Naive-Bayes Algorithm ", nb.score(x_train,y_train)*100)
print("Test Accuracy score of Naive-Bayes Algorithm ", nb.score(x_test,y_test)*100)

rf_pred = rf.predict(x_test)

dt_pred = dt.predict(x_test)

lr_pred = lr.predict(x_test)

knn_pred = knn.predict(x_test)

nb_pred = nb.predict(x_test)

import numpy as np

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, roc_curve

cm = confusion_matrix(y_test, rf_pred)


print('TN {}'.format(cm[0, 0]))
print('FP {}'.format(cm[0, 1]))
print('FN {}'.format(cm[1, 0]))
print('TP {}'.format(cm[1, 1]))

# Accuracy calculation
accuracy = np.divide(np.sum([cm[0, 0], cm[1, 1]]), np.sum(cm)) * 100
print('Accuracy rate {}'.format(accuracy))

# Misclassification rate calculation
misclassification_rate = np.divide(np.sum([cm[0, 1], cm[1, 0]]), np.sum(cm)) * 100
print('Misclassification rate {}'.format(misclassification_rate))

sns.heatmap(cm, annot=True, fmt="d")

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
cm = confusion_matrix(y_test, dt_pred)

print('TN {}'.format(cm[0,0]))
print('FP {}'.format(cm[0,1]))
print('FN {}'.format(cm[1,0]))
print('TP {}'.format(cm[1,1]))
print('Accuracy rate {}'.format(np.divide(np.sum([cm[0,0], cm[1,1]]), np.sum(cm))*100))
print('Misclassification rate {}'.format(np.divide(np.sum([cm[0,1], cm[1,0]]), np.sum(cm))*100))

