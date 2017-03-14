#Import
import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

#Import machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier 

#get titanic's train & test csv files as a DataFrame
titanic_df = pd.read_csv("D:\workspace\kaggle\Titanic\Train.csv")
test_df = pd.read_csv("D:\workspace\kaggle\Titanic\Test.csv")
test_df_Y = pd.read_csv("D:\workspace\kaggle\Titanic\gender_submission.csv")
'''
#preview the data and the infomation
titanic_df.info()
print('----------------------------------------')
test_df.info()
#print titanic_df.describe()
print titanic_df.columns.values
print titanic_df.head()

#Analyze by pivoting categorical features
print titanic_df[['Pclass','Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print titanic_df[['Sex','Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print titanic_df[['SibSp','Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print titanic_df[['Parch','Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print titanic_df[['Embarked','Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#Analyze by visualizing data for numerical features
g = sns.FacetGrid(titanic_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)
plt.show(g)
'''
#drop unnecessary columns, these columns won't be useful in analysis and prediction
titanic_df = titanic_df.drop(['PassengerId','Name','Ticket','Cabin'], axis=1) #axis=0 row; axis=1 column
test_df = test_df.drop(['Name','Ticket','Cabin'], axis=1)
#titanic_df.info()
#print('----------------------------------------')
#test_df.info()

#Sex
titanic_df = titanic_df.join(pd.get_dummies(titanic_df['Sex']))
test_df = test_df.join(pd.get_dummies(test_df['Sex']))
titanic_df.drop(['Sex'], axis=1, inplace=True)
test_df.drop(['Sex'], axis=1, inplace=True)

#Age
titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].mean()).astype(int)
test_df['Age'] = test_df['Age'].fillna(test_df['Age'].mean()).astype(int)

#titanic_df['AgeBand'] = pd.cut(titanic_df['Age'], 5)
#print titanic_df[['AgeBand','Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
titanic_df['Age'].loc[titanic_df['Age'] <= 16] = 0
titanic_df['Age'].loc[(titanic_df['Age'] > 16) & (titanic_df['Age'] <= 32)] = 1
titanic_df['Age'].loc[(titanic_df['Age'] > 32) & (titanic_df['Age'] <= 48)] = 2
titanic_df['Age'].loc[(titanic_df['Age'] > 48) & (titanic_df['Age'] <= 64)] = 3
titanic_df['Age'].loc[titanic_df['Age'] > 64] = 4
#titanic_df.drop(['AgeBand'], axis=1, inplace=True)

test_df['Age'].loc[test_df['Age'] <= 16] = 0
test_df['Age'].loc[(test_df['Age'] > 16) & (test_df['Age'] <= 32)] = 1
test_df['Age'].loc[(test_df['Age'] > 32) & (test_df['Age'] <= 48)] = 2
test_df['Age'].loc[(test_df['Age'] > 48) & (test_df['Age'] <= 64)] = 3
test_df['Age'].loc[test_df['Age'] > 64] = 4

#FamilySize
titanic_df['FamilySize'] = titanic_df['SibSp'] + titanic_df['Parch']
test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch']
titanic_df.drop(['SibSp','Parch'], axis=1, inplace=True)
test_df.drop(['SibSp','Parch'], axis=1, inplace=True)

#Fare
#There is a missing value in test_df
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].mean()).astype(int)
titanic_df['Fare'] = titanic_df['Fare'].astype(int)

#titanic_df['FareBand'] = pd.qcut(titanic_df['Fare'], 4)
#print titanic_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
titanic_df['Fare'].loc[titanic_df['Fare'] <= 7] = 0
titanic_df['Fare'].loc[(titanic_df['Fare'] > 7) & (titanic_df['Fare'] <= 14)] = 1
titanic_df['Fare'].loc[(titanic_df['Fare'] > 14) & (titanic_df['Fare'] <= 31)] = 2
titanic_df['Fare'].loc[titanic_df['Fare'] > 31] = 3
#titanic_df.drop(['FareBand'], axis=1)

test_df['Fare'].loc[test_df['Fare'] <= 7] = 0
test_df['Fare'].loc[(test_df['Fare'] > 7) & (test_df['Fare'] <= 14)] = 1
test_df['Fare'].loc[(test_df['Fare'] > 14) & (test_df['Fare'] <= 31)] = 2
test_df['Fare'].loc[test_df['Fare'] > 31] = 3

#Embarked
#only in titanic_df,fill the missing value with the most occurred value, which is 's'
titanic_df['Embarked'] = titanic_df['Embarked'].fillna('S')
#use get_dummies to achieve one-hot coding
titanic_df = titanic_df.join(pd.get_dummies(titanic_df['Embarked']))
test_df = test_df.join(pd.get_dummies(test_df['Embarked']))
titanic_df.drop(['Embarked'], axis=1, inplace=True)
test_df.drop(['Embarked'], axis=1, inplace=True)

#define training and testing sets
X_train = titanic_df.drop("Survived", axis=1)
Y_train = titanic_df['Survived']
X_test = test_df.drop("PassengerId", axis=1).copy()
Y_test = test_df_Y['Survived']
'''
#Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = logreg.score(X_train, Y_train)

#Get Correlation Coefficient for each feature using Logistic Regression
coeff_df = DataFrame(titanic_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Coefficient Estimate"] = pd.Series(logreg.coef_[0])
print coeff_df

#Support Vector Machines
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = svc.score(X_train, Y_train)
'''
#Random Forests
random_forests = RandomForestClassifier(n_estimators=100)
random_forests.fit(X_train, Y_train)
Y_pred = random_forests.predict(X_test)
acc_random_forest = random_forests.score(X_train, Y_train)
'''
#KNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train,Y_train)
Y_pred = knn.predict(X_test)
acc_knn = knn.score(X_train, Y_train)

#Gaussian Native Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = gaussian.score(X_train, Y_train)

# Perceptron
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = perceptron.score(X_train, Y_train)

# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = sgd.score(X_train, Y_train)

# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = linear_svc.score(X_train, Y_train)

# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = decision_tree.score(X_train, Y_train)

models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
print models.sort_values(by='Score', ascending=False)
'''
submission = pd.DataFrame({
	"PassengerId": test_df["PassengerId"],
	"Survived": Y_pred
	})
submission.to_csv('titanic.csv', index=False)
