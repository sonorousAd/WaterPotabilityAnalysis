import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt



data = pd.read_csv("./water_potability.csv")
#I will round the values to 3 decimal places
data = data.round(3)
#and NAN values are dropped in order to clean data
data = data.dropna()
for cols in data.columns:
    print(cols)

# Separate data
# X will contain all the input variables
X= data.drop('Potability',axis=1)
# Y contais the output
Y = data.Potability
scale = StandardScaler()
X =scale.fit_transform(X)
# Create train and test sets
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size= 0.2, random_state= 10)

# MODELS

lr = LogisticRegression()
lr.fit(xtrain, ytrain)
predicted = lr.predict(xtest)
print( "score for Logistic Regression", lr.score(xtest,ytest) )


dt = DecisionTreeClassifier()
dt.fit(xtrain,ytrain)
prediteddt = dt.predict(xtest)
print( "score for Decision Tree", dt.score(xtest,ytest) )

kn = KNeighborsClassifier()
kn.fit(xtrain, ytrain)
prediteddt = kn.predict(xtest)
print( "score for KNN", kn.score(xtest,ytest) )

# After Plotting all the variables against 'Potability' its easy to see why its hard to accurately
# predict the waters Potability based on the parameters.