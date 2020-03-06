import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
warnings.filterwarnings('ignore')
df = pd.read_csv('BitcoinPrice.csv')
#print(df.head())


df.plot(kind='scatter',x='Date',y='Price')

sns.FacetGrid(df).map(plt.scatter,'Date','Price').add_legend()
plt.show()

sns.pairplot(df)
plt.show()


df.drop(['Date'],1,inplace=True)
#print(df.head())

#Prediction Start
prediction_days = 30
df['Prediction'] = df[['Price']].shift(-prediction_days)

X = np.array(df.drop(['Prediction'],1))
X = X[:len(df) - prediction_days]

#print(X.shape)

Y = np.array(df['Prediction'])
Y = Y[:-prediction_days]
#print(Y.shape)

X_Train , X_Test , Y_Train , Y_Test = train_test_split(X,Y,test_size= 0.2)

p_days_array = np.array(df.drop(['Prediction'],1))[-prediction_days:]
#print(p_days_array)


Rf = RandomForestRegressor(n_estimators=1000,random_state=1)
Rf.fit(X_Train,Y_Train)


print("Random Forest Accuracy Percentage is = ",Rf.score(X_Test,Y_Test)*100)


#print(Y_Test)
#print(Rf.predict(X_Test))

#----------------------------------------------
# from sklearn.svm import SVR
# svm = SVR()
# svm.fit(X_Train,Y_Train)
# print("Support Vector Regression Accuracy = ",svm.score(X_Test,Y_Test)*100)

#---------------------------------------------

# from sklearn.naive_bayes import GaussianNB

# bayes = GaussianNB()

# bayes.fit(X_Train,Y_Train)
# print("Naive Bayes Accuracy = ",bayes.score(X_Test,Y_Test)*100)

#----------------------------------------------

# from sklearn.linear_model import LogisticRegression

# logistic = LogisticRegression()
# logistic.fit(X_Train,Y_Train)
# print("Logistic Regression Accuracy = ",logistic.score(X_Test,Y_Test)*100)


# #----------------------------------------------


from sklearn.linear_model import LinearRegression

linear = LinearRegression()
linear.fit(X_Train,Y_Train)
print("Linear Regression Accuracy = ",linear.score(X_Test,Y_Test)*100)


#------------------------------------------------
from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor()
knn.fit(X_Train,Y_Train)
print("k Nearest Neighbours Accuracy = ",knn.score(X_Test,Y_Test)*100)



#------------------------------------------------
