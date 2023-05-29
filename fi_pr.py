import pandas as pd


data = pd.read_csv(r'C:\Users\ehab\Desktop\archive\IPL_Squad_2023_Auction_Dataset.csv')

print(data)

print(data.isnull().sum())
print(data.duplicated().sum())
data = data.dropna()
print(data.dtypes)


from sklearn import preprocessing
pr_data = preprocessing.LabelEncoder()
dtype = data.dtypes
for i in range(data.shape[1]):
    if dtype[i] == 'object':
        modleEncode = preprocessing.LabelEncoder()
        data[data.columns[i]] = modleEncode.fit_transform(data[data.columns[i]])
print('----------------------------------------------------------')
print('data after encoding : \n ' , data.dtypes)
print('----------------------------------------------------------')


scaler = preprocessing.MinMaxScaler()

scaled_data = scaler.fit_transform(data)

scaled_data = pd.DataFrame(scaled_data ,columns=data.columns)

print('----------------------------------------------------------')
print('data after scaling : \n ' , scaled_data)
print('----------------------------------------------------------')


r = scaled_data.corr()

print(' data correlation : \n ' , r)   

import seaborn as sns
import matplotlib.pyplot as plt

r = scaled_data.corr()

sns.heatmap(r , annot= True)
plt.show()




from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


x = data.drop(['Team'], axis= 1)
y = data.Team.values



X_train, X_test, y_train, y_test = train_test_split(x , y , test_size=0.3 , random_state=54)


clf = DecisionTreeClassifier()



clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)


print('Accuracy:', accuracy_score(y_test, y_pred))


clf = KNeighborsClassifier()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)


print('Accuracy:', accuracy_score(y_test, y_pred))



clf = SVC()


clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)


print('Accuracy:', accuracy_score(y_test, y_pred))


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print('R-squared:', r2)





#team names
#ehab ahmed mohamed hasan
#saeed amr saeed
#omar ahmed farag

