from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.neighbors.nearest_centroid import NearestCentroid
path = "db_world/dbworld_bodies_stemmed.csv"
df = pd.read_csv(path, header = 0)
columns = list(df.columns.values)
df = df.values
words = df[:,:-1]
labels = df[:,-1]
X_train, X_test, Y_train, Y_test = train_test_split(words, labels, test_size=0.2,random_state=50)

from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib import style
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
style.use('ggplot')

# Rocchio Algorithm
clf = NearestCentroid()
clf.fit(X_train,Y_train)
predict = clf.predict(X_test)
accuracy = accuracy_score(Y_test, predict)
print('\nAccuracy of Rocchio:\n')
print (accuracy)
conf_mat = confusion_matrix(Y_test, predict)
print('\nConfusion Matrix: \n',conf_mat)
plt.matshow(conf_mat)
plt.title('Confusion Matrix for test Data\t')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
# Naive Bayes
clf_1 = GaussianNB()
clf_1.fit(X_train, Y_train)
predict_1 = clf_1.predict(X_test)
accuracy_1 = accuracy_score(Y_test, predict_1)
print('\nAccuracy of Naive Bayes:\n')
print (accuracy_1)
conf_mat_DB1 = confusion_matrix(Y_test, predict_1)
print('\nConfusion Matrix: \n',conf_mat_DB1)
plt.matshow(conf_mat_DB1)
plt.title('Confusion Matrix for test Data\t')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
# K nearest neigbor 
clf_2 = KNeighborsClassifier(n_neighbors=3,algorithm='auto',n_jobs=-1)
clf_2.fit(X_train, Y_train)
predict_2 = clf_2.predict(X_test)
accuracy_2 = accuracy_score(Y_test, predict_2)
print('\nAccuracy of KNN:\n')
print (accuracy_2)
conf_mat_DB2 = confusion_matrix(Y_test, predict_2)
print('\nConfusion Matrix: \n',conf_mat_DB2)
plt.matshow(conf_mat_DB2)
plt.title('Confusion Matrix for test Data\t')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
print ("Accuracy Rocchio Algorithm ")
print (clf.score(X_test, Y_test))
print ("Accuracy Naive Bayes ")
print (clf_1.score(X_test, Y_test))
print ("Accuracy KNN ")
print (clf_2.score(X_test, Y_test))

