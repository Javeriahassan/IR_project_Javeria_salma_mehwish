import glob
import sys
import os
import errno
import pandas as pd

path = "SentenceCorpus/labeled_articles"
list_ = []
files = glob.glob(path + "/*.txt")
col = ['Labels', 'Words']
for name in files:
    df = pd.read_csv(name, sep ="\t", names = col,header = None)
    df = df[~df['Labels'].isin(['### abstract ###'])]
    df = df[~df['Labels'].isin(['### introduction ###'])]
    df = df[~df['Words'].isin(['NaN'])]
    list_.append(df)
# Loading Data into DataFrame
frame = pd.DataFrame()
frame = pd.concat(list_)

print(frame)

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
stop = stopwords.words('english')
frame.columns = frame.columns.str.strip()
frame['Words'] = frame['Words'].str.replace('[^\w\s\t]','')
#Word tokenize
frame["Words"] = frame["Words"].fillna("").map(word_tokenize)
print (frame['Words'])
frame['Words'] = frame['Words'].apply(lambda x: [item for item in x if item not in stop])
frame['Words']=[" ".join(Words) for Words in frame['Words'].values]

from sklearn.model_selection import train_test_split
#train test split dividing train test in 80% and 20%
X_train, X_test, Y_train, Y_test = train_test_split(frame['Words'], frame['Labels'], test_size=0.2,random_state=50)
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
vectorizer = CountVectorizer()
vectorizer.fit(X_train)
train_mat = vectorizer.transform(X_train)
print (train_mat.shape)
test_mat = vectorizer.transform(X_test)
print (test_mat.shape)
#Determining tf-idf
tfidf = TfidfTransformer()
tfidf.fit(train_mat)
train_tfmat = tfidf.transform(train_mat)
print (train_tfmat.shape)
test_tfmat = tfidf.transform(test_mat)
print (test_tfmat.shape)
#Naive Bayes
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X = train_tfmat.toarray(), y = Y_train)
predict = clf.predict(test_tfmat.toarray())

from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib import style
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
style.use('ggplot')

accuracy = accuracy_score(Y_test, predict)
#Accuracy
print('\nAccuracy of Naive Bayes:\n')
print (accuracy)
conf_mat = confusion_matrix(Y_test, predict)
print('\nConfusion Matrix: \n',conf_mat)
plt.matshow(conf_mat)
plt.title('Confusion Matrix for test Data\t')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
from sklearn.neighbors import KNeighborsClassifier
clf_1 = KNeighborsClassifier(n_neighbors=5,algorithm='auto',n_jobs=-1)
clf_1.fit(X = train_tfmat.toarray(), y = Y_train)

predict_1 = clf_1.predict(test_tfmat.toarray())
# KNN
accuracy_1 = accuracy_score(Y_test, predict_1)
print('\nAccuracy of KNN:\n')
print (accuracy_1)
conf_mat_1 = confusion_matrix(Y_test, predict_1)
print('\nConfusion Matrix: \n',conf_mat_1)
plt.matshow(conf_mat_1)
plt.title('Confusion Matrix for Test Data\t')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

from sklearn.neighbors import NearestCentroid
clf_2 = NearestCentroid()
clf_2.fit(train_tfmat.toarray(), Y_train)
predict_2 = clf_2.predict(test_tfmat.toarray())
#Rocchio
accuracy_2 = accuracy_score(Y_test, predict_2)
print('\nAccuracy of Rochio:\n')
print (accuracy_2)
conf_mat_2 = confusion_matrix(Y_test, predict_2)
print('\nConfusion Matrix: \n',conf_mat_2)
plt.matshow(conf_mat_1)
plt.title('Confusion Matrix for Test Data\t')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

