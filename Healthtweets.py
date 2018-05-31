import pandas as pd
import numpy as np

file_name = 'dataset/health_tweets_labeled.csv'
df = pd.read_csv(file_name,encoding='ANSI')
df_sample = pd.DataFrame(df)
df.head()
data = np.array(df.iloc[:,0])
target = np.array(df.iloc[:,1])
# # Calculating Term-Document Incidence Matrix:

from sklearn.feature_extraction.text import CountVectorizer

# file_name = 'dataset/health_tweets_labeled.csv'
# df = pd.read_csv(file_name,encoding='ANSI')

vectorizer = CountVectorizer()
x = vectorizer.fit_transform(df['tweet'].values.astype('U'))

arr=x.toarray()
vectorizer.vocabulary_.get('document')
# # TF-IDF
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(arr)
# # KNeighbors
# print(tfidf)
n = 50000
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(arr[:n], target[:n]) 

import winsound as s
s.Beep(1300,2000)
neigh.score(arr[n:], target[n:])
# Roccio 
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.model_selection import cross_val_score
import numpy as np
import winsound as s

n=50000
clf = NearestCentroid()
clf.fit(arr[:n], target[:n])
s.Beep(1300,2000)
NearestCentroid(metric='euclidean', shrink_threshold=None)
s.Beep(1300,2000)
print(clf.score(arr[n:],target[n:]))
s.Beep(1300,2000)
scores = cross_val_score(clf, arr[n:], target[n:], cv=5)
print(scores)
# # Naive-bayes
from sklearn.naive_bayes import GaussianNB
# gnb = GaussianNB()
# y_pred = gnb.fit(arr[:500], target[:500]).predict(arr[500:600])
# print("Number of mislabeled points out of a total %d points : %d"
#       % (arr.shape[0],(target != y_pred).sum()))
n=50000

clf = GaussianNB()
clf.fit(arr[:n], target[:n])
clf.score(arr[n:],target[n:])
clf_pf = GaussianNB()
clf_pf.partial_fit(arr[:n], target[:n], np.unique(target))
clf_pf.score(arr[n:],target[n:])

