# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 23:02:22 2017
Data Mining and Business Intelligence

Quora duplicate pairs detection

"""
"http://stackoverflow.com/questions/31806695/when-to-use-which-fuzz-function-to-compare-2-strings"
os.getcwd()
os.chdir('D:\Kaushik\Uconn Related\UCONN - Study Materials\Semester 2\OPIM 5671 - Data Mining and BI\Projects\Quora - Piars preditcion\Dataset')

import os
from fuzzywuzzy import fuzz 
from fuzzywuzzy import process
import cPickle
import pandas as pd
import numpy as np
import gensim
from nltk.corpus import stopwords
from scipy.spatial.distance import cosine, euclidean, minkowski
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.cross_validation import train_test_split
import re


def f_score(mtx):
    precision = (mtx[0,0] * 1.00)/(mtx[0,0]+mtx[1,0])
    recall = (mtx[0,0]* 1.00)/(mtx[0,1]+mtx[0,1])
    f_stat = (2.00)*((precision*precision)/(precision+recall))
    return round(f_stat,2)

def accuracy(mtx):
    return round((mtx[0,0]+mtx[1,1]) * 1.00 / (mtx[0,0]+mtx[1,1]+mtx[0,1]+mtx[1,0]),2)

    
def round_val(col):
    col_upd = []
    for row in col:    
        if row < 0.5:        
            col_upd = np.append(col_upd,0)
        else:        
            col_upd = np.append(col_upd,1)
    return col_upd

    
def wmd(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return model.wmdistance(s1, s2)


def norm_wmd(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return norm_model.wmdistance(s1, s2)


def sent2vec(s):
    words = str(s).lower().decode('utf-8')
    words = word_tokenize(words)
    stop_words = stopwords.words('english')
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())

    
"Load the data"
data = pd.read_csv('train.csv', sep=',')

list(data)
"Remove the not required columns"
data = data.drop(['id', 'qid1', 'qid2'], axis=1)

"Check the top rows"
data.head()
list(data)

"Check the type of the data"
type(data)

data_p1 = data

""""""""""""""" Phase 1: Feature Extraction """""""""""""""""""""""""""""
"Basic Features"
data_p1['common_words'] = data_p1.apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))), axis=1)
data_p1['len_word_q1'] = data_p1.question1.apply(lambda x: len(str(x).split()))
data_p1['len_word_q2'] = data_p1.question2.apply(lambda x: len(str(x).split()))
data_p1['len_char_q1'] = data_p1.question1.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
data_p1['len_char_q2'] = data_p1.question2.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
data_p1['len_q1'] = data_p1.question1.apply(lambda x: len(str(x)))
data_p1['len_q2'] = data_p1.question2.apply(lambda x: len(str(x)))
data_p1['diff_len'] = data_p1.len_q1 - data.len_q2

list(data_p1)

X = data_p1.iloc[:, [3,4,5,6,7,8,9,10]].values
y = data_p1.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

  
"""""""""""""""""""""Random Forest"""""""""""""""""""""
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)

# Predicting a new result
y_pred_p1_rf = regressor.predict(X_test)
y_pred_p1_rf = round_val(y_pred_p1_rf)
        
cm_p1_rf = confusion_matrix(y_test, y_pred_p1_rf)
accuracy(cm_p1_rf) 
"70%"


""""""""""""""" Phase 2: Feature Extraction """""""""""""""""""""""""""""

data_p2 = data

data_p2 ['fw_qratio'] = data_p2.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
data_p2 ['fw_WRatio'] = data_p2.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
data_p2 ['fw_par_ratio'] = data_p2.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
data_p2 ['fw_par_token_set_ratio'] = data_p2.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
data_p2 ['fw_par_token_sort_ratio'] = data_p2.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
data_p2 ['fw_token_set_ratio'] = data_p2.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
data_p2 ['fw_token_sort_ratio'] = data_p2.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)

data_p2.to_csv('quora_features_extraction_p2.csv', index=False)


list(data_p2)
data_p2_upd = data_p2
data_p2_upd = data_p2_upd.drop(['id','question1','question2'], axis=1)
list(data_p2_upd)

data_p1_p2  = pd.concat([data.reset_index(drop=True), data_p2_upd], axis=1)
list(data_p1_p2)
data_p1_p2 = data_p1_p2.drop(['id'], axis=1)

X = data_p1_p2.iloc[:, [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]].values
y = data_p1_p2.iloc[:, 2].values


" CREATE DATA PARTITIONING "
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

    
"""""""""""""""""""""Random Forest"""""""""""""""""""""
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)

# Predicting a new result
y_pred_p1p2_rf = regressor.predict(X_test)
y_pred_p1p2_rf_upd = round_val(y_pred_p1p2_rf)
        
cm_p1p2_rf = confusion_matrix(y_test, y_pred_p1p2_rf_upd)
accuracy(cm_p1p2_rf)
f_score(cm_p1p2_rf)



data_p1_p2 = pd.read_csv('quora_features_extraction_p2.csv')

"Get the pre-trained models from gensim"
os.chdir('D:\Kaushik\Uconn Related\Study and Research\My Projects\Gensin')
model = gensim.models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
norm_model = gensim.models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
"norm_model.init_sims(replace=True)"

os.chdir('D:\Kaushik\Uconn Related\UCONN - Study Materials\Semester 2\OPIM 5671 - Data Mining and BI\Projects\Quora - Piars preditcion\Dataset')


data_p1_p2.question1.apply(lambda x: len(str(x)))
data_p1_p2['question1'] = data_p1_p2.question1.apply(lambda x: re.sub('[^a-zA-Z0-9 \n\.]', '', x))
data_p1_p2['question2'] = data_p1_p2.question2.apply(lambda x: re.sub('[^a-zA-Z0-9 \n\.]', '', x))


data_p1_p2['question1_uc'] = unicode(data_p1_p2['question1'], "utf-8")
data_p1_p2['question1_uc'] = unicode(data_p1_p2['question1'], "utf-8")
pd.options.mode.chained_assignment = None


data_p1_p2_ss = data_p1_p2.head(100000)
data_p1_p2_ss['q1'] = data_p1_p2_ss.question1.apply(lambda x: re.sub('[^a-zA-Z0-9 \n\.]', '', x))
data_p1_p2_ss['q2'] = data_p1_p2_ss.question2.apply(lambda x: re.sub('[^a-zA-Z0-9 \n\.]', '', x))


list(data_p1_p2_ss)
"calculating word mover distance"
data_p1_p2_ss['wmd'] = data_p1_p2_ss.apply(lambda x: wmd(x['q1'], x['q2']), axis=1)    
data_p1_p2_ss['n_wmd'] = data_p1_p2_ss.apply(lambda x: norm_wmd(x['q1'], x['q2']), axis=1)
data_p1_p2_ss.to_csv('quora_features_extraction_100k_p3.csv', index=False)

""""""""""""" Remove the stopwords - Start """""""""""""""
import nltk
nltk.download()
"Remove Stop words"
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))

def remove_stopword(word):
    return word not in stop

from tqdm import tqdm
question1_vectors = np.zeros((data_p1_p2_ss.shape[0], 300))
error_count = 0
for i, q in tqdm(enumerate(data_p1_p2_ss.q1.values)):
    question1_vectors[i, :] = sent2vec(q)

question2_vectors  = np.zeros((data_p1_p2_ss.shape[0], 300))
for i, q in tqdm(enumerate(data_p1_p2_ss.q2.values)):
    question2_vectors[i, :] = sent2vec(q)

data_p1_p2_ss.to_csv('quora_features_extraction_100k_p4.csv', index=False)

data = data_p1_p2_ss
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis

data['cosine_dist'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]
data['jaccard_dist'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]
data['euclidean_dist'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]
data['minkowski_dist'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]
data.to_csv('quora_features_extraction_100k_p5.csv', index=False)

data_model = pd.read_csv('quora_features_extraction_100k_model.csv')
data_final.isnull().any()
data_round = data_round.dropna()

"Impute missing data"
#data_round = data_round.fillna(method='ffill')


data_final = pd.read_csv('quora_features_extraction_100k_final.csv', sep=',')

X = data_final.iloc[:, [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,22,23,24,25,26,27,28]].values
y = data_final.iloc[:, 2].values


" CREATE DATA PARTITIONING "
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


"""""""""""""""""""""Logistic Regression"""""""""""""""""""""
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred_lr_p1p2 = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm_p1p2_lr = confusion_matrix(y_test, y_pred_lr_p1p2)
accuracy(cm_p1p2_lr)
f_score(cm_p1p2_lr)

"""""""""""""""""""SVM""""""""""""""""""""
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred_svm = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm_svm = confusion_matrix(y_test, y_pred_svm)
"SVM Accuracy"
accuracy(cm_svm )
f_score(cm_svm )
 
"""""""""""""""""""""Random Forest"""""""""""""""""""""
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)

# Predicting a new result
y_pred_p1p2_rf = regressor.predict(X_test)
y_pred_p1p2_rf_upd = round_val(y_pred_p1p2_rf)
cm_p1p2_rf = confusion_matrix(y_test, y_pred_p1p2_rf_upd)
"Random Forest Accuracy"
accuracy(cm_p1p2_rf)
f_score(cm_p1p2_rf)


"""""""""""""""""""""Gradient Boosting - XGBoost"""""""""""""""""""""
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_xg = confusion_matrix(y_test, y_pred)
"XGBoost Accuracy"
accuracy(cm_xg)
f_score(cm_xg)



"PCA - Varliable reduction"
from sklearn.decomposition import PCA
pca = PCA(n_components=6)
fit = pca.fit(X_train)
# summarize components
print("Explained Variance: %s") % fit.explained_variance_ratio_
print(fit.components_)
X_pca = fit.components_




"feature selection - important features selection"
from sklearn.feature_selection import RFE
rfe = RFE(classifier, 10)
rfe = rfe.fit(X_train, y_train)
# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)

" Below are the important features that contributed to accuracy"
#3,7,11,8,10,13
#len q1, len_char_q2, fuzzz_qratio, len_car_q1, common_words, fuzz_partial_ratio
#[ True False False False False False  True  True  True False False False
# False  True False False False False False False  True False]
#[ 1  5  9  6  8 11  1  1  1 15  2 16 13  1  3  7 12 17  4 14  1 10]












