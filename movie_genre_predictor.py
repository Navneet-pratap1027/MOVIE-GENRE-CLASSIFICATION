import pandas as pd
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score


## train data
df_train = pd.read_csv("train_data.txt", sep=':::',header=None,engine='python')
df_train.columns=['ID', 'TITLE', 'GENRE', 'DESCRIPTION']
df_train.dropna(subset=['DESCRIPTION', 'GENRE'], inplace=True)

df_test = pd.read_csv("test_data_solution.txt", sep=':::',header=None,engine='python')
df_test.columns=['ID', 'TITLE', 'GENRE' , 'DESCRIPTION']


## Feature Selection

tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_train = tfidf.fit_transform(df_train['DESCRIPTION'])

# Target labels
y_train = df_train['GENRE']


X_test = tfidf.transform(df_test['DESCRIPTION'])
y_test=df_test['GENRE']


models={
    "Logistic-Regression" : LogisticRegression(),
    "Naive-Bayes" : MultinomialNB(),
    "Support-Vector-Machine" : SVC(kernel="linear")
}


for i in range(len(list(models))):
    model=list(models.values())[i]
    model.fit(X_train,y_train)
    
    
    
    ## Make Prediction
    y_predict_train=model.predict(X_train)
    y_predict_test=model.predict(X_test)
    
    
    ## Train set performance
    y_predict_train_accuracy=accuracy_score(y_train,y_predict_train)
    y_predict_train_f1_score=f1_score(y_train,y_predict_train,average='weighted',zero_division=0)
    y_predict_train_precision=precision_score(y_train,y_predict_train,average='weighted',zero_division=0)
    y_predict_train_recall=recall_score(y_train,y_predict_train,average='weighted',zero_division=0)
    
    ## Test set performance
    y_predict_test_accuracy=accuracy_score(y_test,y_predict_test)
    y_predict_test_f1_score=f1_score(y_test,y_predict_test,average='weighted')
    y_predict_test_precision=precision_score(y_test,y_predict_test,average='weighted')
    y_predict_test_recall=recall_score(y_test,y_predict_test,average='weighted')
    
    print(list(models.keys())[i])
    
    print('Model performance for Training set')
    print("- Accuracy: {:.4f}".format(y_predict_train_accuracy))
    print('- F1 score: {:.4f}'.format(y_predict_train_f1_score))
    
    print('- Precision: {:.4f}'.format(y_predict_train_precision))
    print('- Recall: {:.4f}'.format(y_predict_train_recall))

    
    
    print('----------------------------------')
    
    print('Model performance for Test set')
    print('- Accuracy: {:.4f}'.format(y_predict_test_accuracy))
    print('- F1 score: {:.4f}'.format(y_predict_test_f1_score))
    print('- Precision: {:.4f}'.format(y_predict_test_precision))
    print('- Recall: {:.4f}'.format(y_predict_test_recall))

    
    print('='*35)
    print('\n')
    
    
