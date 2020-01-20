import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import pydotplus
from sklearn.externals.six import StringIO
from IPython.display import Image  

from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import make_scorer, f1_score, roc_curve, auc, accuracy_score, confusion_matrix, classification_report, roc_auc_score

def XySplit(df):
    y = df['Default']
    X = df.drop(columns=['Default'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state=123)


    sm = SMOTE(random_state = 123, sampling_strategy = 1.0)
    X_train,y_train  = sm.fit_sample(X_train, y_train)
    
    scale = StandardScaler()
    scale.fit(X_train)

    X_train_scaled = scale.transform(X_train)
    X_test_scaled = scale.transform(X_test)
    
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled)
    X_test_scaled = pd.DataFrame(X_test_scaled)
    
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled

def stats(df):
    avg_credit_limit = df['CreditLimit'].mean()
    print('Average Credit Limit: $', round(avg_credit_limit, 2))

    total_default_pcnt = df['Default'].sum()/len(df.Default)
    print('Average Chance of Default: ',round((total_default_pcnt * 100), 2), '%')

def plot_feature_importances(model, X_train):
    n_features = X_train.shape[1]
    plt.figure(figsize=(8,8))
    plt.barh(range(n_features), model.feature_importances_, align='center') 
    plt.yticks(np.arange(n_features), X_train.columns.values) 
    plt.xlabel('Feature importance')
    plt.ylabel('Feature')
    
def print_AUC(y_true, y_pred):
    print("AUC Score: {}".format(roc_auc_score(y_true, y_pred)))
    
def scorer():
    f1_scorer = make_scorer(score_func=f1_score,greater_is_better=True)
    return f1_scorer  

def ConfusionMatrix(y_test, pred):
    print('Confusion Matrix \n')
    print(pd.crosstab(y_test, pred, rownames = ['True'], colnames = ['Predicted'], margins = True))
    
def matrix_classification_report(y_test, pred):
    con_mat = ConfusionMatrix(y_test, pred)
    clas_rep = classification_report(y_test, pred)

    print(con_mat, '\n\n','Classification Report \n\n', clas_rep)    
    
def DecisionTree(X_train, X_test, y_train, y_test):
    tree = DecisionTreeClassifier(random_state=123)
    
    tree.fit(X_train, y_train)
    
    pred = tree.predict(X_test)
    prob = tree.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, prob[:,1])
    roc_auc = auc(fpr, tpr)
    
    print("Training Accuracy for Decision Tree Classifier: {:.4}%".format(tree.score(X_train, y_train) * 100))
    print("Testing Accuracy for Decision Tree Classifier: {:.4}%".format(tree.score(X_test, y_test) * 100))
    print("\n")
          
    matrix_classification_report(y_test, pred)
    
    plot_feature_importances(tree, X_train)
    
    
def BaggedTree(X_train, X_test, y_train, y_test, n_estimators_ = 1, max_features_ = 1):
    
    tree = BaggingClassifier(DecisionTreeClassifier(random_state = 123), n_estimators = n_estimators_, 
                             max_features = max_features_)
    
    tree.fit(X_train, y_train)
    pred = tree.predict(X_test)
    prob = tree.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, prob[:,1])
    roc_auc = auc(fpr, tpr)
    
    matrix_classification_report(y_test, pred)

    print("Training Accuracy for Bagging Tree Classifier: {:.4}%".format(tree.score(X_train, y_train) * 100))
    print("Testing Accuracy for Bagging Tree Classifier: {:.4}%".format(tree.score(X_test, y_test) * 100))
    print("\n") 

    print('---------')
    print("AUC Score: {}".format(roc_auc))
    print('---------')
   
    plt.plot(fpr,tpr)

def OptimiseBagging(X_train, X_test, y_train, y_test):
    
    tree = BaggingClassifier(DecisionTreeClassifier(random_state=123))
    
    param_grid = {'n_estimators': [7,8,9,10,11,12],
                  'max_features' : [13,14,15,16,17,18]   
                    }                 

    gs_bt = GridSearchCV(tree, param_grid, cv=5, scoring=scorer())
    
    gs_bt.fit(X_train, y_train)
    
    cvs = pd.DataFrame(gs_bt.cv_results_)
    cvs = cvs.sort_values(by=['rank_test_score'])
    
    print(gs_bt.best_params_)
    
    return cvs.head(6)
       

def RandomForest(X_train, X_test, y_train, y_test, criterion_ = 'gini', max_depth_ = 1, max_features_ = 1, n_estimators_ = 1):
    tree = RandomForestClassifier(random_state=123, criterion = criterion_, max_depth = max_depth_, 
                                  max_features = max_features_, n_estimators = n_estimators_)
    
    tree.fit(X_train, y_train)
    pred = tree.predict(X_test)
    prob = tree.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, prob[:,1])
    roc_auc = auc(fpr, tpr)
    
    matrix_classification_report(y_test, pred)
    
    print("Training Accuracy for Random Forest Classifier {:.4}%".format(tree.score(X_train, y_train) * 100))
    print("Testing Accuracy for Random Forest Classifier: {:.4}%".format(tree.score(X_test, y_test) * 100))
    print("\n") 

    print('---------')
    print("AUC Score: {}".format(roc_auc))
    print('---------')
   
    plt.plot(fpr,tpr)
    
    
def OptimiseForest(X_train, X_test, y_train, y_test):
    tree = RandomForestClassifier(random_state = 123)
    
    param_grid = {'criterion': ['gini','entropy'],
                  'max_depth' : [5,7,9,11],
                  'max_features' : [5,7,9,11],
                  'n_estimators' : [8,10,12,14]
                    }

    gs_rf = GridSearchCV(tree, param_grid, cv=5, scoring=scorer())
    gs_rf.fit(X_train, y_train)   
    
    cvs = pd.DataFrame(gs_rf.cv_results_)
    cvs = cvs.sort_values(by=['rank_test_score'])
    
    print(gs_rf.best_params_)
    
    return cvs.head(6)    
      
def KNN(X_train, X_test, y_train, y_test, n_neighbors_ = 1, leaf_size_ = 1):
    
    tree = KNeighborsClassifier(n_neighbors = n_neighbors_, 
                             leaf_size = leaf_size_)
    
    tree.fit(X_train, y_train)
    
    pred = tree.predict(X_test)
    prob = tree.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, prob[:,1])
    roc_auc = auc(fpr, tpr)
    
    matrix_classification_report(y_test, pred)

    print("Training Accuracy for KNN Classifier: {:.4}%".format(tree.score(X_train, y_train) * 100))
    print("Testing Accuracy for KNN Classifier: {:.4}%".format(tree.score(X_test, y_test) * 100))
    print("\n") 

    print('---------')
    print("AUC Score: {}".format(roc_auc))
    print('---------')
   
    plt.plot(fpr,tpr)   
    
def OptimiseKNN(X_train, X_test, y_train, y_test):
    
    tree = KNeighborsClassifier()

    param_grid = {'n_neighbors':[8,9,10,11,12,13,14],
                  'leaf_size' : [5,10,15,20,25,30,35]   
                    }

    gs_knn = GridSearchCV(tree, param_grid, cv=5, scoring=scorer())
    
    gs_knn.fit(X_train, y_train)    
    
    cvs = pd.DataFrame(gs_knn.cv_results_)
    cvs = cvs.sort_values(by=['rank_test_score'])
    
    print(gs_knn.best_params_)
    
    return cvs.head(6)
    
def LogRegression(X_train, X_test, y_train, y_test, max_iter_ = 1):
    
    tree = LogisticRegression(random_state = 123,max_iter = max_iter_)
    
    tree.fit(X_train, y_train)
    
    pred = tree.predict(X_test)
    prob = tree.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, prob[:,1])
    roc_auc = auc(fpr, tpr)
    
    matrix_classification_report(y_test, pred)

    print("Training Accuracy for Logistic Regression Classifier: {:.4}%".format(tree.score(X_train, y_train) * 100))
    print("Testing Accuracy for Logistic Regression Classifier: {:.4}%".format(tree.score(X_test, y_test) * 100))
    print("\n") 

    print('---------')
    print("AUC Score: {}".format(roc_auc))
    print('---------')
   
    ns_fpr, ns_tpr, _ = roc_curve(y_test, prob[:,1])
    lr_fpr, lr_tpr, _ = roc_curve(y_test, prob[:,1])
    
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(fpr, tpr, marker='.', label='Logistic')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()
    
def OptimiseLogReg(X_train, X_test, y_train, y_test):
    
    tree = LogisticRegression()
    
    param_grid = {'max_iter' : [250,500,1000,2500]
                    }                 

    gs_lr = GridSearchCV(tree, param_grid, cv=5, scoring=scorer())
    
    gs_lr.fit(X_train, y_train)
    
    cvs = pd.DataFrame(gs_lr.cv_results_)
    cvs = cvs.sort_values(by=['rank_test_score'])
    
    print(gs_lr.best_params_)
    
    return cvs.head(6)


def svm_grid_searched(X_train, X_test, y_train, y_test):
    
        param_grid = [
          {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001],'kernel': ['linear']},
          {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
         ]


        scores = ['precision', 'recall']

        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()

            clf = GridSearchCV(
                SVC(max_iter = 10000), param_grid, scoring='%s_macro' % score
            )
            clf.fit(X_train, y_train)

            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()
            print("Grid scores on development set:")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))
            print()

            print("Detailed classification report:")
            print()
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
            y_true, y_pred = y_test, clf.predict(X_test)
            print(classification_report(y_true, y_pred))
            print()
            
def SVM(X_train, X_test, y_train, y_test):
    svc_lin_clf = SVC(C=1, probability = True, max_iter = 20)
    svc_lin_clf.fit(X_train, y_train)
    y_true, y_pred = y_test, svc_lin_clf.predict(X_test)
    y_probs = svc_lin_clf.predict_proba(X_test)
    print(classification_report(y_true, y_pred))
    
    pred = svc_lin_clf.predict(X_test)
    prob = svc_lin_clf.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, prob[:,1])
    roc_auc = auc(fpr, tpr)
    
    matrix_classification_report(y_test, pred)

    ns_probs = [0 for _ in range(len(y_probs[:,1]))]
    ns_auc = roc_auc_score(y_test, ns_probs)
    svm_auc = roc_auc_score(y_test, y_probs[:,1])

    print('---------')
    print("AUC Score: {}".format(roc_auc))
    print('---------')
    
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, y_probs[:,1])
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(fpr, tpr, marker='.', label='Logistic')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()
    
def PCA_DF(df):
    pca = PCA(random_state = 123)

    pca = pca.fit_transform(df)

    pca = pd.DataFrame(pca[:, :8], index=df.index)

    pca_df = pca.join(df['Default'])
    pca_df = pca_df.rename(columns={0:'x1',1:'x2',2:'x3',3:'x4',4:'x5',5:'x6',6:'x7',7:'x8'}, inplace=True)
    
    return pca_df