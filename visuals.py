import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import pydotplus
from sklearn.externals.six import StringIO
from IPython.display import Image  

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import make_scorer, f1_score, roc_curve, auc, accuracy_score, confusion_matrix, classification_report

def bars(df, col):
    col_df = pd.DataFrame(df.groupby([col, 'Default']).size().unstack())
    col_df.plot(kind='bar', stacked = True)
    
def hists(df, col):
    df[col].hist()
    
def PlotDecisionTree(X_train, X_test, y_train, y_test):
    
    tree = DecisionTreeClassifier(max_depth=4, random_state=123)
    
    tree.fit(X_train, y_train)
    
    pred = tree.predict(X_test)
    
    dot_data = StringIO()
    
    export_graphviz(tree, out_file=dot_data,  
                    filled=True, rounded=True,
                    special_characters=True,feature_names=X_train.columns)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    
    image = Image(graph.create_png())
    
    return image

def PlotRocCurve(X_train, X_test, y_train, y_test, model):
    tree = model(random_state=123)
    
    tree.fit(X_train, y_train)

    prob = tree.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, prob[:,1])
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr,tpr)
    
    print('---------')
    print("AUC Score: {}".format(roc_auc))
    print('---------')