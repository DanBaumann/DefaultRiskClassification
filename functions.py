import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_curve, auc, roc_auc_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier

def default_rate(data, variable_name, boolean):
    for b in boolean:
        default_rate = len((data.loc[(data[variable_name] == b) & (data['default_next_month'] == 1)]))/len(data.loc[data[variable_name] == b])
#         male_default_rate = (len((data.loc[(data[variable_name] == boolean[0]) & (data['default_next_month'] == 1)]))/
#                            (len(data.loc[data[variable_name] == boolean[0]])))
    
#     print("Male default rate is {}".format(round(male_default_rate, 2)), '\n\n'
#       "Female default rate is {}".format(round(female_default_rate, 2)))

        print("{} Default rate is : {}".format(b, round(default_rate, 2)))
    

    
def logistic_regression(X_train, y_train, X_test, y_test):

    log_clf = LogisticRegression(solver = 'lbfgs', max_iter = 1000)    
    C = np.logspace(0,4,10)
    param_grid = dict(C = C)    
    opt_model = GridSearchCV(log_clf, param_grid, cv = 10, verbose = 0, scoring = 'roc_auc', n_jobs = -1)
    #   fitting the model
    opt_model.fit(X_train, y_train)
    print("Optimal penalisation paramter is C = {}".format(opt_model.best_params_['C']))    
    best_model = opt_model.best_estimator_
    train_pred = best_model.predict(X_train)
    test_pred = best_model.predict(X_test)

    print("\nRecall for test set is: {0}".format(recall_score(y_test, test_pred)))
    y_score = opt_model.decision_function(X_test)
    fpr, tpr, thresh = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    print("\nAUC is :{0}".format(round(roc_auc, 2)))
    print('\nConfusion Matrix')
    print('----------------')
    display(pd.crosstab(y_test.ravel(), test_pred, rownames=['True'], colnames=['Predicted'], margins=True))

    return fpr, tpr, thresh, y_score  

def linear_svm(X_train, y_train, X_test, y_test):
    lin_svm = Pipeline([("scaler", StandardScaler()), ("linear_svc", LinearSVC(C = 1, max_iter = 100000, loss = "hinge"))])
    # fitting the model & keeping the SMOTE train sample
    lin_svm.fit(X_train, y_train)
    # getting predictions
    train_pred = lin_svm.predict(X_train)
    test_pred = lin_svm.predict(X_test)
    print("\nRecall for test set is: {0}".format(recall_score(y_test, test_pred)))
    # getting scores
    y_score = lin_svm.decision_function(X_test)
    # Plotting roc curve
    fpr, tpr, thresh = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    print("\nAUC is :{0}".format(round(roc_auc, 2)))
    print('\nConfusion Matrix')
    print('----------------')
    display(pd.crosstab(y_test.ravel(), test_pred, rownames=['True'], colnames=['Predicted'], margins=True))
    
    return fpr, tpr, thresh, y_score
                                

def poly_svm(X_train, y_train, X_test, y_test):
    poly_kern = SVC(max_iter=20, probability=True, kernel = 'poly', degree = 2)
    param_grid = {'C': [10**i for i in range(-3,5)], 'gamma': [10**i for i in range(-7,3)]}
    opt_model = GridSearchCV(poly_kern, param_grid, cv = 3, scoring = 'f1'
                        , n_jobs = -1, return_train_score=True)
    opt_model.fit(X_train, y_train)
    best_model = opt_model.best_estimator_
    opt_model.best_params_
    print("Optimal penalisation paramter is C = {}, and optimal gamma is = {}".format(opt_model.best_params_['C'], opt_model.best_params_['gamma']))
    train_pred = best_model.predict(X_train)
    test_pred = best_model.predict(X_test)
    print("\nRecall for test set is: {0}".format(recall_score(y_test, test_pred)))
    y_score = opt_model.decision_function(X_test)
    # Plotting roc curve
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    print("\nAUC is :{0}".format(round(roc_auc, 2)))
    print('\nConfusion Matrix')
    print('----------------')
    display(pd.crosstab(y_test.ravel(), test_pred, rownames=['True'], colnames=['Predicted'], margins=True))

    return fpr, tpr, thresholds, y_score


def tree_clf(X_train, y_train, X_test, y_test):
    tree_clf = DecisionTreeClassifier(random_state = 123)

    param_grid = {'max_depth': range(1, 8+1), 'min_samples_leaf': [5, 10, 15]}
    opt_model = GridSearchCV(tree_clf, param_grid, cv = 3, scoring = 'roc_auc', n_jobs = -1)
    opt_model.fit(X_train, y_train)
    best_model = opt_model.best_estimator_
    opt_model.best_params_      
    print("Optimal max depth paramter is = {}, and optimal min samples leaf is = {}".format(opt_model.best_params_['max_depth'], opt_model.best_params_['min_samples_leaf']))   
    train_pred = best_model.predict(X_train)
    test_pred = best_model.predict(X_test)
    print("\nRecall for test set is: {0}".format(recall_score(y_test, test_pred)))
    proba = opt_model.predict_proba(X_test)[:,1]
    # Plotting roc curve
    fpr, tpr, thresholds = roc_curve(y_test, proba)
    roc_auc = auc(fpr, tpr)
    print("\nAUC is :{0}".format(round(roc_auc, 2)))
    print('\nConfusion Matrix')
    print('----------------')
    display(pd.crosstab(y_test.ravel(), test_pred, rownames=['True'], colnames=['Predicted'], margins=True))

    return fpr, tpr, thresholds, proba      
          

def random_forest(X_train, y_train, X_test, y_test):
    rnd_clf1 = RandomForestClassifier(n_jobs = -1, random_state = 123)

    param_grid = {'n_estimators': [200,300,400,500,600,700,800,900,1000], 'max_leaf_nodes': [5, 10, 15, 20 ,25, 30]}

    opt_model = GridSearchCV(rnd_clf1, param_grid, cv = 3, scoring = 'f1')
    opt_model.fit(X_train, y_train)
    best_model = opt_model.best_estimator_

    opt_model.best_params_
    best_model.fit(X_train, y_train)
    train_pred = best_model.predict(X_train)
    test_pred = best_model.predict(X_test)
    print("Recall for test set is: {0}".format(recall_score(y_test, test_pred)))      
    proba = best_model.predict_proba(X_test)[:,1]
    # Plotting roc curve
    fpr, tpr, thresholds = roc_curve(y_test, proba)      
    roc_auc = auc(fpr, tpr)
    print("\nAUC is :{0}".format(round(roc_auc, 2)))
    print('\nConfusion Matrix')
    print('----------------')
    display(pd.crosstab(y_test.ravel(), test_pred, rownames=['True'], colnames=['Predicted'], margins=True))

    return fpr, tpr, thresholds, proba
    
          
def Voting_Classifier(X_train, y_train, X_test, y_test):
    log_clf = LogisticRegression()
    rnd_clf = RandomForestClassifier()
    svm_clf = SVC()

    voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf),('svc', svm_clf)], voting = 'hard')

    voting_clf.fit(X_train, y_train)
    
    for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
        clf.fit(X_train, y_train)
        test_pred = clf.predict(X_test)
        print(clf.__class__.__name__, accuracy_score(y_test, test_pred))
    print("Recall for test set is: {0}".format(recall_score(y_test, test_pred)))      
    proba = voting_clf.predict_proba(X_test)[:,1]
    # Plotting roc curve
    fpr, tpr, thresholds = roc_curve(y_test, proba)      
    roc_auc = auc(fpr, tpr)
    print("\nAUC is :{0}".format(round(roc_auc, 2)))
    print('\nConfusion Matrix')
    print('----------------')
    display(pd.crosstab(y_test.ravel(), test_pred, rownames=['True'], colnames=['Predicted'], margins=True))

    return fpr, tpr, thresholds, proba
              
    
def plot_roc_curve(fpr, tpr, classifier_name):     
    plt.figure(figsize=(10, 8))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.yticks([i/20.0 for i in range(21)])
    plt.xticks([i/20.0 for i in range(21)], rotation = 'vertical')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) Curve for Test Set: {}'.format(classifier_name))
    plt.legend(loc='lower right')
    print('AUC: {}'.format(auc(fpr, tpr)))

def plot_precision_recall_curve(y_test, proba, classifier_name):
    precision, recall, thresh = precision_recall_curve(y_test, proba)
    plt.plot(recall, precision, marker='.', label=classifier_name)
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    plt.show()
    
    return precision, recall, thresh

def plot_precision_recall_threshold(precision, recall, thresholds): 
    plt.plot(thresholds, precision[:-1], 'b--', label = 'Precision')
    plt.plot(thresholds, recall[:-1], 'g-', label = 'Recall')
    plt.legend()
    plt.grid()
    plt.xlabel('Threshold')
    
    
    
    
    
    
    
    
    
    
    
    
    
