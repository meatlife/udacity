#!/usr/bin/python
# -*- coding: utf-8 -*-
# python = 2.7.11
# scikit-learn = 0.17.1


import sys
import pickle
sys.path.append("./tools/")

import pprint
import pandas as pd
import numpy as np

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit, KFold
from sklearn.preprocessing import MinMaxScaler, Imputer
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
financial_features = ['bonus',
                      'deferred_income',
                      'deferral_payments',
                      'director_fees',
                      'exercised_stock_options',
                      'expenses',
                      'loan_advances',
                      'long_term_incentive',
                      'other',
                      'restricted_stock',
                      'restricted_stock_deferred',
                      'salary',
                      'total_payments',
                      'total_stock_value']

email_features = ['from_messages',
                  'from_this_person_to_poi',
                  'from_poi_to_this_person',
                  'shared_receipt_with_poi',
                  'to_messages']

# You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
# 利用pandas对数据集进行各种分析
df = pd.DataFrame.from_dict(data_dict, orient = 'index').sort_index(axis = 1)
df = df.replace('NaN', np.nan)
print 'The dataset has {} samples and {} variables.'.format(len(df), len(df.iloc[0]))
print 'There are {} poi\'s which account for {} % in all samples.'\
    .format(df['poi'].sum(), np.round(df['poi'].sum()*1.0/len(df), 4)*100)
print '-'* 120
# better use other methods to find the outliers in the notebook such as
#df.describe()
#df[df.isnull().sum(axis = 1) >= 18]
#df['total_stock_value'].plot(kind='hist', bins = 30)
#df.plot('salary','bonus', kind = 'scatter')

data_dict.pop('TOTAL')
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')

### Task 3: Create new feature(s)
# 建立新的邮件变量特征
for name in data_dict:
    if data_dict[name]['from_poi_to_this_person'] != 'NaN' and data_dict[name]['to_messages'] != 'NaN':
        data_dict[name]['fraction_from_poi'] = \
        float(data_dict[name]['from_poi_to_this_person']) / data_dict[name]['to_messages']
    else:
        data_dict[name]['fraction_from_poi'] = 'NaN'

    if data_dict[name]['from_this_person_to_poi'] != 'NaN' and data_dict[name]['from_messages'] != 'NaN':
        data_dict[name]['fraction_to_poi'] = \
        float(data_dict[name]['from_this_person_to_poi']) / data_dict[name]['from_messages']
    else:
        data_dict[name]['fraction_to_poi'] = 'NaN'

new_email_features = ['fraction_to_poi',
                      'fraction_from_poi',
                      'shared_receipt_with_poi']

# 利用对数转换建立新的财务变量特征
log_features = ['bonus',
                'director_fees',
                'exercised_stock_options',
                'expenses',
                'director_fees',
                'loan_advances',
                'long_term_incentive',
                'other',
                'salary',
                'total_payments']

for name in data_dict:
    for feature in log_features:
        if data_dict[name][feature] == 'NaN':
            data_dict[name][feature + '_log'] = 'NaN'
        else:
            data_dict[name][feature + '_log'] = np.log10(float(data_dict[name][feature]))

new_financial_features = ['bonus_log',
                          'deferred_income',
                          'deferral_payments',
                          'director_fees_log',
                          'exercised_stock_options_log',
                          'expenses_log',
                          'loan_advances_log',
                          'long_term_incentive_log',
                          'other_log',
                          'restricted_stock',
                          'restricted_stock_deferred',
                          'salary_log',
                          'total_payments_log',
                          'total_stock_value']

# 经过分析决定选用新特征替换部分原特征
features_list = ['poi'] + new_financial_features + new_email_features

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
def grid_nb(features, labels):
    steps = [('imputer', Imputer(strategy = 'median')),
             ('selector', SelectKBest()),
             ('pca', PCA()),
             ('clf', GaussianNB())]

    pipeline = Pipeline(steps)

    parameters = {'selector__k': np.arange(9, 18),
                  'pca__n_components': np.arange(2, 10)}

    clf_grid = GridSearchCV(pipeline, param_grid = parameters, scoring = 'f1')
    clf_grid.fit(features, labels)
    clf = clf_grid.best_estimator_
    return clf


def grid_svc(features, labels):
    steps = [('imputer', Imputer(strategy = 'median')),
             ('scaler', MinMaxScaler()),
             ('selector', SelectKBest()),
             ('pca', PCA()),
             ('clf', SVC(kernel='sigmoid'))]

    pipeline = Pipeline(steps)

    parameters = {'selector__k': np.arange(9, 18),
                  'pca__n_components': np.arange(2, 10),
                  'clf__C': np.logspace(-3, 3, 7),
                  'clf__gamma': np.logspace(-4, 2, 7)}

    clf_grid = GridSearchCV(pipeline, param_grid = parameters, scoring = 'f1')
    clf_grid.fit(features, labels)
    clf = clf_grid.best_estimator_
    return clf


def grid_dt(features, labels):
    steps = [('imputer', Imputer(strategy = 'median')),
             ('selector', SelectKBest()),
             ('pca', PCA()),
             ('clf', DecisionTreeClassifier())]

    pipeline = Pipeline(steps)

    parameters = {'selector__k': np.arange(9, 18),
                  'pca__n_components': np.arange(2, 10),
                  'clf__criterion': ['gini', 'entropy'],
                  'clf__min_samples_split': np.arange(2, 10),
                  'clf__min_samples_leaf': np.arange(1, 4)}

    clf_grid = GridSearchCV(pipeline, param_grid = parameters, scoring = 'f1')
    clf_grid.fit(features, labels)
    clf = clf_grid.best_estimator_
    return clf


def grid_lr(features, labels):
    steps = [('imputer', Imputer(strategy = 'median')),
             ('selector', SelectKBest()),
             ('pca', PCA()),
             ('clf', LogisticRegression())]

    pipeline = Pipeline(steps)

    parameters = {'selector__k': np.arange(9, 18),
                  'pca__n_components': np.arange(2, 10),
                  'clf__C': np.logspace(-3, 3, 7),
                  'clf__penalty': ['l1', 'l2']}

    clf_grid = GridSearchCV(pipeline, param_grid = parameters, scoring = 'f1')
    clf_grid.fit(features, labels)
    clf = clf_grid.best_estimator_
    return clf

# 对不同算法进行训练及验证
clf = grid_nb(features, labels)

print 'classifier GaussianNB'
print 'clf test by using StratifiedShuffleSplit method'
print '***********************************************'
test_classifier(clf, my_dataset, features_list)
print '-'* 120


clf_svc = grid_svc(features, labels)
print 'classifier SVC'
print 'clf test by using StratifiedShuffleSplit method'
print '***********************************************'
test_classifier(clf_svc, my_dataset, features_list)
print '-'* 120


clf_dt = grid_dt(features, labels)
print 'classifier DecisionTreeClassifier'
print 'clf test by using StratifiedShuffleSplit method'
print '***********************************************'
test_classifier(clf_dt, my_dataset, features_list)
print '-'* 120


clf_lr = grid_lr(features, labels)
print 'classifier LogisticRegression'
print 'clf test by using StratifiedShuffleSplit method'
print '***********************************************'
test_classifier(clf_lr, my_dataset, features_list)
print '-'* 120


# 获取特征重要性分数
def feature_scores(clf, features_list):
    scores = clf.named_steps['selector'].scores_
    index = np.argsort(scores)[::-1]
    for i in range(clf.named_steps['selector'].get_params()['k']):
        print "{}: {}".format(features_list[index[i] + 1], scores[index[i]])

print 'feature scores in the final model (SelectKBest, k = 11)'
print '*********************************'
feature_scores(clf, features_list)
print '-'* 120

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
# KFold 交叉验证法
def kfold_test(clf, features, labels, k=10):
    accuracy_kf = []
    precision_kf = []
    recall_kf = []
    f1_kf = []

    kf = KFold(n=len(features), shuffle=True, n_folds=k, random_state=42)

    for train_index, test_index in kf:
        features_train = []
        features_test = []
        labels_train = []
        labels_test = []

        for ii in train_index:
            features_train.append(features[ii])
            labels_train.append(labels[ii])
        for jj in test_index:
            features_test.append(features[jj])
            labels_test.append(labels[jj])

        clf.fit(features_train, labels_train)
        prediction = clf.predict(features_test)

        accuracy_kf.append(accuracy_score(labels_test, prediction))
        precision_kf.append(precision_score(labels_test, prediction))
        recall_kf.append(recall_score(labels_test, prediction))
        f1_kf.append(f1_score(labels_test, prediction))

    pprint.pprint(clf.named_steps)
    print 'Accuracy: ' + str(np.mean(accuracy_kf))
    print 'Precision: ' + str(np.mean(precision_kf))
    print 'Recall: ' + str(np.mean(recall_kf))
    print 'F1: ' + str(np.mean(f1_kf))

print 'clf test by using KFold method'
kfold_test(clf, features, labels, k=10)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)