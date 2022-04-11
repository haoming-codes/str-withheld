import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.utils import shuffle
from collections import Counter
import networkx as nx
from itertools import combinations, chain, product
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, brier_score_loss, f1_score
import random
import datetime
from operator import itemgetter
import copy
from sklearn.base import clone
from MDLP import MDLP_Discretizer
from sklearn.datasets import load_iris
import sys
from imblearn.under_sampling import RandomUnderSampler
from scipy.io.arff import loadarff 
from pandas.api.types import is_categorical_dtype
from scipy.special import comb
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
import json
simplefilter("ignore", category=ConvergenceWarning)

def powerset(l):
    """Return the powerset of L, excluding the null set"""
    return [list(i) for i in chain.from_iterable(combinations(l, r) for r in range(1, len(l) + 1))]

def add_D_edges(G, D):
    """In Mincut, add edges from s or to t"""
    for (x, y), freq in D.items():
        if y == 1: G.add_edge('s', x, capacity=freq)
        else: G.add_edge(x, 't', capacity=freq)
    return G

def can_report(d1,d2):
    """Returns true if d1 can misreport as d2"""
    none_indices = [i for i, x in enumerate(d2) if x == None]
    if [x for i,x in enumerate(d1) if i not in none_indices] == [x for i,x in enumerate(d2) if i not in none_indices]:
        return True
    return False

def add_inf_edges(G, D):
    """In Mincut, add edges between data points"""
    ds = list(set([x for x,_ in D.keys()]))
    edges = [(d2, d1, np.inf) for d1 in ds for d2 in ds if d1 != d2 and can_report(d1,d2)]
    G.add_weighted_edges_from(edges, 'capacity')
    return G

def pred(x, S, S_bar, testing=False):
    """In Mincut, predict 1 if x can misreport to an accepted point"""
    if x in S: return 1
    elif x in S_bar: return 0
    else:
        in_S = any([can_report(x,s) for s in S if s != 's'])
        if in_S: return 1
        else: return 0

def minCut(X_train, y_train, X_test, y_test):
    """Mincut classifier. Returns prediction and performance"""
    X = [tuple(x) for x in X_train.values.tolist()]
    y = y_train.T.tolist()
    X_test = [tuple(x) for x in X_test.values.tolist()]
    y_test = y_test.T.tolist()
    D = {d:(c/len(y)) for d,c in Counter(list(zip(X,y))).items()}

    # put together the graph and calculating min-cut
    G = nx.DiGraph()
    G = add_D_edges(G, D)
    G = add_inf_edges(G, D)
    cut_value, (S, S_bar) = nx.minimum_cut(G, 's', 't')

    y_preds = [pred(x,S,S_bar,testing=True) for x in X_test]
    test_score = accuracy_score(y_test, y_preds)
    F1 = f1_score(y_test, y_preds)
    y_preds_train = [pred(x,S,S_bar) for x in X]
    test_score_train = accuracy_score(y_train, y_preds_train)
    F1_train = f1_score(y_train, y_preds_train)
    
    return {"PCC": (test_score_train, test_score), "AUC": None, "F1": (F1_train, F1), "Brier": None, 
            "y": (y_train, y_test), "y_pred": (y_preds_train, y_preds), "y_pred_prob": None}

def clusterMajority(X_test, X, y):
    """returns the majority label of X_test in X"""
    X_test = [tuple(x) for x in X_test.values.tolist()]
    y_preds = []
    for row_test in X_test:
        same_rows = [1 if row == row_test else 0 for row in X]
        if sum(same_rows) == 0:
            y_preds.append(0)
            continue
        targets = [target for target, same in zip(y,same_rows) if same==1]
        majority = max(set(targets), key = targets.count)
        y_preds.append(majority)
    return np.array(y_preds)

def clustering(X_train, y_train, X_test, y_test, X_test_dropped, strategic=False):
    """Mincut classifier. Returns prediction and performance"""
    X = [tuple(x) for x in X_train.values.tolist()]
    y = y_train.T.tolist()

    # prediction
    y_preds = clusterMajority(X_test, X, y)
    if strategic:
        for test in X_test_dropped:
            y_preds = np.vstack([y_preds, clusterMajority(test, X, y)])
        # take union of accept on all imputed test data
        y_preds = np.max(y_preds, axis=0)
    test_score = accuracy_score(y_test, y_preds)
    F1 = f1_score(y_test, y_preds)

    y_preds_train = clusterMajority(X_train, X, y)
    test_score_train = accuracy_score(y_train, y_preds_train)
    F1_train = f1_score(y_train, y_preds_train)

    return {"PCC": (test_score_train, test_score), "AUC": None, "F1": (F1_train, F1), "Brier": None, 
            "y": (y_train, y_test), "y_pred": (y_preds_train, y_preds), "y_pred_prob": None}

def add_nan(data, prop):
    """Make (100*prop)% of the data NaN"""
    data_types = data.dtypes.to_dict()
    cols = data.columns
    data = data.values.tolist()
    # np.random.seed(1)
    for i,x in enumerate(data):
        for j,f in enumerate(x):
            if np.random.random() < prop:
                data[i][j] = np.nan
    data = np.array(data)
    return pd.DataFrame(data,columns=cols).astype(data_types)

def noneToNan(data):
    return data.fillna(value=np.nan)

def nanToNone(data):
    return data.replace({np.nan: None})

def nanToMISSING(data):
    data = data.copy()
    cols = data.columns
    for c in cols:
        if pd.api.types.is_categorical_dtype(data[c]):
            if "MISSING" not in data[c].cat.categories.to_list(): 
                data[c] = data[c].cat.add_categories("MISSING")
            data[c] = data[c].fillna("MISSING")
    return data#.replace({np.nan: "MISSING"})

def nanToZERO(data):
    data = data.copy()
    cols = data.columns
    for c in cols:
        if pd.api.types.is_categorical_dtype(data[c]): continue
        data[c] = data[c].fillna(0.)
    return data#.replace({np.nan: "MISSING"})

def jointPredict(X, classifiers, proba=False):
    """Predict 1 if any one of the applicable classifiers predicts 1"""
    y_pred = pd.DataFrame(data=0, index=X.index, columns=["pred"])
    for i, (feature_comb, clf) in enumerate(classifiers):
        # get data-points in the domain of this clf
        X_clf = X[~np.isnan(X[feature_comb]).any(axis=1)][feature_comb]
        if X_clf.shape[0] == 0: continue
        try: 
            if proba:
                new_column = clf.predict_proba(X_clf)[:,1]
            else:
                new_column = clf.predict(X_clf)
        except NotFittedError:
            continue

        current_pred = y_pred.loc[X_clf.index]["pred"].to_numpy()
        max_pred = np.maximum(current_pred, new_column)
        new_column = pd.Series(max_pred, name="pred", index=X_clf.index)
        y_pred.update(new_column)

    y_pred = y_pred.fillna(0)
    return y_pred["pred"].to_numpy()

def jointPredictNonstrategic(X, classifiers, proba=False, helper_X_train=None, helper_y_train=None):
    """Prediction by the reduced-feature classifier"""
    y_pred = pd.DataFrame(index=X.index, columns=["pred"])
    # X_accepted_idx = pd.Index([])
    for feature_comb, clf in classifiers:
        # get data-points in the domain of this clf
        X_clf = X[~np.isnan(X[feature_comb]).any(axis=1) & np.isnan(X[X.columns[~X.columns.isin(feature_comb)]]).all(axis=1)][feature_comb]
        if X_clf.shape[0] == 0: continue
        try: 
            if proba:
                new_column = pd.Series(clf.predict_proba(X_clf)[:,1], name="pred", index=X_clf.index)
            else:
                new_column = pd.Series(clf.predict(X_clf), name="pred", index=X_clf.index)
        except NotFittedError:
            # print("jointPredictNonstrategic() skipped a classifier:", feature_comb)
            continue
        except ValueError as e:
            # print("jointPredictNonstrategic() trained a tmp KNN")
            tmp_X_clf_idx = helper_X_train.index[~np.isnan(helper_X_train[feature_comb]).any(axis=1) & np.isnan(helper_X_train[helper_X_train.columns[~helper_X_train.columns.isin(feature_comb)]]).all(axis=1)]
            abs_bool = np.isin(helper_X_train.index, tmp_X_clf_idx)
            X_train_fixed, y_train_fixed = fixSameLabel(helper_X_train[abs_bool][feature_comb], helper_y_train[abs_bool])
            tmp_clf = KNeighborsClassifier(n_neighbors=len(y_train_fixed))
            tmp_clf.fit(X_train_fixed, y_train_fixed)
            if proba:
                assert clf.classes_[1] == 1
                new_column = pd.Series(tmp_clf.predict_proba(X_clf)[:,1], name="pred", index=X_clf.index)
            else:
                new_column = pd.Series(tmp_clf.predict(X_clf), name="pred", index=X_clf.index)
        y_pred.update(new_column)

    y_pred = y_pred.fillna(0)
    return y_pred["pred"].to_numpy()

def fixSameLabel(X, y):
    """Add a dummy point if all of X are same label"""
    if np.sum(y) not in [len(y),0]:
        return X,y
    row_num = np.random.choice(X.shape[0])
    X = X.append(X.iloc[row_num]).append(X.iloc[row_num])
    y = np.append(np.append(y,1),0)
    return X,y

def ReducedFeature(X_train, y_train, X_test, y_test, X_test_dropped, model=None, strategic=False, col_mapping=None):
    """A reduced-feature classifier, each subclassifier uses model as the internal classifier"""
    if col_mapping == None: col_mapping = {c:[c] for c in X_train.columns}
    p_features = powerset(col_mapping.keys())
    all_classifiers = [] 
    for feature_comb in p_features: 
        features = list(chain.from_iterable([col_mapping[f] for f in feature_comb]))
        all_classifiers.append((features,clone(model)))

    classifiers = all_classifiers
    for i, (feature_comb, clf) in enumerate(classifiers):
        # get rows where only the features of this classifier are not NaN
        X_train_non_nan_idx = X_train.index[~np.isnan(X_train[feature_comb]).any(axis=1) \
            & np.isnan(X_train[X_train.columns[~X_train.columns.isin(feature_comb)]]).all(axis=1)]
        X_train_clf_idx = X_train_non_nan_idx
        abs_bool = np.isin(X_train.index, X_train_clf_idx)
        if (len(X_train_clf_idx) == 0): continue
        X_train_fixed, y_train_fixed = fixSameLabel(X_train[abs_bool][feature_comb], y_train[abs_bool])
        # train the classifier
        clf.fit(X_train_fixed, y_train_fixed)

    # prediction
    y_preds = jointPredictNonstrategic(X_test, classifiers, helper_X_train=X_train, helper_y_train=y_train)
    y_preds_prob = jointPredictNonstrategic(X_test, classifiers, proba=True, helper_X_train=X_train, helper_y_train=y_train)
    if strategic:
        for test in X_test_dropped:
            y_preds_prob = np.vstack([y_preds_prob, jointPredictNonstrategic(test, classifiers, proba=True, helper_X_train=X_train, helper_y_train=y_train)])
            y_preds = np.vstack([y_preds, jointPredictNonstrategic(test, classifiers, helper_X_train=X_train, helper_y_train=y_train)])
        # take union of accept on all imputed test data
        y_preds = np.max(y_preds, axis=0)
        y_preds_prob = np.max(y_preds_prob, axis=0)

    test_score = accuracy_score(y_test, y_preds)
    auc = roc_auc_score(y_test, y_preds_prob)
    brier = -brier_score_loss(y_test, y_preds_prob)
    F1 = f1_score(y_test, y_preds)

    y_preds_prob_train = jointPredictNonstrategic(X_train, classifiers, proba=True, helper_X_train=X_train, helper_y_train=y_train)
    y_preds_train = jointPredictNonstrategic(X_train, classifiers, helper_X_train=X_train, helper_y_train=y_train)
    test_score_train = accuracy_score(y_train, y_preds_train)
    auc_train = roc_auc_score(y_train, y_preds_prob_train)
    brier_train = -brier_score_loss(y_train, y_preds_prob_train)
    F1_train = f1_score(y_train, y_preds_train)

    return {"PCC": (test_score_train, test_score), "AUC": (auc_train, auc), "F1": (F1_train, F1), "Brier": (brier_train, brier), 
            "y": (y_train, y_test), "y_pred": (y_preds_train, y_preds), "y_pred_prob": (y_preds_prob_train, y_preds_prob)}

def HillClimbing(X_train, y_train, X_test, y_test, model=None, rounds=10, stop_criterion=0.0001, clf_order="random", col_mapping=None, approx_subsets=False):
    """A HC classifier, each subclassifier uses model as the internal classifier"""
    if col_mapping == None: col_mapping = {c:[c] for c in X_train.columns}
    all_classifiers = []

    # n = len(col_mapping)
    # rs = range(1, n + 1)
    # num_combs = [comb(n, r, exact=True) for r in rs]
    # prob_combs = [num/sum(num_combs) for num in num_combs]
    # lengths = sorted(np.random.choice(rs, approx_subsets, replace=True, p=prob_combs))
    # for l in lengths:
    #     while True:
    #         subset = np.random.choice(list(col_mapping.keys()), l, replace=False)
    #         features = list(chain.from_iterable([col_mapping[f] for f in subset]))
    #         if features in all_classifiers: continue
    #         all_classifiers.append((features,clone(model)))
    #         break
    if approx_subsets:
        subsets = []
        subset = [list(c) for c in list(combinations(list(col_mapping.keys()), 1))]
        if len(subset)>approx_subsets: subset = random.sample(subset, approx_subsets)
        subsets += subset
        subset = [list(c) for c in list(combinations(list(col_mapping.keys()), 2))]
        if len(subset)>approx_subsets: subset = random.sample(subset, approx_subsets)
        subsets += subset
        for subset in subsets:
            features = list(chain.from_iterable([col_mapping[f] for f in subset]))
            all_classifiers.append((features,clone(model)))
    else: 
        p_features = powerset(col_mapping.keys())
        all_classifiers = [] 
        for feature_comb in p_features: 
            features = list(chain.from_iterable([col_mapping[f] for f in feature_comb]))
            all_classifiers.append((features,clone(model)))

    train_accuracies = []
    test_accuracies = []
    train_accuracies_proba = []
    test_accuracies_proba = []

    if clf_order == "random": 
        random.shuffle(all_classifiers)
        classifiers = all_classifiers
    elif clf_order == "less_feature": 
        classifiers = list(reversed(all_classifiers))
    # print(len(classifiers), [i for i,j in classifiers])
    for round_num in range(rounds):
        print("    Round", round_num)
        for i, (feature_comb, clf) in enumerate(classifiers):
            other_classifiers = classifiers[:i]+classifiers[i+1:]
            # get rows where the features of this classifier are not NaN
            X_train_non_nan_idx = X_train.index[~np.isnan(X_train[feature_comb]).any(axis=1)]
            # get data-points not accepted by any other clf
            X_train_accepted_idx = X_train.index[jointPredict(X_train, other_classifiers).astype(bool)]
            X_train_rejected_idx = X_train.index[~X_train.index.isin(X_train_accepted_idx)]
            # training data is the intersection of the non nan-rows and the rejected
            X_train_clf_idx = X_train_rejected_idx.intersection(X_train_non_nan_idx)
            abs_bool = np.isin(X_train.index, X_train_clf_idx)
            if (len(X_train_clf_idx) == 0): continue
            X_train_fixed, y_train_fixed = fixSameLabel(X_train[abs_bool][feature_comb], y_train[abs_bool])
            # retrain the classifier
            clf.fit(X_train_fixed, y_train_fixed)

        y_preds = jointPredict(X_test, classifiers)
        test_accuracies.append(accuracy_score(y_test, y_preds))
        # check stopping criterion
        if round_num >= 1:
            delta_acc = test_accuracies[-1] - test_accuracies[-2]
            if abs(delta_acc) <= stop_criterion:
                print("    Finished: delta < stop_criterion")
                break
    print("    Finished: after", round_num+1, "rounds")

    y_preds_prob = jointPredict(X_test, classifiers, proba=True)
    y_preds = jointPredict(X_test, classifiers)
    test_score = accuracy_score(y_test, y_preds)
    auc = roc_auc_score(y_test, y_preds_prob)
    brier = -brier_score_loss(y_test, y_preds_prob)
    F1 = f1_score(y_test, y_preds)

    y_preds_prob_train = jointPredict(X_train, classifiers, proba=True)
    y_preds_train = jointPredict(X_train, classifiers)
    test_score_train = accuracy_score(y_train, y_preds_train)
    auc_train = roc_auc_score(y_train, y_preds_prob_train)
    brier_train = -brier_score_loss(y_train, y_preds_prob_train)
    F1_train = f1_score(y_train, y_preds_train)

    return {"PCC": (test_score_train, test_score), "AUC": (auc_train, auc), "F1": (F1_train, F1), "Brier": (brier_train, brier), 
            "y": (y_train, y_test), "y_pred": (y_preds_train, y_preds), "y_pred_prob": (y_preds_prob_train, y_preds_prob)}

def getDroppedXLinear(X, classifier, col_mapping, imp_values, always_misreport=True, X_true=None):
    if X_true is None: X_true = X
    X_true = [{col: val for col, val in zip(X_true.columns,x)} for x in X_true.values.tolist()]
    X_tmp = [{col: val for col, val in zip(X.columns,x)} for x in X.values.tolist()]
    new_X = []
    if isinstance(classifier, list):
        y_preds = classifier[-1].predict(X)
        coefs = {col: coef for col, coef in zip(X.columns, [[clf.coef_[0,i] for clf in classifier] for i in range(len(X.columns))])}
    else:
        y_preds = classifier.predict(X)
        coef = {col: coef for col, coef in zip(X.columns, classifier.coef_[0,:])}
    
    # for every data point
    for row_tmp, row, y in zip(X_tmp,X_true,y_preds):
        new_row = row_tmp.copy()
        # if already accepted, skip this data point
        if y == 1 and always_misreport == False:
            new_X.append(new_row)
            continue
        # for every feature value new_row[f_dummy]
        for f in col_mapping:
            # get coef[f]*data[f]
            if isinstance(classifier, list):
                product_this_row = sum([sum([row_tmp[sub_col]*coef for coef in coefs[sub_col]]) for sub_col in col_mapping[f]])
                product_imp_values = sum([sum([imp_values[sub_col]*coef for coef in coefs[sub_col]]) for sub_col in col_mapping[f]])
                product_true_row = sum([sum([row[sub_col]*coef for coef in coefs[sub_col]]) for sub_col in col_mapping[f]])
            else:
                product_this_row = sum([row_tmp[sub_col]*coef[sub_col] for sub_col in col_mapping[f]])
                product_imp_values = sum([imp_values[sub_col]*coef[sub_col] for sub_col in col_mapping[f]])
                product_true_row = sum([row[sub_col]*coef[sub_col] for sub_col in col_mapping[f]])
            # if imp value better, use imp value
            if product_imp_values > product_true_row:
                for f_dummy in col_mapping[f]:
                    new_row[f_dummy] = imp_values[f_dummy]
            # if true value better, use true value
            elif product_true_row > product_imp_values:
                for f_dummy in col_mapping[f]:
                    new_row[f_dummy] = row[f_dummy]
            # else, no change
        new_X.append(new_row)
    X_dropped_imp = pd.DataFrame([[row[c] for c in X.columns] for row in new_X], index=X.index, columns=X.columns)
    return X_dropped_imp
            

def impClf(X_train, y_train, X_test, y_test, X_test_dropped, model, strategic=False, imp_values=None, col_mapping=None):
    """X_train has been pre-imputed"""
    clf = clone(model)
    clf.fit(X_train, y_train)
    # coef = {col: coef for col, coef in zip(X_train.columns, clf.coef_[0,:])}
    # prediction
    y_preds = clf.predict(X_test)
    y_preds_prob = clf.predict_proba(X_test)[:,1]
    assert clf.classes_[1] == 1
    if strategic:
        if X_test_dropped == None:
            # print(X_test)
            # print(imp_values)
            X_test_dropped_imp = getDroppedXLinear(X_test, clf, col_mapping, imp_values)
            # print(X_test_dropped_imp)
            y_preds = clf.predict(X_test_dropped_imp)
            y_preds_prob = clf.predict_proba(X_test_dropped_imp)[:,1]
        else:
            for test in X_test_dropped:
                y_preds_prob = np.vstack([y_preds_prob, clf.predict_proba(test)[:,1]])
                y_preds = np.vstack([y_preds, clf.predict(test)])
            # take union of accept on all imputed test data
            y_preds = np.max(y_preds, axis=0)
            y_preds_prob = np.max(y_preds_prob, axis=0)

    test_score = accuracy_score(y_test, y_preds)
    auc = roc_auc_score(y_test, y_preds_prob)
    brier = -brier_score_loss(y_test, y_preds_prob)
    F1 = f1_score(y_test, y_preds)

    y_preds_prob_train = clf.predict_proba(X_train)[:,1]
    y_preds_train = clf.predict(X_train)
    test_score_train = accuracy_score(y_train, y_preds_train)
    auc_train = roc_auc_score(y_train, y_preds_prob_train)
    brier_train = -brier_score_loss(y_train, y_preds_prob_train)
    F1_train = f1_score(y_train, y_preds_train)

    return {"PCC": (test_score_train, test_score), "AUC": (auc_train, auc), "F1": (F1_train, F1), "Brier": (brier_train, brier), 
            "y": (y_train, y_test), "y_pred": (y_preds_train, y_preds), "y_pred_prob": (y_preds_prob_train, y_preds_prob)}

def iterative(X_train, y_train, X_train_dropped, X_test, y_test, X_test_dropped, model, strategic_training=False, strategic_testing=False, col_mapping=None, imp_values=None, historic=False):

    clf = clone(model)
    clf.fit(X_train, y_train)
    clfs = []
    if strategic_training:
        if X_train_dropped==None:
            X_train_tmp = X_train.copy()
            itr_count = 0
            if writing:
                f = open(FILE_SPACE+ITERTATIVE_LOG_FILE, "a")
                f.write(str(exp)+":{")
                f.close()
            while itr_count < 100:
                # print train and test error
                y_preds = clf.predict(X_test)
                y_preds_prob = clf.predict_proba(X_test)[:,1]
                assert clf.classes_[1] == 1
                if strategic_testing:
                    # for test in X_test_dropped:
                    #     y_preds_prob = np.vstack([y_preds_prob, clf.predict_proba(test)[:,1]])
                    #     y_preds = np.vstack([y_preds, clf.predict(test)])
                    # y_preds = np.max(y_preds, axis=0)
                    # y_preds_prob = np.max(y_preds_prob, axis=0)
                    X_test_tmp = getDroppedXLinear(X_test, clf, col_mapping, imp_values, always_misreport=False, X_true=X_test)
                    y_preds_prob = clf.predict_proba(X_test_tmp)[:,1]
                    y_preds = clf.predict(X_test_tmp)
                test_score = accuracy_score(y_test, y_preds)
                y_preds_prob_train = clf.predict_proba(X_train)[:,1]
                y_preds_train = clf.predict(X_train)
                test_score_train = accuracy_score(y_train, y_preds_train)
                coef_values = {name: value for name, value in zip(X_train.columns,clf.coef_[0,:])}
                iterative_log={"training_acc": test_score_train, "test_acc": test_score, "coef": coef_values}
                # print(iterative_log["training_acc"], iterative_log["test_acc"])
                if writing:
                    np.set_printoptions(threshold=np.inf)
                    f = open(FILE_SPACE+ITERTATIVE_LOG_FILE, "a")
                    f.write(str(itr_count)+":"+str(iterative_log)+",")
                    f.close()

                itr_count += 1
                if itr_count == 100: break
                encoded = (clf.coef_, clf.intercept_, clf.penalty, clf.C, clf.classes_)
                clf_copy = LogisticRegression(solver='lbfgs', max_iter=2000)
                clf_copy.coef_, clf_copy.intercept_, clf_copy.penalty, clf_copy.C, clf_copy.classes_ = encoded[0], encoded[1], encoded[2], encoded[3], encoded[4] 
                clfs.append(clf_copy)
                # X_train_tmp = X_train_tmp.copy()
                # print(X_train_tmp)
                # print(clf.coef_)
                if historic:
                    X_train_tmp = getDroppedXLinear(X_train_tmp, clfs, col_mapping, imp_values, always_misreport=False, X_true=X_train)
                else:
                    X_train_tmp = getDroppedXLinear(X_train_tmp, clf, col_mapping, imp_values, always_misreport=False, X_true=X_train)
                # print(X_train_tmp)
                # print(X_train_tmp,"\n\n\n")
                clf.fit(X_train_tmp, y_train)
            # print("    training converged in", itr_count, "itrs")
            if writing:
                f = open(FILE_SPACE+ITERTATIVE_LOG_FILE, "a")
                f.write("},")
                f.close()
        else:
            X_train_tmp = X_train.copy()
            X_train_l = X_train.to_numpy().tolist()
            y_train_l = y_train.tolist()
            X_train_dropped_l = [X.to_numpy().tolist() for X in X_train_dropped]
            y_pred_prev = None
            y_pred_prob_prev = None
            y_pred = clf.predict(X_train)
            y_pred_prob = clf.predict_proba(X_train)[:,1]
            y_pred_dropped = [clf.predict(X).tolist() for X in X_train_dropped]
            itr_count = 0
            if writing:
                f = open(FILE_SPACE+ITERTATIVE_LOG_FILE, "a")
                f.write(str(exp)+":{")
                f.close()
            # while not np.array_equal(y_pred, y_pred_prev):
            # while not np.array_equal(y_pred_prob, y_pred_prob_prev):
            while itr_count < 20:
                # print train and test error
                print(itr_count)
                y_preds = clf.predict(X_test)
                y_preds_prob = clf.predict_proba(X_test)[:,1]
                assert clf.classes_[1] == 1
                if strategic_testing:
                    for test in X_test_dropped:
                        y_preds_prob = np.vstack([y_preds_prob, clf.predict_proba(test)[:,1]])
                        y_preds = np.vstack([y_preds, clf.predict(test)])
                    y_preds = np.max(y_preds, axis=0)
                    y_preds_prob = np.max(y_preds_prob, axis=0)
                test_score = accuracy_score(y_test, y_preds)
                y_preds_prob_train = clf.predict_proba(X_train)[:,1]
                y_preds_train = clf.predict(X_train)
                test_score_train = accuracy_score(y_train, y_preds_train)
                coef_values = {name: value for name, value in zip(X_train.columns,clf.coef_[0,:])}
                iterative_log={"training_acc": test_score_train, "test_acc": test_score, "coef": coef_values}
                if writing:
                    np.set_printoptions(threshold=np.inf)
                    f = open(FILE_SPACE+ITERTATIVE_LOG_FILE, "a")
                    f.write(str(itr_count)+":"+str(iterative_log)+",")
                    f.close()

                itr_count += 1
                if itr_count == 20: break
                # y_pred_prev = y_pred
                # y_pred_prob_prev = y_pred_prob
                encoded = (clf.coef_, clf.intercept_, clf.penalty, clf.C, clf.classes_)
                clf_copy = LogisticRegression(solver='lbfgs', max_iter=2000)
                clf_copy.coef_, clf_copy.intercept_, clf_copy.penalty, clf_copy.C, clf_copy.classes_ = encoded[0], encoded[1], encoded[2], encoded[3], encoded[4] 
                clfs.append(clf_copy)
                for i, y_pred_value in enumerate(y_pred):
                    if y_pred_value == 1: continue
                    if   historic == False: row_options = [X[i] for drop_ith, X in enumerate(X_train_dropped_l) if y_pred_dropped[drop_ith][i]==1]
                    elif historic == True:  row_options = [X[i] for drop_ith, X in enumerate(X_train_dropped_l)]
                    if len(row_options) == 0: continue
                    # new_row = random.choice(row_options)
                    if historic == True: y_row = [sum([(clf.predict_proba(np.array(option).reshape(1, -1))[:,1]).item(0) for clf in clfs]) for option in row_options]
                    elif historic == False: y_row = [(clf.predict_proba(np.array(option).reshape(1, -1))[:,1]).item(0) for option in row_options]
                    
                    index = y_row.index(max(y_row))
                    new_row = row_options[index]
                    X_train_l[i] = new_row
                X_train_tmp = pd.DataFrame(X_train_l, index=X_train.index, columns=X_train.columns)
                clf.fit(X_train_tmp, y_train)
                y_pred = clf.predict(X_train)
                # y_pred_prob = clf.predict_proba(X_train)[:,1]
                y_pred_dropped = [clf.predict(X).tolist() for X in X_train_dropped]
            print("    training converged in", itr_count, "itrs")
            if writing:
                f = open(FILE_SPACE+ITERTATIVE_LOG_FILE, "a")
                f.write("},")
                f.close()
    # y_preds = clf.predict(X_test)
    # y_preds_prob = clf.predict_proba(X_test)[:,1]
    # assert clf.classes_[1] == 1
    # if strategic_testing:
    #     for test in X_test_dropped:
    #         y_preds_prob = np.vstack([y_preds_prob, clf.predict_proba(test)[:,1]])
    #         y_preds = np.vstack([y_preds, clf.predict(test)])
    #     y_preds = np.max(y_preds, axis=0)
    #     y_preds_prob = np.max(y_preds_prob, axis=0)
    # print(X_train.columns)
    # print(clf.coef_)
    y_preds = clf.predict(X_test)
    y_preds_prob = clf.predict_proba(X_test)[:,1]
    if strategic_testing:
        X_test_dropped_imp = getDroppedXLinear(X_test, clf, col_mapping, imp_values, always_misreport=False)
        y_preds = clf.predict(X_test_dropped_imp)
        y_preds_prob = clf.predict_proba(X_test_dropped_imp)[:,1]
    test_score = accuracy_score(y_test, y_preds)

    y_preds_prob_train = clf.predict_proba(X_train)[:,1]
    y_preds_train = clf.predict(X_train)
    if strategic_testing:
        X_train_dropped_imp = getDroppedXLinear(X_train, clf, col_mapping, imp_values, always_misreport=False)
        y_preds_train = clf.predict(X_train_dropped_imp)
        y_preds_prob_train = clf.predict_proba(X_train_dropped_imp)[:,1]
    test_score_train = accuracy_score(y_train, y_preds_train)

    test_score = accuracy_score(y_test, y_preds)
    auc = roc_auc_score(y_test, y_preds_prob)
    brier = -brier_score_loss(y_test, y_preds_prob)
    F1 = f1_score(y_test, y_preds)

    y_preds_prob_train = clf.predict_proba(X_train)[:,1]
    y_preds_train = clf.predict(X_train)
    test_score_train = accuracy_score(y_train, y_preds_train)
    auc_train = roc_auc_score(y_train, y_preds_prob_train)
    brier_train = -brier_score_loss(y_train, y_preds_prob_train)
    F1_train = f1_score(y_train, y_preds_train)

    return {"PCC": (test_score_train, test_score), "AUC": (auc_train, auc), "F1": (F1_train, F1), "Brier": (brier_train, brier), 
            "y": (y_train, y_test), "y_pred": (y_preds_train, y_preds), "y_pred_prob": (y_preds_prob_train, y_preds_prob)}
    
def allFeatureDrop(X, cols=None):
    """Return a list of X's with all combination of columns being NaN"""
    if cols == None: cols = X.columns
    data_types = X.dtypes.to_dict()
    X_dropped = []
    p_features = powerset(cols)
    for fs in p_features:
        # cannot drop all feature values
        if len(fs) == len(cols): continue
        X_tmp = X.copy()
        # drop feature values in fs
        X_tmp.loc[:,fs] = np.nan
        X_dropped.append(X_tmp.astype(data_types))
    return X_dropped

def oneHot(X):
    """One-hot encodes X"""
    cols = X.columns
    df1 = pd.get_dummies(X, prefix_sep=ENC_SEP)
    for c in cols:
        df1.loc[X[c].isnull(), df1.columns.str.startswith(c+ENC_SEP)] = np.nan
    return df1

def imputeTrainTest(X_train_nan, X_test_nan, get_imp_values=False, imp_cat_MISSING=False, imp_num_ZERO=False):
    """Imputes training and testing sets with training set mean/mode"""
    data_types = X_train_nan.dtypes.to_dict()
    cols = X_train_nan.columns
    X_train_nan_numerical = X_train_nan.select_dtypes(exclude='category')
    X_train_nan_categorical = X_train_nan.select_dtypes(include='category')
    imp_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    acc_values = {}
    if not X_train_nan_numerical.empty:
        imputer = imp_mean.fit(X_train_nan_numerical)
        values = dict(zip(X_train_nan_numerical.columns, imputer.statistics_))
        if imp_num_ZERO: values = {key: 0. for key in values}
        # print(values)
        acc_values.update(values)
        X_train_nan_numerical_imp = pd.DataFrame(imp_mean.fit_transform(X_train_nan_numerical), columns=X_train_nan_numerical.columns, index=X_train_nan_numerical.index)
        X_test_imp = X_test_nan.fillna(value=values)
        # print("in", X_test_imp)
    else:
        X_train_nan_numerical_imp = X_train_nan_numerical
        X_test_imp = X_test_nan

    if not X_train_nan_categorical.empty:
        imputer = imp_mode.fit(X_train_nan_categorical)
        values = dict(zip(X_train_nan_categorical.columns, imputer.statistics_))
        if imp_cat_MISSING: values = {key: "MISSING" for key in values}
        acc_values.update(values)
        X_train_nan_categorical_imp = pd.DataFrame(imp_mode.fit_transform(X_train_nan_categorical), columns=X_train_nan_categorical.columns, index=X_train_nan_categorical.index)
        X_test_imp = X_test_imp.fillna(value=values)
    else:
        X_train_nan_categorical_imp = X_train_nan_categorical
        X_test_imp = X_test_imp
    assert(len(acc_values)==len(X_train_nan.columns))
    X_train_imp = X_train_nan_numerical_imp.join(X_train_nan_categorical_imp)
    if get_imp_values: return X_train_imp[cols].astype(data_types), X_test_imp[cols].astype(data_types), acc_values
    else: return X_train_imp[cols].astype(data_types), X_test_imp[cols].astype(data_types)

# def modelSelect(clf_name, X_nan, X_dropped, X_nan_dummy, X_dropped_dummy, X_original, X_original_nan, X_original_dummy, y, col_mapping, criterion):
def modelSelect(clf_name, col_mapping, dis_col_mapping, dis_col_mapping_imp, criterion,
            X_MISSING_dummy_scaled, 
            dis_X_MISSING_dummy, 
            X_imp_dummy_scaled, 
            dis_X_imp_dummy, 
            X_nan_dummy_scaled,
            dis_X_nan_dummy,
            X_none_dummy,
            dis_X_none_dummy,
            y,
            X_imp_dropped_dummy_scaled=None,
            dis_X_imp_dropped_dummy=None,
            X_dropped_dummy_scaled=None,
            dis_X_nan_dropped_dummy=None):
    """For each classifier (clf_name), perform model selection in a 5-fold CV"""
    print("  model selection for", clf_name, flush=True)
    if clf_name[:2] == "LR":
        return {"PCC": LogisticRegression(solver='lbfgs', max_iter=2000), "AUC": LogisticRegression(solver='lbfgs', max_iter=2000), "F1": LogisticRegression(solver='lbfgs', max_iter=2000), "Brier": LogisticRegression(solver='lbfgs', max_iter=2000)}
    elif clf_name[:3] == "min":
        return {"PCC": None, "AUC": None, "F1": None, "Brier": None}
    elif clf_name[:10] == "clustering":
        return {"PCC": None, "AUC": None, "F1": None, "Brier": None}
    elif clf_name[:2] == "NN":
        models = [MLPClassifier(max_iter=3000, alpha=a, activation="logistic", hidden_layer_sizes=(h,)) for a in list(np.float_power(10,np.array([-3,-2,-1]))) for h in [5,10,15]]
    elif clf_name[:2] == "RF":
        models = [RandomForestClassifier(n_estimators=n_tree) for n_tree in [100, 200, 300, 400, 500]]
    elif clf_name[:2] == "KN":
        models = [KNeighborsClassifier(n_neighbors=n_nb) for n_nb in [3,5,7,9,11]]
    model_metric_acc = {mod: {metric: [] for metric in criterion} for mod in models}

    # K-fold CV
    kf = KFold(n_splits=K, shuffle=True)
    CV_error = {test: [] for test in LIST_OF_TESTS}
    for CV_itr, (train_index, test_index) in enumerate(kf.split(X_nan_dummy_scaled)):

        # splitting train/test
        X_train_MISSING_dummy_scaled, X_test_MISSING_dummy_scaled = X_MISSING_dummy_scaled.iloc[train_index], X_MISSING_dummy_scaled.iloc[test_index]
        dis_X_train_MISSING_dummy, dis_X_test_MISSING_dummy =  dis_X_MISSING_dummy.iloc[train_index], dis_X_MISSING_dummy.iloc[test_index]
        X_train_imp_dummy_scaled, X_test_imp_dummy_scaled = X_imp_dummy_scaled.iloc[train_index], X_imp_dummy_scaled.iloc[test_index]
        dis_X_train_imp_dummy, dis_X_test_imp_dummy = dis_X_imp_dummy.iloc[train_index], dis_X_imp_dummy.iloc[test_index]
        X_train_nan_dummy_scaled, X_test_nan_dummy_scaled, = X_nan_dummy_scaled.iloc[train_index], X_nan_dummy_scaled.iloc[test_index]
        dis_X_train_nan_dummy, dis_X_test_nan_dummy = dis_X_nan_dummy.iloc[train_index], dis_X_nan_dummy.iloc[test_index]
        X_train_none_dummy, X_test_none_dummy = X_none_dummy.iloc[train_index], X_none_dummy.iloc[test_index]
        dis_X_train_none_dummy, dis_X_test_none_dummy = dis_X_none_dummy.iloc[train_index], dis_X_none_dummy.iloc[test_index]
        if PREPROCESS_DROPPING:
            X_train_imp_dropped_dummy_scaled, X_test_imp_dropped_dummy_scaled = [X.iloc[train_index] for X in X_imp_dropped_dummy_scaled], [X.iloc[test_index] for X in X_imp_dropped_dummy_scaled]
            dis_X_train_imp_dropped_dummy, dis_X_test_imp_dropped_dummy = [X.iloc[train_index] for X in dis_X_imp_dropped_dummy], [X.iloc[test_index] for X in dis_X_imp_dropped_dummy]
            X_train_dropped_dummy_scaled, X_test_dropped_dummy_scaled = [X.iloc[train_index] for X in X_dropped_dummy_scaled], [X.iloc[test_index] for X in X_dropped_dummy_scaled]
            dis_X_train_nan_dropped_dummy, dis_X_test_nan_dropped_dummy = [X.iloc[train_index] for X in dis_X_nan_dropped_dummy], [X.iloc[test_index] for X in dis_X_nan_dropped_dummy]
        y_train, y_test = y[train_index], y[test_index]

        for mod in models:
            if clf_name=="LR, IC":
                acc = LRIC(X_train_MISSING_dummy_scaled, y_train, X_test_MISSING_dummy_scaled, y_test, None, strategic_testing=False)
            elif clf_name=="LR, IC, L2":
                acc = LRIC(X_train_MISSING_dummy_scaled, y_train, X_test_MISSING_dummy_scaled, y_test, None, strategic_testing=False, penalty="L2")
                # acc = LRIC(X_train_MISSING_dummy_scaled, y_train, X_test_MISSING_dummy_scaled, y_test, X_test_MISSING_dropped_dummy_scaled, strategic_testing=True, penalty="L2")
            elif clf_name=="LR, IC, L2, discretized":
                acc = LRIC(dis_X_train_MISSING_dummy, y_train, dis_X_test_MISSING_dummy, y_test, None, strategic_testing=False, penalty="L2")
            elif clf_name== "LR, imputation, strategic":
                acc = impClf(X_train_imp_dummy_scaled, y_train, X_test_imp_dummy_scaled, y_test, None, model=mod, strategic=True, col_mapping=col_mapping, imp_values=X_train_imp_values_dummy_scaled)
            elif clf_name== "LR, imputation, discretized, strategic":
                acc = impClf(dis_X_train_imp_dummy, y_train, dis_X_test_imp_dummy, y_test, None, model=mod, strategic=True, col_mapping=dis_col_mapping_imp, imp_values=dis_X_train_imp_values_dummy)
            elif clf_name[4:] == "imputation, strategic":
                acc = impClf(X_train_imp_dummy_scaled, y_train, X_test_imp_dummy_scaled, y_test, X_test_imp_dropped_dummy_scaled, model=mod, strategic=True)
            elif clf_name[4:] == "imputation, non-strategic":
                acc = impClf(X_train_imp_dummy_scaled, y_train, X_test_imp_dummy_scaled, y_test, [], model=mod, strategic=False)
            elif clf_name[4:] == "imputation, discretized, strategic":
                acc = impClf(dis_X_train_imp_dummy, y_train, dis_X_test_imp_dummy, y_test, dis_X_test_imp_dropped_dummy, model=mod, strategic=True)
            elif clf_name[4:] == "imputation, discretized, non-strategic":
                acc = impClf(dis_X_train_imp_dummy, y_train, dis_X_test_imp_dummy, y_test, [], model=mod, strategic=False)
            # elif clf_name[4:] == "imputation, non-strategic, no fs":
            #     acc = impClf(X_original_train_imp_dummy_scaled, y_train, X_original_test_imp_dummy_scaled, y_test, [], model=mod, strategic=False)
            # elif clf_name[4:] == "imputation, non-strategic, no fs, no dropping":
            #     acc = impClf(X_original_train_dummy_scaled, y_train, X_original_test_dummy_scaled, y_test, [], model=mod, strategic=False)
            elif clf_name[4:] == "reduced-feature, strategic":
                acc = ReducedFeature(X_train_nan_dummy_scaled, y_train, X_test_nan_dummy_scaled, y_test, X_test_dropped_dummy_scaled, model=mod, strategic=True, col_mapping=col_mapping)
            elif clf_name[4:] == "reduced-feature, non-strategic":
                acc = ReducedFeature(X_train_nan_dummy_scaled, y_train, X_test_nan_dummy_scaled, y_test, [], model=mod, strategic=False, col_mapping=col_mapping)
            elif clf_name[4:] == "reduced-feature, discretized, strategic":
                acc = ReducedFeature(dis_X_train_nan_dummy, y_train, dis_X_test_nan_dummy, y_test, dis_X_test_nan_dropped_dummy, model=mod, strategic=True, col_mapping=dis_col_mapping)
            elif clf_name[4:] == "reduced-feature, discretized, non-strategic":
                acc = ReducedFeature(dis_X_train_nan_dummy, y_train, dis_X_test_nan_dummy, y_test, [], model=mod, strategic=False, col_mapping=dis_col_mapping)
            elif clf_name == "min-cut, discretized":
                acc = minCut(dis_X_train_none_dummy, y_train, dis_X_test_none_dummy, y_test)
            elif clf_name == "min-cut":
                acc = minCut(X_train_none_dummy, y_train, X_test_none_dummy, y_test)
            elif clf_name == "clustering, discretized, strategic":
                acc = clustering(dis_X_train_none_dummy, y_train, dis_X_test_none_dummy, y_test, dis_X_test_none_dropped_dummy, strategic=True)
            elif clf_name == "clustering, strategic":
                acc = clustering(X_train_none_dummy, y_train, X_test_none_dummy, y_test, X_test_none_dropped_dummy, strategic=True)
            elif clf_name == "clustering, discretized, non-strategic":
                acc = clustering(dis_X_train_none_dummy, y_train, dis_X_test_none_dummy, y_test, [], strategic=False)
            elif clf_name == "clustering, non-strategic":
                acc = clustering(X_train_none_dummy, y_train, X_test_none_dummy, y_test, [], strategic=False)
            elif clf_name[4:] == "greedy, discretized":
                acc = HillClimbing(dis_X_train_nan_dummy, y_train, dis_X_test_nan_dummy, y_test, model=mod, clf_order="less_feature", col_mapping=dis_col_mapping)
            elif clf_name[4:] == "greedy":
                acc = HillClimbing(X_train_nan_dummy_scaled, y_train, X_test_nan_dummy_scaled, y_test, model=mod, clf_order="less_feature", col_mapping=col_mapping)
            elif clf_name[4:] == "greedy, discretized, approx":
                acc = HillClimbing(dis_X_train_nan_dummy, y_train, dis_X_test_nan_dummy, y_test, model=mod, clf_order="less_feature", col_mapping=dis_col_mapping, approx_subsets=APPROX_NUM_SUBSETS)
            elif clf_name[4:] == "greedy, approx":
                acc = HillClimbing(X_train_nan_dummy_scaled, y_train, X_test_nan_dummy_scaled, y_test, model=mod, clf_order="less_feature", col_mapping=col_mapping, approx_subsets=APPROX_NUM_SUBSETS)
            else:
                print(clf_name, "???", flush=True)
            for metric in criterion:
                # acc[metric][1] is test accuracy
                model_metric_acc[mod][metric].append(acc[metric][1])

    model_metric_cv = {mod: {metric: np.mean(model_metric_acc[mod][metric]) for metric in model_metric_acc[mod]} for mod in model_metric_acc}
    best_mod = {metric: max(model_metric_cv.items(), key = lambda x: x[1][metric])[0] for metric in criterion}
    return best_mod

def scalerTransform(X, scaler):
    return pd.DataFrame(scaler.transform(X), columns=X.columns)

def featureSelect(X, y, num_features=5, cat_only=False):
    """select num_features number of features based on F_SELECT_METHOD"""
    original_cols = X.columns.to_numpy()
    original_X = X
    # (X,_),y = imputeTrainTest(oneHot(X),oneHot(X)),y
    X,y = oneHot(X),y
    selector = SelectKBest(F_SELECT_METHOD, k="all").fit(X, y)
    scores = selector.scores_
    scores[np.isnan(scores)] = -np.inf
    arr1inds = scores.argsort()
    sorted_enc_features = X.columns.to_numpy()[arr1inds[::-1]]
    to_select = []
    for enc_f in sorted_enc_features.tolist():
        f = enc_f.split(ENC_SEP)[0]
        if cat_only and not is_categorical_dtype(original_X[f].dtype):
            continue
        if f not in to_select:
            to_select.append(f)
        if len(to_select) == num_features:
            break
    return to_select

def openDateset(name):
    """read in datasets"""
    if name == GERMANY_DATASET:
        credit_approval = fetch_openml(name=name, as_frame=True)
        X = credit_approval.data
        y = (credit_approval.target == ACCEPT_CLASS).astype(int).values
        return X,y
    elif name == AUSTRALIAN_DATASET:
        credit_approval = fetch_openml(name=name, as_frame=True)
        X = credit_approval.data
        y = (credit_approval.target == ACCEPT_CLASS).astype(int).values
        return X,y
    elif name == TAIWAN_DATASET:
        data = pd.read_excel(name, index_col=0, header=0, skiprows=[1])
        data=data.astype('float').astype({"X2": "category", 
                                          "X3": "category",
                                          "X4": "category", 
                                          # "X6": "category",
                                          # "X7": "category",
                                          # "X8": "category",
                                          # "X9": "category",
                                          # "X10": "category",
                                          # "X11": "category"
                                          })
        data["X2"] = data["X2"].astype(str)
        data["X3"] = data["X3"].astype(str)
        data["X4"] = data["X4"].astype(str)
        data=data.astype({"X2": "category", 
                                          "X3": "category",
                                          "X4": "category", })
        X = data.drop(columns=['Y'])
        # print(X.dtypes.to_dict())
        y = (data['Y'] == ACCEPT_CLASS).astype(int).values
        return X,y
    elif name == POLISH_DATASET:
        raw_data = loadarff('5year.arff')
        data = pd.DataFrame(raw_data[0])
        data=data.astype('float')
        X = data.drop(columns=['class'])
        y = (data['class'] == ACCEPT_CLASS).astype(int).values
        return X,y
    elif name == RESUME_DATASET:
        data = pd.read_csv(name, index_col=0)
        # data = pd.DataFrame(raw_data[0])
        data=data.astype('float')
        X = data.drop(columns=['Interview'])
        y = (data['Interview'] == ACCEPT_CLASS).astype(int).values
        return X,y
    else:
        print("???")

def log_loss(X, Y, clf, penalty=None):
    beta = clf.coef_[0]
    c = clf.intercept_[0]
    Z = np.matmul(X, beta) + c
    Z = np.multiply(-Y, Z)
    loss_vec = np.log(1 + np.exp(Z))
    if penalty=='L2': return 0.1*np.sum(loss_vec) + 0.5*np.dot(beta, beta)
    else: return 0.1*np.sum(loss_vec) #+ 0.5*np.dot(beta, beta)

def fit_IC(X, Y, clf, tol, maxiter, penalty=None, force_negative=False):
    X = X.values
    epoch = 0
    delta = 1
    loss = 1000000
    while(np.abs(delta) > tol and epoch < maxiter):
        clf.fit(X, Y)
        beta = clf.coef_[0]
        # print("{epo}, delta = {change}, coefs = {c}".format(epo=epoch, change=delta, c=clf.coef_[0]))
        if force_negative: clf.coef_[0] = np.minimum(beta, 0)
        else: clf.coef_[0] = np.maximum(beta, 0)
        new_loss = log_loss(X, Y, clf, penalty)
        delta = loss - new_loss
        loss = new_loss
        # print("{epo}, delta = {change}, coefs = {c}".format(epo=epoch, change=delta, c=clf.coef_[0]))
        epoch = epoch + 1

def negateNegativeFeatures(X_train_nan, X_test_nan, y_train):
    X_train_nan, X_test_nan = X_train_nan.copy(), X_test_nan.copy()
    clf = LogisticRegression(solver='lbfgs', max_iter=2000)
    X_train,_ = imputeTrainTest(X_train_nan, X_train_nan)
    clf.fit(X_train, y_train)
    for i, col in enumerate(X_train_nan.columns):
        if clf.coef_[0][i] < 0 and "=" not in col:
            X_train_nan[col+"_NEGATED"] = -X_train_nan[col]
            X_test_nan[col+"_NEGATED"] = -X_test_nan[col]
    return (X_train_nan, X_test_nan)

def LR(X_train, y_train, X_test, y_test, model):
    clf = clone(model)
    clf.fit(X_train, y_train)
    # print(X_train.columns)
    # print(clf.coef_)

    y_preds = clf.predict(X_test)
    y_preds_prob = clf.predict_proba(X_test)[:,1]
    test_score = accuracy_score(y_test, y_preds)
    auc = roc_auc_score(y_test, y_preds_prob)
    brier = -brier_score_loss(y_test, y_preds_prob)
    F1 = f1_score(y_test, y_preds)

    y_preds_prob_train = clf.predict_proba(X_train)[:,1]
    y_preds_train = clf.predict(X_train)
    test_score_train = accuracy_score(y_train, y_preds_train)
    auc_train = roc_auc_score(y_train, y_preds_prob_train)
    brier_train = -brier_score_loss(y_train, y_preds_prob_train)
    F1_train = f1_score(y_train, y_preds_train)

    return {"PCC": (test_score_train, test_score), "AUC": (auc_train, auc), "F1": (F1_train, F1), "Brier": (brier_train, brier), 
            "y": (y_train, y_test), "y_pred": (y_preds_train, y_preds), "y_pred_prob": (y_preds_prob_train, y_preds_prob)}



def LRIC(X_train_MISSING, y_train, X_test_MISSING, y_test, X_test_MISSING_dropped, strategic_testing=False, penalty=None, negate=False):
    if negate: X_train_MISSING_negated, X_test_MISSING_negated = negateNegativeFeatures(X_train_MISSING, X_test_MISSING, y_train)

    if not negate: X_train_ZERO = X_train_MISSING.fillna(0.0)
    else: X_train_ZERO = X_train_MISSING_negated.fillna(0.0)
    for c in X_train_ZERO.columns:
        if "=MISSING" in c:
            X_train_ZERO.drop(c,axis=1,inplace=True)
    
    if not negate: X_test_ZERO = X_test_MISSING.fillna(0.0)
    else: X_test_ZERO = X_test_MISSING_negated.fillna(0.0)
    for c in X_test_ZERO.columns:
        if "=MISSING" in c:
            X_test_ZERO.drop(c,axis=1,inplace=True)
    if X_test_MISSING_dropped:
        for i in range(len(X_test_MISSING_dropped)):
            X_test_MISSING_dropped[i] = X_test_MISSING_dropped[i].fillna(0.0)
            for c in X_test_MISSING_dropped[i].columns:
                if "=MISSING" in c:
                    X_test_MISSING_dropped[i].drop(c,axis=1,inplace=True)
    y_train_flipped = np.ones(len(y_train))-y_train
    y_test_flipped = np.ones(len(y_test))-y_test
    # print(X_train_ZERO)
    scaler = MinMaxScaler().fit(X_train_ZERO)
    X_train_ZERO = scalerTransform(X_train_ZERO, scaler)
    X_test_ZERO = scalerTransform(X_test_ZERO, scaler)

    if penalty=='L2': clf1 = LogisticRegression(fit_intercept=True, max_iter=1, warm_start=True, penalty='l2')
    else: clf1 = LogisticRegression(fit_intercept=True, max_iter=1, warm_start=True)#, penalty='l2')
    fit_IC(X_train_ZERO, y_train, clf1, 0.001, 1000, force_negative=False)
    y_preds_train_nonflipped = clf1.predict(X_train_ZERO)
    test_score_train_nonflipped = accuracy_score(y_train, y_preds_train_nonflipped)
    # print(clf1.coef_[0])
    
    if penalty=='L2': clf2 = LogisticRegression(fit_intercept=True, max_iter=1, warm_start=True, penalty='l2')
    else: clf2 = LogisticRegression(fit_intercept=True, max_iter=1, warm_start=True)#, penalty='l2')
    fit_IC(X_train_ZERO, y_train_flipped, clf2, 0.001, 1000, force_negative=True)
    y_preds_train_flipped = clf2.predict(X_train_ZERO)
    test_score_train_flipped = accuracy_score(y_train_flipped, y_preds_train_flipped)
    # print(clf2.coef_[0])
    
    if test_score_train_nonflipped >= test_score_train_flipped: 
        clf = clf1
        y_train = y_train
        y_test = y_test
    else: 
        clf = clf2
        y_train = y_train_flipped
        y_test = y_test_flipped
    # print(X_train_ZERO.columns)
    # print(clf.coef_[0])
    y_preds = clf.predict(X_test_ZERO)
    y_preds_prob = clf.predict_proba(X_test_ZERO)[:,1]
    y_preds_prob_train = clf.predict_proba(X_train_ZERO)[:,1]
    y_preds_train = clf.predict(X_train_ZERO)
    # print(clf.coef_)
    assert clf.classes_[1] == 1
    if strategic_testing:
        for test in X_test_MISSING_dropped:
            y_preds_prob = np.vstack([y_preds_prob, clf.predict_proba(test)[:,1]])
            y_preds = np.vstack([y_preds, clf.predict(test)])
        if test_score_train_nonflipped > test_score_train_flipped: 
            y_preds = np.max(y_preds, axis=0)
            y_preds_prob = np.max(y_preds_prob, axis=0)
        else:
            y_preds = np.min(y_preds, axis=0)
            y_preds_prob = np.min(y_preds_prob, axis=0)

    test_score = accuracy_score(y_test, y_preds)
    auc = roc_auc_score(y_test, y_preds_prob)
    brier = -brier_score_loss(y_test, y_preds_prob)
    F1 = f1_score(y_test, y_preds)
    test_score_train = accuracy_score(y_train, y_preds_train)
    auc_train = roc_auc_score(y_train, y_preds_prob_train)
    brier_train = -brier_score_loss(y_train, y_preds_prob_train)
    F1_train = f1_score(y_train, y_preds_train)

    return {"PCC": (test_score_train, test_score), "AUC": (auc_train, auc), "F1": (F1_train, F1), "Brier": (brier_train, brier), 
            "y": (y_train, y_test), "y_pred": (y_preds_train, y_preds), "y_pred_prob": (y_preds_prob_train, y_preds_prob)}




writing = False
assert len(sys.argv[1:]) == 4
K=5
NAN_PROP = float(sys.argv[2])
TEST_PROP = 0.5
CATEGORICAL_ONLY = False
if sys.argv[3] == "bal": BALANCED = True
elif sys.argv[3] == "unbal": BALANCED = False
F_SELECT = "f_classif"
if F_SELECT == "f_classif":
    F_SELECT_METHOD = f_classif
elif F_SELECT == "chi2":
    F_SELECT_METHOD = chi2
if (sys.argv[4]) == "preFS": 
    FT_BOOL = True
    PRE_FT = True
elif (sys.argv[4]) == "FS": 
    FT_BOOL = True
    PRE_FT = False
elif (sys.argv[4]) == "NFS": 
    FT_BOOL = False
    PRE_FT = False
FT_TXT = sys.argv[4]
APPROX_NUM_SUBSETS = 30
PREPROCESS_DROPPING = (FT_BOOL==True)
AUSTRALIAN_DATASET = "credit-approval"
GERMANY_DATASET = "credit-g"
TAIWAN_DATASET = "default of credit card clients.xls"
POLISH_DATASET = "polish" # download and unzip https://archive.ics.uci.edu/ml/machine-learning-databases/00365/data.zip
RESUME_DATASET = "resumes_development.csv"
if (sys.argv[1]) == "australia":
    DATASET = AUSTRALIAN_DATASET
elif (sys.argv[1]) == "germany":
    DATASET = GERMANY_DATASET
elif (sys.argv[1]) == "taiwan":
    DATASET = TAIWAN_DATASET
elif (sys.argv[1]) == "poland":
    DATASET = POLISH_DATASET
elif (sys.argv[1]) == "resume":
    DATASET = RESUME_DATASET
else:
    print("Unknown dataset")
ENC_SEP = "="
METRICS = ["PCC"]#, "AUC", "F1", "Brier"]
if DATASET == GERMANY_DATASET:
    ACCEPT_CLASS = "good"
elif DATASET == AUSTRALIAN_DATASET:
    ACCEPT_CLASS = "-"
elif DATASET == TAIWAN_DATASET:
    ACCEPT_CLASS = 0
elif DATASET == POLISH_DATASET:
    ACCEPT_CLASS = 0
elif DATASET == RESUME_DATASET:
    ACCEPT_CLASS = 1

# if (len(sys.argv[1:]) == 4):
WRITE_FILE = "_"+str(sys.argv[1])+"_"+str(sys.argv[2])+"_"+str(sys.argv[3])+"_"+str(sys.argv[4])+"_"+datetime.datetime.now().strftime("%Y%m%d%H%M%S")+"_"+FT_TXT+".txt"
# else:
#     WRITE_FILE = "_"+str(sys.argv[1])+"_"+str(sys.argv[2])+"_"+str(sys.argv[3])+"_"+datetime.datetime.now().strftime("%Y%m%d%H%M%S")+"_"+FT_TXT+".txt"
ITERTATIVE_LOG_FILE = "_"+str(sys.argv[1])+"_"+str(sys.argv[2])+"_"+str(sys.argv[3])+"_iterative_log.txt"
FILE_SPACE = ""#"/usr/project/xtmp/lih14/postnips2020/weekend/"
if FT_BOOL:
    LIST_OF_TESTS = [
                 "LR, original",
                 # "clustering, strategic",
                 # "clustering, discretized, strategic",
                 # "clustering, non-strategic",
                 # "clustering, discretized, non-strategic",
                 # "KN, imputation, strategic",
                 # "KN, imputation, non-strategic",
                 # "KN, imputation, discretized, strategic",
                 # "KN, imputation, discretized, non-strategic",
                 # # "KN, imputation, non-strategic, no fs",
                 # # "KN, imputation, non-strategic, no fs, no dropping",
                 # "KN, reduced-feature, strategic",
                 # "KN, reduced-feature, non-strategic",
                 # "KN, reduced-feature, discretized, strategic",
                 # "KN, reduced-feature, discretized, non-strategic",
                 # "NN, imputation, strategic", 
                 # "NN, imputation, non-strategic", 
                 # "NN, imputation, discretized, strategic",
                 # "NN, imputation, discretized, non-strategic",
                 # # "NN, imputation, non-strategic, no fs",
                 # # "NN, imputation, non-strategic, no fs, no dropping",
                 # "NN, reduced-feature, strategic",
                 # "NN, reduced-feature, non-strategic",
                 # "NN, reduced-feature, discretized, strategic",
                 # "NN, reduced-feature, discretized, non-strategic",
                 # "LR, iterative, strategic, strategic",
                 # "LR, iterative, strategic, strategic, 2",
                 # "LR, iterative, strategic, non-strategic",
                 # "LR, iterative, non-strategic, strategic",
                 # "LR, iterative, non-strategic, non-strategic",
                 # "LR, imputation, strategic", 
                 # "LR, imputation, non-strategic", 
                 # "LR, imputation, discretized, strategic",
                 # "LR, imputation, discretized, non-strategic",
                 # # "LR, imputation, non-strategic, no fs",
                 # # "LR, imputation, non-strategic, no fs, no dropping",
                 # "LR, reduced-feature, strategic",
                 # "LR, reduced-feature, non-strategic",
                 # "LR, reduced-feature, discretized, strategic",
                 # "LR, reduced-feature, discretized, non-strategic",
                 # "RF, imputation, strategic", 
                 # "RF, imputation, non-strategic", 
                 # "RF, imputation, discretized, strategic",
                 # "RF, imputation, discretized, non-strategic",
                 # # "RF, imputation, non-strategic, no fs",
                 # # "RF, imputation, non-strategic, no fs, no dropping",
                 # "RF, reduced-feature, strategic",
                 # "RF, reduced-feature, non-strategic",
                 # "RF, reduced-feature, discretized, strategic",
                 # "RF, reduced-feature, discretized, non-strategic",
                 # "min-cut",
                 # "min-cut, discretized",
                 # "NN, greedy",
                 # "NN, greedy, discretized",
                 # "NN, greedy, approx",
                 # "NN, greedy, discretized, approx",
                 # "LR, greedy",
                 # "LR, greedy, discretized",
                 # "LR, greedy, approx",
                 # "LR, greedy, discretized, approx",
                 # "LR, IC, L2",
                 # "LR, IC, L2, original",
                 # "LR, IC, L2, discretized",
                 # "LR, IC, L2, negated",
                 ]
else:
    LIST_OF_TESTS = [
                 # "LR, original",
                 # "clustering, strategic",
                 # "clustering, discretized, strategic",
                 # "clustering, non-strategic",
                 # "clustering, discretized, non-strategic",
                 # "KN, imputation, strategic",
                 # "KN, imputation, non-strategic",
                 # "KN, imputation, discretized, strategic",
                 # "KN, imputation, discretized, non-strategic",
                 # "KN, imputation, non-strategic, no fs",
                 # "KN, imputation, non-strategic, no fs, no dropping",
                 # "KN, reduced-feature, strategic",
                 # "KN, reduced-feature, non-strategic",
                 # "KN, reduced-feature, discretized, strategic",
                 # "KN, reduced-feature, discretized, non-strategic",
                 # "NN, imputation, strategic", 
                 # "NN, imputation, non-strategic", 
                 # "NN, imputation, discretized, strategic",
                 # "NN, imputation, discretized, non-strategic",
                 # "NN, imputation, non-strategic, no fs",
                 # "NN, imputation, non-strategic, no fs, no dropping",
                 # "NN, reduced-feature, strategic",
                 # "NN, reduced-feature, non-strategic",
                 # "NN, reduced-feature, discretized, strategic",
                 # "NN, reduced-feature, discretized, non-strategic",
                 # "LR, iterative, strategic, strategic",
                 "LR, iterative, strategic, strategic, 2 zero",
                 # "LR, iterative, strategic, non-strategic",
                 # "LR, iterative, non-strategic, strategic",
                 # "LR, iterative, non-strategic, non-strategic",
                 # "LR, imputation, strategic", 
                 # "LR, imputation, strategic, 2 mean", 
                 "LR, imputation, strategic, 2 zero", 
                 "LR, imputation, non-strategic", 
                 # "LR, imputation, discretized, strategic",
                 # "LR, imputation, discretized, non-strategic",
                 # "LR, imputation, non-strategic, no fs",
                 # "LR, imputation, non-strategic, no fs, no dropping",
                 # "LR, reduced-feature, strategic",
                 # "LR, reduced-feature, non-strategic",
                 # "LR, reduced-feature, discretized, strategic",
                 # "LR, reduced-feature, discretized, non-strategic",
                 # "RF, imputation, strategic", 
                 # "RF, imputation, non-strategic", 
                 # "RF, imputation, discretized, strategic",
                 # "RF, imputation, discretized, non-strategic",
                 # "RF, imputation, non-strategic, no fs",
                 # "RF, imputation, non-strategic, no fs, no dropping",
                 # "RF, reduced-feature, strategic",
                 # "RF, reduced-feature, non-strategic",
                 # "RF, reduced-feature, discretized, strategic",
                 # "RF, reduced-feature, discretized, non-strategic",
                 # "min-cut",
                 # "min-cut, discretized",
                 # "NN, greedy",
                 # "NN, greedy, discretized",
                 # "NN, greedy, approx",
                 # "NN, greedy, discretized, approx",
                 # "LR, greedy",
                 # "LR, greedy, discretized",
                 # "LR, greedy, approx",
                 # "LR, greedy, discretized, approx",
                 "LR, IC, L2",
                 # "LR, IC, L2, discretized",
                 # "LR, IC, L2, negated",
                 ]

X_full,y_full = openDateset(DATASET)
X_full,_ = imputeTrainTest(X_full, X_full)
if PRE_FT and FT_BOOL:
    best_features = featureSelect(X_full, y_full, 4, CATEGORICAL_ONLY)
    print("best features", best_features)

if writing:
    f = open(FILE_SPACE+WRITE_FILE, "w")
    f.write("{")
    f.close()
    f = open(FILE_SPACE+ITERTATIVE_LOG_FILE, "w")
    f.write("{")
    f.close()

for exp in range(300):

    print("\nExp", exp, "---------", flush=True)
    test_acc = {test: {m: None for m in METRICS} for test in LIST_OF_TESTS}

    # balancing
    if BALANCED: X_original,y = RandomUnderSampler().fit_resample(X_full, y_full)
    else: X_original,y = X_full, y_full

    # Add None/nan representing missing feature value by nature
    X_nan = add_nan(X_original, NAN_PROP)
    X_MISSING = nanToMISSING(X_nan)
    X_ZERO = nanToZERO(X_nan)
    # print(X_MISSING)
    # print(X_ZERO)

    # splitting train/test
    X_train_nan, X_test_nan, X_train_MISSING, X_test_MISSING, X_train_original, X_test_original, y_train, y_test \
     = train_test_split(X_nan, X_MISSING, X_original, y, test_size=TEST_PROP)

    # feature selection
    if not PRE_FT and FT_BOOL: 
        X,_ = imputeTrainTest(X_train_nan, X_train_nan)
        best_features = featureSelect(X, y_train, 4, CATEGORICAL_ONLY)
        print("best features", best_features)
    if FT_BOOL:
        X_train_original = X_train_original[best_features]
        X_test_original = X_test_original[best_features]
        X_train_nan = X_train_nan[best_features]
        X_test_nan = X_test_nan[best_features]
        X_train_MISSING = X_train_MISSING[best_features]
        X_test_MISSING = X_test_MISSING[best_features]

    # all combinaitons of feature dropping
    if PREPROCESS_DROPPING:
        X_train_dropped, X_test_dropped = allFeatureDrop(X_train_nan), allFeatureDrop(X_test_nan)
        X_train_MISSING_dropped, X_test_MISSING_dropped = [nanToMISSING(X) for X in X_train_dropped], [nanToMISSING(X) for X in X_test_dropped]

    # impute the nan with training set mean
    X_train_imp, X_test_imp, X_train_imp_values = imputeTrainTest(X_train_nan, X_test_nan, get_imp_values=True)
    # X_train_ZERO_imp, X_test_ZERO_imp, X_train_ZERO_imp_values = imputeTrainTest(X_train_nan, X_test_nan, get_imp_values=True, imp_num_ZERO=True)
    X_train_MISSING_imp, X_test_MISSING_imp, X_train_MISSING_imp_values = imputeTrainTest(X_train_MISSING, X_test_MISSING, get_imp_values=True, imp_cat_MISSING=True)
    # print(X_train_MISSING)
    # print(X_test_MISSING)
    # # print(X_train_ZERO_imp)
    # print(X_train_MISSING_imp)
    # print(X_test_MISSING_imp)
    # X_original_train_imp, X_original_test_imp = imputeTrainTest(X_original_train_nan, X_original_test_nan)
    if PREPROCESS_DROPPING: 
        X_test_imp_dropped = [imputeTrainTest(X_train_nan, X)[1] for X in X_test_dropped]
        X_train_imp_dropped = [imputeTrainTest(X_train_nan, X)[1] for X in X_train_dropped]

    # the discretized data
    X_train_nan_y = X_train_nan.assign(y=y_train)
    X_test_nan_y = X_test_nan.assign(y=y_test)
    discretizer = MDLP_Discretizer(dataset=X_train_nan_y, class_label="y")
    dis_X_test_nan = discretizer.apply_cutpoints(X_test_nan_y).drop(columns=['y'])
    dis_X_train_nan = discretizer.apply_cutpoints(X_train_nan_y).drop(columns=['y'])
    if PREPROCESS_DROPPING: 
        dis_X_test_dropped = [discretizer.apply_cutpoints(X.assign(y=y_test)).drop(columns=['y']) for X in X_test_dropped]
        dis_X_train_dropped = [discretizer.apply_cutpoints(X.assign(y=y_test)).drop(columns=['y']) for X in X_train_dropped]
    # the discretized imputed data
    X_train_imp_y = X_train_imp.assign(y=y_train)
    X_test_imp_y = X_test_imp.assign(y=y_test)
    discretizer = MDLP_Discretizer(dataset=X_train_imp_y, class_label="y")
    dis_X_test_imp = discretizer.apply_cutpoints(X_test_imp_y).drop(columns=['y'])
    dis_X_train_imp = discretizer.apply_cutpoints(X_train_imp_y).drop(columns=['y'])
    dis_X_train_imp_values = discretizer.apply_cutpoints(pd.DataFrame.from_dict({key: [X_train_imp_values[key]] for key in X_train_imp.columns})).iloc[0].to_dict()
    if PREPROCESS_DROPPING: 
        dis_X_test_imp_dropped = [discretizer.apply_cutpoints(X.assign(y=y_test)).drop(columns=['y']) for X in X_test_imp_dropped]
        dis_X_train_imp_dropped = [discretizer.apply_cutpoints(X.assign(y=y_test)).drop(columns=['y']) for X in X_train_imp_dropped]
    # discretize and impute the MISSING data
    X_train_MISSING_y = X_train_MISSING.assign(y=y_train)
    X_test_MISSING_y = X_test_MISSING.assign(y=y_test)
    discretizer = MDLP_Discretizer(dataset=X_train_MISSING_y, class_label="y")
    dis_X_train_MISSING = nanToMISSING(discretizer.apply_cutpoints(X_train_MISSING_y).drop(columns=['y']))
    dis_X_test_MISSING = nanToMISSING(discretizer.apply_cutpoints(X_test_MISSING_y).drop(columns=['y']))

    # one-hot encode the dataset
    # X_original_train_imp_dummy, X_original_test_imp_dummy = oneHot(X_original_train_imp), oneHot(X_original_test_imp)
    # X_original_train_dummy, X_original_test_dummy = oneHot(X_original_train), oneHot(X_original_test)
    # X_original_train_nan_dummy, X_original_test_nan_dummy = oneHot(X_original_train_nan), oneHot(X_original_test_nan)
    X_train_original_dummy, X_test_original_dummy = oneHot(X_train_original), oneHot(X_test_original)
    X_train_nan_dummy, X_test_nan_dummy = oneHot(X_train_nan), oneHot(X_test_nan)
    X_train_none_dummy,X_test_none_dummy = nanToNone(X_train_nan_dummy), nanToNone(X_test_nan_dummy)
    X_train_MISSING_dummy, X_test_MISSING_dummy = oneHot(X_train_MISSING), oneHot(X_test_MISSING)
    if PREPROCESS_DROPPING: X_train_dropped_dummy, X_test_dropped_dummy = [oneHot(X) for X in X_train_dropped], [oneHot(X) for X in X_test_dropped]
    if PREPROCESS_DROPPING: X_test_none_dropped_dummy = [nanToNone(X) for X in X_test_dropped_dummy]
    X_train_imp_dummy, X_test_imp_dummy, X_train_imp_values_dummy = oneHot(X_train_imp), oneHot(X_test_imp), oneHot(pd.concat([pd.DataFrame.from_dict({key: [X_train_imp_values[key]] for key in X_train_imp.columns}).astype(X_train_imp.dtypes),X_train_imp])).iloc[0].to_dict()
    X_train_MISSING_imp_values_dummy = oneHot(pd.concat([pd.DataFrame.from_dict({key: [X_train_MISSING_imp_values[key]] for key in X_train_MISSING_imp.columns}).astype(X_train_MISSING_imp.dtypes),X_train_MISSING_imp])).iloc[0].to_dict()
    if PREPROCESS_DROPPING: 
        X_test_imp_dropped_dummy = [oneHot(X) for X in X_test_imp_dropped]
        X_train_imp_dropped_dummy = [oneHot(X) for X in X_train_imp_dropped]
    if PREPROCESS_DROPPING: X_train_MISSING_dropped_dummy = [oneHot(X) for X in X_train_MISSING_dropped]
    if PREPROCESS_DROPPING: X_test_MISSING_dropped_dummy = [oneHot(X) for X in X_test_MISSING_dropped]
    if PREPROCESS_DROPPING: 
        dis_X_test_imp_dropped_dummy = [oneHot(X) for X in dis_X_test_imp_dropped]
        dis_X_train_imp_dropped_dummy = [oneHot(X) for X in dis_X_train_imp_dropped]
    dis_X_train_nan_dummy, dis_X_test_nan_dummy = oneHot(dis_X_train_nan), oneHot(dis_X_test_nan)
    dis_X_train_imp_dummy, dis_X_test_imp_dummy = oneHot(dis_X_train_imp), oneHot(dis_X_test_imp)
    dis_X_train_imp_values_dummy = oneHot(pd.concat([pd.DataFrame.from_dict({key: [dis_X_train_imp_values[key]] for key in dis_X_train_imp.columns}).astype(dis_X_train_imp.dtypes),dis_X_train_imp])).iloc[0].to_dict()
    dis_X_train_none_dummy,dis_X_test_none_dummy = nanToNone(dis_X_train_nan_dummy), nanToNone(dis_X_test_nan_dummy)
    dis_X_train_MISSING_dummy,dis_X_test_MISSING_dummy = oneHot(dis_X_train_MISSING), oneHot(dis_X_test_MISSING)
    if PREPROCESS_DROPPING: 
        dis_X_test_nan_dropped_dummy  = [oneHot(X) for X in dis_X_test_dropped]
        dis_X_train_nan_dropped_dummy  = [oneHot(X) for X in dis_X_train_dropped]
    if PREPROCESS_DROPPING: dis_X_test_none_dropped_dummy = [nanToNone(X) for X in dis_X_test_nan_dropped_dummy]

    # scale the data with training set info
    scaler = StandardScaler().fit(X_train_nan_dummy)
    X_train_original_dummy_scaled = scalerTransform(X_train_original_dummy, scaler)
    X_train_nan_dummy_scaled = scalerTransform(X_train_nan_dummy, scaler)
    X_train_none_dummy_scaled= nanToNone(X_train_nan_dummy_scaled)
    X_test_original_dummy_scaled = scalerTransform(X_test_original_dummy, scaler)
    X_test_nan_dummy_scaled  = scalerTransform(X_test_nan_dummy,  scaler)
    X_test_none_dummy_scaled = nanToNone(X_test_nan_dummy_scaled)
    if PREPROCESS_DROPPING: 
        X_test_dropped_dummy_scaled = [scalerTransform(X, scaler) for X in X_test_dropped_dummy]
        X_train_dropped_dummy_scaled = [scalerTransform(X, scaler) for X in X_train_dropped_dummy]
    scaler = StandardScaler().fit(X_train_MISSING_dummy)
    X_train_MISSING_dummy_scaled=scalerTransform(X_train_MISSING_dummy, scaler)
    X_test_MISSING_dummy_scaled=scalerTransform(X_test_MISSING_dummy, scaler)
    if PREPROCESS_DROPPING: X_test_MISSING_dropped_dummy_scaled = [scalerTransform(X, scaler) for X in X_test_MISSING_dropped_dummy]
    scaler = StandardScaler().fit(X_train_imp_dummy)
    X_train_imp_dummy_scaled = scalerTransform(X_train_imp_dummy, scaler)
    X_test_imp_dummy_scaled  = scalerTransform(X_test_imp_dummy,  scaler)
    X_train_imp_values_dummy_scaled = scalerTransform(pd.DataFrame.from_dict({key: [X_train_imp_values_dummy[key]] for key in X_train_nan_dummy.columns}), scaler).iloc[0].to_dict()
    if PREPROCESS_DROPPING: 
        X_test_imp_dropped_dummy_scaled = [scalerTransform(X, scaler) for X in X_test_imp_dropped_dummy]
        X_train_imp_dropped_dummy_scaled = [scalerTransform(X, scaler) for X in X_train_imp_dropped_dummy]
    # scaler = StandardScaler().fit(X_original_train_dummy)
    # scaler = StandardScaler().fit(X_original_train_dummy)
    # print(X_original_train_dummy)
    # print(X_original_train_dummy.dtypes)
    # X_original_train_dummy_scaled = scalerTransform(X_original_train_dummy, scaler)
    # X_original_test_dummy_scaled  = scalerTransform(X_original_test_dummy, scaler)
    # X_original_train_imp_dummy_scaled = scalerTransform(X_original_train_imp_dummy, scaler)
    # X_original_test_imp_dummy_scaled  = scalerTransform(X_original_test_imp_dummy, scaler)

    # mapping of true cols to one-hot cols
    dis_col_mapping = {c: [c_dummy for c_dummy in dis_X_train_nan_dummy.columns if c_dummy.startswith(c+ENC_SEP) or c_dummy==c] for c in dis_X_train_nan.columns}
    dis_col_mapping_imp = {c: [c_dummy for c_dummy in dis_X_train_imp_dummy.columns if c_dummy.startswith(c+ENC_SEP) or c_dummy==c] for c in dis_X_train_imp.columns}
    col_mapping = {c: [c_dummy for c_dummy in X_train_nan_dummy.columns if c_dummy.startswith(c+ENC_SEP) or c_dummy==c] for c in X_train_nan.columns}
    MISSING_col_mapping = {c: [c_dummy for c_dummy in X_train_MISSING_dummy.columns if c_dummy.startswith(c+ENC_SEP) or c_dummy==c] for c in X_train_MISSING.columns}

    # test each classifier, after cv:
    for clf_name in LIST_OF_TESTS:
        print("Working on", clf_name, flush=True)
        # if not PREPROCESS_DROPPING: 
        #     X_train_dropped = []
        #     X_train_dropped_dummy = []
     #    X_train_nan, X_test_nan, X_train_MISSING, X_test_MISSING, y_train, y_test \
     # = train_test_split(X_nan, X_MISSING, y, test_size=TEST_PROP)
        # metric_model = modelSelect(clf_name, X_train_nan, X_train_dropped, X_train_nan_dummy, X_train_dropped_dummy, X_original_train, X_original_train_nan, X_original_train_dummy, y_train, col_mapping, METRICS)
        if PREPROCESS_DROPPING: 
            metric_model = modelSelect(clf_name, col_mapping=col_mapping, dis_col_mapping=dis_col_mapping, dis_col_mapping_imp=dis_col_mapping_imp, criterion=METRICS,
                X_MISSING_dummy_scaled=X_train_MISSING_dummy_scaled, 
                dis_X_MISSING_dummy=dis_X_train_MISSING_dummy, 
                X_imp_dummy_scaled=X_train_imp_dummy_scaled, 
                dis_X_imp_dummy=dis_X_train_imp_dummy, 
                X_nan_dummy_scaled=X_train_nan_dummy_scaled,
                dis_X_nan_dummy=dis_X_train_nan_dummy,
                X_none_dummy=X_train_none_dummy,
                dis_X_none_dummy=dis_X_train_none_dummy,
                X_imp_dropped_dummy_scaled=X_train_imp_dropped_dummy_scaled,
                dis_X_imp_dropped_dummy=dis_X_train_imp_dropped_dummy,
                X_dropped_dummy_scaled=X_train_dropped_dummy_scaled,
                dis_X_nan_dropped_dummy=dis_X_train_nan_dropped_dummy,
                y=y_train)
        else: 
            metric_model = modelSelect(clf_name, col_mapping=col_mapping, dis_col_mapping=dis_col_mapping, dis_col_mapping_imp=dis_col_mapping_imp, criterion=METRICS,
                X_MISSING_dummy_scaled=X_train_MISSING_dummy_scaled, 
                dis_X_MISSING_dummy=dis_X_train_MISSING_dummy, 
                X_imp_dummy_scaled=X_train_imp_dummy_scaled, 
                dis_X_imp_dummy=dis_X_train_imp_dummy, 
                X_nan_dummy_scaled=X_train_nan_dummy_scaled,
                dis_X_nan_dummy=dis_X_train_nan_dummy,
                X_none_dummy=X_train_none_dummy,
                dis_X_none_dummy=dis_X_train_none_dummy,
                y=y_train)
        print("  Training and testing", clf_name, flush=True)
        for metric, mod in metric_model.items():
            if metric not in METRICS: continue
            if clf_name=="LR, IC":
                acc = LRIC(X_train_MISSING_dummy_scaled, y_train, X_test_MISSING_dummy_scaled, y_test, None, strategic_testing=False)
            elif clf_name=="LR, IC, L2":
                acc = LRIC(X_train_MISSING_dummy_scaled, y_train, X_test_MISSING_dummy_scaled, y_test, None, strategic_testing=False, penalty="L2")
                # acc = LRIC(X_train_MISSING_dummy_scaled, y_train, X_test_MISSING_dummy_scaled, y_test, X_test_MISSING_dropped_dummy_scaled, strategic_testing=True, penalty="L2")
            elif clf_name=="LR, IC, L2, original":
                acc = LRIC(X_train_original_dummy_scaled, y_train, X_test_original_dummy_scaled, y_test, None, strategic_testing=False, penalty="L2")
            elif clf_name=="LR, IC, L2, discretized":
                acc = LRIC(dis_X_train_MISSING_dummy, y_train, dis_X_test_MISSING_dummy, y_test, None, strategic_testing=False, penalty="L2")
            elif clf_name=="LR, IC, L2, negated":
                acc = LRIC(X_train_MISSING_dummy_scaled, y_train, X_test_MISSING_dummy_scaled, y_test, None, strategic_testing=False, penalty="L2", negate=True)
            elif clf_name== "LR, imputation, strategic":
                acc = impClf(X_train_imp_dummy_scaled, y_train, X_test_imp_dummy_scaled, y_test, None, model=mod, strategic=True, col_mapping=col_mapping, imp_values=X_train_imp_values_dummy_scaled)
            elif clf_name== "LR, imputation, strategic, 2 mean":
                acc = impClf(X_train_imp_dummy, y_train, X_test_imp_dummy, y_test, None, model=mod, strategic=True, col_mapping=col_mapping, imp_values=X_train_imp_values_dummy)
            elif clf_name== "LR, imputation, strategic, 2 zero":
                acc = impClf(X_train_imp_dummy, y_train, X_test_imp_dummy, y_test, None, model=mod, strategic=True, col_mapping=col_mapping, imp_values={key: 0. for key in X_train_imp_values_dummy})
            elif clf_name== "LR, imputation, discretized, strategic":
                acc = impClf(dis_X_train_imp_dummy, y_train, dis_X_test_imp_dummy, y_test, None, model=mod, strategic=True, col_mapping=dis_col_mapping_imp, imp_values=dis_X_train_imp_values_dummy)
            elif clf_name[4:] == "imputation, strategic":
                acc = impClf(X_train_imp_dummy_scaled, y_train, X_test_imp_dummy_scaled, y_test, X_test_imp_dropped_dummy_scaled, model=mod, strategic=True)
            elif clf_name[4:] == "imputation, non-strategic":
                acc = impClf(X_train_imp_dummy_scaled, y_train, X_test_imp_dummy_scaled, y_test, [], model=mod, strategic=False)
            elif clf_name[4:] == "imputation, discretized, strategic":
                acc = impClf(dis_X_train_imp_dummy, y_train, dis_X_test_imp_dummy, y_test, dis_X_test_imp_dropped_dummy, model=mod, strategic=True)
            elif clf_name[4:] == "imputation, discretized, non-strategic":
                acc = impClf(dis_X_train_imp_dummy, y_train, dis_X_test_imp_dummy, y_test, [], model=mod, strategic=False)
            # elif clf_name[4:] == "imputation, non-strategic, no fs":
            #     acc = impClf(X_original_train_imp_dummy_scaled, y_train, X_original_test_imp_dummy_scaled, y_test, [], model=mod, strategic=False)
            # elif clf_name[4:] == "imputation, non-strategic, no fs, no dropping":
            #     acc = impClf(X_original_train_dummy_scaled, y_train, X_original_test_dummy_scaled, y_test, [], model=mod, strategic=False)
            elif clf_name[4:] == "reduced-feature, strategic":
                acc = ReducedFeature(X_train_nan_dummy_scaled, y_train, X_test_nan_dummy_scaled, y_test, X_test_dropped_dummy_scaled, model=mod, strategic=True, col_mapping=col_mapping)
            elif clf_name[4:] == "reduced-feature, non-strategic":
                acc = ReducedFeature(X_train_nan_dummy_scaled, y_train, X_test_nan_dummy_scaled, y_test, [], model=mod, strategic=False, col_mapping=col_mapping)
            elif clf_name[4:] == "reduced-feature, discretized, strategic":
                acc = ReducedFeature(dis_X_train_nan_dummy, y_train, dis_X_test_nan_dummy, y_test, dis_X_test_nan_dropped_dummy, model=mod, strategic=True, col_mapping=dis_col_mapping)
            elif clf_name[4:] == "reduced-feature, discretized, non-strategic":
                acc = ReducedFeature(dis_X_train_nan_dummy, y_train, dis_X_test_nan_dummy, y_test, [], model=mod, strategic=False, col_mapping=dis_col_mapping)
            elif clf_name == "min-cut, discretized":
                acc = minCut(dis_X_train_none_dummy, y_train, dis_X_test_none_dummy, y_test)
            elif clf_name == "min-cut":
                acc = minCut(X_train_none_dummy, y_train, X_test_none_dummy, y_test)
            elif clf_name == "clustering, discretized, strategic":
                acc = clustering(dis_X_train_none_dummy, y_train, dis_X_test_none_dummy, y_test, dis_X_test_none_dropped_dummy, strategic=True)
            elif clf_name == "clustering, strategic":
                acc = clustering(X_train_none_dummy, y_train, X_test_none_dummy, y_test, X_test_none_dropped_dummy, strategic=True)
            elif clf_name == "clustering, discretized, non-strategic":
                acc = clustering(dis_X_train_none_dummy, y_train, dis_X_test_none_dummy, y_test, [], strategic=False)
            elif clf_name == "clustering, non-strategic":
                acc = clustering(X_train_none_dummy, y_train, X_test_none_dummy, y_test, [], strategic=False)
            elif clf_name[4:] == "greedy, discretized":
                acc = HillClimbing(dis_X_train_nan_dummy, y_train, dis_X_test_nan_dummy, y_test, model=mod, clf_order="less_feature", col_mapping=dis_col_mapping)
            elif clf_name[4:] == "greedy":
                acc = HillClimbing(X_train_nan_dummy_scaled, y_train, X_test_nan_dummy_scaled, y_test, model=mod, clf_order="less_feature", col_mapping=col_mapping)
            elif clf_name[4:] == "greedy, discretized, approx":
                acc = HillClimbing(dis_X_train_nan_dummy, y_train, dis_X_test_nan_dummy, y_test, model=mod, clf_order="less_feature", col_mapping=dis_col_mapping, approx_subsets=APPROX_NUM_SUBSETS)
            elif clf_name[4:] == "greedy, approx":
                acc = HillClimbing(X_train_nan_dummy_scaled, y_train, X_test_nan_dummy_scaled, y_test, model=mod, clf_order="less_feature", col_mapping=col_mapping, approx_subsets=APPROX_NUM_SUBSETS)
            # elif clf_name == "LR, iterative, strategic, non-strategic":
            #     acc = iterative(X_train_MISSING_dummy, y_train, None, X_test_MISSING_dummy, y_test, None, model=mod, strategic_training=True, strategic_testing=False, col_mapping=col_mapping, imp_values=X_train_imp_values_dummy)
            # elif clf_name[4:] == "iterative, strategic, strategic":
            #     acc = iterative(X_train_MISSING_dummy, y_train, X_train_MISSING_dropped_dummy, X_test_MISSING_dummy, y_test, X_test_MISSING_dropped_dummy, model=mod, strategic_training=True, strategic_testing=True)
            elif clf_name[4:] == "iterative, strategic, strategic":
                acc = iterative(X_train_MISSING_dummy, y_train, None, X_test_MISSING_dummy, y_test, X_test_MISSING_dropped_dummy, model=mod, strategic_training=True, strategic_testing=True, col_mapping=MISSING_col_mapping, imp_values=X_train_MISSING_imp_values_dummy)
            elif clf_name[4:] == "iterative, strategic, strategic, 2":
                acc = iterative(X_train_MISSING_dummy, y_train, X_train_MISSING_dropped_dummy, X_test_MISSING_dummy, y_test, X_test_MISSING_dropped_dummy, model=mod, strategic_training=True, strategic_testing=True, col_mapping=MISSING_col_mapping, imp_values=X_train_MISSING_imp_values_dummy, historic=True)
            elif clf_name[4:] == "iterative, strategic, strategic, 2 zero":
                acc = iterative(X_train_imp_dummy, y_train, None, X_test_imp_dummy, y_test, None, model=mod, strategic_training=True, strategic_testing=True, col_mapping=col_mapping, imp_values={key: 0. for key in X_train_imp_values_dummy}, historic=False)
            # elif clf_name[4:] == "iterative, strategic, non-strategic":
            #     acc = iterative(X_train_MISSING_dummy, y_train, X_train_MISSING_dropped_dummy, X_test_MISSING_dummy, y_test, X_test_MISSING_dropped_dummy, model=mod, strategic_training=True, strategic_testing=False)
            # elif clf_name[4:] == "iterative, non-strategic, strategic":
            #     acc = iterative(X_train_MISSING_dummy, y_train, X_train_MISSING_dropped_dummy, X_test_MISSING_dummy, y_test, X_test_MISSING_dropped_dummy, model=mod, strategic_training=False, strategic_testing=True)
            # elif clf_name[4:] == "iterative, non-strategic, non-strategic":
            #     acc = iterative(X_train_MISSING_dummy, y_train, X_train_MISSING_dropped_dummy, X_test_MISSING_dummy, y_test, X_test_MISSING_dropped_dummy, model=mod, strategic_training=False, strategic_testing=False)
            elif clf_name[4:] == "original":
                acc = LR(X_train_original_dummy_scaled, y_train, X_test_original_dummy_scaled, y_test, model=mod)
            else:
                print(clf_name)
            print("  performance of", clf_name, metric, acc[metric], flush=True)
            test_acc[clf_name][metric] = acc
    if writing:
        np.set_printoptions(threshold=np.inf)
        f = open(FILE_SPACE+WRITE_FILE, "a")
        f.write(str(exp)+":"+str(test_acc)+",")
        f.close()
        # f = open(FILE_SPACE+ITERTATIVE_LOG_FILE, "a")
        # f.write("},")
        # f.close()
