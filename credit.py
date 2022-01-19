import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler
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

def HillClimbing(X_train, y_train, X_test, y_test, model=None, rounds=5, stop_criterion=0.0001, clf_order="random", col_mapping=None):
    """A HC classifier, each subclassifier uses model as the internal classifier"""
    if col_mapping == None: col_mapping = {c:[c] for c in X_train.columns}
    p_features = powerset(col_mapping.keys())
    all_classifiers = [] 
    for feature_comb in p_features: 
        features = list(chain.from_iterable([col_mapping[f] for f in feature_comb]))
        all_classifiers.append((features,clone(model)))

    train_accuracies = []
    test_accuracies = []
    train_accuracies_proba = []
    test_accuracies_proba = []
    for round_num in range(rounds):
        # print("    Round", round_num)
        if clf_order == "random": 
            random.shuffle(all_classifiers)
            classifiers = all_classifiers
        elif clf_order == "less_feature": 
            classifiers = list(reversed(all_classifiers))

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

def impClf(X_train, y_train, X_test, y_test, X_test_dropped, model, strategic=False):
    """X_train has been pre-imputed"""
    clf = clone(model)
    clf.fit(X_train, y_train)
    # prediction
    y_preds = clf.predict(X_test)
    y_preds_prob = clf.predict_proba(X_test)[:,1]
    assert clf.classes_[1] == 1
    if strategic:
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

def imputeTrainTest(X_train_nan, X_test_nan):
    """Imputes training and testing sets with training set mean/mode"""
    data_types = X_train_nan.dtypes.to_dict()
    cols = X_train_nan.columns
    X_train_nan_numerical = X_train_nan.select_dtypes(exclude='category')
    X_train_nan_categorical = X_train_nan.select_dtypes(include='category')
    imp_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    if not X_train_nan_numerical.empty:
        imputer = imp_mean.fit(X_train_nan_numerical)
        values = dict(zip(X_train_nan_numerical.columns, imputer.statistics_))
        X_train_nan_numerical_imp = pd.DataFrame(imp_mean.fit_transform(X_train_nan_numerical), columns=X_train_nan_numerical.columns)
        X_test_imp = X_test_nan.fillna(value=values)
    else:
        X_train_nan_numerical_imp = X_train_nan_numerical
        X_test_imp = X_test_nan

    if not X_train_nan_categorical.empty:
        imputer = imp_mode.fit(X_train_nan_categorical)
        values = dict(zip(X_train_nan_categorical.columns, imputer.statistics_))
        X_train_nan_categorical_imp = pd.DataFrame(imp_mode.fit_transform(X_train_nan_categorical), columns=X_train_nan_categorical.columns)
        X_test_imp = X_test_imp.fillna(value=values)
    else:
        X_train_nan_categorical_imp = X_train_nan_categorical
        X_test_imp = X_test_imp

    X_train_imp = X_train_nan_numerical_imp.join(X_train_nan_categorical_imp)
    return X_train_imp[cols].astype(data_types), X_test_imp[cols].astype(data_types)

def modelSelect(clf_name, X_nan, X_dropped, X_nan_dummy, X_dropped_dummy, X_original, X_original_nan, X_original_dummy, y, col_mapping, criterion):
    """For each classifier (clf_name), perform model selection in a 5-fold CV"""
    print("  model selection for", clf_name, flush=True)
    X_none_dummy = nanToNone(X_nan_dummy)
    if clf_name[:2] == "LR":
        return {"PCC": LogisticRegression(solver='lbfgs'), "AUC": LogisticRegression(solver='lbfgs'), "F1": LogisticRegression(solver='lbfgs'), "Brier": LogisticRegression(solver='lbfgs')}
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
    for CV_itr, (train_index, test_index) in enumerate(kf.split(X_nan)):
        # print("    Fold", CV_itr+1, flush=True)

        # splitting train-test by fold
        X_train_nan, X_train_nan_dummy, X_train_none_dummy, X_original_train, X_original_train_nan, X_original_train_dummy, y_train \
         = X_nan.iloc[train_index], X_nan_dummy.iloc[train_index], X_none_dummy.iloc[train_index], X_original.iloc[train_index], X_original_nan.iloc[train_index], X_original_dummy.iloc[train_index], y[train_index]
        X_test_nan,  X_test_nan_dummy,  X_test_none_dummy,  X_original_test,  X_original_test_nan,  X_original_test_dummy,  y_test \
         = X_nan.iloc[test_index],  X_nan_dummy.iloc[test_index],  X_none_dummy.iloc[test_index],  X_original.iloc[test_index] , X_original_nan.iloc[test_index],  X_original_dummy.iloc[test_index],  y[test_index]
        X_test_dropped = [X.iloc[test_index] for X in X_dropped]
        X_test_dropped_dummy = [X.iloc[test_index] for X in X_dropped_dummy]

        # impute the training set and test set with training set mean
        X_train_imp, X_test_imp = imputeTrainTest(X_train_nan, X_test_nan)
        X_original_train_imp, X_original_test_imp = imputeTrainTest(X_original_train_nan, X_original_test_nan)
        X_test_imp_dropped = [imputeTrainTest(X_train_nan, X)[1] for X in X_test_dropped]

        # the discretized data
        X_train_nan_y = X_train_nan.assign(y=y_train)
        X_test_nan_y = X_test_nan.assign(y=y_test)
        discretizer = MDLP_Discretizer(dataset=X_train_nan_y, class_label="y")
        dis_X_test_nan = discretizer.apply_cutpoints(X_test_nan_y).drop(columns=['y'])
        dis_X_train_nan = discretizer.apply_cutpoints(X_train_nan_y).drop(columns=['y'])
        dis_X_test_dropped = [discretizer.apply_cutpoints(X.assign(y=y_test)).drop(columns=['y']) for X in X_test_dropped]

        # the discretized imputed data
        X_train_imp_y = X_train_imp.assign(y=y_train)
        X_test_imp_y = X_test_imp.assign(y=y_test)
        discretizer = MDLP_Discretizer(dataset=X_train_imp_y, class_label="y")
        dis_X_test_imp = discretizer.apply_cutpoints(X_test_imp_y).drop(columns=['y'])
        dis_X_train_imp = discretizer.apply_cutpoints(X_train_imp_y).drop(columns=['y'])
        dis_X_test_imp_dropped = [discretizer.apply_cutpoints(X.assign(y=y_test)).drop(columns=['y']) for X in X_test_imp_dropped]

        # one-hot encode the dataset
        X_original_train_imp_dummy, X_original_test_imp_dummy = oneHot(X_original_train_imp), oneHot(X_original_test_imp)
        X_train_imp_dummy, X_test_imp_dummy = oneHot(X_train_imp), oneHot(X_test_imp)
        X_test_imp_dropped_dummy = [oneHot(X) for X in X_test_imp_dropped]
        dis_X_test_imp_dropped_dummy = [oneHot(X) for X in dis_X_test_imp_dropped]
        dis_X_train_nan_dummy, dis_X_test_nan_dummy = oneHot(dis_X_train_nan), oneHot(dis_X_test_nan)
        dis_X_train_imp_dummy, dis_X_test_imp_dummy = oneHot(dis_X_train_imp), oneHot(dis_X_test_imp)
        dis_X_test_nan_dropped_dummy  = [oneHot(X) for X in dis_X_test_dropped]

        # scale the data with training set info
        scaler = StandardScaler().fit(X_train_nan_dummy)
        X_train_nan_dummy_scaled = scalerTransform(X_train_nan_dummy, scaler)
        X_test_nan_dummy_scaled  = scalerTransform(X_test_nan_dummy,  scaler)
        X_test_dropped_dummy_scaled = [scalerTransform(X, scaler) for X in X_test_dropped_dummy]
        X_train_imp_dummy_scaled = scalerTransform(X_train_imp_dummy, scaler)
        X_test_imp_dummy_scaled  = scalerTransform(X_test_imp_dummy,  scaler)
        X_test_imp_dropped_dummy_scaled = [scalerTransform(X, scaler) for X in X_test_imp_dropped_dummy]
        scaler = StandardScaler().fit(X_original_train_dummy)
        X_original_train_dummy_scaled = scalerTransform(X_original_train_dummy, scaler)
        X_original_test_dummy_scaled  = scalerTransform(X_original_test_dummy, scaler)
        X_original_train_imp_dummy_scaled = scalerTransform(X_original_train_imp_dummy, scaler)
        X_original_test_imp_dummy_scaled  = scalerTransform(X_original_test_imp_dummy, scaler)

        # mapping of true cols to one-hot cols
        dis_col_mapping = {c: [c_dummy for c_dummy in dis_X_train_nan_dummy.columns if c_dummy.startswith(c+ENC_SEP) or c_dummy==c] for c in dis_X_train_nan.columns}
    
        for mod in models:
            if clf_name[4:] == "imputation, strategic":
                acc = impClf(X_train_imp_dummy_scaled, y_train, X_test_imp_dummy_scaled, y_test, X_test_imp_dropped_dummy_scaled, model=mod, strategic=True)
            elif clf_name[4:] == "imputation, non-strategic":
                acc = impClf(X_train_imp_dummy_scaled, y_train, X_test_imp_dummy_scaled, y_test, [], model=mod, strategic=False)
            elif clf_name[4:] == "imputation, discretzed, strategic":
                acc = impClf(dis_X_train_imp_dummy, y_train, dis_X_test_imp_dummy, y_test, dis_X_test_imp_dropped_dummy, model=mod, strategic=True)
            elif clf_name[4:] == "imputation, discretzed, non-strategic":
                acc = impClf(dis_X_train_imp_dummy, y_train, dis_X_test_imp_dummy, y_test, [], model=mod, strategic=False)
            elif clf_name[4:] == "imputation, non-strategic, no fs":
                acc = impClf(X_original_train_imp_dummy_scaled, y_train, X_original_test_imp_dummy_scaled, y_test, [], model=mod, strategic=False)
            elif clf_name[4:] == "imputation, non-strategic, no fs, no dropping":
                acc = impClf(X_original_train_dummy_scaled, y_train, X_original_test_dummy_scaled, y_test, [], model=mod, strategic=False)
            elif clf_name[4:] == "reduced-feature, strategic":
                acc = ReducedFeature(X_train_nan_dummy_scaled, y_train, X_test_nan_dummy_scaled, y_test, X_test_dropped_dummy_scaled, model=mod, strategic=True, col_mapping=col_mapping)
            elif clf_name[4:] == "reduced-feature, non-strategic":
                acc = ReducedFeature(X_train_nan_dummy_scaled, y_train, X_test_nan_dummy_scaled, y_test, [], model=mod, strategic=False, col_mapping=col_mapping)
            elif clf_name[4:] == "reduced-feature, discretzed, strategic":
                acc = ReducedFeature(dis_X_train_nan_dummy, y_train, dis_X_test_nan_dummy, y_test, dis_X_test_nan_dropped_dummy, model=mod, strategic=True, col_mapping=dis_col_mapping)
            elif clf_name[4:] == "reduced-feature, discretzed, non-strategic":
                acc = ReducedFeature(dis_X_train_nan_dummy, y_train, dis_X_test_nan_dummy, y_test, [], model=mod, strategic=False, col_mapping=dis_col_mapping)
            elif clf_name[4:] == "greedy, discretized":
                acc = HillClimbing(dis_X_train_nan_dummy, y_train, dis_X_test_nan_dummy, y_test, model=mod, clf_order="less_feature", col_mapping=dis_col_mapping)
            elif clf_name[4:] == "greedy":
                acc = HillClimbing(X_train_nan_dummy_scaled, y_train, X_test_nan_dummy_scaled, y_test, model=mod, clf_order="less_feature", col_mapping=col_mapping)
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

def featureSelect(X, y, num_features=5):
    """select num_features number of features based on F_SELECT_METHOD"""
    original_cols = X.columns.to_numpy()
    original_X = X
    (X,_),y = imputeTrainTest(oneHot(X),oneHot(X)),y
    selector = SelectKBest(F_SELECT_METHOD, k="all").fit(X, y)
    scores = selector.scores_
    scores[np.isnan(scores)] = -np.inf
    arr1inds = scores.argsort()
    sorted_enc_features = X.columns.to_numpy()[arr1inds[::-1]]
    to_select = []
    for enc_f in sorted_enc_features.tolist():
        f = enc_f.split(ENC_SEP)[0]
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
    else:
        print("???")


assert len(sys.argv[1:]) == 2 or len(sys.argv[1:]) == 3
K=5
NAN_PROP = float(sys.argv[2])
TEST_PROP = 0.5
if (len(sys.argv[1:]) == 3 and sys.argv[3] == "bal"):
    BALANCED = True
else:
    BALANCED = False
F_SELECT = "f_classif"
if F_SELECT == "f_classif":
    F_SELECT_METHOD = f_classif
elif F_SELECT == "chi2":
    F_SELECT_METHOD = chi2
NUM_FT = 4
AUSTRALIAN_DATASET = "credit-approval"
GERMANY_DATASET = "credit-g"
TAIWAN_DATASET = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
POLISH_DATASET = "polish" # download and unzip https://archive.ics.uci.edu/ml/machine-learning-databases/00365/data.zip
if (sys.argv[1]) == "australia":
    DATASET = AUSTRALIAN_DATASET
elif (sys.argv[1]) == "germany":
    DATASET = GERMANY_DATASET
elif (sys.argv[1]) == "taiwan":
    DATASET = TAIWAN_DATASET
elif (sys.argv[1]) == "poland":
    DATASET = POLISH_DATASET
else:
    print("Unknown dataset")
ENC_SEP = "="
METRICS = ["PCC", "AUC", "F1", "Brier"]
if DATASET == GERMANY_DATASET:
    ACCEPT_CLASS = "good"
elif DATASET == AUSTRALIAN_DATASET:
    ACCEPT_CLASS = "-"
elif DATASET == TAIWAN_DATASET:
    ACCEPT_CLASS = 0
elif DATASET == POLISH_DATASET:
    ACCEPT_CLASS = 0

if (len(sys.argv[1:]) == 3):
    WRITE_FILE = "_"+str(sys.argv[1])+"_"+str(sys.argv[2])+"_"+str(sys.argv[3])+".txt"
else:
    WRITE_FILE = "_"+str(sys.argv[1])+"_"+str(sys.argv[2])+".txt"
FILE_SPACE = ""
LIST_OF_TESTS = [
                 "clustering, strategic",
                 "clustering, discretized, strategic",
                 "clustering, non-strategic",
                 "clustering, discretized, non-strategic",
                 "KN, imputation, strategic",
                 "KN, imputation, non-strategic",
                 "KN, imputation, discretzed, strategic",
                 "KN, imputation, discretzed, non-strategic",
                 "KN, imputation, non-strategic, no fs",
                 "KN, imputation, non-strategic, no fs, no dropping",
                 "KN, reduced-feature, strategic",
                 "KN, reduced-feature, non-strategic",
                 "KN, reduced-feature, discretzed, strategic",
                 "KN, reduced-feature, discretzed, non-strategic",
                 "NN, imputation, strategic", 
                 "NN, imputation, non-strategic", 
                 "NN, imputation, discretzed, strategic",
                 "NN, imputation, discretzed, non-strategic",
                 "NN, imputation, non-strategic, no fs",
                 "NN, imputation, non-strategic, no fs, no dropping",
                 "NN, reduced-feature, strategic",
                 "NN, reduced-feature, non-strategic",
                 "NN, reduced-feature, discretzed, strategic",
                 "NN, reduced-feature, discretzed, non-strategic",
                 "LR, imputation, strategic", 
                 "LR, imputation, non-strategic", 
                 "LR, imputation, discretzed, strategic",
                 "LR, imputation, discretzed, non-strategic",
                 "LR, imputation, non-strategic, no fs",
                 "LR, imputation, non-strategic, no fs, no dropping",
                 "LR, reduced-feature, strategic",
                 "LR, reduced-feature, non-strategic",
                 "LR, reduced-feature, discretzed, strategic",
                 "LR, reduced-feature, discretzed, non-strategic",
                 "RF, imputation, strategic", 
                 "RF, imputation, non-strategic", 
                 "RF, imputation, discretzed, strategic",
                 "RF, imputation, discretzed, non-strategic",
                 "RF, imputation, non-strategic, no fs",
                 "RF, imputation, non-strategic, no fs, no dropping",
                 "RF, reduced-feature, strategic",
                 "RF, reduced-feature, non-strategic",
                 "RF, reduced-feature, discretzed, strategic",
                 "RF, reduced-feature, discretzed, non-strategic",
                 "min-cut",
                 "min-cut, discretized",
                 "NN, greedy",
                 "NN, greedy, discretized",
                 "LR, greedy",
                 "LR, greedy, discretized"
                 ]

X,y_full = openDateset(DATASET)
X,_ = imputeTrainTest(X, X)
best_features = featureSelect(X, y_full, NUM_FT)

f = open(FILE_SPACE+WRITE_FILE, "w")
f.write("{")
f.close()

for exp in range(100):
    print("\nExp", exp, "---------", flush=True)
    test_acc = {test: {m: None for m in METRICS} for test in LIST_OF_TESTS}
    if BALANCED:
        X_original,y = RandomUnderSampler().fit_resample(X, y_full)
    else:
        X_original,y = X, y_full

    # Add None/nan representing missing feature value by nature
    X_original_nan = add_nan(X_original, NAN_PROP)
    X_nan = X_original_nan[best_features]

    # splitting train/test
    X_original_train, X_original_test, X_original_train_nan, X_original_test_nan, X_train_nan, X_test_nan, y_train, y_test \
     = train_test_split(X_original, X_original_nan, X_nan, y, test_size=TEST_PROP)
    
    # all combinaitons of feature dropping
    X_train_dropped, X_test_dropped = allFeatureDrop(X_train_nan), allFeatureDrop(X_test_nan)

    # impute the training set and test set with training set mean
    X_train_imp, X_test_imp = imputeTrainTest(X_train_nan, X_test_nan)
    X_original_train_imp, X_original_test_imp = imputeTrainTest(X_original_train_nan, X_original_test_nan)
    X_test_imp_dropped = [imputeTrainTest(X_train_nan, X)[1] for X in X_test_dropped]

    # the discretized data
    X_train_nan_y = X_train_nan.assign(y=y_train)
    X_test_nan_y = X_test_nan.assign(y=y_test)
    discretizer = MDLP_Discretizer(dataset=X_train_nan_y, class_label="y")
    dis_X_test_nan = discretizer.apply_cutpoints(X_test_nan_y).drop(columns=['y'])
    dis_X_train_nan = discretizer.apply_cutpoints(X_train_nan_y).drop(columns=['y'])
    dis_X_test_dropped = [discretizer.apply_cutpoints(X.assign(y=y_test)).drop(columns=['y']) for X in X_test_dropped]

    # the discretized imputed data
    X_train_imp_y = X_train_imp.assign(y=y_train)
    X_test_imp_y = X_test_imp.assign(y=y_test)
    discretizer = MDLP_Discretizer(dataset=X_train_imp_y, class_label="y")
    dis_X_test_imp = discretizer.apply_cutpoints(X_test_imp_y).drop(columns=['y'])
    dis_X_train_imp = discretizer.apply_cutpoints(X_train_imp_y).drop(columns=['y'])
    dis_X_test_imp_dropped = [discretizer.apply_cutpoints(X.assign(y=y_test)).drop(columns=['y']) for X in X_test_imp_dropped]

    # one-hot encode the dataset
    X_original_train_imp_dummy, X_original_test_imp_dummy = oneHot(X_original_train_imp), oneHot(X_original_test_imp)
    X_original_train_dummy, X_original_test_dummy = oneHot(X_original_train), oneHot(X_original_test)
    X_original_train_nan_dummy, X_original_test_nan_dummy = oneHot(X_original_train_nan), oneHot(X_original_test_nan)
    X_train_nan_dummy, X_test_nan_dummy = oneHot(X_train_nan), oneHot(X_test_nan)
    X_train_none_dummy,X_test_none_dummy = nanToNone(X_train_nan_dummy), nanToNone(X_test_nan_dummy)
    X_train_dropped_dummy, X_test_dropped_dummy = [oneHot(X) for X in X_train_dropped], [oneHot(X) for X in X_test_dropped]
    X_test_none_dropped_dummy = [nanToNone(X) for X in X_test_dropped_dummy]
    X_train_imp_dummy, X_test_imp_dummy = oneHot(X_train_imp), oneHot(X_test_imp)
    X_test_imp_dropped_dummy = [oneHot(X) for X in X_test_imp_dropped]
    dis_X_test_imp_dropped_dummy = [oneHot(X) for X in dis_X_test_imp_dropped]
    dis_X_train_nan_dummy, dis_X_test_nan_dummy = oneHot(dis_X_train_nan), oneHot(dis_X_test_nan)
    dis_X_train_imp_dummy, dis_X_test_imp_dummy = oneHot(dis_X_train_imp), oneHot(dis_X_test_imp)
    dis_X_train_none_dummy,dis_X_test_none_dummy = nanToNone(dis_X_train_nan_dummy), nanToNone(dis_X_test_nan_dummy)
    dis_X_test_nan_dropped_dummy  = [oneHot(X) for X in dis_X_test_dropped]
    dis_X_test_none_dropped_dummy = [nanToNone(X) for X in dis_X_test_nan_dropped_dummy]

    # scale the data with training set info
    scaler = StandardScaler().fit(X_train_nan_dummy)
    X_train_nan_dummy_scaled = scalerTransform(X_train_nan_dummy, scaler)
    X_train_none_dummy_scaled= nanToNone(X_train_nan_dummy_scaled)
    X_test_nan_dummy_scaled  = scalerTransform(X_test_nan_dummy,  scaler)
    X_test_none_dummy_scaled = nanToNone(X_test_nan_dummy_scaled)
    X_test_dropped_dummy_scaled = [scalerTransform(X, scaler) for X in X_test_dropped_dummy]
    X_train_imp_dummy_scaled = scalerTransform(X_train_imp_dummy, scaler)
    X_test_imp_dummy_scaled  = scalerTransform(X_test_imp_dummy,  scaler)
    X_test_imp_dropped_dummy_scaled = [scalerTransform(X, scaler) for X in X_test_imp_dropped_dummy]
    scaler = StandardScaler().fit(X_original_train_dummy)
    X_original_train_dummy_scaled = scalerTransform(X_original_train_dummy, scaler)
    X_original_test_dummy_scaled  = scalerTransform(X_original_test_dummy, scaler)
    X_original_train_imp_dummy_scaled = scalerTransform(X_original_train_imp_dummy, scaler)
    X_original_test_imp_dummy_scaled  = scalerTransform(X_original_test_imp_dummy, scaler)

    # mapping of true cols to one-hot cols
    dis_col_mapping = {c: [c_dummy for c_dummy in dis_X_train_nan_dummy.columns if c_dummy.startswith(c+ENC_SEP) or c_dummy==c] for c in dis_X_train_nan.columns}
    col_mapping = {c: [c_dummy for c_dummy in X_train_nan_dummy.columns if c_dummy.startswith(c+ENC_SEP) or c_dummy==c] for c in X_train_nan.columns}

    # test each classifier, after cv:
    for clf_name in LIST_OF_TESTS:
        print("Working on", clf_name, flush=True)
        metric_model = modelSelect(clf_name, X_train_nan, X_train_dropped, X_train_nan_dummy, X_train_dropped_dummy, X_original_train, X_original_train_nan, X_original_train_dummy, y_train, col_mapping, METRICS)
        print("  Training and testing", clf_name, flush=True)
        for metric, mod in metric_model.items():
            if clf_name[4:] == "imputation, strategic":
                acc = impClf(X_train_imp_dummy_scaled, y_train, X_test_imp_dummy_scaled, y_test, X_test_imp_dropped_dummy_scaled, model=mod, strategic=True)
            elif clf_name[4:] == "imputation, non-strategic":
                acc = impClf(X_train_imp_dummy_scaled, y_train, X_test_imp_dummy_scaled, y_test, [], model=mod, strategic=False)
            elif clf_name[4:] == "imputation, discretzed, strategic":
                acc = impClf(dis_X_train_imp_dummy, y_train, dis_X_test_imp_dummy, y_test, dis_X_test_imp_dropped_dummy, model=mod, strategic=True)
            elif clf_name[4:] == "imputation, discretzed, non-strategic":
                acc = impClf(dis_X_train_imp_dummy, y_train, dis_X_test_imp_dummy, y_test, [], model=mod, strategic=False)
            elif clf_name[4:] == "imputation, non-strategic, no fs":
                acc = impClf(X_original_train_imp_dummy_scaled, y_train, X_original_test_imp_dummy_scaled, y_test, [], model=mod, strategic=False)
            elif clf_name[4:] == "imputation, non-strategic, no fs, no dropping":
                acc = impClf(X_original_train_dummy_scaled, y_train, X_original_test_dummy_scaled, y_test, [], model=mod, strategic=False)
            elif clf_name[4:] == "reduced-feature, strategic":
                acc = ReducedFeature(X_train_nan_dummy_scaled, y_train, X_test_nan_dummy_scaled, y_test, X_test_dropped_dummy_scaled, model=mod, strategic=True, col_mapping=col_mapping)
            elif clf_name[4:] == "reduced-feature, non-strategic":
                acc = ReducedFeature(X_train_nan_dummy_scaled, y_train, X_test_nan_dummy_scaled, y_test, [], model=mod, strategic=False, col_mapping=col_mapping)
            elif clf_name[4:] == "reduced-feature, discretzed, strategic":
                acc = ReducedFeature(dis_X_train_nan_dummy, y_train, dis_X_test_nan_dummy, y_test, dis_X_test_nan_dropped_dummy, model=mod, strategic=True, col_mapping=dis_col_mapping)
            elif clf_name[4:] == "reduced-feature, discretzed, non-strategic":
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
            else:
                print(clf_name)
            print("  performance of", clf_name, metric, acc[metric], flush=True)
            test_acc[clf_name][metric] = acc

    np.set_printoptions(threshold=np.inf)
    f = open(FILE_SPACE+WRITE_FILE, "a")
    f.write(str(exp)+":"+str(test_acc)+",")
    f.close()
