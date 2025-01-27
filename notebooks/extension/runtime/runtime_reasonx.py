# standard imports
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image
from xgboost import XGBClassifier

# local imports
sys.path.append('../src/') # local path
import reasonx

from helper_functions import read_adult, read_give_me_some_credit, read_south_german_credit, read_credit_card_default, read_australian_credit

from neighborhood import naive_neighborhood_instance

simplified=False
continuous_only=False
dataset = "adult"

# read dataset
if dataset == "gmsc":
    df, pred_atts, target, df_code = read_give_me_some_credit(continuous_only=continuous_only, simplified=simplified)
if dataset == "sgc":
    df, pred_atts, target, df_code = read_south_german_credit(continuous_only=continuous_only, simplified=simplified)
if dataset == "adult":
    df, pred_atts, target, df_code = read_adult(continuous_only=continuous_only, simplified=simplified)
if dataset == "dccc":
    df, pred_atts, target, df_code = read_credit_card_default(continuous_only=continuous_only, simplified=simplified)
if dataset == "aca":
    df, pred_atts, target, df_code = read_australian_credit(continuous_only=continuous_only, simplified=simplified)

# encode df
df_encoded_onehot = df_code.fit_transform(df)
# encoded atts names
encoded_pred_atts = df_code.encoded_atts(pred_atts)

case = "c"
constraints = []
#constraints = "F.capitalgain=CF.capitalgain"
confidence_f = 0
tree_depth = 3

# split predictive and target
X, y = df_encoded_onehot[encoded_pred_atts], df_encoded_onehot[target]
# retain test sets
X1, XT1, y1, yt1 = train_test_split(X, y, test_size=0.3, random_state=42)
X2, XT2, y2, yt2 = train_test_split(X, y, test_size=0.3, random_state=24)

if case == "a":
    clf1 = DecisionTreeClassifier(max_depth=tree_depth)
    clf1.fit(X1, y1)

    tree_ = clf1

if case == "b":
    # learn ML model
    xgb = XGBClassifier(random_state = 0)
    xgb.fit(X1, y1)
    xgb_label = xgb.predict(XT1)

    xgb_label_df = pd.Series(data=xgb_label)

    # split the test set (XT1/xgb_labels) in two parts
    XT1_train, XT1_test, xgb_label_train, xgb_label_test = train_test_split(XT1, xgb_label_df, test_size=0.3, random_state=42)

    # splitpoint for explanations (case b)
    # train a surrogate decision tree
    clf2 = DecisionTreeClassifier(max_depth=tree_depth)
    clf2.fit(XT1_train, xgb_label_train)

    tree_ = clf2

if case == "c":
    # learn ML model
    xgb = XGBClassifier(random_state = 0)
    xgb.fit(X1, y1)
    xgb_label = xgb.predict(XT1)

    ml_model_local = xgb

instances = 1
evaluation_list = []

# splitpoint for explanations (case a)

if case == "a":
    r = reasonx.ReasonX(pred_atts, target, df_code, verbose = 1)
    r.model(tree_)
    
    for i in range(instances):
        # return only one instance (case c)
        r.instance('F', features=XT1.iloc[i:i+1], label=yt1.iloc[i], minconf = confidence_f)

        r.instance('CF', label=1-yt1.iloc[i], minconf = 0)

        if len(constraints) > 0:
            r.constraint(constraints)

        r.solveopt(minimize='l1norm(F, CF)', eps = 0.01)
        r.solveopt(minimize='linfnorm(F, CF)', eps = 0.01)

if case == "b":
    r = reasonx.ReasonX(pred_atts, target, df_code, verbose = 1)
    r.model(tree_)
    
    for i in range(instances):
        # return only one instance (case c)
        r.instance('F', features=XT1_test.iloc[i:i+1], label=xgb_label_test.iloc[i], minconf = confidence_f)
        r.instance('CF', label=1-xgb_label_test.iloc[i], minconf = 0)

        if len(constraints) > 0:
            r.constraint(constraints)

        r.solveopt(minimize='l1norm(F, CF)', eps = 0.01)
        r.solveopt(minimize='linfnorm(F, CF)', eps = 0.01)

if case == "c": 
    fidelity = []
    for i in range(instances): 
        # pick data instance
        features=XT1.iloc[i:i+1]
        # relevant label is the predicted label by ML model
        label=xgb_label[i]
        data_numpy = XT1.to_numpy()

        # splitpoint for explanations (case c)

        # neighborhood generation
        N = 5000
        C = int(df_encoded_onehot.shape[1] * 2 / 3)
        neigh = naive_neighborhood_instance(features.to_numpy(), C, N, np.transpose(data_numpy), 42)

        # predict labels of neigh
        label_neigh = ml_model_local.predict(neigh)

        # split neigh
        neigh_train, neigh_test, neigh_label_train, neigh_label_test = train_test_split(neigh, label_neigh, test_size=0.3, random_state=42)

        # train surrogate DT
        clf2 = DecisionTreeClassifier(max_depth=tree_depth)
        clf2.fit(neigh_train, neigh_label_train)

        # execute the evaluation
        r = reasonx.ReasonX(pred_atts, target, df_code, verbose = 1)
        r.model(clf2)

        # return only one instance (case c)
        r.instance('F', features=features, label=label, minconf = confidence_f)
        r.instance('CF', label=1-label, minconf = 0)

        if len(constraints) > 0:
            r.constraint(constraints)

        r.solveopt(minimize='l1norm(F, CF)', eps = 0.01)
        r.solveopt(minimize='linfnorm(F, CF)', eps = 0.01)
