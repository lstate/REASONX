from anchor import utils
from anchor import anchor_tabular

import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image
from xgboost import XGBClassifier

# local imports
sys.path.append('../src/') # local path

from helper_functions import read_adult, read_give_me_some_credit, read_south_german_credit

# dataset execution, copied
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

# encode df
df_encoded_onehot = df_code.fit_transform(df)
# encoded atts names
encoded_pred_atts = df_code.encoded_atts(pred_atts)
df_encoded_onehot.head()
novel = df_encoded_onehot * 1

# dataset partition, convert datasets to numpy, train DT

X, y = novel[encoded_pred_atts], novel[target]
# retain test sets
X1, XT1, y1, yt1 = train_test_split(X, y, test_size=0.3, random_state=42)
X2, XT2, y2, yt2 = train_test_split(X, y, test_size=0.3, random_state=24)

X1_numpy = X1.to_numpy()
y1_numpy = y1.to_numpy()

tree_numpy = DecisionTreeClassifier(max_depth = 3)
tree_numpy.fit(X1_numpy, y1_numpy)

xgb = XGBClassifier(random_state = 0)
xgb.fit(X1_numpy, y1_numpy)
xgb_label = xgb.predict(XT1.to_numpy())

ml_model = xgb

# splitpoint for explanations

# initialize the explanator on encoded data
explainer = anchor_tabular.AnchorTabularExplainer(
    ["0", "1"],
    encoded_pred_atts,
    X1_numpy)

instances = 1
threshold = [0.99]

numpy_test = XT1.to_numpy()

predict_fn = lambda x: ml_model.predict(x)

for j in range(len(threshold)):
    for i in range(instances):
        # prediction
        # uses assigned labels from above (initialization of explainer)
        #print("Prediction: ", explainer.class_names[predict_fn(numpy_test[i].reshape(1,-1))[0]])
        
        # generate explanation
        #exp = explainer.explain_instance(numpy_test[i], ml_model.predict, threshold=threshold[j])
        exp = explainer.explain_instance(numpy_test[i], predict_fn, threshold=threshold[j])
