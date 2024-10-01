import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from IPython.display import Image
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# local imports
sys.path.append('../src/') # local path

from helper_functions import read_adult, read_give_me_some_credit, read_south_german_credit

import dice_ml

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
    df, pred_atts, target, df_code, nominal_atts, ordinal_atts, continous_atts = read_adult(continuous_only=continuous_only, simplified=simplified, return_feature_type=True)

# encode df
df_encoded_onehot = df_code.fit_transform(df)
# encoded atts names
encoded_pred_atts = df_code.encoded_atts(pred_atts)

# prepare the dataset for dice
# class must be an integer (and not an object)
df_dice = df.drop([target], axis=1)
df_dice[target] = df_encoded_onehot[target]
df_target = df_dice[target]

train_dataset, test_dataset, y_train, y_test = train_test_split(df_dice, df_target, test_size=0.3, random_state=42)

x_train = train_dataset.drop(target, axis=1)
x_test = test_dataset.drop(target, axis=1)

d  = dice_ml.Data(dataframe=train_dataset, continuous_features=continous_atts, outcome_name=target)
numerical = continous_atts
categorical = x_train.columns.difference(numerical)

categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

# apply transformation on columns separately
transformations = ColumnTransformer(transformers=[('cat', categorical_transformer, categorical)])

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
clf = Pipeline(steps=[('preprocessor', transformations), ('classifier', XGBClassifier(random_state=0))])

model = clf.fit(x_train, y_train)

# splitpoint for explanations

# Using sklearn backend
m = dice_ml.Model(model=model, backend="sklearn")
# Using method=random for generating CFs
exp = dice_ml.Dice(d, m, method="random")

instances = 1
n_total_ce = [5]

# DiCE allows tunable parameters proximity_weight (default: 0.5) and diversity_weight (default: 1.0) to handle proximity and diversity respectively.
proximity_weight=0.5
diversity_weight=0

#features_to_vary = []
features_to_vary=["sex", "race", "workclass", "education", "age", "capitalloss", "hoursperweek"]

for k in range(len(n_total_ce)):
    for i in range(instances):
        if len(features_to_vary) > 0:
            e1 = exp.generate_counterfactuals(x_test[i:i+1], total_CFs=n_total_ce[k], desired_class="opposite", features_to_vary=features_to_vary, proximity_weight=proximity_weight, diversity_weight=diversity_weight)
        else:
            e1 = exp.generate_counterfactuals(x_test[i:i+1], total_CFs=n_total_ce[k], desired_class="opposite", proximity_weight=proximity_weight, diversity_weight=diversity_weight)
        
        # results as dataframe
        s = e1.cf_examples_list[0].final_cfs_df
