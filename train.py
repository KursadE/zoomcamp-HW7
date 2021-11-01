#!/usr/bin/env python
# coding: utf-8

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle

import seaborn as sns
from matplotlib import pyplot as plt

from IPython.display import display
from sklearn.metrics import mutual_info_score

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# ** Parameters **
categorical = ['occupation', 'channel_code', 'credit_product', 'age_cat','vintage_cat']
train_columns = categorical
max_depth = 10
min_samples_leaf = 20
output_file = f'tree_model_depth={max_depth}.bin'


print('Reading data')
!wget https://github.com/KursadE/zoomcamp-HW7/blob/main/traindata_creditcard.csv
df = pd.read_csv("traindata_creditcard.csv")


# ** Data Preparation and Data Cleaning **

print('Preparing data')

df.columns = df.columns.str.lower()

categorical = ['gender', 'region_code', 'occupation', 'channel_code', 'credit_product', 'is_active']

for col in df[categorical].columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')

df['credit_product'] = df['credit_product'].fillna('Unk')

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

df_full_train = df_full_train.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train['is_lead']
y_val = df_val['is_lead']
y_test = df_test['is_lead']


y_full_train = df_full_train['is_lead']
del df_full_train['is_lead']
del df_test['is_lead']

# ** Feature Engineering **
def cat(df):
    df['age_cat'] = pd.cut(df['age'],bins=[20,29,39,49,59,69,79,89],
       labels=['20-29','30-39','40-49','50-59','60-69','70-79','80-89'])
    df['vintage_cat'] = pd.cut(df['vintage'],bins=[0,19,39,59,79,99,119,139],
       labels=['0-19','20-39','40-59','60-79','80-99','100-119','120-139'])
    df['avg_account_balance_cat'] = pd.qcut(df['avg_account_balance'], q=10, 
                                                  labels=['0','1','2','3','4','5','6','7','8','9'])

cat(df_train)
cat(df_val)
cat(df_test)
cat(df_full_train)

# ** Defining Decision Tree Model **

def train(df, y, max_depth, min_samples_leaf):
    dicts = df[train_columns].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    model.fit(X_train, y)
    
    return dv, model

def predict(df, dv, model):
    dicts = df[train_columns].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

# ** Training Model **

print(f'Training Decision Tree with max_depth={max_depth} and min_samples_leaf={min_samples_leaf}')

dv, model = train(df_train, y_train, max_depth, min_samples_leaf)

y_pred = predict(df_val, dv, model)

print(f'auc score for training={roc_auc_score(y_val, y_pred)}')


# ** Training Final Model **

print(f'Training Final Decision Tree with max_depth={max_depth} and min_samples_leaf={min_samples_leaf}')

dv, model = train(df_full_train, y_full_train, max_depth, min_samples_leaf)

y_pred = predict(df_test, dv, model)

print(f'auc score for final model={roc_auc_score(y_test, y_pred)}')


# ** Saving the Model **

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'The model is saved to {output_file}')
