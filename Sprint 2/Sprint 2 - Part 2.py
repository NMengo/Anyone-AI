import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, precision_score, recall_score, f1_score, accuracy_score,\
    classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.linear_model import SGDRegressor, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from pandas.plotting import scatter_matrix
# from Sprint2 import search_best_hyperparameters

def search_best_hyperparameters(train, train_targets, model, **kwargs):
    result = {
        "hyperparameters": None,
        "mae": None
    }
    # Complete your code here
    param_grid = [
        {str(key): value for key, value in kwargs.items()}
    ]
    grid_search = GridSearchCV(model, param_grid, cv=10, scoring='neg_mean_absolute_error', return_train_score=True)
    grid_search.fit(train, train_targets.ravel())

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    result.update({'hyperparameters': best_params, 'mae': best_score})

    return result


all_nba_df = pd.read_csv("https://raw.githubusercontent.com/anyoneai/notebooks/main/datasets/all_nba_1990_2017_dataset.csv"
                         , index_col=0)
all_nba_df = all_nba_df.reset_index(drop=True)

print(f"Number of rows: {len(all_nba_df)}")
print(f"Number of seasons: {all_nba_df['season_id'].nunique()}")
print(f"Number of players: {all_nba_df['player_id'].nunique()}")
print(f"Number of all-nba selections: {len(all_nba_df[all_nba_df['all_nba']==1])}")
print(f"Number of non-selected: {len(all_nba_df[all_nba_df['all_nba']==0])}")

players_x = all_nba_df.drop(['all_nba', 'season_id', 'player_id', 'player_season', 'player', 'season', 'season_start']
                            , axis=1)
players_y = all_nba_df['all_nba'].copy()

X_train, X_test, y_train, y_test = train_test_split(players_x, players_y,
                                                    random_state=42, stratify=all_nba_df['all_nba'])

comp = pd.DataFrame({
    'Overall': all_nba_df['all_nba'].value_counts() / len(all_nba_df),
    'Stratified': y_test.value_counts() / len(y_test),
}).sort_index()
print(comp)

log_reg = LogisticRegression(random_state=42)

class BaseLineModel():
    def fit(self, xtrain, ytrain):
        return self

    def predict(self, X):
        predictions = np.zeros((len(X), 1))
        return predictions

model = BaseLineModel()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
baseline_error = mean_absolute_error(y_test, predictions)
baseline_accuracy = accuracy_score(y_test, predictions)
print('---------------------------------------------------------------------')
print(f"Baseline model Accuracy: {baseline_accuracy}")

# 2.4
"""
In general, accuracy is not the only metric we should look at, since it may be misleading.
In this particular case, we have a class imbalance. Accuracy is not a good metric to evaluate model performance.

We could resample data in order to improve the imbalance.
Also, using another metric such as F1 which is a better fit, since it measures also the type of error made.
"""

players_x.hist(bins=50, figsize=(20, 15))

# 2.5
# Function to Scale only numerical features from a given dataset, using a given Scaling method.
def numerical_scaler(features_df, scaling_method):
    columns_drop_list = []
    features_df = features_df.reset_index(drop=True)
    for column in features_df:
        if features_df[str(column)].dtypes in ['float', 'float64', 'int64']:
            pass
        else:
            columns_drop_list.append(column)

    num_attr = features_df.drop(columns_drop_list, axis=1)
    num_attr_columns = num_attr.columns
    scaler = scaling_method
    num_attr = pd.DataFrame(scaler.fit_transform(num_attr), columns=num_attr_columns)
    num_attr = num_attr.reset_index(drop=True)
    features_df = pd.concat([features_df[columns_drop_list], num_attr], axis=1)

    return features_df

players_x = numerical_scaler(players_x, StandardScaler())

# 2.6
players_x2 = players_x.drop('team', axis=1)
X_train2, X_test2, y_train2, y_test2 = train_test_split(players_x2, players_y,
                                                    random_state=42, stratify=all_nba_df['all_nba'])
log_reg.fit(X_train2, y_train2)
log_reg_baseline_pred = log_reg.predict(X_test2)

# 2.7
logreg_base_acc = accuracy_score(y_test2, log_reg_baseline_pred)
logreg_base_prec = precision_score(y_test2, log_reg_baseline_pred)
logreg_base_recall = recall_score(y_test2, log_reg_baseline_pred)
logreg_base_f1 = f1_score(y_test2, log_reg_baseline_pred)

print('---------------------------------------------------------------------')
print(f"Accuracy: {logreg_base_acc}")
print(f"Precision: {logreg_base_prec}")
print(f"Recall: {logreg_base_recall}")
print(f"F1: {logreg_base_f1}")

# 2.8
def encode_df(features_df, column_tobe_encoded: str):
    encode = OneHotEncoder()
    enc_df = pd.DataFrame(encode.fit_transform(features_df[[column_tobe_encoded]]).toarray())
    enc_df.columns = encode.get_feature_names_out([column_tobe_encoded])
    enc_df.columns = enc_df.columns.str.replace(column_tobe_encoded+'_', '')
    features_df = features_df.reset_index(drop=True)
    features_df = features_df.drop(column_tobe_encoded, axis=1)
    features_df = pd.concat([features_df, enc_df], axis=1)

    return features_df

players_x = encode_df(players_x, 'team')

X_train, X_test, y_train, y_test = train_test_split(players_x, players_y,
                                                    random_state=42, stratify=all_nba_df['all_nba'])

# 2.9
log_reg = LogisticRegression(random_state=42)
result = search_best_hyperparameters(X_train, y_train, log_reg, tol=[0.00001, 0.0001, 0.001, 0.01],
                                     C=[0.001,0.01,0.1,1,10,100], max_iter=[100, 1000, 10000, 100000])

best_params = result['hyperparameters']
best_params.update({'random_state':42})

def tuned_model_evaluation(X_train, y_train, X_test, y_test, model, params:dict):
    tuned_model = model(**{key:value for key, value in params.items()})
    tuned_model.fit(X_train, y_train)
    tuned_predictions = tuned_model.predict(X_test)
    tuned_f1 = f1_score(y_test, tuned_predictions)
    tuned_recall = recall_score(y_test, tuned_predictions)
    tuned_precision = precision_score(y_test, tuned_predictions)

    return tuned_recall, tuned_precision, tuned_f1, tuned_predictions, tuned_model

log_reg_tuned = tuned_model_evaluation(X_train, y_train, X_test, y_test, LogisticRegression, best_params)
log_reg_tuned_precision = log_reg_tuned[1]
log_reg_tuned_recall = log_reg_tuned[0]
log_reg_tuned_f1 = log_reg_tuned[2]
log_reg_tuned_predictions = log_reg_tuned[3]

print('---------------------------------------------------------------------')
print(f"Precision Tuned Log Reg: {log_reg_tuned_precision}")
print(f"Recall Tuned Log Reg: {log_reg_tuned_recall}")
print(f"F1 Tuned Log Reg: {log_reg_tuned_f1}")


#=======================================================================================================================
all_nba_2018_df = pd.read_csv(
    "https://raw.githubusercontent.com/anyoneai/notebooks/main/datasets/all_nba_2018_dataset.csv", index_col=0)
all_nba_2018_selections = pd.read_csv(
    "https://raw.githubusercontent.com/anyoneai/notebooks/main/datasets/all_nba_2018_selections.csv", index_col=0)

def add_missing_teams(complete_df, incomplete_df):
    missing_columns = []
    for column in complete_df:
        if column in incomplete_df.columns:
            pass
        else:
            missing_columns.append(column)

    incomplete_df[[missing_columns]] = float(0)

    return incomplete_df

all_nba_2018_df = all_nba_2018_df.iloc[:, 6:]
all_nba_2018_df = numerical_scaler(all_nba_2018_df, StandardScaler())
all_nba_2018_df = encode_df(all_nba_2018_df, 'team')
all_nba_2018_df = add_missing_teams(players_x, all_nba_2018_df)

X_test_2018 = all_nba_2018_df
y_test_2018 = all_nba_2018_selections['all_nba'].copy()
y_test_2018 = y_test_2018.reset_index(drop=True)

# 2.11
new_test_log_reg = tuned_model_evaluation(X_train, y_train, all_nba_2018_df, y_test_2018, LogisticRegression, best_params)
new_test_recall = new_test_log_reg[0]
new_test_precision = new_test_log_reg[1]
new_test_f1 = new_test_log_reg[2]
new_test_predictions = new_test_log_reg[3]
log_reg_model = new_test_log_reg[4]



print('---------------------------------------------------------------------')
print(f"Precision Tuned Log Reg - New Test Set: {new_test_precision}")
print(f"Recall Tuned Log Reg - New Test Set: {new_test_recall}")
print(f"F1 Tuned Log Reg - New Test Set: {new_test_f1}")
print(classification_report(y_test_2018, new_test_predictions))

# 2.12
"""
With a set with higher percentage of positives, the Precision is higher and the Recall lower.
While when the positives percentage is low, the Precision is much lower and Recall much higher.

So, in the first case, the cases where the model predicted positive, it was really positive but failed to find them all.
In the second case, the model predicted most of the positives but also returned a high percentage of false positives.

In other words, the model is prone to predicting false positives.
"""

# 2.13
prob = log_reg_model.predict_proba(all_nba_2018_df)[:,1]

all_nba_2018_predictions = log_reg_model.predict(all_nba_2018_df)
all_nba_2018_selections = all_nba_2018_selections.reset_index(drop=True)
all_nba_2018_selections['Prediction'] = all_nba_2018_predictions
all_nba_2018_selections['all_nba_prob'] = prob
all_nba_2018_selections['all_nba_prob'] = all_nba_2018_selections['all_nba_prob'].round(3)

# 2.14
all_nba_2018_selections = all_nba_2018_selections.sort_values('all_nba_prob', ascending=False)
print(all_nba_2018_selections.head(15))

# 2.15
true_positive_selection = all_nba_2018_selections[(all_nba_2018_selections['all_nba']==1) &
                                                  (all_nba_2018_selections['Prediction']==1)]['player']
true_negative_selection = all_nba_2018_selections[(all_nba_2018_selections['all_nba']==1) &
                                                  (all_nba_2018_selections['Prediction']==0)]['player']
false_positive_selection = all_nba_2018_selections[(all_nba_2018_selections['all_nba']==0) &
                                                   (all_nba_2018_selections['Prediction']==1)]['player']

print(f"True Positive players: {true_positive_selection}")
print(f"True Negative players: {true_negative_selection}")
print(f"False Positive players: {false_positive_selection}")

# 2.16

class AllNbaSingleSeasonClassifier():
    def __init__(self, model):
        self._model = model

    def predict(self, df_features: pd.DataFrame):
        prob = self._model.predict_proba(df_features)[:,1]
        df_features['prob'] = prob
        df_features = df_features.sort_values('prob', ascending=False)
        df_features['prediction'] = 0
        df_features.iloc[0:15, -1] = 1
        predictions = df_features['prediction'].sort_index(ascending=True).to_numpy()
        return predictions


# 2.17
last_model = AllNbaSingleSeasonClassifier(log_reg_model)
last_model_predictions = last_model.predict(all_nba_2018_df)
last_model_f1 = f1_score(y_test_2018, last_model_predictions)
last_model_recall = recall_score(y_test_2018, last_model_predictions)
last_model_precision = precision_score(y_test_2018, last_model_predictions)
print('---------------------------------------------------------------------')
print(f"Precision Last Log Reg - New Test Set: {last_model_precision}")
print(f"Recall Last Log Reg - New Test Set: {last_model_recall}")
print(f"F1 Last Log Reg - New Test Set: {last_model_f1}")