import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.linear_model import SGDRegressor, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from pandas.plotting import scatter_matrix
from Sprint2 import search_best_hyperparameters
import warnings

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
print(f"Baseline model Accuracy: {baseline_accuracy}")

# 2.4
"""
Accuracy is not the only metric we should look at, since it's insight may be not only be insufficient but also wrong.
If model is overfitting badly we should expect error to tend to 0.

In which case, in order to check if it's the case and also leaving Test set untouched, we should create a validation set.
Manually, by splitting train set into subsets or directly with sci-kit learn k-fold cross-validation.
"""

players_x.hist(bins=50, figsize=(20, 15))

# 2.5
num_attr = players_x.drop('team', axis=1)
std_scal = StandardScaler()
num_attr = pd.DataFrame(std_scal.fit_transform(num_attr), columns=players_x.columns[1:])
players_x = pd.concat([players_x['team'], num_attr], axis=1)

X_train, X_test, y_train, y_test = train_test_split(players_x, players_y,
                                                    random_state=42, stratify=all_nba_df['all_nba'])
# 2.6
X_train2 = X_train.drop('team', axis=1)
X_test2 = X_test.drop('team', axis=1)
log_reg.fit(X_train2, y_train)
log_reg_baseline_pred = log_reg.predict(X_test2)

# 2.7
logreg_base_acc = accuracy_score(y_test, log_reg_baseline_pred)
logreg_base_prec = precision_score(y_test, log_reg_baseline_pred)
logreg_base_recall = recall_score(y_test, log_reg_baseline_pred)
logreg_base_f1 = f1_score(y_test, log_reg_baseline_pred)

print(f"Accuracy: {logreg_base_acc}")
print(f"Precision: {logreg_base_prec}")
print(f"Recall: {logreg_base_recall}")
print(f"F1: {logreg_base_f1}")

# 2.8
encode = OneHotEncoder()
enc_df = pd.DataFrame(encode.fit_transform(players_x[['team']]).toarray())
enc_df.columns = encode.get_feature_names_out(['team'])
enc_df.columns = enc_df.columns.str.replace('team_', '')
players_x = players_x.reset_index(drop=True)
players_x = players_x.drop('team', axis=1)
players_x = pd.concat([players_x, enc_df], axis=1)

X_train, X_test, y_train, y_test = train_test_split(players_x, players_y,
                                                    random_state=42, stratify=all_nba_df['all_nba'])

# 2.9
log_reg = LogisticRegression(random_state=42)
result = search_best_hyperparameters(X_train, y_train, log_reg, tol=[0.00001, 0.0001, 0.001, 0.01],
                                     C=[0.001,0.01,0.1,1,10,100], max_iter=[100, 1000, 10000, 100000])

best_params = result['hyperparameters']
tol = best_params['tol']
C = best_params['C']
max_iter = best_params['max_iter']

log_reg_tuned = LogisticRegression(tol=tol, C=C, max_iter=max_iter)
log_reg_tuned.fit(X_train, y_train)
log_reg_tuned_predictions = log_reg_tuned.predict(X_test)
log_reg_tuned_f1 = f1_score(y_test, log_reg_tuned_predictions)
log_reg_tuned_recall = recall_score(y_test, log_reg_tuned_predictions)
log_reg_tuned_precision = precision_score(y_test, log_reg_tuned_predictions)

print(f"F1 Tuned Log Reg: {log_reg_tuned_f1}")
