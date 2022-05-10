import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from scipy import stats
data = pd.read_csv("https://raw.githubusercontent.com/anyoneai/notebooks/main/datasets/project2_players_df.csv")
data.dropna(subset=["PTS"], inplace=True)
data = data.dropna(subset=['TEAM_NAME'])
data.reset_index(drop=True,inplace=True)

# fig, axs = plt.subplots(ncols=5, figsize=(40,10))
# sns.regplot(x='PTS', y='Salary', data=data, ax=axs[0]).set(title='Score Vs Salary')
# sns.regplot(x='AST', y='Salary', data=data, ax=axs[1]).set(title='Assists Vs Salary')
# sns.regplot(x='REB', y='Salary', data=data, ax=axs[2]).set(title='Rebounds Vs Salary')
# sns.regplot(x='STL', y='Salary', data=data, ax=axs[3]).set(title='Steals Vs Salary')
# sns.regplot(x='BLK', y='Salary', data=data, ax=axs[4]).set(title='Blocks Vs Salary')
# plt.show()

players = data[['PTS', 'Salary']].copy()
players['PTS_cat'] = pd.cut(players['PTS'], bins=[0, 5, 10, 15, 30], labels=[1, 2, 3, 4], include_lowest=True)
players['PTS_cat'].hist()

# Stratified approach
players = players.reset_index(drop=True)
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(players, players['PTS_cat']):
    strat_train_set = players.loc[train_index]
    strat_test_set = players.loc[test_index]

# Random approach
train_set, test_set = train_test_split(players, test_size=0.2, random_state=42)

# Sampling bias comparison
comp = pd.DataFrame({
    'Overall': players['PTS_cat'].value_counts() / len(players),
    'Stratified': strat_test_set['PTS_cat'].value_counts() / len(strat_test_set),
    'Random': test_set['PTS_cat'].value_counts() / len(test_set)
}).sort_index()
comp['Rand %error'] = ((comp['Random'] - comp['Overall']) / comp['Overall']) * 100
comp['Strat %error'] = ((comp['Stratified'] - comp['Overall']) / comp['Overall']) * 100
print(comp)

# Removing Clustering to clean data again.
strat_train_set = strat_train_set.drop('PTS_cat', axis=1)
strat_test_set = strat_test_set.drop('PTS_cat', axis=1)

# Random and Stratified Train Sets
players = strat_train_set.drop('Salary', axis=1)
X_train = strat_train_set.drop('Salary', axis=1)
players_target = strat_train_set['Salary'].copy()
y_train = strat_train_set['Salary'].copy()

# Random and Stratified Test Sets
players_test = strat_test_set.drop('Salary', axis=1)
X_test = strat_test_set.drop('Salary', axis=1)
players_target_test = strat_test_set['Salary'].copy()
y_test = strat_test_set['Salary'].copy()

# ======================================================================================================================
class BaselineModel():
    """A baseline model that always returns the same value, the mean of the players salary in the train data"""

    def fit(self, players_target):
        self.mean = int(np.mean(players_target))

    def predict(self, X):
        predictions = np.array([self.mean] * len(X.index))
        return (predictions)
model = BaselineModel()
model.fit(players_target)
baseline_prediction = model.predict(players_test)
baseline_error = mean_absolute_error(players_target_test, baseline_prediction)
baseline_error = int(round(baseline_error))


# ======================================================================================================================
# Feature Scaling
"""
Since there are no extreme outliers in PTS, using Min-Max Scaling seems the right fit.
"""
minmaxscal = MinMaxScaler()
players = minmaxscal.fit_transform(players.values.reshape(-1,1))
players_test = minmaxscal.fit_transform(players_test.values.reshape(-1,1))
X_train = minmaxscal.fit_transform(X_train.values.reshape(-1,1))
X_test = minmaxscal.fit_transform(X_test.values.reshape(-1,1))


# ======================================================================================================================
# Testing & evaluating models

# Stratified approach
sgd_reg = SGDRegressor(random_state=42)
sgd_reg.fit(players, players_target.ravel())
sgd_predictions = sgd_reg.predict(players_test)
sgd_mse = mean_absolute_error(players_target_test, sgd_predictions)

# Random approach
sgd_regN = SGDRegressor(random_state=42)
sgd_regN.fit(X_train, y_train.ravel())
sgdN_predictions = sgd_regN.predict(X_test)
sgdN_mse = mean_absolute_error(y_test, sgdN_predictions)


# ======================================================================================================================
# Tuning models
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


# ======================================================================================================================
# Tuned model test & evaluation
sgd_reg = SGDRegressor(random_state=42)
result = search_best_hyperparameters(players, players_target, sgd_reg, max_iter = [1000, 10000, 1000000],
                                     eta0 = [0.0001, 0.001, 0.01, 0.1])
best_params = result['hyperparameters']
eta = best_params['eta0']
max_ite = best_params['max_iter']

sgd_reg_tuned = SGDRegressor(random_state=42, eta0=eta, max_iter=max_ite)
sgd_reg_tuned.fit(players, players_target)
sgd_tuned_predictions = sgd_reg_tuned.predict(players_test)
best_mae = mean_absolute_error(players_target_test, sgd_tuned_predictions)

print("Mean Absolute Error for each model:")
print(f"Baseline: {baseline_error}")
print(f"Default Random SGDRegressor: {sgdN_mse}")
print(f"Default Stratified SGDRegressor: {sgd_mse}")
print(f"Best SGDRegressor: {best_mae}")


# ======================================================================================================================
# Multivariate SGDRegressor

# Feature Engineering
players_c = data[["PTS", "REB", "AST", "BLK", "SEASON_EXP", "POSITION", "DRAFT_NUMBER", "Salary", "PERSON_ID", "TEAM_NAME"]] .copy()
players_c = players_c.set_index('PERSON_ID')
minmaxscal = MinMaxScaler()
columns = players_c[["PTS", "REB", "AST", "BLK", "SEASON_EXP"]].values.reshape(-1, 5)
columns = minmaxscal.fit_transform(columns)
players_c[["PTS", "REB", "AST", "BLK", "SEASON_EXP"]] = columns

players_c['DRAFT_NUMBER'] = players_c['DRAFT_NUMBER'].replace('Undrafted', 0)
players_c['DRAFT_NUMBER'] = (players_c['DRAFT_NUMBER'].fillna(0)).astype('int64')
players_c['draf_bin'] = pd.cut(players_c['DRAFT_NUMBER'],
                               bins=[0, 0.1, 15, 30, 60], include_lowest=True, labels=['firstround_lottery',
                                                                       'firstround_non_lottery',
                                                                       'second_round',
                                                                       'undrafted'])

players_c = pd.get_dummies(players_c, columns=['DRAFT_NUMBER', 'POSITION', 'TEAM_NAME'])


# ======================================================================================================================
# Train, Test split & Model evaluation
players_x = players_c.drop(['Salary', 'draf_bin'], axis=1)
players_y = players_c['Salary'].copy()
X_train, X_test, y_train, y_test = train_test_split(players_x, players_y, test_size=0.2, random_state=42)

sgd_reg_multi = SGDRegressor(random_state=42)
result2 = search_best_hyperparameters(X_train, y_train, sgd_reg_multi,
                                      loss = ['squared_loss', 'squared_epsilon_insensitive', 'huber', 'epsilon_insensitive'],
                                      penalty = ['l1', 'l2', 'elasticnet'],
                                      max_iter = [100, 500, 1000])
best_params = result2['hyperparameters']
loss = best_params['loss']
penalty = best_params['penalty']
max_iter = best_params['max_iter']
sgd_multi_tuned = SGDRegressor(random_state=42, loss=loss, max_iter=max_iter, penalty=penalty)
sgd_multi_tuned.fit(X_train, y_train)
sgd_multi_tuned_predictions = sgd_multi_tuned.predict(X_test)
best_mae = mean_absolute_error(y_test, sgd_multi_tuned_predictions)

print(f'Multi SGDRegressor: {best_mae}')

"""
Model with Stratified Sampling seems to be doing slightly better than a Multivariate regression.
"""

# ======================================================================================================================
# Multivariate Decision Tree

dec_tree = DecisionTreeRegressor(random_state=42)
result3 = search_best_hyperparameters(X_train, y_train, dec_tree, max_depth = [5, 10, 15, 20, 50],
                                      min_samples_leaf = [2, 10, 20, 50], max_features = [5, 10])
best_params = result3['hyperparameters']
max_depth = best_params['max_depth']
min_samples_leaf = best_params['min_samples_leaf']
max_features = best_params['max_features']
dec_tree2 = DecisionTreeRegressor(random_state=42, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                  max_features=max_features)
dec_tree2.fit(X_train, y_train)
dec_tree2_predictions = dec_tree2.predict(X_test)
best_mae = mean_absolute_error(y_test, dec_tree2_predictions)

print(f'Multi Decision Tree: {best_mae}')

"""
Now there's a clear improvement in the result with the Decision Tree.
This means that a non-linear approach was the right fit due to the Draft variable that "spoils" the linear correlation
of the rest of players.
Maybe we could run a linear model for non-rookie players and a non-linear model for rookie ones, to improve results.
"""


