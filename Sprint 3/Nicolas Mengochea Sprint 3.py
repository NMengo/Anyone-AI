import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, precision_score, recall_score, f1_score, accuracy_score,\
    classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.linear_model import SGDRegressor, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from pandas.plotting import scatter_matrix
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.impute import SimpleImputer
from scipy.stats import iqr
from datetime import datetime as dt
import warnings
warnings.filterwarnings('ignore')
pd.options.display.max_rows = None

application_test = pd.read_csv('DataSets/application_test.csv')
application_train = pd.read_csv('DataSets/application_train.csv')
application_test.insert(1, 'TARGET', np.zeros(len(application_test)))


def equalize_train_test(train, test):
    train['NAME_INCOME_TYPE'] = train['NAME_INCOME_TYPE'].replace(['Maternity leave'], 'Working')
    # train['CODE_GENDER'] = train['CODE_GENDER'].replace(['XNA'], 'F')
    train = train.replace(['XNA', 'XAP'], np.nan)
    test = test.replace(['XNA', 'XAP'], np.nan)
    train['DAYS_EMPLOYED'].replace(365243, np.nan, inplace = True)
    test['DAYS_EMPLOYED'].replace(365243, np.nan, inplace = True)
    train['NAME_FAMILY_STATUS'] = train['NAME_FAMILY_STATUS'].replace(['Unknown'], 'Married')
    return train,test

def get_categoricals(train, criteria='Int and narrowed objects'):
    """
    The criteria for categoricals selection will impact in feature encoding.
    """
    df_categoricals = pd.DataFrame()
    df_categoricals['column_name'] = train.columns
    df_categoricals['dtype'] = np.array(train.dtypes)
    df_categoricals['n_uniques'] = np.array(train.nunique())
    # -----------------------------------------------------------------------------------
    # Filters
    type_object = df_categoricals['dtype'] == 'object'
    type_int = df_categoricals['dtype'] == ('int64')
    nunique_lt_2 = df_categoricals['n_uniques'] <= 2
    # -----------------------------------------------------------------------------------
    # Object nunique bounds
    objects_nuniques = df_categoricals[type_object]['n_uniques']
    upper_bound_nuniques = (np.quantile(objects_nuniques, 0.75) - np.quantile(objects_nuniques, 0.25)) * 1.5 \
                           + (np.quantile(objects_nuniques, 0.75))
    not_outlier_nuniques = df_categoricals['n_uniques'] < upper_bound_nuniques
    # -----------------------------------------------------------------------------------
    # Criteria options
    if criteria == 'Int and narrowed objects':
        df_categoricals = df_categoricals[
            (type_int & nunique_lt_2) |
            (type_object & not_outlier_nuniques)]
    elif criteria == 'Objects':
        df_categoricals = df_categoricals[type_object]
    elif criteria == 'Only narrow objects':
        df_categoricals = df_categoricals[(type_object & not_outlier_nuniques)]

    df_categoricals.set_index('column_name', inplace=True)
    return df_categoricals


def get_numerical_bounds(train, categoricals, index, bound_mult=1.5):
    df_numerical = pd.DataFrame()
    df_numerical['column_name'] = train.columns
    df_numerical['dtype'] = np.array(train.dtypes)
    df_numerical = df_numerical[df_numerical['dtype'] != 'object']
    df_numerical.reset_index(inplace=True, drop=True)

    for i in range(len(df_numerical)):
        df_numerical.loc[i, 'upper bound'] = np.quantile(train[df_numerical.loc[i, 'column_name']], 0.75) + iqr(
            train[df_numerical.loc[i, 'column_name']]) * bound_mult
        df_numerical.loc[i, 'lower bound'] = np.quantile(train[df_numerical.loc[i, 'column_name']], 0.25) - iqr(
            train[df_numerical.loc[i, 'column_name']]) * bound_mult
        df_numerical = df_numerical.fillna(0)

    df_numerical = pd.merge(df_numerical, categoricals, on=['column_name'], how="outer", indicator=True).query(
        '_merge=="left_only"')
    df_numerical = df_numerical[df_numerical['upper bound'] > 10]
    df_numerical.drop(['dtype_y', 'n_uniques', '_merge'], axis=1, inplace=True)
    df_numerical.set_index('column_name', inplace=True)
    df_numerical.drop([index, 'HOUR_APPR_PROCESS_START'], axis=0, inplace=True)
    return df_numerical


def manage_outliers(df_numerical, train, outl_treatment='Filter'):
    print(f"Original DF Lenght: {len(train)}")
    for column in train:
        if outl_treatment == 'Filter':
            if column in list(df_numerical.index):
                train = train[train[column] < df_numerical.loc[column, 'upper bound']]
                train = train[train[column] > df_numerical.loc[column, 'lower bound']]
        elif outl_treatment == 'Mean':
            if column in list(df_numerical.index):
                values_inside_range = train[column][(train[column] < df_numerical.loc[column, 'upper bound']) &
                                            (train[column] > df_numerical.loc[column, 'lower bound'])]
                mean_vir = float(values_inside_range.mean())
                train[column][(train[column] > df_numerical.loc[column, 'upper bound']) &
                                       (train[column] < df_numerical.loc[column, 'lower bound'])] = mean_vir
        else:
            break
    print(f"Post processing DF Lenght: {len(train)}")
    return train


def imputing_values(train, test, categoricals, object_treatment='Mode'):
    imp = SimpleImputer(missing_values=np.nan, strategy='median')

    for column in train:
        if train[column].dtypes != 'object':
            train[column] = imp.fit_transform(train[column].values.reshape(-1, 1))
            test[column] = imp.transform(test[column].values.reshape(-1, 1))

    if object_treatment == 'Drop rows if nans':
        for column in train:
            if train[column].dtypes == 'object':
                train = train.dropna(axis=0, how='any', subset=column)
        return train, test, categoricals
    elif object_treatment == 'Drop columns if nans':
        for column in train:
            if train[column].dtypes == 'object' and train[column].isna().any():
                train = train.drop(column, axis=1)
                categoricals = categoricals[categoricals.index != column]
            if test[column].dtypes == 'object' and test[column].isna().any():
                test = test.drop(column, axis=1)
        train, test = train.align(test, join='inner', axis=1)
        return train, test, categoricals
    elif object_treatment == 'Mode':
        imp2 = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        for column in train:
            if train[column].dtypes == 'object':
                train[column] = imp2.fit_transform(train[column].values.reshape(-1, 1))
                test[column] = imp2.transform(test[column].values.reshape(-1, 1))
        return train, test, categoricals
    else:
        raise ValueError('Wrong parameter')


def feature_encoding(train, test, categoricals):
    lb = LabelBinarizer()
    oh = OneHotEncoder(handle_unknown='ignore')
    trindex = train.index
    teindex = test.index
    for column_name in list(categoricals.index):
        try:
            if categoricals.loc[column_name, 'n_uniques'] <= 2:
                train[column_name] = lb.fit_transform(train[column_name])
                test[column_name] = lb.transform(test[column_name])
            elif categoricals.loc[column_name, 'n_uniques'] > 2:
                # Train Set --------------------------------------------------------------------------------------------
                train_encoded = pd.DataFrame(oh.fit_transform(train[[column_name]]).toarray(), index=trindex)
                train_encoded.columns = oh.get_feature_names_out([column_name])
                train_encoded.columns = train_encoded.columns.str.replace(column_name + '_', '')
                # Test Set ---------------------------------------------------------------------------------------------
                test_encoded = pd.DataFrame(oh.transform(test[[column_name]]).toarray(), index=teindex)
                test_encoded.columns = oh.get_feature_names_out([column_name])
                test_encoded.columns = test_encoded.columns.str.replace(column_name + '_', '')
                train, test = train.drop(column_name, axis=1), test.drop(column_name, axis=1)
                train, test = pd.concat([train, train_encoded], axis=1), pd.concat([test, test_encoded], axis=1)
            else:
                pass
        except:
            pass
    return train, test


# Function to Scale only numerical features from a given dataset, using a given Scaling method.
def numerical_scaler(train, test, scaling_method):
    columns_drop_list = []
    trindex = train.index
    teindex = test.index
    for column in train:
        print(column)
        if train[column].dtypes in ['float', 'float64', 'int64']:
            pass
        else:
            columns_drop_list.append(column)

    columns_drop_list.append('TARGET') # IF MINMAXSCALER, COMMENT. #####################################################
    train_numeric = train.drop(columns_drop_list, axis=1)
    test_numeric = test.drop(columns_drop_list, axis=1)
    train_numeric_columns = train_numeric.columns
    test_numeric_columns = test_numeric.columns
    scaler = scaling_method
    train_numeric = pd.DataFrame(scaler.fit_transform(train_numeric), columns=train_numeric_columns, index=trindex)
    test_numeric = pd.DataFrame(scaler.transform(test_numeric), columns=test_numeric_columns, index=teindex)
    train, test = (pd.concat([train[columns_drop_list], train_numeric], axis=1),
                   pd.concat([test[columns_drop_list], test_numeric], axis=1))
    return train, test

# Manual align between Dfs.
def train_test_column_dif(train:pd.DataFrame, test:pd.DataFrame):
    columns_train = list(train.columns)
    columns_test = list(test.columns)

    for column in columns_train:
        if column not in columns_test:
            test[column] = np.zeros(len(test))

    for column in columns_test:
        if column not in columns_train:
            train[column] = np.zeros(len(train))
    return train, test


def preprocessing(train: pd.DataFrame, test, index: str):
    train, test = equalize_train_test(train, test)
    categoricals = get_categoricals(train, criteria='Objects')                                                          # 'Int and narrowed objects' / 'Objects' / 'Only narrow objects'
    df_numerical = get_numerical_bounds(train, categoricals, index, bound_mult=1)                                       # Any
    train = manage_outliers(df_numerical, train, outl_treatment='Filter')                                               # None(only 365243) / Filter / Mean
    train, test, categoricals = imputing_values(train, test, categoricals, object_treatment='Mode')                     # Drop rows if nans / Drop columns if nans / Mode
    train, test = feature_encoding(train, test, categoricals)
    # ============================================
    train[index], test[index] = train[index].astype('int64'), test[index].astype('int64')
    train.set_index(index, inplace=True)
    test.set_index(index, inplace=True)
    # ============================================
    train, test = numerical_scaler(train, test, StandardScaler())                                                       # MinMaxScaler() / StandardScaler()
    train, test = train.select_dtypes(exclude=['object']), test.select_dtypes(exclude=['object'])
    train, test = train_test_column_dif(train, test)
    return train, test

def search_best_hyperparameters(search_method, train, train_targets, model, param_grid, cv=3, n_iter=None, scoring=None):
    # param_grid = [{str(key): value for key, value in kwargs.items()}]
    search = search_method(model, param_grid, cv=cv, n_iter=n_iter, scoring=scoring, return_train_score=True)
    search.fit(train, train_targets.ravel())

    best_params = search.best_params_
    best_params.update({'random_state': 42})
    best_score = search.best_score_
    results = search.cv_results_

    return best_params, best_score, results

def tuned_model_evaluation(X_train, y_train, X_test, model, params:dict):
    tuned_model = model(**{key:value for key, value in params.items()})
    tuned_model.fit(X_train, y_train)
    tuned_proba = tuned_model.predict_proba(X_test)[:, 1]
    tuned_predictions = pd.DataFrame({'SK_ID_CURR':X_test.index, 'TARGET':tuned_proba})
    tuned_predictions.to_csv('tuned_predictions_rfc.csv', index=False)

    return tuned_predictions

application_train, application_test = preprocessing(application_train, application_test, 'SK_ID_CURR')

X_train = application_train.drop('TARGET', axis=1)
y_train = application_train['TARGET'].copy()
ytrainhead = y_train.head()
X_test = application_test.drop('TARGET', axis=1)

# ======================================================================================================================
# Logistic Regressor - Default
lr = LogisticRegression(C=0.1, random_state=42)
lr.fit(X_train, y_train)
predict_proba = lr.predict_proba(X_test)[:, 1]
predictions_df = pd.DataFrame({'SK_ID_CURR':X_test.index, 'TARGET':predict_proba})
predictions_df.to_csv('predictions_df.csv', index=False)

# Private Score: 0.73472

"""
**N° 1 Try:

Categoricals (encoding related): Int and narrowed objects
Bounds: 1.5
Outlier treatment: None
Nan object treatment: Mode
Scaler: MinMaxScaler

**Result: 0.73113
"""

"""
**N° 2 Try:

Categoricals (encoding related): Objects
Bounds: 1.5
Outlier treatment: None
Nan object treatment: Mode
Scaler: StandardScaler

**Result: 0.73385
"""

"""
**N° 3 Try:

Categoricals (encoding related): Objects
Bounds: 1.5
Outlier treatment: None
Nan object treatment: Mode
Scaler: MinMaxScaler

**Result: 0.73038
"""

"""
**N° 4 Try:

Categoricals (encoding related): Only narrow objects
Bounds: 1.5
Outlier treatment: None
Nan object treatment: Mode
Scaler: StandardScaler

**Result: 0.73197
"""

"""
**N° 5 Try:

Categoricals (encoding related): Objects
Bounds: 1.5
Outlier treatment: Filter
Nan object treatment: Mode
Scaler: StandardScaler

**Result: 0.73434
"""

"""
**N° 6 Try:

Categoricals (encoding related): Objects
Bounds: 2
Outlier treatment: Filter
Nan object treatment: Mode
Scaler: StandardScaler

**Result: 0.73426
"""

"""
**N° 7 Try:

Categoricals (encoding related): Objects
Bounds: 1
Outlier treatment: Filter
Nan object treatment: Mode
Scaler: StandardScaler

**Result: 0.73472
"""

"""
**N° 8 Try:
**Even though with 0.5 bounds the result is slightly better I think there's way more risk of overfitting, and it is not worth it.**

Categoricals (encoding related): Objects
Bounds: 0.5
Outlier treatment: Filter
Nan object treatment: Mode
Scaler: StandardScaler

**Result: 0.73478
"""

"""
**N° 9 Try:
Categoricals (encoding related): Objects
Bounds: 1
Outlier treatment: Mean
Nan object treatment: Mode
Scaler: StandardScaler

**Result: 0.73385
"""

"""
**N° 10 Try:
Categoricals (encoding related): Objects
Bounds: 1
Outlier treatment: Filter
Nan object treatment: Drop rows if nans
Scaler: StandardScaler

**Result: 0.61486
"""

"""
**N° 12 Try:
Categoricals (encoding related): Objects
Bounds: 2.5
Outlier treatment: Mean
Nan object treatment: Drop rows if nans
Scaler: StandardScaler

**Result: 0.61212
"""

"""
**N° 13 Try:
Categoricals (encoding related): Objects
Bounds: 1
Outlier treatment: Filter
Nan object treatment: Drop columns if nans
Scaler: StandardScaler

**Result: 0.72924
"""

"""
**N° 14 Try:
Categoricals (encoding related): Objects
Bounds: 2
Outlier treatment: Mean
Nan object treatment: Drop columns if nans
Scaler: StandardScaler

**Result: 0.72880
"""

"""
**N° 15 Try:
**Even though with 0.5 bounds the result is slightly better I think there's way more risk of overfitting, and it is not worth it.**
Categoricals (encoding related): Objects
Bounds: 1
Outlier treatment: Filter
Nan object treatment: Mode
Scaler: StandardScaler

**Result: 0.73472
"""

# ======================================================================================================================
# Random Forest Classifier - Default
start = dt.now()
rfc = RandomForestClassifier(random_state=42, n_jobs=-1)
rfc.fit(X_train, y_train)
predict_proba_rfc = rfc.predict_proba(X_test)[:, 1]
predictions_rfc = pd.DataFrame({'SK_ID_CURR':X_test.index, 'TARGET':predict_proba_rfc})
predictions_rfc.to_csv('predictions_rfc.csv', index=False)

# Private Score: 0.68799

# ======================================================================================================================
# Random Forest Classifier - Tuned
param_grid = {
 'bootstrap': [True, False],
 'max_depth': [10, 50, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [100, 200]
}

best_params, best_score, results  = search_best_hyperparameters(RandomizedSearchCV, X_train, y_train, rfc, param_grid,
                                                                cv=3, n_iter=10, scoring='roc_auc')

tuned_predictions_rfc = tuned_model_evaluation(X_train, y_train, X_test, RandomForestClassifier, best_params)
end = dt.now()

# Private Score: 0.72011