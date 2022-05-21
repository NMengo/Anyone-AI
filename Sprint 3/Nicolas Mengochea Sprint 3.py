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
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelBinarizer
from sklearn.impute import SimpleImputer
from scipy.stats import iqr
import warnings
warnings.filterwarnings('ignore')
pd.options.display.max_rows = None

# application_test = pd.read_csv('DataSets/application_test.csv')
# application_train = pd.read_csv('DataSets/application_train.csv')
# application_test.insert(1, 'TARGET', np.zeros(len(application_test)))


def equalize_train_test(train):
    train['NAME_INCOME_TYPE'] = train['NAME_INCOME_TYPE'].replace(['Maternity leave'], 'Working')
    train['CODE_GENDER'] = train['CODE_GENDER'].replace(['XNA'], 'F')
    train['NAME_FAMILY_STATUS'] = train['NAME_FAMILY_STATUS'].replace(['Unknown'], 'Married')
    return train

def get_categoricals(train):
    df_categoricals = pd.DataFrame()
    df_categoricals['column_name'] = train.columns
    df_categoricals['dtype'] = np.array(train.dtypes)
    df_categoricals['n_uniques'] = np.array(train.nunique())
    type_object = df_categoricals['dtype'] == 'object'
    type_int = df_categoricals['dtype'] == ('int64')
    nunique_lt_2 = df_categoricals['n_uniques'] <= 2
    objects_nuniques = df_categoricals[type_object]['n_uniques']
    upper_bound = (np.quantile(objects_nuniques, 0.75) - np.quantile(objects_nuniques, 0.25))*1.5 \
                  + (np.quantile(objects_nuniques, 0.75))
    not_outlier = df_categoricals['n_uniques'] < upper_bound

    df_categoricals = df_categoricals[
        (type_int & nunique_lt_2) |
        (type_object & not_outlier)
    ]
    df_categoricals.set_index('column_name', inplace=True)
    return df_categoricals


def get_numerical_bounds(train, categoricals):
    df_numerical = pd.DataFrame()
    df_numerical['column_name'] = train.columns
    df_numerical['dtype'] = np.array(train.dtypes)
    df_numerical = df_numerical[df_numerical['dtype'] != 'object']
    df_numerical.reset_index(inplace=True, drop=True)

    for i in range(len(df_numerical)):
        df_numerical.loc[i, 'upper bound'] = np.quantile(train[df_numerical.loc[i, 'column_name']], 0.75) + iqr(
            train[df_numerical.loc[i, 'column_name']]) * 1.5
        df_numerical.loc[i, 'lower bound'] = np.quantile(train[df_numerical.loc[i, 'column_name']], 0.25) - iqr(
            train[df_numerical.loc[i, 'column_name']]) * 1.5
        df_numerical = df_numerical.fillna(0)

    df_numerical = pd.merge(df_numerical, categoricals, on=['column_name'], how="outer", indicator=True).query(
        '_merge=="left_only"')
    df_numerical = df_numerical[df_numerical['upper bound'] > 10]
    df_numerical.drop(['dtype_y', 'n_uniques', '_merge'], axis=1, inplace=True)
    df_numerical.set_index('column_name', inplace=True)
    df_numerical.drop(['SK_ID_CURR', 'HOUR_APPR_PROCESS_START'], axis=0, inplace=True)
    return df_numerical


def remove_outliers(df_numerical, train):
    print(f"Original DF Lenght: {len(train)}")
    for column in train:
        if column in list(df_numerical.index):
            train = train[train[column] < df_numerical.loc[column, 'upper bound']]
            train = train[train[column] > df_numerical.loc[column, 'lower bound']]
    print(f"Post processing DF Lenght: {len(train)}")
    return train


def imputing_values(train, test, object_treatment=None):
    print('-----------------------------------------------------------------------')
    print('Original Df:')
    print(train.isna().sum())
    imp = SimpleImputer(missing_values=np.nan, strategy='median')

    for column in train:
        if train[column].dtypes != 'object':
            train[column] = imp.fit_transform(train[column].values.reshape(-1, 1))
            test[column] = imp.transform(test[column].values.reshape(-1, 1))
    print('-----------------------------------------------------------------------')
    print('Df with numerical attributes imputed:')
    print(train.isna().sum())

    if not object_treatment:
        print('-----------------------------------------------------------------------')
        print('Df with both numerical and object attributes imputed:')
        print(train.isna().sum())
        print('-----------------------------------------------------------------------')
        print(test.isna().sum())
        return train, test
    elif object_treatment == 'Drop':
        train, test = train.dropna(), test.dropna()
        print('-----------------------------------------------------------------------')
        print('Df with both numerical and object attributes imputed:')
        print(train.isna().sum())
        print('-----------------------------------------------------------------------')
        print(test.isna().sum())
        return train, test
    elif object_treatment == 'Mode':
        imp2 = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        for column in train:
            if train[column].dtypes == 'object':
                train[column] = imp2.fit_transform(train[column].values.reshape(-1, 1))
                test[column] = imp2.transform(test[column].values.reshape(-1, 1))
        print('-----------------------------------------------------------------------')
        print('Df with both numerical and object attributes imputed:')
        print(train.isna().sum())
        print('-----------------------------------------------------------------------')
        print(test.isna().sum())
        return train, test
    else:
        raise ValueError('Wrong parameter')


def feature_encoding(train, test, categoricals):
    lb = LabelBinarizer()
    oh = OneHotEncoder(handle_unknown='ignore')
    trindex = train.index
    teindex = test.index
    for column_name in list(categoricals.index):
        if categoricals.loc[column_name, 'n_uniques'] == 2:
            train[column_name] = lb.fit_transform(train[column_name])
            test[column_name] = lb.transform(test[column_name])
        elif categoricals.loc[column_name, 'n_uniques'] > 2:
            enc_df = pd.DataFrame(oh.fit_transform(train[[column_name]]).toarray(), index=trindex)
            enc_df.columns = oh.get_feature_names_out([column_name])
            enc_df.columns = enc_df.columns.str.replace(column_name + '_', '')
            enc_df_test = pd.DataFrame(oh.transform(test[[column_name]]).toarray(), index=teindex)
            enc_df_test.columns = oh.get_feature_names_out([column_name])
            enc_df_test.columns = enc_df_test.columns.str.replace(column_name + '_', '')
            # train, test = train.reset_index(drop=True), test.reset_index(drop=True)
            train, test = train.drop(column_name, axis=1), test.drop(column_name, axis=1)
            train, test = pd.concat([train, enc_df], axis=1), pd.concat([test, enc_df_test], axis=1)
        else:
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

    num_attr = train.drop(columns_drop_list, axis=1)
    num_attr_test = test.drop(columns_drop_list, axis=1)
    num_attr_columns = num_attr.columns
    scaler = scaling_method
    num_attr = pd.DataFrame(scaler.fit_transform(num_attr), columns=num_attr_columns, index=trindex)
    num_attr_test = pd.DataFrame(scaler.transform(num_attr_test), columns=num_attr_columns, index=teindex)
    # num_attr = num_attr.reset_index(drop=True)
    # num_attr_test = num_attr_test.reset_index(drop=True)
    train, test = pd.concat([train[columns_drop_list], num_attr], axis=1), pd.concat([test[columns_drop_list], num_attr_test], axis=1)
    return train, test


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
    train = equalize_train_test(train)
    categoricals = get_categoricals(train)
    df_numerical = get_numerical_bounds(train, categoricals)
    train = remove_outliers(df_numerical, train)
    train, test = imputing_values(train, test, object_treatment='Mode')
    train, test = feature_encoding(train, test, categoricals)
    # ============================================
    train[index], test[index] = train[index].astype('int64'), test[index].astype('int64')
    train.set_index(index, inplace=True)
    test.set_index(index, inplace=True)
    # ============================================
    train, test = numerical_scaler(train, test, MinMaxScaler())
    train, test = train.select_dtypes(exclude=['object']), test.select_dtypes(exclude=['object'])
    train, test = train_test_column_dif(train, test)
    return train, test


# application_train, application_test = preprocessing(application_train, application_test, 'SK_ID_CURR')
# application_train.to_csv('application_train_mod.csv')
# application_test.to_csv('application_test_mod.csv')


application_train, application_test = pd.read_csv('application_train_mod.csv'), pd.read_csv('application_test_mod.csv')
application_train, application_test = application_train.set_index('SK_ID_CURR'), application_test.set_index('SK_ID_CURR')

X_train = application_train.drop('TARGET', axis=1)
y_train = application_train['TARGET'].copy()
X_test = application_test.drop('TARGET', axis=1)

X_testhead = X_test.head()

lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)
predictions = lr.predict(X_test)
full_proba = lr.predict_proba(X_test)
predict_proba = lr.predict_proba(X_test)[:, 1]
predictions_df = pd.DataFrame({'SK_ID_CURR':X_test.index, 'TARGET':predict_proba})
predictions_df.to_csv('predictions_df.csv', index=False)

apphead = application_train.head()
apphead2 = application_test.head()