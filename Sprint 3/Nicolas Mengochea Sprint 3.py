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


application_test = pd.read_csv('DataSets/application_test.csv')
application_train = pd.read_csv('DataSets/application_train.csv')


def get_categoricals(df_features):
    df_categoricals = pd.DataFrame()
    df_categoricals['column_name'] = df_features.columns
    df_categoricals['dtype'] = np.array(df_features.dtypes)
    df_categoricals['n_uniques'] = np.array(df_features.nunique())
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


def get_numerical_bounds(df_features):
    df_numerical = pd.DataFrame()
    df_numerical['column_name'] = df_features.columns
    df_numerical['dtype'] = np.array(df_features.dtypes)
    df_numerical = df_numerical[df_numerical['dtype'] != 'object']
    df_numerical.reset_index(inplace=True, drop=True)

    for i in range(len(df_numerical)):
        df_numerical.loc[i, 'upper bound'] = np.quantile(df_features[df_numerical.loc[i, 'column_name']], 0.75) + iqr(
            df_features[df_numerical.loc[i, 'column_name']]) * 1.5
        df_numerical.loc[i, 'lower bound'] = np.quantile(df_features[df_numerical.loc[i, 'column_name']], 0.25) - iqr(
            df_features[df_numerical.loc[i, 'column_name']]) * 1.5
        df_numerical = df_numerical.fillna(0)

    df_numerical = pd.merge(df_numerical, categoricals, on=['column_name'], how="outer", indicator=True).query(
        '_merge=="left_only"')
    df_numerical = df_numerical[df_numerical['upper bound'] > 10]
    df_numerical.drop(['dtype_y', 'n_uniques', '_merge'], axis=1, inplace=True)
    df_numerical.drop([0, 22], axis=0, inplace=True)
    df_numerical.set_index('column_name', inplace=True)
    return df_numerical


def remove_outliers(df_numerical, df_features):
    print(f"Original DF Lenght: {len(df_features)}")
    for column in df_features:
        if column in list(df_numerical.index):
            df_features = df_features[df_features[column] < df_numerical.loc[column, 'upper bound']]
            df_features = df_features[df_features[column] > df_numerical.loc[column, 'lower bound']]
    print(f"Post processing DF Lenght: {len(df_features)}")
    return df_features


def imputing_values(df_features, object_treatment=None):
    print('Original Df:')
    print('-----------------------------------------------------------------------')
    print(df_features.isna().sum())
    imp = SimpleImputer(missing_values=np.nan, strategy='median')

    for column in df_features:
        if df_features[column].dtypes != 'object':
            df_features[column] = imp.fit_transform(df_features[column].values.reshape(-1, 1))
    print('Df with numerical attributes imputed:')
    print('-----------------------------------------------------------------------')
    print(df_features.isna().sum())

    if not object_treatment:
        print('Df with both numerical and object attributes imputed:')
        print('-----------------------------------------------------------------------')
        print(df_features.isna().sum())
        return df_features
    elif object_treatment == 'Drop':
        df_features = df_features.dropna()
        print('Df with both numerical and object attributes imputed:')
        print('-----------------------------------------------------------------------')
        print(df_features.isna().sum())
        return df_features
    elif object_treatment == 'Mode':
        imp2 = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        for column in df_features:
            df_features[column] = imp2.fit_transform(df_features[column].values.reshape(-1, 1))
        print('Df with both numerical and object attributes imputed:')
        print('-----------------------------------------------------------------------')
        print(df_features.isna().sum())
        return df_features
    else:
        raise ValueError('Wrong parameter')


def feature_encoding(df_features, categoricals):
    lb = LabelBinarizer()
    oh = OneHotEncoder()
    for column_name in list(categoricals.index):
        if categoricals.loc[column_name, 'n_uniques'] == 2:
            df_features[column_name] = lb.fit_transform(df_features[column_name])
        else:
            enc_df = pd.DataFrame(oh.fit_transform(df_features[[column_name]]).toarray())
            enc_df.columns = oh.get_feature_names_out([column_name])
            enc_df.columns = enc_df.columns.str.replace(column_name + '_', '')
            df_features = df_features.reset_index(drop=True)
            df_features = df_features.drop(column_name, axis=1)
            df_features = pd.concat([df_features, enc_df], axis=1)
    return df_features


categoricals = get_categoricals(application_train)
df_numerical = get_numerical_bounds(application_train)
application_train = remove_outliers(df_numerical, application_train)
application_train = imputing_values(application_train, object_treatment= 'Mode')
application_train = feature_encoding(application_train, categoricals)
app_head = application_train.head(5)