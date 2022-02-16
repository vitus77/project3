import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sqlite3
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from category_encoders import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
import pickle

conn = sqlite3.connect('apart.db')
cur = conn.cursor()

query = cur.execute("SELECT * FROM apt_deal")
columns = [column[0] for column in query.description]

df = pd.DataFrame.from_records(data = query.fetchall(), columns = columns)

cur.close()
conn.close()

conn = sqlite3.connect('apart.db')
cur = conn.cursor()

query = cur.execute("SELECT * FROM code")
columns = [col[0] for col in query.description]

df2 = pd.DataFrame.from_records(data = query.fetchall(), columns = columns)

cur.close()
conn.close()

df.drop('id', axis = 1, inplace = True)
df['거래일'] = df['거래일'].str.replace('-', '').astype(int)

df_ml = pd.merge(df, df2, left_on = '지역코드', right_on = '법정동코드', how = 'outer')

df_ml.drop(index = [431570, 431571], inplace = True)
to_int = ['지역코드', '거래일', '층', '건축년도', '거래금액']
df_ml[to_int] = df_ml[to_int].astype(int)
df_ml = df_ml.sort_values(by='거래일').reset_index(drop = True)
df_ml.drop('법정동코드', axis = 1, inplace = True)
df_ml = df_ml[['지역코드', '도시', '시군구', '법정동', '거래일', '아파트', '전용면적', '층', '건축년도', '거래금액']]

target = '거래금액'
features = df_ml.drop(columns = target).columns
baseline = df_ml[target].mean()

df_ml[target] = np.log1p(df_ml[target])

mask_train = (df_ml['거래일'] <20200000)
mask_test = (df_ml['거래일'] >= 20200000)

train = df_ml[mask_train]
test = df_ml[mask_test]

X_train = train[features]
y_train = train[target]
X_test = test[features]
y_test = test[target]

def Regression_Accuracy(y, pred):
    mae = mean_absolute_error(y, pred)
    mse = mean_squared_error(y, pred)
    r2 = r2_score(y, pred)
    print('MAE: ', mae,  'MSE: ', mse, 'r2 : ', r2)

pipe = Pipeline([
                 ('preprocessing', make_pipeline(OrdinalEncoder(), SimpleImputer())),
                 ('xgb', XGBRegressor(random_state=2, objective='reg:squarederror'))
])

dists = {
    'preprocessing__simpleimputer__strategy': ['mean', 'median'],
    'xgb__learning_rate': [0.01, 0.1],
    'xgb__n_estimators': [50, 100, 300],
    'xgb__max_depth': [3, 4, 5, 6, 7, 8]
}

clf = RandomizedSearchCV(
    pipe,
    param_distributions = dists,
    n_iter = 10,
    cv = 3,
    scoring = 'neg_mean_squared_error',
    verbose = 1,
    n_jobs = -1
)

clf.fit(X_train, y_train)
pipe = clf.best_estimator_

y_pred = pipe.predict(X_test)

y_test_exp = np.expm1(y_test)
y_pred_exp = np.expm1(y_pred)

Regression_Accuracy(y_test_exp, y_pred_exp)

def Predict_Price(code, state, city, dong, date, name, area, floor, year):
    df = pd.DataFrame(data = [[code, state, city, dong, date, name, area, floor, year]], columns = ['지역코드', '도시', '시군구', '법정동', '거래일', '아파트', '전용면적', '층', '건축년도'])

    y_pred = np.expm1(pipe.predict(df)[0])
    return y_pred

Predict_Price(29200, '광주광역시', '광산구', '신가동', 20200101, '도시공사', 84.83, 2, 2002)

with open('pipe.pkl', 'wb') as pickle_file:
    pickle.dump(pipe, pickle_file)

with open('pipe.pkl','rb') as pickle_file:
    pipe = pickle.load(pickle_file)

Predict_Price(29200, '광주광역시', '광산구', '신가동', 20200101, '도시공사', 84.83, 2, 2002)