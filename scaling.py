import pandas as pd
import numpy as np

data = pd.read_csv('data/ApartDeal.csv', low_memory=False)

data = data.sample(frac = 0.1)
data = data.drop('지번', axis = 1)
data['지역코드'] = data['지역코드'].astype(np.int32)
data['전용면적'] = data['전용면적'].astype(np.float32)
data['건축년도'] = data['건축년도'].astype(np.int16)
data['거래금액'] = data['거래금액'].astype(np.int32)
data['거래일'] = pd.to_datetime(data['거래일'])

data.to_csv('data/Apart Deal.csv', index = False)

data = pd.read_csv('data/법정동코드 전체자료.txt', sep = '\t', encoding = 'cp949')

data['법정동코드'] = data['법정동코드'].astype(str)
data['법정동코드'] = data['법정동코드'].str[:5]
data[['도시', '시군구', '동']] = pd.DataFrame(data['법정동명'].str.split(' ', 2).tolist())

idx = data[data['폐지여부'] == '폐지'].index
data.drop(index = idx, inplace = True)
data.drop('폐지여부', axis = 1, inplace = True)

temp1 = data[data['동'].isna() == True].copy()
temp2 = data[data['법정동명'].str.endswith('구')].copy()

temp2.dropna(inplace = True)
temp_list = temp2['시군구'].unique().tolist()
for temp in temp_list:
  idx = temp1[temp1['시군구'] == temp].index
  temp1.drop(index = idx, inplace = True)

temp2['시군구'] = temp2['시군구'] + ' ' + temp2['동']

df = pd.merge(temp1, temp2, how = 'outer')

df.drop(['법정동명', '동'], axis = 1, inplace = True)
idx = list(range(83, 110))
df.drop(index = idx, inplace = True)
df.dropna(inplace = True)

df.to_csv('data/Code Table.csv', index = False)