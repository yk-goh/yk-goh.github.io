---
layout: single
title: "Conventional Diff-in-Diff"
categories: Causal_Inference
sidebar: true
use_math: true
---

```python
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' %x)
np.set_printoptions(suppress=True)
from matplotlib import font_manager, rc
```

```python
df = pd.read_csv('241228_template_did.csv', index_col=0)

df.dropna(inplace=True)
print(df.shape)
df.drop_duplicates(inplace=True, subset=['join_sn', 'did_credit_change'])
print(df.shape)
```


    (9567841, 6)
    (615820, 6)

## 1. 전처리

```python
df = df[['join_sn', 'did_credit_change']]
df['hl'] = df['did_credit_change'].apply(lambda x: 1 if x == 'high_low' else 0)

df.head()

df['did_credit_change'].value_counts()
```

```
high_high    571870
high_low      43950
Name: did_credit_change, dtype: int64
```
표본은 고신용 유지 차주 약 57만 명, 신용도 하락 차주 약 4만 명으로 구성되어 있다. 


pre 시점과 post 시점에 해당하는 컬럼을 생성한다. 한편 `hl`은 처치를 나타내는 변수로서 통제집단(고신용 유지)이면 `hl==0`, 처치집단(신용도 하락)이면 `hl==1`이다. 
```python
df_credit_pre = df[['join_sn', 'hl']]

df_201912 = df_credit_pre.assign(ym=201912)
df_202112 = df_credit_pre.assign(ym=202112)
df_credit = pd.concat([df_201912, df_202112], ignore_index=True)
```

차주별 보험의 보유여부 테이블과 보유개수 테이블을 만든다.
```python
df_cnt = pd.read_csv('../INSURANCE_CNT_CONTRACT_I_VER3.csv')
```

```python
# 생명보험(종신+정기)
df_cnt['life'] = df_cnt['i_cnt_whole']+df_cnt['i_cnt_term']
# 건강보험(질병+암)
df_cnt['disease'] = df_cnt['i_cnt_disease'] + df_cnt['i_cnt_cancer']
# 저축성보험
df_cnt['saving'] = df_cnt['i_cnt_pen_sv'] + df_cnt['i_cnt_pen'] + df_cnt['i_cnt_sv'] + df_cnt['i_cnt_ed']
# 화재보험
df_cnt['fire'] = df_cnt['i_cnt_fire'] + df_cnt['i_cnt_liability'] + df_cnt['i_cnt_fire_2']
```

행(row) 단위로 작동하는 람다식보다 벡터연산이 훨씬 빠르다. 
```python
df_cnt_binary = df_cnt.copy()
df_cnt_binary = df_cnt_binary.set_index(['join_sn', 'ym'])
# df_cnt_binary = df_cnt_binary.set_index(['join_sn', 'ym']).applymap(lambda x: 1 if x > 0 else 0).reset_index()
df_cnt_binary = (df_cnt_binary > 0).astype(int).reset_index()
df_cnt_binary.head(2)
```
<table border="1">
  <thead>
    <tr>
      <th>join_sn</th>
      <th>ym</th>
      <th>i_cnt_all</th>
      <th>...</th>
      <th>life</th>
      <th>disease</th>
      <th>saving</th>
      <th>fire</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>101</td>
      <td>201803</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>101</td>
      <td>201806</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>


```python
df_cnt_binary.isna().sum()
```

```
join_sn            0
ym                 0
i_cnt_all          0
i_cnt_whole        0
i_cnt_term         0
i_cnt_disease      0
i_cnt_health       0
i_cnt_cancer       0
i_cnt_nursing      0
i_cnt_child        0
i_cnt_dent         0
i_cnt_pen_sv       0
i_cnt_pen          0
i_cnt_sv           0
i_cnt_ed           0
i_cnt_drive        0
i_cnt_tour         0
i_cnt_golf         0
i_cnt_med          0
i_cnt_car          0
i_cnt_fire         0
i_cnt_liability    0
i_cnt_fire_2       0
i_cnt_else         0
life               0
disease            0
saving             0
fire               0
dtype: int64
```

```python
df_cnt_binary.shape
```

```
(53373350, 28)
```

컬럼명을 재설정한다. i_cnt_\*가 보유개수를 의미한다면 보유여부를 나타내는 컬럼은 i_has_\*이다.
```python
colnames = ['join_sn', 'ym', 'i_has_all', 'i_has_whole', 'i_has_term',
       'i_has_disease', 'i_has_health', 'i_has_cancer', 'i_has_nursing',
       'i_has_child', 'i_has_dent', 'i_has_pen_sv', 'i_has_pen', 'i_has_sv',
       'i_has_ed', 'i_has_drive', 'i_has_tour', 'i_has_golf', 'i_has_med',
       'i_has_car', 'i_has_fire', 'i_has_liability', 'i_has_fire_2',
       'i_has_else', 'has_life', 'has_disease', 'has_saving', 'has_fire']
df_cnt_binary.columns = colnames
df_cnt_binary.columns
```

```
Out[10]:

Index(['join_sn', 'ym', 'i_has_all', 'i_has_whole', 'i_has_term',
       'i_has_disease', 'i_has_health', 'i_has_cancer', 'i_has_nursing',
       'i_has_child', 'i_has_dent', 'i_has_pen_sv', 'i_has_pen', 'i_has_sv',
       'i_has_ed', 'i_has_drive', 'i_has_tour', 'i_has_golf', 'i_has_med',
       'i_has_car', 'i_has_fire', 'i_has_liability', 'i_has_fire_2',
       'i_has_else', 'has_life', 'has_disease', 'has_saving', 'has_fire'],
      dtype='object')
```

pre 시점에 해당하는 2019년 12월과 post 시점에 해당하는 2021년 12월의 보험 데이터만 필터링 한다.
```python
df_cnt_filtered = df_cnt[(df_cnt['ym']==201912) | (df_cnt['ym']==202112)]
df_cnt_binary_filtered = df_cnt_binary[(df_cnt_binary['ym']==201912) | (df_cnt_binary['ym']==202112)]

df_cnt_filtered.head(2)
```
<table border="1">
  <thead>
    <tr>
      <th>join_sn</th>
      <th>ym</th>
      <th>i_cnt_all</th>
      <th>...</th>
      <th>life</th>
      <th>disease</th>
      <th>saving</th>
      <th>fire</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>101</td>
      <td>201912</td>
      <td>6</td>
      <td>...</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>101</td>
      <td>202112</td>
      <td>7</td>
      <td>...</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>


신용 테이블에 보험 보유여부와 보유개수 테이블을 머지한다. 패널데이터이므로 단위(차주번호)와 시점을 모두 key로 사용해야 한다. 
```python
df_credit_cnt = df_credit.merge(df_cnt_filtered, how='left', on=['join_sn', 'ym'])
# df_credit_cnt['post'] = df_credit_cnt['ym'].apply(lambda x: 1 if x==202112 else 0)
df_credit_cnt['post'] = (df_credit_cnt['ym'] == 202112).astype(int)

df_credit_bin = df_credit.merge(df_cnt_binary_filtered, how='left', on=['join_sn', 'ym'])
# df_credit_bin['post'] = df_credit_bin['ym'].apply(lambda x: 1 if x==202112 else 0)
df_credit_bin['post'] = (df_credit_bin['ym'] == 202112).astype(int)

print("보유여부 df_credit_bin 테이블: ", df_credit_bin.shape, "보유개수 df_credit_cnt 테이블", df_credit_cnt.shape)
print(df_credit_cnt.columns)
print(df_credit_bin.columns)
```

```
보유여부 df_credit_bin 테이블:  (1231640, 30) 보유개수 df_credit_cnt 테이블 (1231640, 30)
Index(['join_sn', 'hl', 'ym', 'i_cnt_all', 'i_cnt_whole', 'i_cnt_term',
       'i_cnt_disease', 'i_cnt_health', 'i_cnt_cancer', 'i_cnt_nursing',
       'i_cnt_child', 'i_cnt_dent', 'i_cnt_pen_sv', 'i_cnt_pen', 'i_cnt_sv',
       'i_cnt_ed', 'i_cnt_drive', 'i_cnt_tour', 'i_cnt_golf', 'i_cnt_med',
       'i_cnt_car', 'i_cnt_fire', 'i_cnt_liability', 'i_cnt_fire_2',
       'i_cnt_else', 'life', 'disease', 'saving', 'fire', 'post'],
      dtype='object')
Index(['join_sn', 'hl', 'ym', 'i_has_all', 'i_has_whole', 'i_has_term',
       'i_has_disease', 'i_has_health', 'i_has_cancer', 'i_has_nursing',
       'i_has_child', 'i_has_dent', 'i_has_pen_sv', 'i_has_pen', 'i_has_sv',
       'i_has_ed', 'i_has_drive', 'i_has_tour', 'i_has_golf', 'i_has_med',
       'i_has_car', 'i_has_fire', 'i_has_liability', 'i_has_fire_2',
       'i_has_else', 'has_life', 'has_disease', 'has_saving', 'has_fire',
       'post'],
      dtype='object')
```
CreDB의 샘플링 기준에 의하여 보험정보가 있거나 대출정보가 있으면 표본에 포함된다. 따라서 어떤 차주는 대출 혹은 보험 중 하나가 없을 수 있다. 위의 코드에서 신용 테이블을 left, 보험 테이블을 right로 하여 left join 하였기 때문에 대출을 가지고 있으나 보험 정보가 존재하지 않는 차주가 데이터프레임에 있다. 

```python
# 대출은 있으나 보험이 하나도 없는 차주-시점
print(df_credit_bin.isna().sum())
print(df_credit_cnt.isna().sum())
```

```
join_sn                0
hl                     0
ym                     0
i_has_all          94950
...
has_life           94950
has_disease        94950
has_saving         94950
has_fire           94950
post                   0
dtype: int64
join_sn                0
hl                     0
ym                     0
i_cnt_all          94950
...
life               94950
disease            94950
saving             94950
fire               94950
post                   0
dtype: int64
```

대출은 있으나 보험이 하나도 없는 차주-시점은 보험보유 '없음' 보험개수 0개로 바꾼다
```python
df_credit_bin = df_credit_bin.fillna(0)
df_credit_cnt = df_credit_cnt.fillna(0)
# print(df_credit_bin.isna().sum())
# print(df_credit_cnt.isna().sum())
```


```python
print(df_credit_bin[df_credit_bin['ym']==201912]['i_has_all'].value_counts())
print(df_credit_bin[df_credit_bin['ym']==202112]['i_has_all'].value_counts())
print(df_credit_bin[df_credit_bin['post']==0]['i_has_all'].value_counts())
print(df_credit_bin[df_credit_bin['post']==1]['i_has_all'].value_counts())

# print(df_credit_cnt[df_credit_cnt['ym']==201912]['i_cnt_all'].value_counts())
# print(df_credit_cnt[df_credit_cnt['ym']==202112]['i_cnt_all'].value_counts())
# print(df_credit_cnt[df_credit_cnt['post']==0]['i_cnt_all'].value_counts())
# print(df_credit_cnt[df_credit_cnt['post']==1]['i_cnt_all'].value_counts())
```

```
1.000    564298
0.000     51522
Name: i_has_all, dtype: int64
1.000    572392
0.000     43428
Name: i_has_all, dtype: int64
1.000    564298
0.000     51522
Name: i_has_all, dtype: int64
1.000    572392
0.000     43428
Name: i_has_all, dtype: int64
```

보유여부 확인 시 사용할 컬럼, 보유개수 확인 시 사용할 컬럼을 정의한다.

```python
cnt_columns = ['i_cnt_all', 'i_cnt_whole', 'i_cnt_term',
       'i_cnt_disease', 'i_cnt_health', 'i_cnt_cancer', 'i_cnt_nursing',
       'i_cnt_child', 'i_cnt_dent', 'i_cnt_pen_sv', 'i_cnt_pen', 'i_cnt_sv',
       'i_cnt_ed', 'i_cnt_drive', 'i_cnt_tour', 'i_cnt_golf', 'i_cnt_med',
       'i_cnt_car', 'i_cnt_fire', 'i_cnt_liability', 'i_cnt_fire_2',
       'i_cnt_else', 'life', 'disease', 'saving', 'fire']
has_columns = ['i_has_all', 'i_has_whole', 'i_has_term',
       'i_has_disease', 'i_has_health', 'i_has_cancer', 'i_has_nursing',
       'i_has_child', 'i_has_dent', 'i_has_pen_sv', 'i_has_pen', 'i_has_sv',
       'i_has_ed', 'i_has_drive', 'i_has_tour', 'i_has_golf', 'i_has_med',
       'i_has_car', 'i_has_fire', 'i_has_liability', 'i_has_fire_2',
       'i_has_else', 'has_life', 'has_disease', 'has_saving', 'has_fire']
```


## 2. 보유여부 - OLS for DiD

```python
import statsmodels.formula.api as smf

results = []
for col in has_columns: 
    
    did_data = (df_credit_bin
                .groupby(['join_sn', 'post'])
                .agg({col: 'max', 'hl': 'max'})
                .reset_index()
               )

    formula = f'{col} ~ hl*post'
    model = smf.ols(formula, data = did_data).fit(cov_type='cluster', cov_kwds={'groups': did_data['join_sn']})
    results.append({
        'Column': col,
        'Intercept': model.params['Intercept'],
        'Intercept SE': model.bse['Intercept'],
        'D': model.params['hl'],
        'D SE': model.bse['hl'],
        'Post': model.params['post'],
        'Post SE': model.bse['post'],
        'D:Post': model.params['hl:post'],
        'D:Post SE': model.bse['hl:post'],
        'rsquared_adj':model.rsquared_adj,
        'observations': int(model.nobs)
    })

results_df = pd.DataFrame(results)
results_df.transpose()
```
![2시점 이중차분법-보유여부](/images/capstonePRJ/2timepointsDiD/2timepointsDiD_ols1.png)


## 3 보유개수 - OLS for DiD

```python
import statsmodels.formula.api as smf
df_credit_cnt_no = df_credit_cnt.copy()
results = [] 

for col in cnt_columns:
    # threshold
    threshold = df_credit_cnt_no[col].quantile(0.99)
    df_credit_cnt_no.loc[df_credit_cnt_no[col] > threshold, col] = None
    # drop na(outlier)
    did_data = (df_credit_cnt_no
                .dropna()
                .groupby(['join_sn', 'post'])
                .agg({col: 'max', 'hl': 'max'})
                .reset_index()
               )

    formula = f'{col} ~ hl*post'
    model = smf.ols(formula, data = did_data).fit(cov_type='cluster', cov_kwds={'groups': did_data['join_sn']})
    
    results.append({
        'Column': col,
        'Intercept': model.params['Intercept'],
        'Intercept SE': model.bse['Intercept'],
        'D': model.params['hl'],
        'D SE': model.bse['hl'],
        'Post': model.params['post'],
        'Post SE': model.bse['post'],
        'D:Post': model.params['hl:post'],
        'D:Post SE': model.bse['hl:post'],
        'rsquared_adj':model.rsquared_adj,
        'observations': int(model.nobs)
    })

results_df = pd.DataFrame(results)
results_df.transpose()
```
![2시점 이중차분법-보유개수](/images/capstonePRJ/2timepointsDiD/2timepointsDiD_ols2.png)

```python
import statsmodels.formula.api as smf
df_credit_cnt_no = df_credit_cnt.copy()

for col in cnt_columns:
    # threshold
    threshold = df_credit_cnt_no[col].quantile(0.99)
    df_credit_cnt_no.loc[df_credit_cnt_no[col] > threshold, col] = None
    # drop na(outlier)
    did_data = (df_credit_cnt_no
                .dropna()
                .groupby(['join_sn', 'post'])
                .agg({col: 'max', 'hl': 'max'})
                .reset_index()
               )

    formula = f'{col} ~ hl*post'
    model = smf.ols(formula, data = did_data).fit(cov_type='cluster', cov_kwds={'groups': did_data['join_sn']})
    
    print(model.summary())
```

```
OLS Regression Results                            
==============================================================================
Dep. Variable:              i_cnt_all   R-squared:                       0.005
Model:                            OLS   Adj. R-squared:                  0.005
Method:                 Least Squares   F-statistic:                 1.243e+04
Date:                Mon, 06 Jan 2025   Prob (F-statistic):               0.00
Time:                        18:59:55   Log-Likelihood:            -2.9728e+06
No. Observations:             1220129   AIC:                         5.946e+06
Df Residuals:                 1220125   BIC:                         5.946e+06
Df Model:                           3                                         
Covariance Type:              cluster                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept      3.4882      0.004    978.861      0.000       3.481       3.495
hl            -0.0409      0.014     -2.950      0.003      -0.068      -0.014
post           0.4058      0.002    191.414      0.000       0.402       0.410
hl:post       -0.1711      0.010    -17.723      0.000      -0.190      -0.152
==============================================================================
Omnibus:                   226433.166   Durbin-Watson:                   1.173
Prob(Omnibus):                  0.000   Jarque-Bera (JB):           413756.128
Skew:                           1.174   Prob(JB):                         0.00
Kurtosis:                       4.621   Cond. No.                         10.2
==============================================================================

Notes:
[1] Standard Errors are robust to cluster correlation (cluster)

                            OLS Regression Results                            
==============================================================================
Dep. Variable:                   life   R-squared:                       0.000
Model:                            OLS   Adj. R-squared:                  0.000
Method:                 Least Squares   F-statistic:                     184.7
Date:                Mon, 06 Jan 2025   Prob (F-statistic):          1.08e-119
Time:                        19:00:43   Log-Likelihood:            -1.0422e+06
No. Observations:             1143393   AIC:                         2.084e+06
Df Residuals:                 1143389   BIC:                         2.085e+06
Df Model:                           3                                         
Covariance Type:              cluster                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept      0.3084      0.001    376.023      0.000       0.307       0.310
hl            -0.0217      0.003     -7.344      0.000      -0.027      -0.016
post           0.0110      0.001     19.850      0.000       0.010       0.012
hl:post       -0.0214      0.002     -9.034      0.000      -0.026      -0.017
==============================================================================
Omnibus:                   457835.671   Durbin-Watson:                   1.231
Prob(Omnibus):                  0.000   Jarque-Bera (JB):          1657297.853
Skew:                           2.062   Prob(JB):                         0.00
Kurtosis:                       7.216   Cond. No.                         10.2
==============================================================================

Notes:
[1] Standard Errors are robust to cluster correlation (cluster)
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                disease   R-squared:                       0.002
Model:                            OLS   Adj. R-squared:                  0.002
Method:                 Least Squares   F-statistic:                     3527.
Date:                Mon, 06 Jan 2025   Prob (F-statistic):               0.00
Time:                        19:00:45   Log-Likelihood:            -1.8462e+06
No. Observations:             1142909   AIC:                         3.692e+06
Df Residuals:                 1142905   BIC:                         3.692e+06
Df Model:                           3                                         
Covariance Type:              cluster                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept      1.0173      0.002    633.717      0.000       1.014       1.020
hl             0.1310      0.007     20.059      0.000       0.118       0.144
post           0.1097      0.001    100.266      0.000       0.108       0.112
hl:post       -0.0478      0.005     -9.351      0.000      -0.058      -0.038
==============================================================================
Omnibus:                   280553.346   Durbin-Watson:                   1.223
Prob(Omnibus):                  0.000   Jarque-Bera (JB):           628360.769
Skew:                           1.410   Prob(JB):                         0.00
Kurtosis:                       5.290   Cond. No.                         10.2
==============================================================================

Notes:
[1] Standard Errors are robust to cluster correlation (cluster)

```






