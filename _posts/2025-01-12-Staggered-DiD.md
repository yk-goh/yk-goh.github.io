---
layout: single
title: "Staggered DiD"
categories: Causal Inference
sidebar: true
use_math: true
---

```python
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import statsmodels.formula.api as smf
import re 
from linearmodels.panel import PanelOLS
pd.set_option('display.max_column', None)
```
# 1. 신용도 변화 테이블 확인
```python
df = pd.read_csv('../250111_df_credit_low.csv', index_col=0)
```

```python
df.iloc[:, 1:].head(3)
```

<table>
  <thead>
    <tr>
      <th> </th>
      <th>ym</th>
      <th>spread_score</th>
      <th>exp_default</th>
      <th>future_default</th>
      <th>y_pred_sm</th>
      <th>credit_low</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>201906</td>
      <td>0.627814</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.488615</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>201906</td>
      <td>0.633201</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.492191</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>201906</td>
      <td>0.651423</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.984594</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>


신용도 하락 시점의 분포를 확인한다
```python
df_min_ym = df[df['credit_low']==1].groupby('join_sn').min('ym')
print(df_min_ym.shape)
```

```
(56175, 6)
```

```python
df_min_ym.columns
```

```
Index(['ym', 'spread_score', 'exp_default', 'future_default', 'y_pred_sm',
       'credit_low'],
      dtype='object')
```


```python
df_min_ym['ym'].value_counts()
# 총 15개 시점
# 201906부터 있음
```

```
ym
201912    5598
201909    5297
202006    5200
202009    4658
202003    4624
202112    3551
202203    3394
202012    3390
202103    3261
202206    3027
202209    2946
202303    2874
202212    2836
202106    2802
202109    2717
Name: count, dtype: int64
```

```python
len_yms = len(df_min_ym['ym'].unique())
pd.to_datetime(df_min_ym['ym'], format='%Y%m').hist(bins=len_yms)
```

이미지 자리
![](/images/capstonePRJ/staggered_did/ym_hist.png)


# 2. 전처리
## 2.1 보유여부, 보유개수 테이블 만들기
```python
df_cnt_pre = pd.read_csv('../../INSURANCE_CNT_CONTRACT_I_VER3.csv')
```

```python
# 생명보험(종신+정기)
df_cnt_pre['life'] = df_cnt_pre['i_cnt_whole']+df_cnt_pre['i_cnt_term']
# 건강보험(질병+암)
df_cnt_pre['disease'] = df_cnt_pre['i_cnt_disease'] + df_cnt_pre['i_cnt_cancer']
# 저축성보험
df_cnt_pre['saving'] = df_cnt_pre['i_cnt_pen_sv'] + df_cnt_pre['i_cnt_pen'] + df_cnt_pre['i_cnt_sv'] + df_cnt_pre['i_cnt_ed']

df_cnt_pre.iloc[:, 1:].head(3)
```

<table>
  <thead>
    <tr>
      <th> </th>
      <th>ym</th>
      <th>i_cnt_all</th>
      <th>...</th>
      <th>life</th>
      <th>disease</th>
      <th>saving</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>201803</td>
      <td>7</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>201806</td>
      <td>7</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>201809</td>
      <td>6</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>


```python
target_cols = ['join_sn', 'ym', 'i_cnt_all','life', 'disease', 'i_cnt_health', 'saving']
df_cnt = df_cnt_pre[target_cols]
df_cnt.rename(columns={'i_cnt_all': 'all', 'i_cnt_health': 'hurt'}, inplace=True)
# 보유개수 테이블
df_cnt.iloc[:, 1:].head(3)
```
<table>
  <thead>
    <tr>
      <th> </th>
      <th>ym</th>
      <th>all</th>
      <th>life</th>
      <th>disease</th>
      <th>hurt</th>
      <th>saving</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>201803</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>201806</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>201809</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>


## 2.2 신용도 변화 테이블과 조인하기
```python
df_credit_cnt = df.merge(df_cnt, how='left', on=['join_sn', 'ym'])

print("보유개수 df_credit_cnt 테이블", df_credit_cnt.shape)
print(df_credit_cnt.columns)
```

```
보유개수 df_credit_cnt 테이블 (14540106, 12)
Index(['join_sn', 'ym', 'spread_score', 'exp_default', 'future_default',
       'y_pred_sm', 'credit_low', 'all', 'life', 'disease', 'hurt', 'saving'],
      dtype='object')
```

```python
# 대출은 있으나 보험이 하나도 없는 차주-시점 확인하기
# print(df_credit_bin.isna().sum())
# print(df_credit_cnt.isna().sum())

# 대출은 있으나 보험이 하나도 없는 차주-시점은 보험보유 '없음' 보험개수 "0개"로 바꾼다
df_credit_cnt = df_credit_cnt.fillna(0)
print(df_credit_cnt.isna().sum())
```

```
join_sn           0
ym                0
spread_score      0
exp_default       0
future_default    0
y_pred_sm         0
credit_low        0
all               0
life              0
disease           0
hurt              0
saving            0
dtype: int64
```

```python
# 보유여부 테이블
df_credit_bin = df_credit_cnt.copy()
df_credit_bin.rename(columns = {'all': 'has_all', 'life': 'has_life', 'disease': 'has_disease', 'hurt': 'has_hurt', 'saving': 'has_saving'}, inplace=True)

cols_to_transform = ['has_all', 'has_life', 'has_disease', 'has_hurt', 'has_saving']
df_credit_bin[cols_to_transform] = (df_credit_bin[cols_to_transform] > 0).astype(int)

df_credit_bin.iloc[:, 1:].head(3)
```
<table>
  <thead>
    <tr>
      <th> </th>
      <th>ym</th>
      <th>spread_score</th>
      <th>exp_default</th>
      <th>future_default</th>
      <th>y_pred_sm</th>
      <th>credit_low</th>
      <th>has_all</th>
      <th>has_life</th>
      <th>has_disease</th>
      <th>has_hurt</th>
      <th>has_saving</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>201906</td>
      <td>0.627814</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.488615</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>201906</td>
      <td>0.633201</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.492191</td>
      <td>0.0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>201906</td>
      <td>0.651423</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.984594</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>



# 3.1 보유여부 staggered DiD


```python
target_cols = ['has_all', 'has_life', 'has_disease', 'has_hurt', 'has_saving']

df_credit_bin_temp = df_credit_bin.set_index(['join_sn', 'ym']) # Entity FE, Time FE를 위해 인덱스 지정 

for col in target_cols:
    # 모형 만들고 적합 **************************************************
    formula = f"{col} ~ credit_low + EntityEffects + TimeEffects"
    model = PanelOLS.from_formula(formula, data=df_credit_bin_temp)
    result = model.fit(cov_type='clustered', cluster_entity=True)

    print(result.summary)
```

```

                          PanelOLS Estimation Summary                           
================================================================================
Dep. Variable:                has_all   R-squared:                        0.0004
Estimator:                   PanelOLS   R-squared (Between):             -0.0012
No. Observations:            14540106   R-squared (Within):            1.045e-05
Date:                Sun, Jan 12 2025   R-squared (Overall):             -0.0014
Time:                        21:56:39   Log-likelihood                  9.89e+06
Cov. Estimator:             Clustered                                           
                                        F-statistic:                      4733.3
Entities:                     1204332   P-value                           0.0000
Avg Obs:                       12.073   Distribution:              F(1,13335758)
Min Obs:                       1.0000                                           
Max Obs:                       16.000   F-statistic (robust):             593.58
                                        P-value                           0.0000
Time periods:                      16   Distribution:              F(1,13335758)
Avg Obs:                    9.088e+05                                           
Min Obs:                    8.835e+05                                           
Max Obs:                    9.248e+05                                           
                                                                                
                             Parameter Estimates                              
==============================================================================
            Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
------------------------------------------------------------------------------
credit_low    -0.0241     0.0010    -24.363     0.0000     -0.0260     -0.0222
==============================================================================

F-test for Poolability: 47.769
P-value: 0.0000
Distribution: F(1204346,13335758)

Included effects: Entity, Time






                          PanelOLS Estimation Summary                           
================================================================================
Dep. Variable:               has_life   R-squared:                        0.0002
Estimator:                   PanelOLS   R-squared (Between):             -0.0011
No. Observations:            14540106   R-squared (Within):            1.238e-05
Date:                Sun, Jan 12 2025   R-squared (Overall):             -0.0012
Time:                        21:58:55   Log-likelihood                 6.115e+06
Cov. Estimator:             Clustered                                           
                                        F-statistic:                      2170.2
Entities:                     1204332   P-value                           0.0000
Avg Obs:                       12.073   Distribution:              F(1,13335758)
Min Obs:                       1.0000                                           
Max Obs:                       16.000   F-statistic (robust):             311.04
                                        P-value                           0.0000
Time periods:                      16   Distribution:              F(1,13335758)
Avg Obs:                    9.088e+05                                           
Min Obs:                    8.835e+05                                           
Max Obs:                    9.248e+05                                           
                                                                                
                             Parameter Estimates                              
==============================================================================
            Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
------------------------------------------------------------------------------
credit_low    -0.0212     0.0012    -17.636     0.0000     -0.0235     -0.0188
==============================================================================

F-test for Poolability: 72.745
P-value: 0.0000
Distribution: F(1204346,13335758)

Included effects: Entity, Time






                          PanelOLS Estimation Summary                           
================================================================================
Dep. Variable:            has_disease   R-squared:                        0.0003
Estimator:                   PanelOLS   R-squared (Between):             -0.0017
No. Observations:            14540106   R-squared (Within):              -0.0003
Date:                Sun, Jan 12 2025   R-squared (Overall):             -0.0019
Time:                        22:01:05   Log-likelihood                 5.264e+06
Cov. Estimator:             Clustered                                           
                                        F-statistic:                      4175.1
Entities:                     1204332   P-value                           0.0000
Avg Obs:                       12.073   Distribution:              F(1,13335758)
Min Obs:                       1.0000                                           
Max Obs:                       16.000   F-statistic (robust):             548.28
                                        P-value                           0.0000
Time periods:                      16   Distribution:              F(1,13335758)
Avg Obs:                    9.088e+05                                           
Min Obs:                    8.835e+05                                           
Max Obs:                    9.248e+05                                           
                                                                                
                             Parameter Estimates                              
==============================================================================
            Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
------------------------------------------------------------------------------
credit_low    -0.0311     0.0013    -23.415     0.0000     -0.0337     -0.0285
==============================================================================

F-test for Poolability: 81.275
P-value: 0.0000
Distribution: F(1204346,13335758)

Included effects: Entity, Time






                          PanelOLS Estimation Summary                           
================================================================================
Dep. Variable:               has_hurt   R-squared:                        0.0002
Estimator:                   PanelOLS   R-squared (Between):             -0.0016
No. Observations:            14540106   R-squared (Within):              -0.0004
Date:                Sun, Jan 12 2025   R-squared (Overall):             -0.0018
Time:                        22:03:17   Log-likelihood                 2.041e+06
Cov. Estimator:             Clustered                                           
                                        F-statistic:                      2521.1
Entities:                     1204332   P-value                           0.0000
Avg Obs:                       12.073   Distribution:              F(1,13335758)
Min Obs:                       1.0000                                           
Max Obs:                       16.000   F-statistic (robust):             410.83
                                        P-value                           0.0000
Time periods:                      16   Distribution:              F(1,13335758)
Avg Obs:                    9.088e+05                                           
Min Obs:                    8.835e+05                                           
Max Obs:                    9.248e+05                                           
                                                                                
                             Parameter Estimates                              
==============================================================================
            Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
------------------------------------------------------------------------------
credit_low    -0.0302     0.0015    -20.269     0.0000     -0.0331     -0.0273
==============================================================================

F-test for Poolability: 50.871
P-value: 0.0000
Distribution: F(1204346,13335758)

Included effects: Entity, Time






                          PanelOLS Estimation Summary                           
================================================================================
Dep. Variable:             has_saving   R-squared:                     9.229e-06
Estimator:                   PanelOLS   R-squared (Between):             -0.0001
No. Observations:            14540106   R-squared (Within):            9.295e-05
Date:                Sun, Jan 12 2025   R-squared (Overall):             -0.0001
Time:                        22:05:30   Log-likelihood                 8.355e+06
Cov. Estimator:             Clustered                                           
                                        F-statistic:                      123.08
Entities:                     1204332   P-value                           0.0000
Avg Obs:                       12.073   Distribution:              F(1,13335758)
Min Obs:                       1.0000                                           
Max Obs:                       16.000   F-statistic (robust):             24.824
                                        P-value                           0.0000
Time periods:                      16   Distribution:              F(1,13335758)
Avg Obs:                    9.088e+05                                           
Min Obs:                    8.835e+05                                           
Max Obs:                    9.248e+05                                           
                                                                                
                             Parameter Estimates                              
==============================================================================
            Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
------------------------------------------------------------------------------
credit_low    -0.0043     0.0009    -4.9823     0.0000     -0.0060     -0.0026
==============================================================================

F-test for Poolability: 83.133
P-value: 0.0000
Distribution: F(1204346,13335758)

Included effects: Entity, Time
```




# 3.2 보유개수 Staggered DiD
```python
target_cols = ['all', 'life', 'disease', 'hurt', 'saving']

df_credit_cnt_temp = df_credit_cnt.set_index(['join_sn', 'ym']) # Entity FE, Time FE를 위해 인덱스 지정 

for col in target_cols:
    # 1% 이상치 처리 *************************************
    threshold = df_credit_cnt_temp[col].quantile(0.99)
    df_credit_cnt_temp.loc[df_credit_cnt_temp[col] > threshold, col] = None # =threshold로 하면 cap 지정 방식으로 이상치 처리
    # 모형 만들고 적합 **************************************************
    formula = f"{col} ~ credit_low + EntityEffects + TimeEffects"
    model = PanelOLS.from_formula(formula, data=df_credit_cnt_temp)
    result = model.fit(cov_type='clustered', cluster_entity=True)

    print(result.summary)
```

```



                          PanelOLS Estimation Summary                           
================================================================================
Dep. Variable:                    all   R-squared:                        0.0008
Estimator:                   PanelOLS   R-squared (Between):             -0.0022
No. Observations:            14402750   R-squared (Within):              -0.0016
Date:                Sun, Jan 12 2025   R-squared (Overall):             -0.0026
Time:                        22:24:16   Log-likelihood                -1.763e+07
Cov. Estimator:             Clustered                                           
                                        F-statistic:                   1.043e+04
Entities:                     1199619   P-value                           0.0000
Avg Obs:                       12.006   Distribution:              F(1,13203115)
Min Obs:                       1.0000                                           
Max Obs:                       16.000   F-statistic (robust):             1129.5
                                        P-value                           0.0000
Time periods:                      16   Distribution:              F(1,13203115)
Avg Obs:                    9.002e+05                                           
Min Obs:                    8.771e+05                                           
Max Obs:                    9.153e+05                                           
                                                                                
                             Parameter Estimates                              
==============================================================================
            Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
------------------------------------------------------------------------------
credit_low    -0.2422     0.0072    -33.608     0.0000     -0.2563     -0.2280
==============================================================================

F-test for Poolability: 115.08
P-value: 0.0000
Distribution: F(1199633,13203115)

Included effects: Entity, Time








                          PanelOLS Estimation Summary                           
================================================================================
Dep. Variable:                   life   R-squared:                        0.0002
Estimator:                   PanelOLS   R-squared (Between):             -0.0011
No. Observations:            14424732   R-squared (Within):           -1.044e-05
Date:                Sun, Jan 12 2025   R-squared (Overall):             -0.0012
Time:                        22:26:54   Log-likelihood                 1.149e+06
Cov. Estimator:             Clustered                                           
                                        F-statistic:                      2343.6
Entities:                     1199811   P-value                           0.0000
Avg Obs:                       12.023   Distribution:              F(1,13224905)
Min Obs:                       1.0000                                           
Max Obs:                       16.000   F-statistic (robust):             324.73
                                        P-value                           0.0000
Time periods:                      16   Distribution:              F(1,13224905)
Avg Obs:                    9.015e+05                                           
Min Obs:                    8.773e+05                                           
Max Obs:                    9.171e+05                                           
                                                                                
                             Parameter Estimates                              
==============================================================================
            Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
------------------------------------------------------------------------------
credit_low    -0.0311     0.0017    -18.020     0.0000     -0.0345     -0.0277
==============================================================================

F-test for Poolability: 74.838
P-value: 0.0000
Distribution: F(1199825,13224905)

Included effects: Entity, Time










                          PanelOLS Estimation Summary                           
================================================================================
Dep. Variable:                disease   R-squared:                        0.0003
Estimator:                   PanelOLS   R-squared (Between):             -0.0017
No. Observations:            14426200   R-squared (Within):              -0.0007
Date:                Sun, Jan 12 2025   R-squared (Overall):             -0.0019
Time:                        22:29:29   Log-likelihood                -7.998e+06
Cov. Estimator:             Clustered                                           
                                        F-statistic:                      4066.4
Entities:                     1200824   P-value                           0.0000
Avg Obs:                       12.014   Distribution:              F(1,13225360)
Min Obs:                       1.0000                                           
Max Obs:                       16.000   F-statistic (robust):             436.82
                                        P-value                           0.0000
Time periods:                      16   Distribution:              F(1,13225360)
Avg Obs:                    9.016e+05                                           
Min Obs:                    8.785e+05                                           
Max Obs:                    9.169e+05                                           
                                                                                
                             Parameter Estimates                              
==============================================================================
            Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
------------------------------------------------------------------------------
credit_low    -0.0774     0.0037    -20.900     0.0000     -0.0847     -0.0701
==============================================================================

F-test for Poolability: 100.60
P-value: 0.0000
Distribution: F(1200838,13225360)

Included effects: Entity, Time









                          PanelOLS Estimation Summary                           
================================================================================
Dep. Variable:                   hurt   R-squared:                        0.0003
Estimator:                   PanelOLS   R-squared (Between):             -0.0020
No. Observations:            14404732   R-squared (Within):              -0.0008
Date:                Sun, Jan 12 2025   R-squared (Overall):             -0.0022
Time:                        22:32:01   Log-likelihood                 -6.45e+06
Cov. Estimator:             Clustered                                           
                                        F-statistic:                      4064.1
Entities:                     1200478   P-value                           0.0000
Avg Obs:                       11.999   Distribution:              F(1,13204238)
Min Obs:                       1.0000                                           
Max Obs:                       16.000   F-statistic (robust):             653.75
                                        P-value                           0.0000
Time periods:                      16   Distribution:              F(1,13204238)
Avg Obs:                    9.003e+05                                           
Min Obs:                    8.782e+05                                           
Max Obs:                    9.149e+05                                           
                                                                                
                             Parameter Estimates                              
==============================================================================
            Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
------------------------------------------------------------------------------
credit_low    -0.0695     0.0027    -25.569     0.0000     -0.0748     -0.0642
==============================================================================

F-test for Poolability: 57.547
P-value: 0.0000
Distribution: F(1200492,13204238)

Included effects: Entity, Time






                          PanelOLS Estimation Summary                           
================================================================================
Dep. Variable:                 saving   R-squared:                     1.359e-06
Estimator:                   PanelOLS   R-squared (Between):          -3.569e-05
No. Observations:            14396914   R-squared (Within):             3.49e-05
Date:                Sun, Jan 12 2025   R-squared (Overall):           -3.66e-05
Time:                        22:34:31   Log-likelihood                  3.19e+06
Cov. Estimator:             Clustered                                           
                                        F-statistic:                      17.945
Entities:                     1197017   P-value                           0.0000
Avg Obs:                       12.027   Distribution:              F(1,13199881)
Min Obs:                       1.0000                                           
Max Obs:                       16.000   F-statistic (robust):             4.0371
                                        P-value                           0.0445
Time periods:                      16   Distribution:              F(1,13199881)
Avg Obs:                    8.998e+05                                           
Min Obs:                    8.728e+05                                           
Max Obs:                    9.169e+05                                           
                                                                                
                             Parameter Estimates                              
==============================================================================
            Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
------------------------------------------------------------------------------
credit_low    -0.0024     0.0012    -2.0093     0.0445     -0.0046  -5.772e-05
==============================================================================

F-test for Poolability: 92.016
P-value: 0.0000
Distribution: F(1197031,13199881)

Included effects: Entity, Time
```




### 이상치 처리 방식이 truncate가 아닌 cap인 경우
```python
target_cols = ['life', 'disease', 'hurt']

df_credit_cnt_temp = df_credit_cnt.set_index(['join_sn', 'ym']) # Entity FE, Time FE를 위해 인덱스 지정 

for col in target_cols:
    # 1% 이상치 처리 *************************************
    threshold = df_credit_cnt_temp[col].quantile(0.99)
    df_credit_cnt_temp.loc[df_credit_cnt_temp[col] > threshold, col] = threshold # =threshold로 하면 cap 지정 방식으로 이상치 처리
    # 모형 만들고 적합 **************************************************
    formula = f"{col} ~ credit_low + EntityEffects + TimeEffects"
    model = PanelOLS.from_formula(formula, data=df_credit_cnt_temp)
    result = model.fit(cov_type='clustered', cluster_entity=True)

    print(result.summary)
```

```



                          PanelOLS Estimation Summary                           
================================================================================
Dep. Variable:                   life   R-squared:                        0.0002
Estimator:                   PanelOLS   R-squared (Between):             -0.0011
No. Observations:            14540106   R-squared (Within):           -5.923e-06
Date:                Tue, Jan 14 2025   R-squared (Overall):             -0.0011
Time:                        15:41:18   Log-likelihood                 8.421e+05
Cov. Estimator:             Clustered                                           
                                        F-statistic:                      2488.1
Entities:                     1204332   P-value                           0.0000
Avg Obs:                       12.073   Distribution:              F(1,13335758)
Min Obs:                       1.0000                                           
Max Obs:                       16.000   F-statistic (robust):             326.37
                                        P-value                           0.0000
Time periods:                      16   Distribution:              F(1,13335758)
Avg Obs:                    9.088e+05                                           
Min Obs:                    8.835e+05                                           
Max Obs:                    9.248e+05                                           
                                                                                
                             Parameter Estimates                              
==============================================================================
            Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
------------------------------------------------------------------------------
credit_low    -0.0326     0.0018    -18.066     0.0000     -0.0361     -0.0290
==============================================================================

F-test for Poolability: 82.828
P-value: 0.0000
Distribution: F(1204346,13335758)

Included effects: Entity, Time









                          PanelOLS Estimation Summary                           
================================================================================
Dep. Variable:                disease   R-squared:                        0.0003
Estimator:                   PanelOLS   R-squared (Between):             -0.0017
No. Observations:            14540106   R-squared (Within):              -0.0007
Date:                Tue, Jan 14 2025   R-squared (Overall):             -0.0019
Time:                        15:43:23   Log-likelihood                -8.338e+06
Cov. Estimator:             Clustered                                           
                                        F-statistic:                      4124.0
Entities:                     1204332   P-value                           0.0000
Avg Obs:                       12.073   Distribution:              F(1,13335758)
Min Obs:                       1.0000                                           
Max Obs:                       16.000   F-statistic (robust):             422.69
                                        P-value                           0.0000
Time periods:                      16   Distribution:              F(1,13335758)
Avg Obs:                    9.088e+05                                           
Min Obs:                    8.835e+05                                           
Max Obs:                    9.248e+05                                           
                                                                                
                             Parameter Estimates                              
==============================================================================
            Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
------------------------------------------------------------------------------
credit_low    -0.0788     0.0038    -20.559     0.0000     -0.0863     -0.0713
==============================================================================

F-test for Poolability: 111.98
P-value: 0.0000
Distribution: F(1204346,13335758)

Included effects: Entity, Time








                          PanelOLS Estimation Summary                           
================================================================================
Dep. Variable:                   hurt   R-squared:                        0.0003
Estimator:                   PanelOLS   R-squared (Between):             -0.0020
No. Observations:            14540106   R-squared (Within):              -0.0008
Date:                Tue, Jan 14 2025   R-squared (Overall):             -0.0021
Time:                        15:45:27   Log-likelihood                -6.642e+06
Cov. Estimator:             Clustered                                           
                                        F-statistic:                      4090.1
Entities:                     1204332   P-value                           0.0000
Avg Obs:                       12.073   Distribution:              F(1,13335758)
Min Obs:                       1.0000                                           
Max Obs:                       16.000   F-statistic (robust):             637.83
                                        P-value                           0.0000
Time periods:                      16   Distribution:              F(1,13335758)
Avg Obs:                    9.088e+05                                           
Min Obs:                    8.835e+05                                           
Max Obs:                    9.248e+05                                           
                                                                                
                             Parameter Estimates                              
==============================================================================
            Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
------------------------------------------------------------------------------
credit_low    -0.0699     0.0028    -25.255     0.0000     -0.0753     -0.0644
==============================================================================

F-test for Poolability: 63.104
P-value: 0.0000
Distribution: F(1204346,13335758)

Included effects: Entity, Time
```


