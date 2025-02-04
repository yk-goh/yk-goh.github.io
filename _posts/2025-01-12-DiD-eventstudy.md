---
layout: single
title: "Eventstudy plot: validating and visualizing parallel trend assumption"
categories: Causal_Inference
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

# 1. DiD 템플릿 가져오기
```python
df = pd.read_csv('241228_template_did.csv', index_col=0)

df.dropna(inplace=True)
print(df.shape)
df.drop_duplicates(inplace=True, subset=['join_sn', 'did_credit_change'])
print(df.shape)

df = df[['join_sn', 'did_credit_change']]
df['hl'] = (df['did_credit_change'] == 'high_low').astype(int)

df['did_credit_change'].value_counts()
```

```
(9567841, 6)
(615820, 6)


did_credit_change
high_high    571870
high_low      43950
Name: count, dtype: int64
```

```python
df_credit_pre = df[['join_sn', 'hl']]

df_201812 = df_credit_pre.assign(ym=201812)
df_201903 = df_credit_pre.assign(ym=201903)
df_201906 = df_credit_pre.assign(ym=201906)
df_201909 = df_credit_pre.assign(ym=201909)
df_201912 = df_credit_pre.assign(ym=201912)
df_202003 = df_credit_pre.assign(ym=202003)
df_202006 = df_credit_pre.assign(ym=202006)
df_202009 = df_credit_pre.assign(ym=202009)
df_202012 = df_credit_pre.assign(ym=202012)
df_202103 = df_credit_pre.assign(ym=202103)
df_202106 = df_credit_pre.assign(ym=202106)
df_202109 = df_credit_pre.assign(ym=202109)
df_202112 = df_credit_pre.assign(ym=202112)
df_202203 = df_credit_pre.assign(ym=202203)
df_202206 = df_credit_pre.assign(ym=202206)
df_202209 = df_credit_pre.assign(ym=202209)
df_202212 = df_credit_pre.assign(ym=202212)
df_credit = pd.concat([df_201812, df_201903, df_201906, df_201909, df_201912, df_202003, df_202006, df_202009, df_202012, df_202103, df_202106, df_202109, df_202112, df_202203, df_202206, df_202209, df_202212], ignore_index=True)

print(df_credit.shape)
df_credit.head()
```
    (10468940, 3)
  <table>
    <thead>
      <tr>
        <!-- 인덱스를 표시하는 열 추가 -->
        <th> </th>
        <th>join_sn</th>
        <th>hl</th>
        <th>ym</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>0</td>
        <td>3</td>
        <td>0</td>
        <td>201812</td>
      </tr>
      <tr>
        <td>1</td>
        <td>8</td>
        <td>0</td>
        <td>201812</td>
      </tr>
      <tr>
        <td>2</td>
        <td>13</td>
        <td>0</td>
        <td>201812</td>
      </tr>
      <tr>
        <td>3</td>
        <td>17</td>
        <td>0</td>
        <td>201812</td>
      </tr>
      <tr>
        <td>4</td>
        <td>21</td>
        <td>0</td>
        <td>201812</td>
      </tr>
    </tbody>
  </table>

# 2. 전처리
## 2.1 보유여부 및 보유개수 테이블 만들기
```python
df_cnt_pre = pd.read_csv('../INSURANCE_CNT_CONTRACT_I_VER3.csv')
```

```python
# 생명보험(종신+정기)
df_cnt_pre['life'] = df_cnt_pre['i_cnt_whole']+df_cnt_pre['i_cnt_term']
# 건강보험(질병+암)
df_cnt_pre['disease'] = df_cnt_pre['i_cnt_disease'] + df_cnt_pre['i_cnt_cancer']
# 저축성보험
df_cnt_pre['saving'] = df_cnt_pre['i_cnt_pen_sv'] + df_cnt_pre['i_cnt_pen'] + df_cnt_pre['i_cnt_sv'] + df_cnt_pre['i_cnt_ed']
```

```python
target_cols = ['join_sn', 'ym', 'i_cnt_all','life', 'disease', 'i_cnt_health', 'saving']
df_cnt = df_cnt_pre[target_cols]
df_cnt.rename(columns={'i_cnt_all': 'all', 'i_cnt_health': 'hurt'}, inplace=True)
# 보유개수 테이블
df_cnt.head(3)
```
<table>
  <thead>
    <tr>
      <th> </th>
      <th>join_sn</th>
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
      <td>1</td>
      <td>201803</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
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
      <td>1</td>
      <td>201809</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>

```python
df_credit_cnt = df_credit.merge(df_cnt, how='left', on=['join_sn', 'ym'])

print("보유개수 df_credit_cnt 테이블", df_credit_cnt.shape)
print(df_credit_cnt.columns)
```

```
보유개수 df_credit_cnt 테이블 (10468940, 8)
Index(['join_sn', 'hl', 'ym', 'all', 'life', 'disease', 'hurt', 'saving'], dtype='object')
```

```python
# 대출은 있으나 보험이 하나도 없는 차주-시점은 보험보유 '없음' 보험개수 0개로 바꾼다
df_credit_cnt = df_credit_cnt.fillna(0)
print(df_credit_cnt.isna().sum())
```

```
join_sn    0
hl         0
ym         0
all        0
life       0
disease    0
hurt       0
saving     0
dtype: int64
```

## 2.2 event time 정의
```python
print(df_credit_cnt.shape)
event_ym = 202003
df_credit_cnt['year'] = df_credit_cnt['ym']//100
df_credit_cnt['month'] = df_credit_cnt['ym']%100
event_year = event_ym // 100
event_month = event_ym % 100

df_credit_cnt['event_time_month'] = ( df_credit_cnt['year'] - event_year)*12 + ( df_credit_cnt['month'] - event_month )
df_credit_cnt['event_time'] = df_credit_cnt['event_time_month']/3
print(df_credit_cnt.shape)
df_credit_cnt.head(3)
```
    (10468940, 12)
    (10468940, 12)
<table>
  <thead>
    <tr>
      <th> </th>
      <th>join_sn</th>
      <th>hl</th>
      <th>ym</th>
      <th>all</th>
      <th>life</th>
      <th>disease</th>
      <th>hurt</th>
      <th>saving</th>
      <th>year</th>
      <th>month</th>
      <th>event_time_month</th>
      <th>event_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>201812</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2018</td>
      <td>12</td>
      <td>-15</td>
      <td>-5.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>8</td>
      <td>0</td>
      <td>201812</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2018</td>
      <td>12</td>
      <td>-15</td>
      <td>-5.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>13</td>
      <td>0</td>
      <td>201812</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2018</td>
      <td>12</td>
      <td>-15</td>
      <td>-5.0</td>
    </tr>
  </tbody>
</table>

    

## 2.3 event time의 더미변수 생성
```python
# 보유개수 

df_credit_cnt_et = pd.get_dummies(df_credit_cnt, columns=['event_time'], prefix='et')
# if event_time is NaN, all dummy columns are False

# 변수명으로 사용할 수 없는 -와 .를 수정한다 
et_columns = [col for col in df_credit_cnt_et.columns if col.startswith('et_')]
new_et_columns = {col: col.replace('.0', '').replace('-', '9') for col in et_columns}
# et_915는 eventtime -15 시점을 뜻함 
df_credit_cnt_et.rename(columns = new_et_columns, inplace=True)

# all et_* columns are set to False if Never-treated
cols_to_modify = [col for col in df_credit_cnt_et.columns if col.startswith('et_')]
df_credit_cnt_et.loc[df_credit_cnt_et['hl']==0, cols_to_modify] = False
df_credit_cnt_et.head(3)
```
<table>
  <thead>
    <tr>
      <th>Index</th>
      <th>join_sn</th>
      <th>hl</th>
      <th>ym</th>
      <th>all</th>
      <th>life</th>
      <th>disease</th>
      <th>hurt</th>
      <th>saving</th>
      <th>year</th>
      <th>month</th>
      <th>event_time_month</th>
      <th>et_95</th>
      <th>et_94</th>
      <th>et_93</th>
      <th>et_92</th>
      <th>et_91</th>
      <th>et_0</th>
      <th>et_1</th>
      <th>et_2</th>
      <th>...</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>201812</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2018</td>
      <td>12</td>
      <td>-15</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
    </tr>
    <tr>
      <td>1</td>
      <td>8</td>
      <td>0</td>
      <td>201812</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2018</td>
      <td>12</td>
      <td>-15</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
    </tr>
    <tr>
      <td>2</td>
      <td>13</td>
      <td>0</td>
      <td>201812</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2018</td>
      <td>12</td>
      <td>-15</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
    </tr>
  </tbody>
</table>



보유개수 테이블로부터 보유여부 테이블 생성
```python
df_credit_bin_et = df_credit_cnt_et.copy()
# 보유여부 테이블
df_credit_bin_et.rename(columns={'all': 'has_all', 'life': 'has_life', 'disease': 'has_disease', 'hurt': 'has_hurt', 'saving': 'has_saving'}, inplace=True)
cols_to_transform = ['has_all', 'has_life', 'has_disease', 'has_hurt', 'has_saving']
df_credit_bin_et[cols_to_transform] = (df_credit_bin_et[cols_to_transform] > 0).astype(int)

df_credit_bin_et.head(3)
```
<table>
  <thead>
    <tr>
      <th>Index</th>
      <th>join_sn</th>
      <th>hl</th>
      <th>ym</th>
      <th>all</th>
      <th>life</th>
      <th>disease</th>
      <th>hurt</th>
      <th>saving</th>
      <th>year</th>
      <th>month</th>
      <th>event_time_month</th>
      <th>et_95</th>
      <th>et_94</th>
      <th>et_93</th>
      <th>et_92</th>
      <th>et_91</th>
      <th>et_0</th>
      <th>et_1</th>
      <th>et_2</th>
      <th>...</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>201812</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2018</td>
      <td>12</td>
      <td>-15</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
    </tr>
    <tr>
      <td>1</td>
      <td>8</td>
      <td>0</td>
      <td>201812</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2018</td>
      <td>12</td>
      <td>-15</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
    </tr>
    <tr>
      <td>2</td>
      <td>13</td>
      <td>0</td>
      <td>201812</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2018</td>
      <td>12</td>
      <td>-15</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
    </tr>
  </tbody>
</table>


# 3.1 보유여부의 eventstudy plot
```python
print(df_credit_cnt_et.shape)
```

```
(10468940, 28)
```

```python
target_cols = ['has_life', 'has_disease', 'has_hurt']
# 베이스라인을 -1 시점으로 지정
event_cols = [c for c in df_credit_bin_et if c.startswith('et_') and c != 'et_91']

df_credit_bin_et_temp = df_credit_bin_et.set_index(['join_sn', 'ym']) # Entity FE, Time FE를 위해 인덱스 지정 

for col in target_cols:
    # 모형 만들고 적합 **************************************************
    formula = f"{col} ~ EntityEffects + TimeEffects +" + " + ".join(event_cols)
    model = PanelOLS.from_formula(formula, data=df_credit_bin_et_temp)
    result = model.fit(cov_type='clustered', cluster_entity=True)

    # 숫자 추출 **************************************************
    params = result.params
    std_errors = result.std_errors
    t_values = result.tstats
    p_values = result.pvalues
    conf_int = result.conf_int()

    # event_coefs라는 테이블로 만든다
    variables = [var for var in params.index if var.startswith('et')]
     
    data = {
        'variable': variables,
        'coefficient': [params[var] for var in variables],
        'standard_error': [std_errors[var] for var in variables],
        'T-value': [t_values[var] for var in variables],
        'p-value': [p_values[var] for var in variables],
        'lower_ci': [conf_int.loc[var, 'lower'] for var in variables],
        'upper_ci': [conf_int.loc[var, 'upper'] for var in variables],
    }
    event_coefs = pd.DataFrame(data)
    
    # 변수명으로부터 event time에 해당하는 숫자 추출하기
    event_coefs['event_time'] = event_coefs['variable'].apply(lambda x: int(re.findall(r'et_(\d+)', x)[0])) # 정규표현식 
    # event time에 해당하는 숫자를 알맞게 바꾸기
    event_coefs.loc[event_coefs['event_time']==95, 'event_time'] = -5
    event_coefs.loc[event_coefs['event_time']==94, 'event_time'] = -4
    event_coefs.loc[event_coefs['event_time']==93, 'event_time'] = -3
    event_coefs.loc[event_coefs['event_time']==92, 'event_time'] = -2 # 베이스라인 시점(-1)은 event coefs 테이블에 없음 

    # 플롯 **************************************************
    event_coefs = event_coefs.sort_values('event_time')
    print(event_coefs)
    
    plt.figure(figsize=(12,6))
    # point
    plt.scatter(event_coefs['event_time'], event_coefs['coefficient'], color='blue', label='Point Estimate')
    # Confidence Interval
    for i, row in event_coefs.iterrows():
        plt.plot([row['event_time'], row['event_time']], [row['lower_ci'], row['upper_ci']], color='blue')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.axvline(x=-1, color='purple', linestyle='--') # x = -1
    plt.title(f'{col} insurance Event-Study DiD plot: baseline=et-1') # baseline = -1 시점 
    plt.xlabel('Event Time')
    plt.ylabel('coefficient')
    plt.show()
```

```



   variable  coefficient  standard_error    T-value       p-value  lower_ci  \
0     et_95     0.007027        0.001259   5.583112  2.362591e-08  0.004560   
1     et_94     0.004814        0.001121   4.293379  1.759761e-05  0.002616   
2     et_93     0.003809        0.000923   4.128185  3.656406e-05  0.002001   
3     et_92     0.002085        0.000676   3.083326  2.047010e-03  0.000760   
4      et_0    -0.001805        0.000688  -2.624788  8.670317e-03 -0.003152   
5      et_1    -0.002443        0.000957  -2.554275  1.064094e-02 -0.004318   
6      et_2    -0.003099        0.001145  -2.705509  6.819993e-03 -0.005344   
7      et_3    -0.005957        0.001274  -4.676479  2.918467e-06 -0.008454   
8      et_4    -0.011107        0.001387  -8.006840  1.110223e-15 -0.013825   
9      et_5    -0.015363        0.001493 -10.287466  0.000000e+00 -0.018290   
10     et_6    -0.018482        0.001566 -11.805604  0.000000e+00 -0.021550   
11     et_7    -0.019453        0.001628 -11.946228  0.000000e+00 -0.022645   
12     et_8    -0.021342        0.001682 -12.685755  0.000000e+00 -0.024639   
13     et_9    -0.022571        0.001727 -13.072700  0.000000e+00 -0.025955   
14    et_10    -0.030948        0.001849 -16.741330  0.000000e+00 -0.034571   
15    et_11    -0.033222        0.001888 -17.592005  0.000000e+00 -0.036924   

    upper_ci  event_time  
0   0.009493          -5  
1   0.007011          -4  
2   0.005618          -3  
3   0.003410          -2  
4  -0.000457           0  
5  -0.000569           1  
6  -0.000854           2  
7  -0.003460           3  
8  -0.008388           4  
9  -0.012436           5  
10 -0.015414           6  
11 -0.016262           7  
12 -0.018044           8  
13 -0.019187           9  
14 -0.027325          10  
15 -0.029521          11  
```


![has_life_insurance](/images/capstonePRJ/2timepointsDiD_eventstudyplot/2timepoints_bin_life.png)






```
   variable  coefficient  standard_error    T-value       p-value  lower_ci  \
0     et_95     0.005121        0.001452   3.526179  4.216037e-04  0.002274   
1     et_94     0.003525        0.001307   2.695814  7.021708e-03  0.000962   
2     et_93     0.002816        0.001083   2.600370  9.312350e-03  0.000694   
3     et_92     0.001777        0.000821   2.163070  3.053582e-02  0.000167   
4      et_0    -0.003603        0.000791  -4.555275  5.231762e-06 -0.005153   
5      et_1    -0.006970        0.001067  -6.529703  6.590328e-11 -0.009062   
6      et_2    -0.008321        0.001269  -6.559022  5.416445e-11 -0.010807   
7      et_3    -0.010334        0.001414  -7.308945  2.693401e-13 -0.013105   
8      et_4    -0.013398        0.001562  -8.578170  0.000000e+00 -0.016459   
9      et_5    -0.016540        0.001673  -9.888195  0.000000e+00 -0.019818   
10     et_6    -0.019039        0.001747 -10.897731  0.000000e+00 -0.022463   
11     et_7    -0.022913        0.001816 -12.620316  0.000000e+00 -0.026472   
12     et_8    -0.025102        0.001875 -13.387868  0.000000e+00 -0.028776   
13     et_9    -0.026931        0.001937 -13.900668  0.000000e+00 -0.030728   
14    et_10    -0.028508        0.001998 -14.269991  0.000000e+00 -0.032424   
15    et_11    -0.031281        0.002053 -15.238449  0.000000e+00 -0.035305   

    upper_ci  event_time  
0   0.007967          -5  
1   0.006087          -4  
2   0.004939          -3  
3   0.003387          -2  
4  -0.002053           0  
5  -0.004878           1  
6  -0.005834           2  
7  -0.007563           3  
8  -0.010337           4  
9  -0.013261           5  
10 -0.015614           6  
11 -0.019355           7  
12 -0.021427           8  
13 -0.023134           9  
14 -0.024593          10  
15 -0.027258          11  
```
![has_health_insurance](/images/capstonePRJ/2timepointsDiD_eventstudyplot/2timepoints_bin_disease.png)



```
   variable  coefficient  standard_error    T-value       p-value  lower_ci  \
0     et_95     0.004474        0.001617   2.766915  5.658962e-03  0.001305   
1     et_94     0.001679        0.001427   1.176434  2.394215e-01 -0.001118   
2     et_93     0.000495        0.001181   0.418835  6.753365e-01 -0.001820   
3     et_92    -0.000673        0.000872  -0.771855  4.402005e-01 -0.002381   
4      et_0    -0.003691        0.000918  -4.022360  5.761845e-05 -0.005490   
5      et_1    -0.005113        0.001242  -4.118325  3.816387e-05 -0.007547   
6      et_2    -0.005482        0.001454  -3.769469  1.635962e-04 -0.008332   
7      et_3    -0.008692        0.001625  -5.349173  8.835889e-08 -0.011877   
8      et_4    -0.012517        0.001758  -7.119553  1.082912e-12 -0.015963   
9      et_5    -0.029936        0.001929 -15.517688  0.000000e+00 -0.033717   
10     et_6    -0.030178        0.002009 -15.023617  0.000000e+00 -0.034116   
11     et_7    -0.026740        0.002050 -13.044286  0.000000e+00 -0.030758   
12     et_8    -0.028719        0.002117 -13.564987  0.000000e+00 -0.032869   
13     et_9    -0.033355        0.002170 -15.368082  0.000000e+00 -0.037609   
14    et_10    -0.034808        0.002220 -15.677472  0.000000e+00 -0.039160   
15    et_11    -0.038364        0.002281 -16.817695  0.000000e+00 -0.042835   

    upper_ci  event_time  
0   0.007643          -5  
1   0.004477          -4  
2   0.002810          -3  
3   0.001035          -2  
4  -0.001893           0  
5  -0.002680           1  
6  -0.002632           2  
7  -0.005507           3  
8  -0.009071           4  
9  -0.026155           5  
10 -0.026241           6  
11 -0.022722           7  
12 -0.024570           8  
13 -0.029101           9  
14 -0.030457          10  
15 -0.033893          11  
```
![has_hurt_insurance](/images/capstonePRJ/2timepointsDiD_eventstudyplot/2timepoints_bin_hurt.png)





# 3.2 보유개수의 eventstudy plot 
```python
target_cols = ['life', 'disease', 'hurt']
# 베이스라인을 -1 시점으로 지정
event_cols = [c for c in df_credit_cnt_et if c.startswith('et_') and c != 'et_91']

df_credit_cnt_et_temp = df_credit_cnt_et.set_index(['join_sn', 'ym']) # Entity FE, Time FE를 위해 인덱스 지정 

for col in target_cols:
    # 1% 이상치 처리 *************************************
    threshold = df_credit_cnt_et_temp[col].quantile(0.99)
    df_credit_cnt_et_temp.loc[df_credit_cnt_et_temp[col] > threshold, col] = None # =threshold로 하면 cap 지정 방식으로 이상치 처리
    # 모형 만들고 적합 **************************************************
    formula = f"{col} ~ EntityEffects + TimeEffects +" + " + ".join(event_cols)
    model = PanelOLS.from_formula(formula, data=df_credit_cnt_et_temp)
    result = model.fit(cov_type='clustered', cluster_entity=True)

    # 숫자 추출 **************************************************
    params = result.params
    std_errors = result.std_errors
    t_values = result.tstats
    p_values = result.pvalues
    conf_int = result.conf_int()

    # event_coefs라는 테이블로 만든다
    variables = [var for var in params.index if var.startswith('et')]
     
    data = {
        'variable': variables,
        'coefficient': [params[var] for var in variables],
        'standard_error': [std_errors[var] for var in variables],
        'T-value': [t_values[var] for var in variables],
        'p-value': [p_values[var] for var in variables],
        'lower_ci': [conf_int.loc[var, 'lower'] for var in variables],
        'upper_ci': [conf_int.loc[var, 'upper'] for var in variables],
    }
    event_coefs = pd.DataFrame(data)
    
    # 변수명으로부터 event time에 해당하는 숫자 추출하기
    event_coefs['event_time'] = event_coefs['variable'].apply(lambda x: int(re.findall(r'et_(\d+)', x)[0])) # 정규표현식 
    # event time에 해당하는 숫자를 알맞게 바꾸기
    event_coefs.loc[event_coefs['event_time']==915, 'event_time'] = -15
    event_coefs.loc[event_coefs['event_time']==914, 'event_time'] = -14
    event_coefs.loc[event_coefs['event_time']==913, 'event_time'] = -13
    event_coefs.loc[event_coefs['event_time']==912, 'event_time'] = -12
    event_coefs.loc[event_coefs['event_time']==911, 'event_time'] = -11
    event_coefs.loc[event_coefs['event_time']==910, 'event_time'] = -10
    event_coefs.loc[event_coefs['event_time']==99, 'event_time'] = -9
    event_coefs.loc[event_coefs['event_time']==98, 'event_time'] = -8
    event_coefs.loc[event_coefs['event_time']==97, 'event_time'] = -7
    event_coefs.loc[event_coefs['event_time']==96, 'event_time'] = -6
    event_coefs.loc[event_coefs['event_time']==95, 'event_time'] = -5
    event_coefs.loc[event_coefs['event_time']==94, 'event_time'] = -4
    event_coefs.loc[event_coefs['event_time']==93, 'event_time'] = -3
    event_coefs.loc[event_coefs['event_time']==92, 'event_time'] = -2 # 베이스라인 시점(-1)은 event coefs 테이블에 없음 

    # 플롯 **************************************************
    event_coefs = event_coefs.sort_values('event_time')
    print(event_coefs)
    
    plt.figure(figsize=(12,6))
    # point
    plt.scatter(event_coefs['event_time'], event_coefs['coefficient'], color='blue', label='Point Estimate')
    # Confidence Interval
    for i, row in event_coefs.iterrows():
        plt.plot([row['event_time'], row['event_time']], [row['lower_ci'], row['upper_ci']], color='blue')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.axvline(x=-1, color='purple', linestyle='--') # x = -1
    plt.title(f'{col} insurance Count: Event-Study DiD plot: baseline=et-1') # baseline = -1 시점 
    plt.xlabel('Event Time')
    plt.ylabel('coefficient')
    plt.show()
```

```
   variable  coefficient  standard_error    T-value       p-value  lower_ci  \
0     et_95     0.011554        0.001787   6.465466  1.009921e-10  0.008052   
1     et_94     0.008080        0.001585   5.098499  3.423645e-07  0.004974   
2     et_93     0.007597        0.001314   5.780040  7.468533e-09  0.005021   
3     et_92     0.004212        0.000961   4.381137  1.180627e-05  0.002328   
4      et_0    -0.002447        0.000979  -2.498218  1.248195e-02 -0.004366   
5      et_1    -0.003710        0.001354  -2.740419  6.136094e-03 -0.006364   
6      et_2    -0.004682        0.001644  -2.848028  4.399113e-03 -0.007904   
7      et_3    -0.009364        0.001818  -5.149866  2.606771e-07 -0.012928   
8      et_4    -0.015710        0.001982  -7.924437  2.220446e-15 -0.019596   
9      et_5    -0.020998        0.002132  -9.849106  0.000000e+00 -0.025176   
10     et_6    -0.025212        0.002225 -11.328886  0.000000e+00 -0.029574   
11     et_7    -0.026801        0.002307 -11.617876  0.000000e+00 -0.031322   
12     et_8    -0.029327        0.002387 -12.284130  0.000000e+00 -0.034006   
13     et_9    -0.031638        0.002451 -12.910058  0.000000e+00 -0.036441   
14    et_10    -0.044623        0.002608 -17.109793  0.000000e+00 -0.049735   
15    et_11    -0.048010        0.002672 -17.970459  0.000000e+00 -0.053246   

    upper_ci  event_time  
0   0.015057          -5  
1   0.011186          -4  
2   0.010173          -3  
3   0.006097          -2  
4  -0.000527           0  
5  -0.001057           1  
6  -0.001460           2  
7  -0.005800           3  
8  -0.011825           4  
9  -0.016819           5  
10 -0.020850           6  
11 -0.022279           7  
12 -0.024648           8  
13 -0.026835           9  
14 -0.039511          10  
15 -0.042774          11  
```
![count_life_insurance](/images/capstonePRJ/2timepointsDiD_eventstudyplot/2timepoints_count_life.png)


```
   variable  coefficient  standard_error    T-value       p-value  lower_ci  \
0     et_95     0.000878        0.003593   0.244343  8.069654e-01 -0.006163   
1     et_94     0.001179        0.003233   0.364678  7.153521e-01 -0.005158   
2     et_93     0.001744        0.002635   0.661830  5.080802e-01 -0.003420   
3     et_92     0.000798        0.001964   0.406164  6.846220e-01 -0.003052   
4      et_0    -0.009508        0.001913  -4.969848  6.700662e-07 -0.013258   
5      et_1    -0.014669        0.002621  -5.597368  2.176365e-08 -0.019806   
6      et_2    -0.016061        0.003190  -5.035358  4.769646e-07 -0.022313   
7      et_3    -0.021971        0.003606  -6.092840  1.109290e-09 -0.029039   
8      et_4    -0.031589        0.004013  -7.870795  3.552714e-15 -0.039456   
9      et_5    -0.037323        0.004346  -8.588062  0.000000e+00 -0.045841   
10     et_6    -0.044560        0.004566  -9.760226  0.000000e+00 -0.053509   
11     et_7    -0.051582        0.004792 -10.764953  0.000000e+00 -0.060974   
12     et_8    -0.056413        0.004973 -11.342949  0.000000e+00 -0.066161   
13     et_9    -0.060348        0.005154 -11.709118  0.000000e+00 -0.070450   
14    et_10    -0.060062        0.005393 -11.136373  0.000000e+00 -0.070632   
15    et_11    -0.070222        0.005533 -12.692485  0.000000e+00 -0.081066   

    upper_ci  event_time  
0   0.007919          -5  
1   0.007517          -4  
2   0.006908          -3  
3   0.004648          -2  
4  -0.005758           0  
5  -0.009533           1  
6  -0.009810           2  
7  -0.014904           3  
8  -0.023723           4  
9  -0.028805           5  
10 -0.035612           6  
11 -0.042191           7  
12 -0.046666           8  
13 -0.050247           9  
14 -0.049491          10  
15 -0.059379          11  
```
![count_disease_insurance](/images/capstonePRJ/2timepointsDiD_eventstudyplot/2timepoints_count_disease.png)



```
   variable  coefficient  standard_error    T-value       p-value  lower_ci  \
0     et_95     0.008967        0.002697   3.325293  8.832592e-04  0.003682   
1     et_94     0.004533        0.002396   1.891876  5.850757e-02 -0.000163   
2     et_93     0.002616        0.001967   1.329799  1.835845e-01 -0.001239   
3     et_92    -0.000425        0.001435  -0.296190  7.670850e-01 -0.003236   
4      et_0    -0.006020        0.001497  -4.021120  5.792264e-05 -0.008954   
5      et_1    -0.010122        0.002020  -5.010902  5.417646e-07 -0.014081   
6      et_2    -0.011869        0.002410  -4.924638  8.451790e-07 -0.016593   
7      et_3    -0.016351        0.002731  -5.987619  2.129431e-09 -0.021703   
8      et_4    -0.024842        0.002979  -8.338937  0.000000e+00 -0.030681   
9      et_5    -0.088176        0.003345 -26.356914  0.000000e+00 -0.094733   
10     et_6    -0.093692        0.003542 -26.451352  0.000000e+00 -0.100634   
11     et_7    -0.072232        0.003608 -20.019993  0.000000e+00 -0.079304   
12     et_8    -0.081785        0.003774 -21.669355  0.000000e+00 -0.089182   
13     et_9    -0.089510        0.003875 -23.100898  0.000000e+00 -0.097105   
14    et_10    -0.095493        0.003998 -23.882515  0.000000e+00 -0.103330   
15    et_11    -0.103482        0.004142 -24.980609  0.000000e+00 -0.111601   

    upper_ci  event_time  
0   0.014253          -5  
1   0.009230          -4  
2   0.006471          -3  
3   0.002387          -2  
4  -0.003086           0  
5  -0.006163           1  
6  -0.007145           2  
7  -0.010999           3  
8  -0.019004           4  
9  -0.081619           5  
10 -0.086750           6  
11 -0.065161           7  
12 -0.074388           8  
13 -0.081916           9  
14 -0.087657          10  
15 -0.095363          11  
```
![count_hurt_insurance](/images/capstonePRJ/2timepointsDiD_eventstudyplot/2timepoints_count_hurt.png)





