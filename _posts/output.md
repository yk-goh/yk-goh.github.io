```
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

```
df = pd.read_csv('241228_template_did.csv', index_col=0)

df.dropna(inplace=True)
print(df.shape)
df.drop_duplicates(inplace=True, subset=['join_sn', 'did_credit_change'])
print(df.shape)
```

```
df = df[['join_sn', 'did_credit_change']]
df['hl'] = df['did_credit_change'].apply(lambda x: 1 if x == 'high_low' else 0)

df.head()

df['did_credit_change'].value_counts()
```

```
df_credit_pre = df[['join_sn', 'hl']]

df_201912 = df_credit_pre.assign(ym=201912)
df_202112 = df_credit_pre.assign(ym=202112)
df_credit = pd.concat([df_201912, df_202112], ignore_index=True)
```

```
df_cnt = pd.read_csv('../INSURANCE_CNT_CONTRACT_I_VER3.csv')
```

```
# 생명보험(종신+정기)
df_cnt['life'] = df_cnt['i_cnt_whole']+df_cnt['i_cnt_term']
# 건강보험(질병+암)
df_cnt['disease'] = df_cnt['i_cnt_disease'] + df_cnt['i_cnt_cancer']
# 저축성보험
df_cnt['saving'] = df_cnt['i_cnt_pen_sv'] + df_cnt['i_cnt_pen'] + df_cnt['i_cnt_sv'] + df_cnt['i_cnt_ed']
# 화재보험
df_cnt['fire'] = df_cnt['i_cnt_fire'] + df_cnt['i_cnt_liability'] + df_cnt['i_cnt_fire_2']
```

```
df_cnt_binary = df_cnt.copy()
df_cnt_binary = df_cnt_binary.set_index(['join_sn', 'ym'])
# df_cnt_binary = df_cnt_binary.set_index(['join_sn', 'ym']).applymap(lambda x: 1 if x > 0 else 0).reset_index()
df_cnt_binary = (df_cnt_binary > 0).astype(int).reset_index()
df_cnt_binary.head(2)
```

```
df_cnt_binary.isna().sum()
```

```
df_cnt_binary.shape
```

```
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
df_cnt_filtered = df_cnt[(df_cnt['ym']==201912) | (df_cnt['ym']==202112)]
df_cnt_binary_filtered = df_cnt_binary[(df_cnt_binary['ym']==201912) | (df_cnt_binary['ym']==202112)]

df_cnt_filtered.head(2)
```

```
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
# 대출은 있으나 보험이 하나도 없는 차주-시점
print(df_credit_bin.isna().sum())
print(df_credit_cnt.isna().sum())
```

```
# 대출은 있으나 보험이 하나도 없는 차주-시점은 보험보유 '없음' 보험개수 0개로 바꾼다
df_credit_bin = df_credit_bin.fillna(0)
df_credit_cnt = df_credit_cnt.fillna(0)
print(df_credit_bin.isna().sum())
print(df_credit_cnt.isna().sum())
```

```
print(df_credit_bin[df_credit_bin['ym']==201912]['i_has_all'].value_counts())
print(df_credit_bin[df_credit_bin['ym']==202112]['i_has_all'].value_counts())
print(df_credit_bin[df_credit_bin['post']==0]['i_has_all'].value_counts())
print(df_credit_bin[df_credit_bin['post']==1]['i_has_all'].value_counts())

print(df_credit_cnt[df_credit_cnt['ym']==201912]['i_cnt_all'].value_counts())
print(df_credit_cnt[df_credit_cnt['ym']==202112]['i_cnt_all'].value_counts())
print(df_credit_cnt[df_credit_cnt['post']==0]['i_cnt_all'].value_counts())
print(df_credit_cnt[df_credit_cnt['post']==1]['i_cnt_all'].value_counts())
```

```
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

```
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

```
df_credit_bin.head(2)
```

```
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

```
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

#### 차주번호와 hl (hl=0이면 고고, hl=1이면 고저)¶

## 1. 전처리¶

보유 여부, 보유 개수 테이블 만들기


2019 12, 2021 12만 필터링


신용 테이블에 보유여부, 보유개수 테이블 쪼인


보유여부 확인 시 사용할 컬럼, 보유개수 확인 시 사용할 컬럼


# 1.1 보유여부 - ols¶

# 2.1 보유개수 - ols¶