---
layout: single
title: "Retrieving CounterFactuals"
categories: XAI
sidebar: true
use_math: true
---

# 문제: bank 데이터를 이용하여 CFP를 구한다. 

프로그램 시작 시 tf.compat.v1.disable_v2_behavior() 함수를 가장 먼저 실행하여 alibi explainer가 작동할 수 있도록 한다.


```python
import pandas as pd 
import numpy as np
import alibi
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf 
tf.compat.v1.disable_v2_behavior()

np.set_printoptions(suppress=True, precision=3)
```

## 1. 데이터 확인
bank 데이터셋은 은행의 마케팅캠페인 정보와 고객이 해당 캠페인 결과 은행의 정기예금 상품에 가입했는지를 기록한 데이터셋이다. 16개의 특성변수가 있으며 목적변수는 y이다. 


```python
path = 'bank.csv'
df = pd.read_csv(path, header=0, index_col=None, delimiter=';')
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day</th>
      <th>month</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30</td>
      <td>unemployed</td>
      <td>married</td>
      <td>primary</td>
      <td>no</td>
      <td>1787</td>
      <td>no</td>
      <td>no</td>
      <td>cellular</td>
      <td>19</td>
      <td>oct</td>
      <td>79</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33</td>
      <td>services</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>4789</td>
      <td>yes</td>
      <td>yes</td>
      <td>cellular</td>
      <td>11</td>
      <td>may</td>
      <td>220</td>
      <td>1</td>
      <td>339</td>
      <td>4</td>
      <td>failure</td>
      <td>no</td>
    </tr>
    <tr>
      <th>2</th>
      <td>35</td>
      <td>management</td>
      <td>single</td>
      <td>tertiary</td>
      <td>no</td>
      <td>1350</td>
      <td>yes</td>
      <td>no</td>
      <td>cellular</td>
      <td>16</td>
      <td>apr</td>
      <td>185</td>
      <td>1</td>
      <td>330</td>
      <td>1</td>
      <td>failure</td>
      <td>no</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30</td>
      <td>management</td>
      <td>married</td>
      <td>tertiary</td>
      <td>no</td>
      <td>1476</td>
      <td>yes</td>
      <td>yes</td>
      <td>unknown</td>
      <td>3</td>
      <td>jun</td>
      <td>199</td>
      <td>4</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
    <tr>
      <th>4</th>
      <td>59</td>
      <td>blue-collar</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>0</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>226</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>balance</th>
      <th>day</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4521.000000</td>
      <td>4521.000000</td>
      <td>4521.000000</td>
      <td>4521.000000</td>
      <td>4521.000000</td>
      <td>4521.000000</td>
      <td>4521.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>41.170095</td>
      <td>1422.657819</td>
      <td>15.915284</td>
      <td>263.961292</td>
      <td>2.793630</td>
      <td>39.766645</td>
      <td>0.542579</td>
    </tr>
    <tr>
      <th>std</th>
      <td>10.576211</td>
      <td>3009.638142</td>
      <td>8.247667</td>
      <td>259.856633</td>
      <td>3.109807</td>
      <td>100.121124</td>
      <td>1.693562</td>
    </tr>
    <tr>
      <th>min</th>
      <td>19.000000</td>
      <td>-3313.000000</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>33.000000</td>
      <td>69.000000</td>
      <td>9.000000</td>
      <td>104.000000</td>
      <td>1.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>39.000000</td>
      <td>444.000000</td>
      <td>16.000000</td>
      <td>185.000000</td>
      <td>2.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>49.000000</td>
      <td>1480.000000</td>
      <td>21.000000</td>
      <td>329.000000</td>
      <td>3.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>87.000000</td>
      <td>71188.000000</td>
      <td>31.000000</td>
      <td>3025.000000</td>
      <td>50.000000</td>
      <td>871.000000</td>
      <td>25.000000</td>
    </tr>
  </tbody>
</table>
</div>



목적변수를 살펴보면, 약 8:1로 no가 더 많은 불균형 데이터이다


```python
df['y'].value_counts()
```




    y
    no     4000
    yes     521
    Name: count, dtype: int64



다음으로 특성변수의 타입을 확인한다. 
- categorical variables: index 1(job), 2(marital), 3(education), 4(default), 6(housing), 7(loan), 8(contact), 10(month), 15(poutcome)
- numeric variables: index 0(age), 5(balance), 9(day), 11(duration), 12(campaign), 13(pdays), 14(previous)

일자(day)는 csv 파일에 숫자로 저장되어 있어서 pandas가 int로 읽어왔다. 그러나 월(month)이 다를 때 날짜 간 대소 관계가 성립하기 어렵다. 
- "(1월) 31일은 (2월) 1일보다 30만큼 크다" 혹은 "31배 크다"라고 말할 수 없다.

그러므로 특성변수 day를 숫자형이 아닌 범주형으로 취급하기로 한다. 그러나 범주가 31개이면 one-hot 인코딩 했을 때 너무 sparse 해져 모형 성능에 악영향을 줄 수 있다.

따라서 일자를 초순(1일~10일), 중순(11일~20일), 말(21일~31일)로 범주화 하기로 한다. 이 과제에서 다루는 bank 데이터셋이 개인 재정과 관련된 데이터셋이고, 개인의 수입/지출 사이클은 일 단위보다는 주로 월 단위로 이루어지므로(월급, 공과금 납부 등) 일자(day) 변수를 범주화 하여도 무리는 없을 것으로 보인다.


```python
df.dtypes
```




    age           int64
    job          object
    marital      object
    education    object
    default      object
    balance       int64
    housing      object
    loan         object
    contact      object
    day           int64
    month        object
    duration      int64
    campaign      int64
    pdays         int64
    previous      int64
    poutcome     object
    y            object
    dtype: object



## 2. 전처리

### 2.1 day 변수 범주화
데이터프레임에 day_cat이라는 범주형 변수를 생성한다.


```python
def get_day_cat(day):
    if day <= 10:
        day_cat = 'early'
    elif day <= 20:
        day_cat = 'mid'
    else: 
        day_cat = 'late'
    return day_cat

df['day_cat'] = df['day'].apply(get_day_cat)

df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>balance</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>day</th>
      <th>month</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>y</th>
      <th>day_cat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30</td>
      <td>unemployed</td>
      <td>married</td>
      <td>primary</td>
      <td>no</td>
      <td>1787</td>
      <td>no</td>
      <td>no</td>
      <td>cellular</td>
      <td>19</td>
      <td>oct</td>
      <td>79</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
      <td>mid</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33</td>
      <td>services</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>4789</td>
      <td>yes</td>
      <td>yes</td>
      <td>cellular</td>
      <td>11</td>
      <td>may</td>
      <td>220</td>
      <td>1</td>
      <td>339</td>
      <td>4</td>
      <td>failure</td>
      <td>no</td>
      <td>mid</td>
    </tr>
    <tr>
      <th>2</th>
      <td>35</td>
      <td>management</td>
      <td>single</td>
      <td>tertiary</td>
      <td>no</td>
      <td>1350</td>
      <td>yes</td>
      <td>no</td>
      <td>cellular</td>
      <td>16</td>
      <td>apr</td>
      <td>185</td>
      <td>1</td>
      <td>330</td>
      <td>1</td>
      <td>failure</td>
      <td>no</td>
      <td>mid</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30</td>
      <td>management</td>
      <td>married</td>
      <td>tertiary</td>
      <td>no</td>
      <td>1476</td>
      <td>yes</td>
      <td>yes</td>
      <td>unknown</td>
      <td>3</td>
      <td>jun</td>
      <td>199</td>
      <td>4</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
      <td>early</td>
    </tr>
    <tr>
      <th>4</th>
      <td>59</td>
      <td>blue-collar</td>
      <td>married</td>
      <td>secondary</td>
      <td>no</td>
      <td>0</td>
      <td>yes</td>
      <td>no</td>
      <td>unknown</td>
      <td>5</td>
      <td>may</td>
      <td>226</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>unknown</td>
      <td>no</td>
      <td>early</td>
    </tr>
  </tbody>
</table>
</div>



### 2.2 특성변수의 순서 변경
전처리 과정의 용이성을 위해 범주형 특성변수끼리, 실수형 특성변수끼리 모이도록 순서를 변경한다. 범주형 특성변수 열 개가 앞부분에, 실수형 특성변수 여섯 개는 뒷부분에 위치하도록 한다. 한편, 목적변수가 ‘yes’ ‘no’로 기입되어 있는데 이를 1과 0으로 바꿔준다.


```python
df_cat = df[['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'day_cat']]
df_num = df[['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']]
df_f = pd.concat([df_cat, df_num], axis=1)
df_t = df['y'].map({'yes': 1, 'no':0})
```

### 2.3 범주형 특성변수 전처리 (1) category_map 만들기
category map을 Ordinal Encoding보다 먼저 만들어야 한다. alibi.utils 라이브러리의 gen_category_map 함수를 사용하여 category map을 만든다. alibi 특성 상 입력 데이터는 numpy 형이어야 하므로 pandas 데이터프레임에 .to_numpy()를 사용하여 numpy로 캐스트한다. 


```python
feature_names = df_f.columns.to_list()
categorical_index = list(range(10)) # index 0 thru 9 are categorical columns 

category_map = alibi.utils.gen_category_map(data=df_f.to_numpy(), categorical_columns=categorical_index)
category_map
```




    {0: ['admin.',
      'blue-collar',
      'entrepreneur',
      'housemaid',
      'management',
      'retired',
      'self-employed',
      'services',
      'student',
      'technician',
      'unemployed',
      'unknown'],
     1: ['divorced', 'married', 'single'],
     2: ['primary', 'secondary', 'tertiary', 'unknown'],
     3: ['no', 'yes'],
     4: ['no', 'yes'],
     5: ['no', 'yes'],
     6: ['cellular', 'telephone', 'unknown'],
     7: ['apr',
      'aug',
      'dec',
      'feb',
      'jan',
      'jul',
      'jun',
      'mar',
      'may',
      'nov',
      'oct',
      'sep'],
     8: ['failure', 'other', 'success', 'unknown'],
     9: ['early', 'late', 'mid']}



### 2.3 범주형 특성변수 전처리 (2) Ordinal Encoding
sklearn에서도 OneHotEncoder를 제공하지만 이 과제에서는 ablibi.utils의 함수인 ord_to_ohe를 사용하여 one-hot 인코딩을 하고자 한다. 이에 범주형 특성변수를 먼저 순서형 정수로 ordinal encoding 한다. 아래 프로그램을 보면 OrdinalEncoder를 통과한 df_cat은 numpy로 형변환 되었음을 알 수 있다. 


```python
oe = OrdinalEncoder()
df_cat = oe.fit_transform(df_cat)
print(type(df_cat))
print(type(df_num))
```

    <class 'numpy.ndarray'>
    <class 'pandas.core.frame.DataFrame'>


범주형 특성변수가 순서형 정수로 바뀌었음을 확인할 수 있다.


```python
X = np.c_[df_cat[:,:], df_num.to_numpy()]
y = df_t.to_numpy()

X[0] # 확인 완료
```




    array([  10.,    1.,    0.,    0.,    0.,    0.,    0.,   10.,    3.,
              2.,   30., 1787.,   79.,    1.,   -1.,    0.])



### 2.3 범주형 특성변수 전처리 (3) one-hot encoding
alibi.utility 함수인 ord_to_ohe에 사용되는 `cat_vars_ord`를 생성한다. cat_vars_ord는 범주형 특성변수의 인덱스를 key로 하고 해당 특성변수의 범주값의 개수를 value로 하는 dictionary이다. 

`{특성변수 index: 범주값의 개수}`


```python
cat_vars_ord = {}
n_categories = len(list(category_map.keys()))

for i in range(n_categories):
    cat_vars_ord[i] = len(np.unique(X[:, i]))
    
print(cat_vars_ord)
```

    {0: 12, 1: 3, 2: 4, 3: 2, 4: 2, 5: 2, 6: 3, 7: 12, 8: 4, 9: 3}


alibi.utility 함수인 `ord_to_ohe`를 사용하여 ordinal(순서형 정수) 변수를 one-hot encoding 한다. 이 함수의 첫번째 파라미터의 인자로는 순서형 정수인 데이터를, 두번째 파라미터의 인자로는 앞서 생성한 cat_vars_ord 딕셔너리를 넘겨준다. `ord_to_ohe(X, cat_vars_ord)`

`ord_to_ohe` 함수의 리턴은 다음과 같다. 
- ord_to_ohe\[0\]은 원핫인코딩 된 자료를 제공하고
- ord_to_ohe\[1\]은 dictionary로, key는 원핫인코딩 된 특성변수의 시작칼럼 인덱스, value는 범주값의 개수이다.


```python
from alibi.utils import ord_to_ohe, ohe_to_ord

cat_vars_ohe = ord_to_ohe(X[:, :-6], cat_vars_ord)[1]
print(cat_vars_ohe)

X_cat_ohe = ord_to_ohe(X[:, :-6], cat_vars_ord)[0]
print(X_cat_ohe[0])
```

    {0: 12, 12: 3, 15: 4, 19: 2, 21: 2, 23: 2, 25: 3, 28: 12, 40: 4, 44: 3}
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 1. 0. 1. 0. 0. 0. 1. 0. 1. 0. 1.
     0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 1. 0. 0. 1.]


### 2.4 `실수형` 특성변수 전처리: 표준화
특성변수값을 -1부터 1 사이로 표준화 하는 방법은 다음과 같다. 
$$x_{scaled} = (1-(-1))\times \frac{x-min(x)}{max(x)-min(x)} +(-1)$$


```python
X_num = X[:, -6: ].astype(np.float32, copy=False)
xmin, xmax = X_num.min(axis=0), X_num.max(axis=0)
lower, upper = -1.0, 1.0
X_num_scaled = 2*(X_num-xmin)/(xmax-xmin) - 1
# X_num_scaled = (upper - lower) * (X_num - xmin) / (xmax - xmin)  + lower

X_num_scaled.shape
```




    (4521, 6)



### 2.5 전처리된 (범주형, 실수형) 특성변수 결합
47개의 원핫 인코딩한 변수 + 6개의 실수형 변수를 연결하여 하나의 X 데이터로 만든다. 총 53개의 변수가 있음을 확인할 수 있다.


```python
X = np.c_[X_cat_ohe, X_num_scaled].astype(dtype=np.float32, copy=False)
print(X.shape) 
```

    (4521, 53)


### 2.6 학습데이터와 시험데이터 분할
불균형 데이터임에 유의하여 층화(stratify)를 사용한다.


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, stratify=y)
print(X_train.shape, X_test.shape)
```

    (3164, 53) (1357, 53)


## 3. 딥러닝 분류모형 생성
정기예금 상품에 가입했는지(1) 아닌지(0)를 예측하는 분류 모형을 만든다. 불균형 데이터이기 때문에 metric으로는 precision이나 f1 score가 적합할 것이나, tensorflow v1이 지원하는 metric이 제한적이어서 모형 컴파일 시 ‘accuracy’로 지정하였다. 


```python

tf.get_logger().setLevel(40)
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model 
from tensorflow.keras.utils import to_categorical
```


```python
def fwd():
    x_in = Input(shape=(53,))
    x = Dense(64, activation='relu')(x_in)
    x = Dropout(.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(.2)(x)
    x_out = Dense(2, activation='softmax')(x)

    model = Model(inputs=x_in, outputs=x_out)
    model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['AUC'])
    return model 
```


```python
model = fwd()
model.summary()
```

    Model: "model_1"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input_2 (InputLayer)        [(None, 53)]              0         
                                                                     
     dense_4 (Dense)             (None, 64)                3456      
                                                                     
     dropout_3 (Dropout)         (None, 64)                0         
                                                                     
     dense_5 (Dense)             (None, 64)                4160      
                                                                     
     dropout_4 (Dropout)         (None, 64)                0         
                                                                     
     dense_6 (Dense)             (None, 64)                4160      
                                                                     
     dropout_5 (Dropout)         (None, 64)                0         
                                                                     
     dense_7 (Dense)             (None, 2)                 130       
                                                                     
    =================================================================
    Total params: 11,906
    Trainable params: 11,906
    Non-trainable params: 0
    _________________________________________________________________


모형 출력층에 노드가 두 개이므로(yes, no) 모형에 학습데이터를 적합할 때에도 to_categorical 함수를 사용하여 y_train을 범주화 한다.


```python
y_train_ohe = to_categorical(y_train)
print(y_train_ohe.shape)
model.fit(X_train, y_train_ohe, batch_size=64, epochs=30, verbose=0)

```

    (3164, 2)





    <keras.callbacks.History at 0x255f2b9f970>




```python
y_test_ohe = to_categorical(y_test)
print(X_train.shape, X_test.shape, y_train_ohe.shape, y_test_ohe.shape)
```

    (3164, 53) (1357, 53) (3164, 2) (1357, 2)


적합된 모형의 성능 평가 결과, precision이 0.8998로 나타났다. 


```python
loss, accuracy = model.evaluate(X_test, y_test_ohe)
print(f'Accuracy: {accuracy}, Loss: {loss}')

from sklearn.metrics import f1_score, precision_score

y_pred_proba = model.predict(X_test)
y_pred = np.round(y_pred_proba)
precision = precision_score(y_test_ohe, y_pred, average='samples')
f1 = f1_score(y_test_ohe, y_pred, average='samples')

print(f'Precision: {precision:.4f}')
print(f'F1 Score: {f1:.4f}')
```

    Accuracy: 0.9660272598266602, Loss: 0.24388542793850382
    Precision: 0.8998
    F1 Score: 0.8998





```python
model.save('model_bank.h5', save_format='h5')
```

## 4. AutoEncoder 모형 생성

x_proto가 예측레이블의 데이터분포를 벗어나지 않도록 하는 autoencoder 손실을 추가하기 위해 autoencoder 모형을 정의하고 학습한다. 오토인코더 모형은 특성변수를 줄이는 과정에서 perturbation을 일으킨다. 그리고 인풋과 아웃풋 모두 X_train을 주어 학습시킴으로써 특성변수를 다시 늘렸을 때 원본과 얼마나 유사한가를 손실로 정의한다. 따라서 오토인코더 모형이 아래 5.번의 CounterfactualProto 동기화에 사용되면 모형은 이 손실을 최소화 하기 위해 데이터의 분포를 벗어나지 않는, 핍진성 있는 cf 표본을 생성하게 된다. 


```python
def ae_model():
    '''encoder'''
    x_in = Input(shape=(53,))
    x = Dense(64, activation='relu')(x_in)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    encoded = Dense(8, activation=None)(x)
    encoder = Model(x_in, encoded)
    '''decoder'''
    dec_in = Input(shape=(8,))
    x = Dense(16, activation='relu')(dec_in)
    x = Dense(32, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    decoded = Dense(53, activation=None)(x)
    decoder = Model(dec_in, decoded)
    '''AutoEncoder : encoder + decoder'''
    x_out = decoder(encoder(x_in))
    autoencoder = Model(x_in, x_out)
    autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
    return autoencoder, encoder, decoder
```


```python
ae, enc, dec = ae_model()
ae.fit(X_train, X_train, batch_size=64, epochs=100, validation_data=(X_test, X_test), verbose=0)

ae.save('ae_bank.h5', save_format='h5')
enc.save('enc_bank.h5', save_format='h5')
```




## 5. CFP 구하기

### 5.1 cf 동기화
먼저 CounterfactualProto 클래스를 호출하여 동기화 한다. 이 때 beta는 L1 규제화의 가중치, gamma는 오토인코더 손실의 가중치, theta는 x_proto 손실의 가중치이다. x_proto를 구하는 인코더 모형은 앞서 정의한 오토인코더 모형의 인코더(enc)로 지정한다. 학습에 사용되는 유사성(거리) 척도는 ‘abdm’으로 범주형 변수의 거리를 둘씩 짝지어서(pair-wise) 그 외 변수들과의 맥락을 고려하여 계산하는 방식이다.


```python
idx = 3
X = X_test[idx].reshape((1,) + X_test[0].shape)
print(X.shape)

```

    (1, 53)



```python
shape = X.shape 
beta = .1 # Weight for L1 regularization term 
gamma = 10 # Weight for autoencoder loss
theta = .1 # Weight for prototype loss
c_init = 1.
c_steps = 5
max_iterations = 500

rng = (-1., 1.) 
rng_shape = (1,) + df_f.shape[1:]
feature_range = ((np.ones(rng_shape) * rng[0]).astype(np.float32), (np.ones(rng_shape) * rng[1]).astype(np.float32))
```


```python
from alibi.explainers import CounterfactualProto

cf = CounterfactualProto(model, shape, beta=beta, 
                        enc_model=enc, ae_model=ae, gamma=gamma, 
                        theta=theta, 
                        cat_vars=cat_vars_ohe, ohe=True, 
                        max_iterations=max_iterations, feature_range=feature_range,
                        c_init=c_init, c_steps=c_steps)

cf.fit(X_train, d_type='abdm')
```






    CounterfactualProto(meta={
      'name': 'CounterfactualProto',
      'type': ['blackbox', 'tensorflow', 'keras'],
      'explanations': ['local'],
      'params': {
                  'kappa': 0.0,
                  'beta': 0.1,
                  'gamma': 10,
                  'theta': 0.1,
                  'cat_vars': {
                                0: 12,
                                12: 3,
                                15: 4,
                                19: 2,
                                21: 2,
                                23: 2,
                                25: 3,
                                28: 12,
                                40: 4,
                                44: 3}
                              ,
                  'ohe': True,
                  'use_kdtree': False,
                  'learning_rate_init': 0.01,
                  'max_iterations': 500,
                  'c_init': 1.0,
                  'c_steps': 5,
                  'eps': (0.001, 0.001),
                  'clip': (-1000.0, 1000.0),
                  'update_num_grad': 1,
                  'write_dir': None,
                  'feature_range': (array([[-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
            -1., -1., -1.]], dtype=float32), array([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]],
          dtype=float32)),
                  'shape': (1, 53),
                  'is_model': True,
                  'is_ae': True,
                  'is_enc': True,
                  'enc_or_kdtree': True,
                  'is_cat': True,
                  'trustscore_kwargs': None,
                  'd_type': 'abdm',
                  'w': None,
                  'disc_perc': (25, 50, 75),
                  'standardize_cat_vars': False,
                  'smooth': 1.0,
                  'center': True,
                  'update_feature_range': True}
                ,
      'version': '0.9.6'}
    )



### 5.2 describe_instance() 함수 정의


```python
from alibi.utils import ohe_to_ord
target_names = ['no', 'yes']

def describe_instance(X, explanation, eps=1e-2):
    print("Original instance: {} -- proba: {}".format(target_names[explanation.orig_class], explanation.orig_proba[0]))
    print("Counterfactual instance: {} -- proba: {}".format(target_names[explanation.cf['class']], explanation.cf['proba'][0]))

    print("\n=====Counterfactual perturbations=====")

    print("\n-----Categorical-----")
    X_orig_ord = ohe_to_ord(X, cat_vars_ohe)[0]
    X_cf_ord = ohe_to_ord(explanation.cf['X'], cat_vars_ohe)[0]

    delta_cat = {}
    for i, (_, v) in enumerate(category_map.items()):
        cat_orig = v[int(X_orig_ord[0, i])]
        cat_cf = v[int(X_cf_ord[0, i])]
        if cat_orig != cat_cf: 
            delta_cat[feature_names[i]] = [cat_orig, cat_cf]
    if delta_cat:
        for k, v in delta_cat.items():
            print(f'ㅁ {k}: {v[0]} ---> {v[1]}')

    print("\n-----Numerical-----")
    delta_num = X_cf_ord[0, -6:] - X_orig_ord[0, -6:]
    n_keys = len(list(cat_vars_ord.keys()))
    for i in range(delta_num.shape[0]):
        if np.abs(delta_num[i]) > eps:
            print('ㅁ {}: {:.2f} ---> {:.2f}'.format(feature_names[i+n_keys], X_orig_ord[0, i+n_keys], X_cf_ord[0, i+n_keys]))
```

### 5.3 특정 표본의 Counterfactual 구하기
이제 표본의 Counterfactual을 구한다. 

- 아래 index 17 고객의 경우, 은행의 마케팅캠페인 결과 정기예금 상품에 가입하지 않을 확률이 0.976으로 예측되었다. 그러나 duration(가장 최근 연락의 통화시간)을 약간 높이면 가입할 확률이 가입하지 않을 확률보다 약간 우세하게(0.501) 변하는 것으로 예상되고 있다.
- 인덱스 30 표본의 경우도 정기예금 상품에 가입하지 않을 확률이 높게(0.999) 예측되었다. 하지만 직업이 실업자가 아닌 은퇴이고 통화시간이 약간 더 길었다면 상품에 가입할 확률이 0.692인 것으로 나타났다. 그러나 직업이 실업자에서 은퇴자로 바뀌는 것은 상식적으로 불가능할 뿐만 아니라 비즈니스(은행)가 영향을 줄 수 없는 부분이다. 이러한 점이 CounterfactualProto 모형의 한계로 작용한다. 


```python
idx = 42
X = X_test[idx].reshape((1,) + X_test[0].shape)

explanation = cf.explain(X)

print(explanation.data.keys())
# print(explanation.data['cf'].keys())
```

    dict_keys(['cf', 'all', 'orig_class', 'orig_proba', 'id_proto'])



```python
describe_instance(X, explanation)
```

    Original instance: no -- proba: [1. 0.]
    Counterfactual instance: yes -- proba: [0.303 0.697]
    
    =====Counterfactual perturbations=====
    
    -----Categorical-----
    ㅁ job: entrepreneur ---> retired
    ㅁ education: secondary ---> primary
    ㅁ contact: telephone ---> cellular
    ㅁ month: jul ---> aug
    ㅁ day_cat: mid ---> late
    
    -----Numerical-----
    ㅁ duration: -0.82 ---> -0.17
    ㅁ campaign: -0.92 ---> -0.96



```python
idx = 10
X = X_test[idx].reshape((1,) + X_test[0].shape)
explanation = cf.explain(X)
describe_instance(X, explanation)
```

    Original instance: no -- proba: [1. 0.]
    Counterfactual instance: yes -- proba: [0.302 0.698]
    
    =====Counterfactual perturbations=====
    
    -----Categorical-----
    ㅁ job: blue-collar ---> retired
    ㅁ education: secondary ---> tertiary
    ㅁ housing: yes ---> no
    
    -----Numerical-----
    ㅁ age: -0.15 ---> 0.05
    ㅁ duration: -0.95 ---> -0.52



```python
idx = 17
X = X_test[idx].reshape((1,) + X_test[0].shape)
explanation = cf.explain(X)
describe_instance(X, explanation)
```

    Original instance: no -- proba: [0.976 0.024]
    Counterfactual instance: yes -- proba: [0.499 0.501]
    
    =====Counterfactual perturbations=====
    
    -----Categorical-----
    
    -----Numerical-----
    ㅁ duration: -0.74 ---> -0.52



```python
idx = 20
X = X_test[idx].reshape((1,) + X_test[0].shape)
explanation = cf.explain(X)
describe_instance(X, explanation)
```

    Original instance: yes -- proba: [0.478 0.522]
    Counterfactual instance: no -- proba: [0.502 0.498]
    
    =====Counterfactual perturbations=====
    
    -----Categorical-----
    ㅁ job: admin. ---> services
    
    -----Numerical-----
    ㅁ duration: -0.92 ---> -0.94



```python
idx = 30
X = X_test[idx].reshape((1,) + X_test[0].shape)
explanation = cf.explain(X)
describe_instance(X, explanation)
```

    Original instance: no -- proba: [0.999 0.001]
    Counterfactual instance: yes -- proba: [0.308 0.692]
    
    =====Counterfactual perturbations=====
    
    -----Categorical-----
    ㅁ job: unemployed ---> retired
    
    -----Numerical-----
    ㅁ duration: -0.95 ---> -0.66



```python
idx = 40
X = X_test[idx].reshape((1,) + X_test[0].shape)
explanation = cf.explain(X)
describe_instance(X, explanation)
```

    Original instance: no -- proba: [0.664 0.336]
    Counterfactual instance: yes -- proba: [0.495 0.505]
    
    =====Counterfactual perturbations=====
    
    -----Categorical-----
    ㅁ job: technician ---> unemployed
    
    -----Numerical-----



