---
layout: single
title: "Shap Explainer"
categories: XAI
sidebar: true
use_math: true
---

# Shap Explainer on Boston-housing data

## 0. import libraries and data


```python
# %pip install shap
```


```python
import pandas as pd
import numpy as np
import shap 
from sklearn.linear_model import LinearRegression

# scientific notation(자연상수 e를 사용해서 표현하는 것)을 사용하지 않고, 소수점 아래 세 자리까지 표기한다.
np.set_printoptions(suppress=True, precision=3)
# shap 라이브러리 시각화를 위한 자바스크립트 활성화 
shap.initjs()

```




## 1. 전처리

Boston 데이터셋을 pandas DataFrame으로 불러오고 학습데이터와 시험데이터로 나눈다. 한편 Linear Regression 모형을 사용할 예정이므로 특성변수를 표준화 하는 StandardScaler와 LinearRegression을 묶어 pipeline을 생성한다. 


```python
df = pd.read_csv('/Users/ykgoh/Documents/Module 4/XAI/materials/section3_rev/Boston.csv', index_col=0)
df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>PRICE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1.0</td>
      <td>296.0</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
      <td>21.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
      <td>34.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>394.63</td>
      <td>2.94</td>
      <td>33.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>396.90</td>
      <td>5.33</td>
      <td>36.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
X = df.iloc[:,:-1]
y = df['PRICE']
```


```python
# Shap value를 계산할 background data 
X_bg = X[100:200]
```


```python
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import make_pipeline

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)
print(X_train.shape)

pipeline = make_pipeline(StandardScaler(), LinearRegression())
pipeline.fit(X_train, y_train)
```

    (354, 13)




## 2. Global Explainer


```python
explainer_ex = shap.Explainer(pipeline.predict, X_train) # data 1
shap_values_ex = explainer_ex(X_test) # data 2 

shap_values_ex.shape
```




    (152, 13)



### 2.1 회귀계수의 부호와 shap value 평균의 관계
적합된 선형회귀 모형의 회귀계수를 살펴보면 음(-)의 값을 갖는 특성변수는 CRIM, NOX, AGE, DIS, TAX, PTRATIO, LSTAT이고 양(+)의 값을 갖는 특성변수는 ZN, INDUS, CHAS, RM, RAD, B임을 확인할 수 있다. 


```python
lm = pipeline.named_steps['linearregression']
coef = lm.coef_

print(X_test.columns)
print(coef)

```

    Index(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
           'PTRATIO', 'B', 'LSTAT'],
          dtype='object')
    [-1.108  0.808  0.343  0.814 -1.798  2.914 -0.299 -2.943  2.094 -1.447
     -2.052  1.024 -3.886]


회귀 계수를 살펴보면, 

<span style="color:blue">(-) CRIM, NOX, AGE, DIS, TAX, PTRATIO, LSTAT </span> <span style="color:red">(+) ZN, INDUS, CHAS, RM, RAD, B</span>



그런데 shap.Explainer로 구한 SHAP value를 bar 차트로 시각화 한 결과를 살펴보면 다음과 같다. 아래는 Global Explainer로, 왼쪽 차트의 경우 SHAP value 절댓값의 평균이므로 여러 표본에 대한 기여도가 상쇄되지 않으며 오른쪽 차트의 경우 SHAP value의 단순평균이라 각 특성변수의 여러 표본에 대한 기여도가 상쇄되어 나타난다. 따라서 왼쪽 차트로부터 특성변수 중요도를 파악할 수 있으며, LSTAT(주거지구 내 저소득층 비율), DIS(주요 업무지구까지 거리), RM(평균 방 개수) 순으로 중요함을 알 수 있다. 
- 주택가격 예측치에 RM은 평균적으로 -0.85만큼 기여하였음 : 회귀계수 부호와 불일치
    - RM의 회귀계수는 2.914인데 bar chart에서 mean(SHAP value)는 -0.85로 나타남
- RM의 shap value가 `음수(-)`인 표본이 `양수(+)`인 표본보다 많기 때문에 RM 특성변수의 shap value를 단순평균 했을 때에는 그 부호(-0.85)가 회귀계수와 반대로 나온다.
- 아래 beeswarm plot을 보면, RM의 경우 `shap value < 0`인 표본이 많음을 확인할 수 있다. 


```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2)
plt.sca(axes[0]) #set the current axis
"""절댓값의 평균이므로 여러 표본에 대한 기여도가 +/- 상쇄되지 않는다"""
shap.plots.bar(shap_values_ex, show=False)

plt.sca(axes[1])
"""shap value들의 단순평균이라 각 특성변수의 기여도가 상쇄된다"""
shap.plots.bar(shap_values_ex.mean(axis=0), show=False)
plt.tight_layout() # subplot이 겹치지 않도록 조정 
plt.show()
```


    
![png](/images/m4/a4_2_shapExplainer/output_16_0.png)
    


한편, 우측의 mean(SHAP value) 플롯에 의하여 주택가격 예측치에 RM(평균 방 개수) 특성변수가 평균적으로 -0.85만큼 기여한 것으로 나타나, 회귀계수(2.914) 부호와 일치하지 않음을 발견하였다. 이러한 현상을 이해하기 위해 전체적인 분포를 보여주는 beeswarm 플롯을 시각화 한다. shap.plots의 beeswarm 함수는 Explanation 클래스의 객체만 받기 때문에 shap.Explainer로 SHAP value를 구했을 때에만 사용할 수 있다. 

아래 플롯을 보면, RM의 SHAP value가 음수(-)인 표본이 양수(+)인 표본보다 많음을 알 수 있다. 이러한 이유로 특성변수 RM의 SHAP value를 단순평균 했을 때에는 그(-0.85) 부호가 회귀계수와 반대로 나왔다. 



```python
shap.plots.beeswarm(shap_values_ex)
```


    
![png](/images/m4/a4_2_shapExplainer/output_18_0.png)
    


### 2.2 방향성과 선형관계 확인
다음으로 scatter plot을 그려 특성변수가 예측치에 영향을 주는 방향을 확인하려 한다. Boston housing 데이터셋의 LSTAT 변수는 표본(주거지구) 내 저소득층 비율을 의미한다. 앞서 선형회귀 모형의 coef_ 속성을 통해 LSTAT의 회귀계수가 -3.886임을 확인한 바 있다(즉, 주거지구 내 저소득층 비율이 높을수록 해당 주거지구의 평균 주택가격이 낮다). LSTAT과 LSTAT의 SHAP value를 X, Y 축으로 하는 scatter plot을 그려 확인했을 때 LSTAT 값이 클수록 SHAP value가 작아, 회귀계수와 방향성이 일치함을 알 수 있다. 또한 선형회귀 모형을 사용했기 때문에 나타나는 선형성도 scatter plot에서 확인할 수 있다. 


```python
shap.plots.scatter(shap_values_ex[:, 'LSTAT'])

```


    
![png](/images/m4/a4_2_shapExplainer/output_20_0.png)
    



```python
shap.partial_dependence_plot("RM", pipeline.predict, X_train, ice=False, model_expected_value=True, feature_expected_value=True)
```


    
![png](/images/m4/a4_2_shapExplainer/output_21_0.png)
    


\[상단 scatter plot\]
- LSTAT은 회귀계수가 -3.886이다. LSTAT 값이 클수록 shap value 값이 작아, 회귀계수와 방향성이 일치함을 확인할 수 있다. 
- 또한 선형성도 확인할 수 있다

\[partial dependence plot\]
- 회귀에서만 사용할 수 있음(분류 모형에서는 사용 불가)
- shap value로 PD를 계산한 것임. X_train 데이터셋으로부터 shap value를 계산하였음
- PD란.. 특성변수 간 독립성을 가정하고 $x_i$를 제외한 특성변수를 임의표본으로 대체하여 $x_{-i}$의 효과를 제거하는 것

## 3. Local Explainer
다음으로 특정한 표본 하나의 SHAP value를 시각화 한다. 아래는 SHAP value를 구한 시험데이터 중 인덱스 1 표본의 주택가격 예측치에 각 특성변수값이 어떻게 기여했는지 보여주는 waterfall plot이다. 이 표본의 LSTAT 값은 3.53으로 전체 분포에서 상당히 낮은 편인데, 이러한 특성이 주택가격을 기댓값(24.269)보다 높게 끌어올리는 데에 주요한 역할을 하였다. 또한 해당 주거지구가 찰스 강가에 위치한 것도(CHAS=1) 주택가격 예측치를 높이는 데 기여하였다. 모든 특성변수값을 종합했을 때, 이 표본의 주택가격은 기댓값보다 12 이상 높은 36.495로 예측 되었다. 


```python
shap.plots.waterfall(shap_values_ex[1 , :])
```


    
![png](/images/m4/a4_2_shapExplainer/output_24_0.png)
    


다른 방식으로 확인할 수 있다.


```python
index=1
print("data of given X: \n", shap_values_ex.data[index])
print("base value: \n", shap_values_ex.base_values[index]) # 사실 인덱스에 상관없이 모두 같은 값임 
print("shap values: \n", shap_values_ex.values[index])
shap_values_sum = np.sum(shap_values_ex.values[index])
print('shap values sum: \n', shap_values_sum)
print("====================================================================")
pred = pipeline.predict(X_train[index:index+1])
print("prediction: ", pred)
print("baseline + shap values sum: ", shap_values_ex.base_values[0] + shap_values_sum)

```

    data of given X: 
     [  0.056  40.      6.41    1.      0.447   6.758  32.9     4.078   4.
     254.     17.6   396.9     3.53 ]
    base value: 
     24.269217332919638
    shap values: 
     [ 0.464  0.984 -0.234  2.777  1.639  1.462  0.38  -0.675 -1.272  1.273
      0.406  0.647  4.375]
    shap values sum: 
     12.225796510567347
    ====================================================================
    prediction:  [23.703]
    baseline + shap values sum:  36.49501384348699


아래 force plot은 목적변수에 대한 상방압력과 하방압력을 시각화한다.


```python
shap.plots.force(shap_values_ex[1])
```


![png](/images/m4/a4_2_shapExplainer/shap_l.png)
    



