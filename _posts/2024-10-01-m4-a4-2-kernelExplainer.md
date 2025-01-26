---
layout: single
title: "Kernel Explainer"
categories: XAI
sidebar: true
use_math: true
---

# Kernel Explainer on Boston-housing data

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



```python
df = pd.read_csv('../materials/section3_rev/Boston.csv', index_col=0)
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



## 1. 전처리


```python
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import make_pipeline

X = df.iloc[:,:-1]
y = df['PRICE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

model_linear = LinearRegression().fit(X_train_std, y_train)
```

Kernel Explainer는 특성변수의 수가 많아지면서 계산부담이 기하급수적으로 증가하는 shap.Explainer의 문제를 극복하기 위해 등장하였다. 아래 프로그램은 shap.kmeans 함수값을 KernelExplainer의 data 파라미터로 주었는데, 이렇게 하면 전체 학습데이터를 사용하는 것보다 연산 속도가 빠르면서 SHAP value를 계산함에 있어 표본의 임의성을 최소화 할 수 있다. 


```python
explainer_k = shap.KernelExplainer(model_linear.predict, shap.kmeans(X_train_std, 100)) 

shap_values_k = explainer_k.shap_values(X_test_std)
expected_k = explainer_k.expected_value

print(expected_k)
print(shap_values_k.shape)
```

    100%|██████████| 152/152 [00:59<00:00,  2.54it/s]

    23.01500216449717
    (152, 13)


    


## 2. Kernel Explainer: Global

### 2.1 summary plot
아래 summary plot은 Kernel Explainer를 이용하여 표준화된 학습데이터에서 계산한 SHAP value를 보여준다. 그러므로 이 값들은 Global Explainer이다. max_display=10으로 지정하여 SHAP value의 절대값 평균이 큰(즉, 중요도가 높은) 순서로 특성변수를 10개 보여주고 있다. 앞서 shap.Explainer로 구한 특성변수 중요도와 같은 순서로 나타났다.  


```python
feature_names = X_test.columns.to_list()
shap.summary_plot(shap_values_k, features=X_test_std, feature_names=feature_names, max_display=10)
# feature_names를 별도로 지정하지 않으면 features로 들어간 데이터셋이 pandas인 경우에 한해 features에 지정된 특성변수명을 사용함 
```


    
![png](/images/m4/a4_2_kernelExplainer/output_11_0.png)
    


### 2.2 dependence plot


```python
X_test.head()
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>173</th>
      <td>0.09178</td>
      <td>0.0</td>
      <td>4.05</td>
      <td>0.0</td>
      <td>0.510</td>
      <td>6.416</td>
      <td>84.1</td>
      <td>2.6463</td>
      <td>5.0</td>
      <td>296.0</td>
      <td>16.6</td>
      <td>395.50</td>
      <td>9.04</td>
    </tr>
    <tr>
      <th>274</th>
      <td>0.05644</td>
      <td>40.0</td>
      <td>6.41</td>
      <td>1.0</td>
      <td>0.447</td>
      <td>6.758</td>
      <td>32.9</td>
      <td>4.0776</td>
      <td>4.0</td>
      <td>254.0</td>
      <td>17.6</td>
      <td>396.90</td>
      <td>3.53</td>
    </tr>
    <tr>
      <th>491</th>
      <td>0.10574</td>
      <td>0.0</td>
      <td>27.74</td>
      <td>0.0</td>
      <td>0.609</td>
      <td>5.983</td>
      <td>98.8</td>
      <td>1.8681</td>
      <td>4.0</td>
      <td>711.0</td>
      <td>20.1</td>
      <td>390.11</td>
      <td>18.07</td>
    </tr>
    <tr>
      <th>72</th>
      <td>0.09164</td>
      <td>0.0</td>
      <td>10.81</td>
      <td>0.0</td>
      <td>0.413</td>
      <td>6.065</td>
      <td>7.8</td>
      <td>5.2873</td>
      <td>4.0</td>
      <td>305.0</td>
      <td>19.2</td>
      <td>390.91</td>
      <td>5.52</td>
    </tr>
    <tr>
      <th>452</th>
      <td>5.09017</td>
      <td>0.0</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.713</td>
      <td>6.297</td>
      <td>91.8</td>
      <td>2.3682</td>
      <td>24.0</td>
      <td>666.0</td>
      <td>20.2</td>
      <td>385.09</td>
      <td>17.27</td>
    </tr>
  </tbody>
</table>
</div>




```python
shap.dependence_plot('PTRATIO', shap_values_k, features=X_test, interaction_index=None)
```


    
![png](/images/m4/a4_2_kernelExplainer/output_14_0.png)
    


## 3. Kernel Explainer: Local
### 3.1 한 표본의 Local Explainer
force plot은 특정 표본의 shap value를 보여준다. 


```python
shap.force_plot(expected_k, shap_values_k[0, :], features=X_test.iloc[0])
```



![png](/images/m4/a4_2_kernelExplainer/kernel_g.png)
    




단, 위의 force plot은 특성변수값(예: RM=6.416)만 보여주고 shap values를 보여주지 않는다. shap values는 다음과 같이 확인할 수 있다.
- LSTAT의 shap value는 1.851이다.


```python
print(X_test.iloc[0])
print(shap_values_k[0, :])
```

    CRIM         0.09178
    ZN           0.00000
    INDUS        4.05000
    CHAS         0.00000
    NOX          0.51000
    RM           6.41600
    AGE         84.10000
    DIS          2.64630
    RAD          5.00000
    TAX        296.00000
    PTRATIO     16.60000
    B          395.50000
    LSTAT        9.04000
    Name: 173, dtype: float64
    [ 0.452 -0.402 -0.35  -0.229  0.73   0.359 -0.166  1.552 -1.081  0.968
      1.528  0.421  1.851]


### 3.2 여러 표본의 Local Explainer
force plot으로 2개 이상 표본을 시각화 할 수 있다. 다음은 force plot으로 학습데이터 전체를 시각화 한 결과이다. 
- 학습데이터 내 모든 표본의 shap value를 하나의 figure로 시각화 하고 sample order by output value를 클릭하였다.


```python
shap.force_plot(expected_k, shap_values_k, features = X_test, feature_names=feature_names)
```



![png](/images/m4/a4_2_kernelExplainer/kernel_l.png)
    
출력 결과의 dropdown box에서 가로축은 DIS, 세로축은 DIS effects를 클릭하였다. 세로축에서 주택가격 예측치의 기댓값이 23.02임을 확인할 수 있다. 한편 DIS가 약 3.7일 때를 기준으로 이보다 작으면(즉 업무지구와 거리가 가까우면) 주택가격 예측치에 긍정적(+) 영향을 주고, 이보다 크면(업무지구와 거리가 멀면) 부정적(-) 영향을 주는 것을 볼 수 있다. 
