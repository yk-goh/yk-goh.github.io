---
layout: single
title: "Partition Explainer"
categories: XAI
sidebar: true
use_math: true
---

# Partition Explainer on Boston Housing data

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



## 1. 전처리


```python
X = df.iloc[:,:-1]
y = df['PRICE']

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)
```

## 2. Partition Explainer

특성변수가 아주 많은 이미지 데이터나 텍스트 데이터의 SHAP value를 구할 때에는 계산부담이 매우 크다. 계산부담을 줄이기 위해 특성변수를 그룹화 하여 coalition의 수를 줄이는 방식이 가능하고, shap 라이브러리에서는 PartitionExplainer가 이러한 방식을 지원한다. 그룹화된 coalition을 기반으로 계산된 SHAP value를 Owen value라고 한다. 
shap 라이브러리에서는 Owen value의 토큰 단위를 masker로 조절한다. (일반적으로 언어모형의 입력 단위인 토큰은 단어(word) 단위가 아닌데, 사람이 SHAP(Owen) value를 이해하려면 단위가 단어여야 하기 때문이다. )
masker는 SHAP value 연산 시 특성변수를 임시로 없애는 역할을 한다. 특성변수를 마스킹 함으로써 해당 특성변수가 모델 예측에 갖는 영향력을 측정할 수 있다. 

아래 프로그램은 LGBM 모형에 학습데이터를 적합하고, shap.maskers.Partition 클래스를 이용하여 학습데이터로부터 마스킹을 생성한다. 클러스터링 방법은 기본값인 ‘correlation’으로 두었으며, Tabular 데이터이므로 적절한 방법으로 보인다. PartitionExplainer를 인스턴스화 할 때에는 masker 파라미터를 반드시 지정해야 한다. (사실 아래에서 사용하는 모형이 tree-based model이라는 점, 사용하는 데이터셋이 표본 수 506개, 특성변수 13개인 간단한 Tabular data라는 점 때문에 masker의 필요성은 낮다고 생각된다.)  


```python
import lightgbm as lgb

model_lgbm = lgb.LGBMRegressor()
model_lgbm.fit(X_train, y_train)

masker = shap.maskers.Partition(X_train)  # clustering = 'correlation' by default
explainer_partition = shap.PartitionExplainer(model_lgbm.predict, masker=masker)

```

    [LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.011103 seconds.
    You can set `force_row_wise=true` to remove the overhead.
    And if memory is not enough, you can set `force_col_wise=true`.
    [LightGBM] [Info] Total Bins 898
    [LightGBM] [Info] Number of data points in the train set: 354, number of used features: 13
    [LightGBM] [Info] Start training from score 23.015819
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
   


PartitionExplainer의 explainer는 .values 속성은 Owen value를 나타내며 .base_values 속성은 기댓값을 나타낸다(정의 상 모든 표본에 동일한 값이다). 따라서 특정 표본에 대한 Local explainer를 시각화 할 때에는 인덱스를 지정해야 한다. 


```python
of_partition = explainer_partition(X_test)
print(of_partition)

shap_values_partition = of_partition.values
expected_partition = of_partition.base_values

print(shap_values_partition.shape)
print(expected_partition.shape)
```

    PartitionExplainer explainer: 153it [00:11,  1.35it/s]                         

    .values =
    array([[ 0.299, -0.013,  0.573, ...,  0.514,  0.202,  0.155],
           [-0.228,  0.051, -0.2  , ..., -0.104,  0.065, 10.557],
           [ 0.458, -0.017, -0.207, ..., -0.777,  0.238, -4.671],
           ...,
           [ 0.251, -0.016, -0.29 , ..., -0.1  ,  0.611,  4.899],
           [ 0.505, -0.015, -0.425, ..., -0.562, -1.243, -4.532],
           [ 0.001, -0.013, -0.337, ..., -0.565,  0.251, -5.081]])
    
    .base_values =
    array([24.635, 24.635, 24.635, 24.635, 24.635, 24.635, 24.635, 24.635,
           24.635, 24.635, 24.635, 24.635, 24.635, 24.635, 24.635, 24.635,
           24.635, 24.635, 24.635, 24.635, 24.635, 24.635, 24.635, 24.635,
           24.635, 24.635, 24.635, 24.635, 24.635, 24.635, 24.635, 24.635,
           24.635, 24.635, 24.635, 24.635, 24.635, 24.635, 24.635, 24.635,
           24.635, 24.635, 24.635, 24.635, 24.635, 24.635, 24.635, 24.635,
           24.635, 24.635, 24.635, 24.635, 24.635, 24.635, 24.635, 24.635,
           24.635, 24.635, 24.635, 24.635, 24.635, 24.635, 24.635, 24.635,
           24.635, 24.635, 24.635, 24.635, 24.635, 24.635, 24.635, 24.635,
           24.635, 24.635, 24.635, 24.635, 24.635, 24.635, 24.635, 24.635,
           24.635, 24.635, 24.635, 24.635, 24.635, 24.635, 24.635, 24.635,
           24.635, 24.635, 24.635, 24.635, 24.635, 24.635, 24.635, 24.635,
           24.635, 24.635, 24.635, 24.635, 24.635, 24.635, 24.635, 24.635,
           24.635, 24.635, 24.635, 24.635, 24.635, 24.635, 24.635, 24.635,
           24.635, 24.635, 24.635, 24.635, 24.635, 24.635, 24.635, 24.635,
           24.635, 24.635, 24.635, 24.635, 24.635, 24.635, 24.635, 24.635,
           24.635, 24.635, 24.635, 24.635, 24.635, 24.635, 24.635, 24.635,
           24.635, 24.635, 24.635, 24.635, 24.635, 24.635, 24.635, 24.635,
           24.635, 24.635, 24.635, 24.635, 24.635, 24.635, 24.635, 24.635])
    
    .data =
    array([[  0.092,   0.   ,   4.05 , ...,  16.6  , 395.5  ,   9.04 ],
           [  0.056,  40.   ,   6.41 , ...,  17.6  , 396.9  ,   3.53 ],
           [  0.106,   0.   ,  27.74 , ...,  20.1  , 390.11 ,  18.07 ],
           ...,
           [  0.527,   0.   ,   6.2  , ...,  17.4  , 382.   ,   4.63 ],
           [  5.581,   0.   ,  18.1  , ...,  20.2  , 100.19 ,  16.22 ],
           [  9.925,   0.   ,  18.1  , ...,  20.2  , 388.52 ,  16.44 ]])
    (152, 13)
    (152,)


    


## 3. Partition Explainer: Global
shap 라이브러리에는 plots 모듈이 있는데, 여기 있는 함수 중 일부는 Explanation 클래스에 속하는 객체만 받기 때문에 shap.Explainer로 구현한 객체에만 적용할 수 있다. Kernel, Tree, Deep, Gradient, Partition Explainer에는 적용할 수 있는 플롯이 한계가 있다.


```python
shap.summary_plot(shap_values_partition, features=X_test, feature_names=feature_names, max_display=10)
```


    
![png](/images/m4/a4_2_partitionExplainer/output_13_0.png)
    



```python
shap.plots.violin(shap_values_partition, plot_type='layered_violin')
```



    
![png](/images/m4/a4_2_partitionExplainer/output_14_1.png)
    


## 4. Partition Explainer: Local


```python
shap.force_plot(expected_partition[0], shap_values_partition[0], features=X_test.iloc[0, :])
```




![png](/images/m4/a4_2_partitionExplainer/partition_l.png)



앞서 Deep Explainer, Gradient Explainer에서 살펴본 학습데이터의 인덱스 0 표본에 대하여 Partition Explainer로 구한 Owen value를 살펴보면 위 그림과 같다. 기댓값은 24.63으로 두 explainer와 유사하다. 그러나 예측치는 25.23으로 Deep Explainer(29.13), Gradient Explainer(29.08)와 차이가 있다. 또한 특성변수의 중요도와 예측치에 대한 방향이 두 explainer와 다르다. 예를 들어 특성변수 AGE의 경우 Deep/Gradient explainer에서 구한 SHAP value는 양(+)의 값을 가지나 Partition explainer에서 구한 Owen value는 음(-)의 값을 가진다. 특성변수 RM의 경우 Deep/Gradient explainer에서 구한 SHAP value의 절댓값이 아주 작아 force plot에 레이블이 생략되었지만 Partition explainer로 구한 Owen value로는 절댓값이 가장 크다. 

이렇듯 Explainer마다 SHAP(Owen) value의 크기(특성변수 중요도)와 방향이 다르게 나타나므로 여러 Explainer를 비교하고 설명하고자 하는 모형에 적합한 것을 사용해야 할 것이다. 
