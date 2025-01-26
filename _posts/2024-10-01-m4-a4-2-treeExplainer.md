---
layout: single
title: "Tree Explainer"
categories: XAI
sidebar: true
use_math: true
---


# Tree Explainer on Boston-housing data

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
from sklearn.model_selection import train_test_split 
X = df.iloc[:,:-1]
y = df['PRICE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)
```

## 2. Tree Explainer
Tree Explainer는 decision tree 모형에 특화된 shap explainer이다. 여러 의사결정나무에서 구한 shap value들을 평균하여 특성변수의 shap value로 정의한다.

Tree Explainer를 인스턴스화 할 때 모형과 함께 데이터를 지정하면 특성변수가 독립이라는 가정 아래 $E(f(x)\mid S)$를 추정하는 intervention 방식으로 작동한다. 따라서 아래 프로그램에서는 TreeExplainer의 파라미터에 모형만 지정함으로써 특성변수의 독립성 가정을 하지 않고 주어진 coalition의 tree-path만으로 $E(f(x)\mid S)$를 추정하는 tree_path_dependent 방식으로 TreeExplainer를 인스턴스화 하였다. 

Tree explainer와 Deep explainer는 shap.Explainer, KernelExplainer처럼 model.predict, model.predict_proba를 사용할 수 없으므로 shap value의 axiom 중 하나인 additivity를 위배하는 경우가 드물게 있다. 이를 무시하기 위해 check_additivity=False를 부여하여 에러를 방지한다
 - additivity: prediction = baseline(expected) + sum(shap values)인 성질


```python
from sklearn.ensemble import RandomForestRegressor

model_tree = RandomForestRegressor()
model_tree.fit(X_train, y_train)
```





```python
explainer_tree = shap.TreeExplainer(model_tree)
shap_values_tree = explainer_tree.shap_values(X_test, check_additivity=False)
print(shap_values_tree.shape)
```

    (152, 13)


## 3. Tree Explainer: beeswarm plot

X_test에서 구한 shap value 분포를 보여준다. 중요도가 큰 특성변수가 상단에 위치한다


```python
shap.summary_plot(shap_values_tree, features=X_test)
```


    
![png](/images/m4/a4_2_treeExplainer/output_12_0.png)
    


plot_type에 bar를 지정하면 특성변수 별 shap value의 절대값 평균을 구하고 바 차트로 나타낸다


```python
shap.summary_plot(shap_values_tree, features=X_test, plot_type='bar')
```


    
![png](/images/m4/a4_2_treeExplainer/output_14_0.png)
    


## 4. Tree Explainer: Dependence plot
다음은 특성변수 RM의 dependence plot이다. 사용한 모형이 선형이 아니므로 특성변수값의 변화에 따라 shap value가 선형으로 증감하지 않는다


```python
shap.dependence_plot("RM", shap_values_tree, features=X_test, interaction_index=None)
```


    
![png](/images/m4/a4_2_treeExplainer/output_16_0.png)
    


dependence plot을 사용하여 특성변수의 교호작용을 살펴본다. 

먼저 RM과 AGE(주거지구 내, 1940년 이전에 지은 주택의 비율)의 dependence plot을 보면 방 개수가 6개 내외일 때, 주택의 연식(AGE)이 짧을수록 RM의 SHAP value가 낮아 주택가격 예측치가 낮다. 이는 상식에 위배되며 summary plot으로 구한 global explainer 결과와도 상반된다. RM과 AGE 사이 confounding factor가 존재할 가능성이 있다. 


```python
shap.dependence_plot("RM", shap_values_tree, features=X_test, interaction_index=6)
# interaction index=6에 해당하는 것이 LSTAT 변수이다 
```


    
![png](/images/m4/a4_2_treeExplainer/output_18_0.png)
    


아래 for loop은 13개 특성변수에 대하여 가장 교호작용이 뚜렷한 변수와 그 시각화 결과를 제공한다. 

- `CRIM ~ LSTAT`: 전반적으로 저소득층 비율이 낮은 주거지구가 범죄율 또한 낮다. 두 변수가 상관성을 보이나 교호작용은 딱히 드러나지 않는다. 한편 LSTAT이 15 이상일 때 범죄율이 높을수록 LSTAT의 shap value가 낮아 주택가격 예측치에 음(-)의 방향으로 기여하고 있다. 
- `AGE ~ LSTAT`: 전반적으로 연식이 짧은 주택은 저소득층 비율이 낮은 주거지구에 분포하고 연식이 긴 주택은 저소득층 비율이 높은 주거지구에 분포함을 알 수 있다. 두 변수의 교호작용은 드러나지 않으며, 다만 상관성이 높은 것으로 보인다.


```python
for name in X_train.columns:
    shap.dependence_plot(name, shap_values_tree, X_test) 
    # index 12 : LSTAT: % lower status of the population
```


    
![png](/images/m4/a4_2_treeExplainer/output_20_0.png)
    



    
![png](/images/m4/a4_2_treeExplainer/output_20_1.png)
    



    
![png](/images/m4/a4_2_treeExplainer/output_20_2.png)
    



    
![png](/images/m4/a4_2_treeExplainer/output_20_3.png)
    



    
![png](/images/m4/a4_2_treeExplainer/output_20_4.png)
    



    
![png](/images/m4/a4_2_treeExplainer/output_20_5.png)
    



    
![png](/images/m4/a4_2_treeExplainer/output_20_6.png)
    



    
![png](/images/m4/a4_2_treeExplainer/output_20_7.png)
    



    
![png](/images/m4/a4_2_treeExplainer/output_20_8.png)
    



    
![png](/images/m4/a4_2_treeExplainer/output_20_9.png)
    



    
![png](/images/m4/a4_2_treeExplainer/output_20_10.png)
    



    
![png](/images/m4/a4_2_treeExplainer/output_20_11.png)
    



    
![png](/images/m4/a4_2_treeExplainer/output_20_12.png)
    




