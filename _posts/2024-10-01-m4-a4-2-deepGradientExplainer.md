---
layout: single
title: "Deep Explainer and Gradient Explainer"
categories: XAI
sidebar: true
use_math: true
---


# Deep Explainer and Gradient Explainer on Boston-housing data

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
feature_names = X_test.columns.to_list()
```

## 2. Deep Explainer
Deep Explainer는 딥러닝 모형에 특화된 알고리즘을 이용하여 SHAP value를 추정한다. 이는 입력 텐서가 서로 독립이라고 가정하며, 딥러닝 모형의 모든 은닉층이 입력 텐서를 선형결합만 한다고 가정한다는 한계가 있다(활성함수에 의한 비선형결합을 고려하지 못한다). 

DeepExplainer를 사용하기 위해 먼저 간단한 딥러닝 모형을 만든다. 이 모형으로 DeepExplainer, GradientExplainer의 SHAP value를 각각 구할 예정이다. 



```python
import tensorflow as tf
from tensorflow.keras import layers, models

input_shape = X_train.shape[1:]

inputs = tf.keras.Input(shape=input_shape)
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(32, activation='relu')(x)
outputs = layers.Dense(1)(x)

model_dl = models.Model(inputs=inputs, outputs=outputs)
model_dl.compile(optimizer='adam', loss='mse', metrics=['mae'])

model_dl.summary()
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "functional_4"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer_4 (<span style="color: #0087ff; text-decoration-color: #0087ff">InputLayer</span>)      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">13</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_12 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)             │           <span style="color: #00af00; text-decoration-color: #00af00">896</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_13 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)             │         <span style="color: #00af00; text-decoration-color: #00af00">2,080</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_14 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)              │            <span style="color: #00af00; text-decoration-color: #00af00">33</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">3,009</span> (11.75 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">3,009</span> (11.75 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



모형의 성능을 판단하는 metric을 mean absolute error로 정의하고 시험데이터셋을 validation data로 지정해 손실함수값과 MAE를 구한다. mae=5.2746으로, 약간 과대적합이 일어났다. 


```python
model_dl.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=16, epochs=30)
```

    Epoch 1/30
    [1m23/23[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 3ms/step - loss: 2051.5322 - mae: 36.6595 - val_loss: 253.1929 - val_mae: 14.0662
    ...
    Epoch 30/30
    [1m23/23[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 907us/step - loss: 36.6940 - mae: 4.5424 - val_loss: 37.8127 - val_mae: 5.0780





    <keras.src.callbacks.history.History at 0x33a6c2490>



한편 앞서 Kernel Expaliner를 적용할 때 생성한 선형회귀(LinearRegression) 모형과 딥러닝 모형의 MAE를 비교하면, 선형회귀 모형보다(MAE=3.163) 성능이 낮음을 확인할 수 있다. 딥러닝 모형에 은닉층을 추가하고 Dropout 층을 추가함으로써 성능을 개선하고 과대적합을 완화할 수 있을 테지만 이 과제에서는 우선 DeepExplainer 적용으로 넘어가기로 한다.


```python
from sklearn.metrics import mean_absolute_error
y_pred_linear = model_linear.predict(X_test_std)
mae_linear = mean_absolute_error(y_test, y_pred_linear)
print("MAE of linear regression model: ", mae_linear)
```

    MAE of linear regression model:  3.1627098714574053


## 2.1 Deep Explainer: Global

아래 프로그램은 Deep Explainer를 구할 background 데이터로 X_train에서 랜덤하게 뽑은 100개의 표본을 지정한다. 한편, 딥러닝 모형으로부터 구한 SHAP value의 shape은 (표본수, 특성변수 수, 예측함수의 출력 dimension)이다. 그런데 이 딥러닝 모형이 회귀 모형이므로 출력 dimension(마지막 MLP층의 노드 수)은 1이다. 즉 SHAP value의 shape이 (152, 13, 1)로 주어졌으므로 (152, 13)으로 reshape 해야 한다. 


```python
X_train = X_train.to_numpy()
background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
```


```python
explainer_deep = shap.DeepExplainer(model=model_dl, data=background)
shap_values_deep = explainer_deep.shap_values(X_test.to_numpy())
print(shap_values_deep.shape)
# (152, 13, 1) 표본 수, 특성변수 수, 딥러닝 모형에서 마지막 MLP층의 노드 수

shap_values_deep = shap_values_deep.reshape(152, -1)
print(shap_values_deep.shape)
```

    (152, 13, 1)
    (152, 13)


    /opt/miniconda3/envs/env_xai_pthn39/lib/python3.9/site-packages/shap/explainers/_deep/deep_tf.py:99: UserWarning: Your TensorFlow version is newer than 2.4.0 and so graph support has been removed in eager mode and some static graphs may not be supported. See PR #1483 for discussion.
      warnings.warn("Your TensorFlow version is newer than 2.4.0 and so graph support has been removed in eager mode and some static graphs may not be supported. See PR #1483 for discussion.")



```python
shap.summary_plot(shap_values_deep, features=X_test, feature_names=feature_names, max_display=10)
```


    
![png](/images/m4/a4_2_deepGradientExplainer/output_17_0.png)
    


## 2.2 Deep Explainer: Local

다음은 표본 하나의 force plot으로, Local Explainer에 해당한다. 기댓값(평균)은 24.61이고 LSTAT, B, INDUS 변수가 주택가격 예측치를 상승시키는 데에 중요한 기여를 하고 있다. 예측치 하락에는 ZN, TAX 변수가 주요하게 작용하였다. 예측치를 상승/하락시키는 요인을 종합하였을 때, 이 표본의 주택가격 예측치는 29.13이다. 


```python
expected_deep = explainer_deep.expected_value.numpy()

idx = 0

print(expected_deep)
print(shap_values_deep.shape)

shap.force_plot(expected_deep[0], shap_values_deep[idx], features=X_test.iloc[0, :])
```

    [24.608]
    (152, 13)



![png](/images/m4/a4_2_deepGradientExplainer/deep_2_2.png)





## 3. Gradient Explainer

Gradient Explainer는 Integrated Gradients가 확장된 Expected Gradients를 사용해 모형을 설명한다. IG는 하나의 baseline에서 특성변수값까지의 구간을 등분하여 각 point에서 구한 미분값(gradient)을 적분하는 방식이다. 이렇게 하면 비선형 활성함수에 의한 기여도 왜곡을 처리할 수 있다. 그렇지만 IG는 엄밀히는 SHAP value가 아니다. IG가 SHAP value를 근사하도록 수정한 것이 Expected Gradients이다. Expected Gradients는 background data에서 샘플링한 여러 기준값을 적용하고, 각 기준값(baseline)에서의 IG를 계산하여 그 값들의 평균으로 구한다. 딥러닝 모형에서 활성함수에 의한 비선형 결합을 고려하지 못하는 Deep Explainer와 달리 Gradient Explainer는 모형에 비선형 결합이 있어도 기여도의 왜곡 없이 SHAP value를 계산할 수 있기 때문에 두 Explainer를 함께 살펴봐야 한다. 


```python
explainer_ge = shap.GradientExplainer(model_dl, background)

shap_values_ge = explainer_ge.shap_values(X_test.to_numpy())
print(shap_values_ge.shape)
# (152, 13, 1) 표본 수, 특성변수 수, 딥러닝 모형에서 마지막 MLP층의 노드 수

shap_values_ge = shap_values_ge.reshape(152, -1)
print(shap_values_ge.shape)
```

    (152, 13, 1)
    (152, 13)


DeepExplainer를 사용하여 단일 표본(X_test의 인덱스 0)의 local explainer를 구현한 것처럼, GradientExplainer를 사용하여 동일한 표본의 explainer를 구현하고 시각화 하였다. GradientExplainer 객체에는 expected_value를 구하는 속성이 없기 때문에 별도로 계산해야 한다.

Deep Explainer와 Gradient Explainer로 구한 기댓값과 SHAP value를 비교하면 다음과 같다. 
- DeepExplainer로 구한 기댓값(24.61)과 GradientExplainer로 구한 기댓값(24.23)에 약간 차이가 있다. 
- DeepExplainer로 구한 SHAP value의 경우 특성변수 ZN의 하방 영향이, GradientExplainer로 구한 SHAP value의 경우 특성변수 TAX의 하방 영향이 좀더 크게 나타났지만 전반적으로 유사한 결과를 보여주고 있다.



```python
expected_ge = model_dl.predict(X_train).mean()
print(expected_ge)

idx = 0

print(shap_values_ge.shape)

shap.force_plot(expected_ge, shap_values_ge[idx], features=X_test.iloc[0, :])
```

    [1m12/12[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 584us/step
    24.225084
    (152, 13)



![png](/images/m4/a4_2_deepGradientExplainer/deep_3.png)


