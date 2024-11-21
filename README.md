# DAA
DAA group pratical work

## Table of Contents

1. [Data Description](#data-description)
2. [Data Exploration](#data-exploration)
3. [Data Preprocessing](#data-preprocessing)
4. [Data Interpretability](#data-interpretability)
5. [Useful Links / Refs](#useful-links--refs)

## Data Description
### Transition
- **CN**: Cognitive Normal, estado normal
- **MCI**: Mild Cognitive Impairment, estado entre o avanço normal da perda de memoria com a idade e um certo declinio serio de demencia
- **AD**: Azlheimer Disease, forma mais comum de demencia nas pessoas mais velhas


## Data Exploration
### Shape 
- **Rows**: 305
- **Columns**: 2181

### Columns
#### Age
- **Max**: 91
- **Min**: 55.3
- **Mean**: 75.1

#### Sex
- **Man**: 173 (57%)
- **Woman**: 132 (43%)

Relativamente Balanceado

#### Transition
- **CN-CN**: 96 (31.5%)
- **MCI-MCI**: 71 (23.3%)
- **MCI-AD**: 68 (22.3%)
- **AD-AD**: 60 (19.7%)
- **CN-MCI**: 10 (3.2%)

![Figura1](Images/target_distribution.png)

Muito desbalanceado, o que nos indica que é preciso ter cuidado com parâmetros e features.

Analisamos então a distribuição e correlação da idade e do sexo em relação à nossa coluna target:

![Figura2](Images/sex_age_distribution.png)
![Figura3](Images/sex_age_correlation.png)

### NaN Values
Não foram encontrados nenhuns NaN values nos datasets.

### Nunique Features
Encontramos algumas features constantes, no entanto decidimos não remover precipitadamente sem uma análise prévia, excluíndo no entanto apenas as features relacionadas com versões das tecnologias usadas.

### Non Numeric Features
Após a análise das features do tipo `object`, reparamos que existiam features capazes de ser transformadas em uma ou mais features numéricas, podendo trazer informação relevante para os futuros modelos.


No que toca a exploração de dados, foi feita uma análise mais profunda não mencionada por não trazer informação considerada relevante numa exploração global.

## Data Preprocessing
### Feature Target
No que toca a pré-processamento de dados, começamos por fazer `encoding` da nossa feature **target**, transformando-a numa feature numérica.

### Object Features
Seguidamente, como mencionado em [Data Exploration](#non-numeric-features), transformamos features do tipo `object` em várias features novas, servem de exemplo as seguintes transformações:

| diagnostics_Mask-original_BoundingBox | 
|-|
| (103, 113, 93, 36, 30, 71)| 
| (32, 104, 3, 54, 10, 89)| 

| x_min | y_min | largura | altura | profundidade | extra
|-|-|-|-|-|-|
|103|113|93|36|30|71 
|32|104|3|54|10|89 



As restantes features não numéricas foram removidas.

### Data Scaler
De modo a **padronizar** os dados, recorremos ao método `StandardScaler`. Desta forma, os valores das features são ajustados para uma escala comum, com média 0 e desvio padrão igual a 1, conseguindo **melhorar** assim o desempenho de modelos sensíveis à escala.


Por termos um **dataset limpo**, pouco pré-processamento de dados foi necessário, sendo portanto agora mais relevante uma análise das features.

## Data Interpretability
Para analisar a contribuição de cada feature para as previsões finais, recorremos à tecnologia **SHAP** capaz de nos ajudar a entender os outputs dos nossos modelos para cada classe.

Para concretização desta análise, recorremos ao modelo `XGBoost`, um modelo baseado em `DecisionTrees`. 

Por ser um modelo muito **eficiente**, **robusto**, altamente **eficaz** em encontrar **padrões** e **relações complexas** entre features e ainda lidar bem com **desbalanceamento de classes**, consideramos ser a escolha mais adequada para o problema. 

**A análise aos shap values ainda nao esta concluida, ainda nao pensei no que vamos falar sobre isto na apresentação.**


## Useful Links / Refs
1. **Alzheimer**: [Data](http://adni.loni.usc.edu/)
2. **PyRadiomics**: [Documentation](https://pyradiomics.readthedocs.io/en/latest/features.html)
3. **IQR ouliers method**: [IQR](https://builtin.com/articles/1-5-iqr-rule)
4. **SHAP documentation**: [SHAP](https://shap.readthedocs.io/en/latest/)
5. **Hyperparameter Tuning**: [Tuning](https://www.geeksforgeeks.org/hyperparameter-tuning/)
6. **Bayesian Optimization**: [Bayes](https://www.geeksforgeeks.org/catboost-bayesian-optimization/)
7. **Different Boosting Methods**: [Boost](https://www.geeksforgeeks.org/gradientboosting-vs-adaboost-vs-xgboost-vs-catboost-vs-lightgbm/)
9. **Feature Engineering**: [FeatureEng](https://www.geeksforgeeks.org/what-is-feature-engineering/)
10. **Skewness & Kurtosis** [SkewKurt](https://vivekrai1011.medium.com/skewness-and-kurtosis-in-machine-learning-c19f79e2d7a5)
11. **Going Deep with SHAP Values** [Deep SHAP](https://medium.com/biased-algorithms/shap-values-for-multiclass-classification-2a1b93f69c63)