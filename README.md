# DAA
DAA group pratical work

## Table of Contents

1. [Data Description](#data-description)
2. [Data Exploration](#data-exploration)
3. [Data Preprocessing](#data-preprocessing)
4. [Undefined](#undefined)
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

Relatively balanced

#### Transition
- **CN-CN**: 96
- **MCI-MCI**: 71
- **MCI-AD**: 68
- **AD-AD**: 60
- **CN-MCI**: 10

#### Diagnostics Versions
- **PyRadiomics**: 2.2.0
- **Numpy**: 1.18.5
- **SimpleITK**: 1.2.4
- **PyWavelet**: 1.1.1
- **Python**: 3.7.7

#### Diagnostic Images
- **Dimensionality**: 3
- **Spacing**: (1.0, 1.0, 1.0)
- **Size**: (256, 256, 256)
- **Minimum**: ainda nada concluído
- **Mean**: ainda nada concluído
- **Maximum**: ainda nada concluído

#### Diagnostic Masks
- **Spacing**: (1.0, 1.0, 1.0)
- **Size**: (256, 256, 256)
- **BoundingBox**: todos os valores são diferentes
- **VoxelNum**: praticamente todos os valores são diferentes
- **VolumeNum**: [1,2,3,4] -> Outliers: [1,3,4]
- **CenterOfMassIndex**: todos os valores são diferentes
- **CenterOfMass**: todos os valores são diferentes

#### Configurations
- **Settings**: valor único para todas as entradas 
- **EnabledImageTypes**: valor único para todas as entradas

Analisando estes três últimos conjuntos de dados `Diagnostics`, percebemos que certas features com entradas todas iguais podem ser removidas por se tratarem de informação claramente irrelevante, pois apresentam uma correlation completamente **nula** para o nosso `Target`.


## Data Preprocessing



## Undefined


## Useful Links / Refs
1. **Alzheimer**: [Data](http://adni.loni.usc.edu/)
2. **PyRadiomics**: [Documentation](https://pyradiomics.readthedocs.io/en/latest/features.html)
3. **IQR ouliers method**: [IQR](https://builtin.com/articles/1-5-iqr-rule)
4. **SHAP documentation**: [SHAP](https://shap.readthedocs.io/en/latest/)
5. **Hyperparameter Tuning**: [Tuning](https://www.geeksforgeeks.org/hyperparameter-tuning/)
6. **Bayesian Optimization**: [Bayes](https://www.geeksforgeeks.org/catboost-bayesian-optimization/)
7. **Different Boosting Methods**: [Boost](https://www.geeksforgeeks.org/gradientboosting-vs-adaboost-vs-xgboost-vs-catboost-vs-lightgbm/)
8. **CatBoost Hyperparams**: [CatBoost](https://www.geeksforgeeks.org/catboost-parameters-and-hyperparameters/)
9. **Feature Engineering**: [FeatureEng](https://www.geeksforgeeks.org/what-is-feature-engineering/)