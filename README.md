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
- **BoundingBox**: ainda nada concluído
- **VoxelNum**: ainda nada concluído
- **VolumeNum**: [1,2,3,4] -> Outliers: [1,3,4]
- **CenterOfMassIndex**: ainda nada concluído
- **CenterOfMass**: ainda nada concluído

#### Configurations
- **Settings**: valor único para todas as entradas 
- **EnabledImageTypes**: valor único para todas as entradas

unica conclusao aqui é que *CenterOfMassIndex* e *CenterOfMass* têm exatamente os mesmos valores, dar `drop` posteriormente.

## Data Preprocessing

- `remove_nunique_values()` com este método removemos todas as colunas que têm todas as entradas iguais. 2181 -> 2034 features


## Undefined


## Useful Links / Refs
1. **Alzheimer**: [Data](http://adni.loni.usc.edu/)
2. **PyRadiomics**: [Documentation](https://pyradiomics.readthedocs.io/)
3. **IQR ouliers method**: [IQR](https://builtin.com/articles/1-5-iqr-rule)