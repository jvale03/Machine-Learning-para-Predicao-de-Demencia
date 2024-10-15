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


## Data Preprocessing

## Undefined


## Useful Links / Refs
1. **Alzheimer**: [Data](http://adni.loni.usc.edu/)
2. **PyRadiomics**: [Documentation](https://pyradiomics.readthedocs.io/)
3. **IQR ouliers method**: [IQR](https://builtin.com/articles/1-5-iqr-rule)