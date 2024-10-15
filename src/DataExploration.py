import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

main_df = pd.read_csv("../Dataset/train_radiomics_occipital_CONTROL.csv")

####################
# Matplotlib Plots #

def show_histogram(title="histogram",df=main_df):
    plt.figure(figsize=(13,8))
    plt.subplots_adjust(bottom=0.17)
    plt.title(title)
    sns.histplot(df)
    plt.show()

def show_boxplot(title="boxplot",columns=main_df.columns,df=main_df):
    plt.figure(figsize=(13,8))
    plt.subplots_adjust(bottom=0.17)
    df[columns].boxplot()
    plt.xticks(rotation=15)
    plt.title(title)
    plt.show()

def show_heatmap(title="correlation heatmap",df=main_df.select_dtypes(include="number")):
    plt.figure(figsize=(13,8))
    plt.subplots_adjust(bottom=0.17)
    plt.title(title)
    sns.heatmap(df.corr(),annot=True,cmap="coolwarm",linewidths=0.5)
    plt.show()


############
# Outliers #

# IQR method 
## considera como outliers os dados que estao 1.5*IQR acima e abaixo do primeiro e terceiro quartil, respetivamente. IQR = Q3 - Q1
def detect_outliers(column, df=main_df):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filtrar os outliers
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

    show_boxplot(title=f"{column} boxplot",df=df,columns=[column])

    return outliers[column]

# esta função faz um loop de 7 em 7 colunas por todas as colunas do dataset para uma analise geral dos outliers.
# esta operacao é demorada e nao muito boa porque 2181 / 7 = 300 vezes
def explore_outliers(df=main_df):
    number_df = df.select_dtypes(include="number")
    n_columns = number_df.columns
    # normalização para ser visualmente perceptivel nos plots
    scaler = MinMaxScaler()
    number_df_scaled = pd.DataFrame(scaler.fit_transform(number_df),columns=n_columns)

    for i in range(0, len(n_columns),7):
        show_boxplot(df=number_df_scaled,columns=n_columns[i:i+7])

#####################
# Basic Exploration #

def main_exploration(df=main_df):
    print(main_df.shape) 
    # output: 305 linhas, 2181 colunas

def categorical_exploration(df=main_df):
    """
    Transition description:
    CN - Cognitive Normal, estado normal
    MCI - Mild Cognitive Impairment, estado entre o avanço normal da perda de memoria com a idade e um certo declinio serio de demencia
    AD - Azlheimer Disease, forma mais comum de demencia nas pessoas mais velhas
    """
    categorical_df = df[["Sex","Transition"]]

    for column in categorical_df.columns:
        show_histogram(f"{column} histogram",categorical_df[column])


def numerical_exploration(df=main_df):
    # age exploration
    age_exploration = df["Age"].describe()
    print(age_exploration)
    show_histogram("Age Histogram",df["Age"])
    print(detect_outliers("Age"))

    # show_heatmap()


def diagnostics_versions_explorer(df=main_df):
    diagnostics_versions_columns = ["diagnostics_Versions_PyRadiomics","diagnostics_Versions_Numpy","diagnostics_Versions_SimpleITK","diagnostics_Versions_PyWavelet","diagnostics_Versions_Python"] 

    diagnostics_df = df[diagnostics_versions_columns]

    for column in diagnostics_df.columns:
        print(column,": ")
        values = diagnostics_df[column].unique()
        print(values)



# main_info()
# categorical_exploration()
# numerical_exploration()
# explore_outliers()
