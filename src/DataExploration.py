import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

main_df = pd.read_csv("../Dataset/train_radiomics_occipital_CONTROL.csv")

####################
# Matplotlib Plots #

def show_histogram(title="histogram",df=main_df):
    plt.title(title)
    sns.histplot(df)
    plt.show()

def show_boxplot(title="boxplot",columns=main_df.columns,df=main_df):
    df[columns].boxplot()
    plt.title(title)
    plt.show()

def show_heatmap(title="correlation heatmap",df=main_df.select_dtypes(include="number")):
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
    return outliers[column]


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
    show_boxplot("Outliers Boxplot",["Age"],df)
    print(detect_outliers("Age"))



    show_heatmap(df=df[['Age', 'lbp-3D-k_ngtdm_Busyness', 'lbp-3D-k_ngtdm_Coarseness']])


#main_info()
# categorical_exploration()
numerical_exploration()