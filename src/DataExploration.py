import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

main_df = pd.read_csv("../Dataset/train_radiomics_occipital_CONTROL.csv")

def main_info(df=main_df):
    print(main_df.shape) 
    # output: 305 linhas, 2181 colunas

def show_histogram(title="histogram",df=main_df):
    plt.title(title)
    sns.histplot(df)
    plt.show()

def show_boxplot(columns=main_df.columns,df=main_df):
    df[columns].boxplot()
    plt.title("Outliers Boxplot")
    plt.show()

# IQR method 
## considera como outliers os dados que estao 1.5*IQR acima e abaixo do primeiro e terceiro quartil, respetivamente. IQR = Q3 - Q1
def outliers_detection(column, df=main_df):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filtrar os outliers
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers[column]

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

    """
    output:
        173 (43%) - man
        132 (57%) - woman
        # relativamente equilibrado

        CN-CN      96
        MCI-MCI    71
        MCI-AD     68
        AD-AD      60
        CN-MCI     10
    """



def numerical_exploration(df=main_df):
    age_exploration = df["Age"].describe()
    show_histogram("Age Histogram",df["Age"])
    """
    output:
        max - 91
        min - 55.3
        mean - 75.1
    """
    show_boxplot(["Age"],df)
    print(outliers_detection("Age"))



#main_info()
# categorical_exploration()
numerical_exploration()