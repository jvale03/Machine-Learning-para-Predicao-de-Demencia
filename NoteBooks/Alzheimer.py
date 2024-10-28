#!/usr/bin/env python
# coding: utf-8

# In[53]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# In[2]:


control_df = pd.read_csv("../Dataset/train_radiomics_occipital_CONTROL.csv")
train_df = pd.read_csv("../Dataset/train_radiomics_hipocamp.csv")
test_df = pd.read_csv("../Dataset/test_radiomics_hipocamp.csv")
dummy_df = pd.read_csv("../Dataset/dummy_submission.csv")


# # Data Exploration and Preprocessing

# ## Matplotlib Plots

# In[3]:


def show_histogram(df,title="histogram"):
    plt.figure(figsize=(13,8))
    plt.subplots_adjust(bottom=0.17)
    plt.title(title)
    sns.histplot(df)
    plt.show()


# In[4]:


def show_pie(df,title="pie"):
    labels = df.unique().tolist()
    counts = df.value_counts()
    sizes = [counts[var_cat] for var_cat in labels]
    _, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct="%1.1f%%",shadow=True)
    ax1.axis("equal")
    plt.title(title)
    plt.show()


# In[5]:


def show_boxplot(df,title="boxplot"):
    plt.figure(figsize=(13,8))
    plt.subplots_adjust(bottom=0.17)
    df.boxplot()
    plt.xticks(rotation=15)
    plt.title(title)
    plt.show()


# In[6]:


def show_heatmap(df,title="correlation heatmap"):
    df = df.select_dtypes(include="number")
    plt.figure(figsize=(13,8))
    plt.subplots_adjust(bottom=0.25,left=0.22,right=0.95)
    plt.xticks(rotation=15)
    plt.title(title)
    sns.heatmap(df.corr(),annot=True,cmap="coolwarm",linewidths=0.5)
    plt.show()


# In[7]:


def show_jointplot(df,x_label,y_label,title="jointplot",hue="Transition_code"):
    sns.jointplot(data=df,x=x_label,y=y_label,hue=hue)
    plt.show()


# In[8]:


def show_catplot(df, x_label, y_label, title="catplot", hue="Transition_code"):
    sns.catplot(data=df, x=x_label, y=y_label, hue=hue)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


# In[9]:


def show_pairplot(df,hue="Transition_code"):
    sns.pairplot(df,hue=hue)
    plt.show()


# ## Outliers

# In[10]:


def explore_outliers(df,columns):
    number_df = df[columns].select_dtypes(include="number")
    n_columns = number_df.columns
    # normalização para ser visualmente perceptivel nos plots
    scaler = MinMaxScaler()
    number_df_scaled = pd.DataFrame(scaler.fit_transform(number_df),columns=n_columns)

    for i in range(0, len(n_columns),7):
        show_boxplot(df=number_df_scaled[n_columns[i:i+7]])
        
# esta função faz um loop de 7 em 7 colunas por todas as colunas do dataset para uma analise geral dos outliers.
# esta operacao é demorada e nao muito boa porque 2181 / 7 = 300 vezes


# In[11]:


def detect_outliers(df,column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filtrar os outliers
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

    show_boxplot(title=f"{column} boxplot",df=df[[column]])

    print(outliers[column])

# IQR method 
## considera como outliers os dados que estao 1.5*IQR acima e abaixo do primeiro e terceiro quartil, respetivamente. IQR = Q3 - Q1


# ## Category Encoder and Decoder
# bastante útil para poder codificar e posteriormente decodificar categorical features

# In[12]:


def target_encoder(df, target="Transition"):
    le_make = LabelEncoder()
    df[f"{target}_code"] = le_make.fit_transform(df[target])
    return le_make

def target_decoder(le_make, preds):
    return le_make.inverse_transform(preds)


# In[13]:


le_make_control = target_encoder(control_df)
le_make_train = target_encoder(train_df)


# ## Basic Exploration

# In[14]:


def main_exploration(df):
    print(df.shape) 


# In[15]:


main_exploration(control_df)
main_exploration(train_df)
main_exploration(test_df)


# In[16]:


def numerical_exploration(df):
    # age exploration
    show_histogram(df["Age"],title="Histogram")
    print(df["Age"].describe())
    detect_outliers(df,"Age")


# In[17]:


numerical_exploration(control_df)
numerical_exploration(train_df)


# In[18]:


def categorical_exploration(df):
    """
    Transition description:
    CN - Cognitive Normal, estado normal
    MCI - Mild Cognitive Impairment, estado entre o avanço normal da perda de memoria com a idade e um certo declinio serio de demencia
    AD - Azlheimer Disease, forma mais comum de demencia nas pessoas mais velhas
    """
    categorical_df = df[["Sex","Transition"]]

    for column in categorical_df.columns:
        show_histogram(categorical_df[column],title=f"{column} histogram")
        print(df[column].value_counts())


# In[19]:


categorical_exploration(train_df)


# In[20]:


show_catplot(train_df, "Age", "Transition", hue="Sex")
show_heatmap(train_df[["Age","Transition_code","Sex"]])


# In[21]:


sns.heatmap(train_df.isnull(), yticklabels=False, cbar=False, cmap="viridis")


# ## Diagnostic Exploration

# In[22]:


diagnostics_configs_columns = ["diagnostics_Configuration_Settings","diagnostics_Configuration_EnabledImageTypes"]
def diagnostics_configs(df):
    for col in diagnostics_configs_columns:
        print(len(df[col].unique()))


# In[23]:


diagnostics_configs(control_df)
diagnostics_configs(train_df)


# In[24]:


diagnostics_versions_columns = ["diagnostics_Versions_PyRadiomics","diagnostics_Versions_Numpy","diagnostics_Versions_SimpleITK","diagnostics_Versions_PyWavelet","diagnostics_Versions_Python"] 
def diagnostics_versions_explorer(df):
    for column in diagnostics_versions_columns:
        print(column,": ")
        values = df[column].unique()
        print(values)


# In[25]:


diagnostics_versions_explorer(control_df)
diagnostics_versions_explorer(train_df)


# In[26]:


diagnostics_image_columns = ["diagnostics_Image-original_Mean","diagnostics_Image-original_Minimum","diagnostics_Image-original_Maximum"]
def diagnostics_image_explorer(df):

    for column in diagnostics_image_columns:
        show_histogram(df=df[column],title=column)

    explore_outliers(df,diagnostics_image_columns)

    print(df[diagnostics_image_columns].describe())
    print(df[diagnostics_image_columns].info())
    


# In[27]:


diagnostics_image_explorer(control_df)
diagnostics_image_explorer(train_df)


# In[28]:


diagnostics_mask_columns = ["diagnostics_Mask-original_BoundingBox","diagnostics_Mask-original_VoxelNum","diagnostics_Mask-original_VolumeNum","diagnostics_Mask-original_CenterOfMassIndex","diagnostics_Mask-original_CenterOfMass"]
def diagnostics_mask_explorer(df):

    
    for column in diagnostics_mask_columns:
        print(column,": ")
        values = df[column].unique()
        print(len(values))
        show_histogram(title=column,df=df[column])
    

    detect_outliers(df,"diagnostics_Mask-original_VoxelNum")
    detect_outliers(df,"diagnostics_Mask-original_VolumeNum")
    explore_outliers(df,columns=diagnostics_mask_columns)
    
    print(df[diagnostics_mask_columns].describe(),"\n")
    print(df[diagnostics_mask_columns].info())


# In[29]:


diagnostics_mask_explorer(control_df)
diagnostics_mask_explorer(train_df)


# In[30]:


diagnostics = ["diagnostics_Mask-original_Spacing","diagnostics_Mask-original_Size","diagnostics_Mask-original_BoundingBox","diagnostics_Mask-original_VoxelNum","diagnostics_Mask-original_VolumeNum","diagnostics_Mask-original_CenterOfMassIndex","diagnostics_Mask-original_CenterOfMass","diagnostics_Image-original_Spacing","diagnostics_Image-original_Size","diagnostics_Image-original_Mean","diagnostics_Image-original_Maximum"]
def masks_images_correlation(df):
    show_heatmap(df=df[diagnostics])   


# In[31]:


masks_images_correlation(control_df)


# ## Drop Unnecessary Columns

# In[32]:


unnecessary_columns = diagnostics_versions_columns + diagnostics_configs_columns +["diagnostics_Image-original_Dimensionality","diagnostics_Image-original_Minimum","diagnostics_Image-original_Size","diagnostics_Mask-original_Spacing","diagnostics_Image-original_Spacing","diagnostics_Mask-original_Size","diagnostics_Image-original_Hash","diagnostics_Mask-original_Hash","ID","Image","Mask",'diagnostics_Mask-original_CenterOfMassIndex']

unnecessary_df = pd.DataFrame()
for col in unnecessary_columns+["Transition"]:
    le_make = LabelEncoder()
    unnecessary_df[f"{col}_code"] = le_make.fit_transform(train_df[col])

show_heatmap(unnecessary_df)


# In[33]:


control_df = control_df.drop(columns=unnecessary_columns,axis=1,errors="ignore")
train_df = train_df.drop(columns=unnecessary_columns,axis=1,errors="ignore")
test_df = test_df.drop(columns=unnecessary_columns,axis=1,errors="ignore")


# In[34]:


main_exploration(train_df)


# ## Top Correlations Function
# Esta função devolve as colunas mais/menos correlacionadas com a feature target desejada

# In[35]:


def top_correlations(df, target="Transition_code",starts_with=None,number=10,ascending=False):
    if starts_with == None:
        corr_columns = df.select_dtypes(include=["int64","float64"]).columns
    else:
        corr_columns = df.columns[df.columns.str.startswith(starts_with)]

    corr_matrix = df[corr_columns].corrwith(df[target])

    top_features = corr_matrix.sort_values(ascending=ascending).head(number).index.tolist()
    top_features.append(target)
    top_features = pd.Index(top_features)
    
    return top_features

corr_columns = train_df.select_dtypes(include=["int64","float64"]).columns
corr_matrix = train_df[corr_columns].corrwith(train_df["Transition_code"])


# ## Nunique Columns PreProcessing

# In[36]:


nunique_columns = train_df.columns[train_df.nunique() == 1].tolist()
nunique_columns.append("Transition_code")
nunique_corr_columns = top_correlations(train_df[nunique_columns],number=10000)

show_heatmap(train_df[nunique_corr_columns])

main_exploration(train_df)
main_exploration(control_df)
main_exploration(test_df)

nunique_columns.remove("Transition_code")
train_df = train_df.drop(columns=nunique_columns, errors="ignore")
test_df = test_df.drop(columns=nunique_columns, errors="ignore")
control_df = control_df.drop(columns=nunique_columns, errors="ignore")


# ## Non Numeric Exploration

# In[37]:


def non_numeric_exploration(df):
    non_numeric_columns = train_df.select_dtypes(exclude=['int64', 'float64']).columns
    return non_numeric_columns


# In[38]:


non_numeric_columns = non_numeric_exploration(train_df)
print(train_df[non_numeric_columns].head())


# ## Non Numerical Columns PreProcessing

# In[39]:


# Separar a coluna de BoundingBox em várias colunas
train_df[['x_min', 'y_min', 'largura', 'altura', 'profundidade', 'extra']] = train_df['diagnostics_Mask-original_BoundingBox'].str.strip('()').str.split(',', expand=True).astype(float)


# Separar a coluna de CenterOfMassIndex em várias colunas
train_df[['x_center', 'y_center', 'z_center']] = train_df['diagnostics_Mask-original_CenterOfMass'].str.strip('()').str.split(',', expand=True).astype(float)


# In[40]:


# Separar a coluna de BoundingBox em várias colunas
test_df[['x_min', 'y_min', 'largura', 'altura', 'profundidade', 'extra']] = test_df['diagnostics_Mask-original_BoundingBox'].str.strip('()').str.split(',', expand=True).astype(float)


# Separar a coluna de CenterOfMassIndex em várias colunas
test_df[['x_center', 'y_center', 'z_center']] = test_df['diagnostics_Mask-original_CenterOfMass'].str.strip('()').str.split(',', expand=True).astype(float)


# In[41]:


# Separar a coluna de BoundingBox em várias colunas
control_df[['x_min', 'y_min', 'largura', 'altura', 'profundidade', 'extra']] = control_df['diagnostics_Mask-original_BoundingBox'].str.strip('()').str.split(',', expand=True).astype(float)


# Separar a coluna de CenterOfMassIndex em várias colunas
control_df[['x_center', 'y_center', 'z_center']] = control_df['diagnostics_Mask-original_CenterOfMass'].str.strip('()').str.split(',', expand=True).astype(float)


# In[42]:


train_df = train_df.drop(['diagnostics_Mask-original_BoundingBox', 'diagnostics_Mask-original_CenterOfMass'], axis=1, errors="ignore")
test_df = test_df.drop(['diagnostics_Mask-original_BoundingBox', 'diagnostics_Mask-original_CenterOfMass'], axis=1, errors="ignore")
control_df = control_df.drop(['diagnostics_Mask-original_BoundingBox', 'diagnostics_Mask-original_CenterOfMass'], axis=1, errors="ignore")


# In[43]:


new_numeric_columns = ['x_min', 'y_min', 'largura', 'altura', 'profundidade', 'extra','x_center', 'y_center', 'z_center',"Transition_code"]
show_heatmap(train_df[new_numeric_columns])


# ## Numeric Diagnostics Corr

# In[44]:


diagnostics_corr = top_correlations(train_df, starts_with="diagnostics")
show_heatmap(train_df[diagnostics_corr])


# ## Radiomics

# In[45]:


rad_corr = top_correlations(train_df,starts_with="lbp",number=20)
show_heatmap(train_df[rad_corr])


# ## Top Correlations
# Retorna as X colunas com melhor correlação com o nosso target

# In[46]:


top_features = top_correlations(train_df)
show_heatmap(train_df[top_features])


# # Data Processing

# ## Very Low Correlations Processing
# Estas colunas têm um valor de correlação com o target tão baixo que podem ser praticamente dispensadas

# In[47]:


low_corr_columns_absolute = corr_matrix[(corr_matrix.abs() < 0.05)].index.tolist() #retorna as colunas maiores que 0.05 ou menores que -0.05
low_corr_columns = corr_matrix[(corr_matrix < 0.05)].index.tolist() #retorna apenas as colunas maiores que 0.05
print(len(low_corr_columns_absolute))
print(len(low_corr_columns))


# In[61]:


low_corr_train_df = train_df.drop(columns=low_corr_columns,axis=1,errors="ignore")
low_corr_control_df = control_df.drop(columns=low_corr_columns,axis=1,errors="ignore")
low_corr_test_df = test_df.drop(columns=low_corr_columns,axis=1,errors="ignore")

low_corr_abs_train_df = train_df.drop(columns=low_corr_columns_absolute,axis=1,errors="ignore")
low_corr_abs_control_df = control_df.drop(columns=low_corr_columns_absolute,axis=1,errors="ignore")
low_corr_abs_test_df = test_df.drop(columns=low_corr_columns_absolute,axis=1,errors="ignore")


# In[49]:


main_exploration(train_df_low)
main_exploration(control_df_low)
main_exploration(test_df_low)
print("---low---")
main_exploration(train_df_low_abs)
main_exploration(control_df_low_abs)
main_exploration(test_df_low_abs)


# ## Data Scaler

# In[56]:


from sklearn.preprocessing import StandardScaler

def data_scaler(df):
    scaler_df = df.drop(columns=["Transition","Transition_code"],errors="ignore")
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(scaler_df),columns=scaler_df.columns)
    return df_scaled


# ### PCA method

# In[57]:


from sklearn.decomposition import PCA

def pca_applier(train_df, test_df):
    # 1. Ajustar o PCA no dataset de treino para reter 95% da variância
    pca = PCA(n_components=0.95)
    train_pca_features = pca.fit_transform(train_df)

    # 2. Determinar o número de componentes que explicam pelo menos 95% da variância
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    components_number = np.argmax(cumulative_variance >= 0.95) + 1

    # 3. Criar um novo PCA com o número exato de componentes e ajustar no treino
    final_pca = PCA(n_components=components_number)
    train_reduced = final_pca.fit_transform(train_df)

    # 4. Transformar o dataset de teste com o PCA ajustado no treino
    test_reduced = final_pca.transform(test_df)

    # 5. Converter ambos para DataFrames para fácil visualização e tratamento
    train_reduced_df = pd.DataFrame(train_reduced, columns=[f"PC{i+1}" for i in range(components_number)])
    test_reduced_df = pd.DataFrame(test_reduced, columns=[f"PC{i+1}" for i in range(components_number)])

    return train_reduced_df, test_reduced_df


# In[62]:


pca_train_df = data_scaler(train_df)
pca_test_df = data_scaler(test_df)
pca_train_df, pca_test_df = pca_applier(pca_train_df,pca_test_df)


# ## Data Normalizer

# In[ ]:


from sklearn.preprocessing import MinMaxScaler

def data_normalizer(df):
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    
    return df_normalized


# # Testing Phase

# In[60]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import pandas as pd

df = new_train_df
df_test = new_test_df

# Codificar a variável 'Transition' para análise de correlação (convertendo categorias em números)
df['Transition_code'] = train_df['Transition'].astype('category').cat.codes


# Separar os dados em features e target para treino e teste
X_train = df.drop(columns=["Transition_code","Transition"],axis=1,errors="ignore")
X_test = df_test
y_train = df['Transition_code']



# Treinar um modelo Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = rf_model.predict(X_test)

# Criar o mapeamento inverso para transformar os números em categorias originais
inverse_transition_map = dict(enumerate(train_df['Transition'].astype('category').cat.categories))

# Reverter as previsões para as categorias originais
y_pred_original = pd.Series(y_pred).map(inverse_transition_map)


# Atualizar a coluna 'Result' no dataset 'dummy_df' com as previsões
dummy_df["Result"] = y_pred_original.values

# Guardar o dataset atualizado
dummy_df.to_csv("../Dataset/dummy_submission.csv", index=False)


print(y_pred_original)

print("Coluna 'Result' atualizada com sucesso!")


# In[ ]:




