#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ast

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV


# In[ ]:


control_df = pd.read_csv("../Dataset/train_radiomics_occipital_CONTROL.csv")
train_df = pd.read_csv("../Dataset/train_radiomics_hipocamp.csv")
test_df = pd.read_csv("../Dataset/test_radiomics_hipocamp.csv")
dummy_df = pd.read_csv("../Dataset/dummy_submission.csv")


# # Data Exploration and Preprocessing

# ## Matplotlib Plots

# In[ ]:


def show_histogram(df,title="histogram"):
    plt.figure(figsize=(13,8))
    plt.subplots_adjust(bottom=0.17)
    plt.title(title)
    sns.histplot(df)
    plt.show()


# In[ ]:


def show_pie(df,title="pie"):
    labels = df.unique().tolist()
    counts = df.value_counts()
    sizes = [counts[var_cat] for var_cat in labels]
    _, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct="%1.1f%%",shadow=True)
    ax1.axis("equal")
    plt.title(title)
    plt.show()


# In[ ]:


def show_boxplot(df,title="boxplot"):
    plt.figure(figsize=(13,8))
    plt.subplots_adjust(bottom=0.17)
    df.boxplot()
    plt.xticks(rotation=15)
    plt.title(title)
    plt.show()


# In[ ]:


def show_heatmap(df,title="correlation heatmap"):
    df = df.select_dtypes(include="number")
    plt.figure(figsize=(13,8))
    plt.subplots_adjust(bottom=0.25,left=0.22,right=0.95)
    plt.xticks(rotation=15)
    plt.title(title)
    sns.heatmap(df.corr(),annot=True,cmap="coolwarm",linewidths=0.5)
    plt.show()


# In[ ]:


def show_jointplot(df,x_label,y_label,title="jointplot",hue="Transition_code"):
    sns.jointplot(data=df,x=x_label,y=y_label,hue=hue)
    plt.show()


# In[ ]:


def show_catplot(df, x_label, y_label, title="catplot", hue="Transition_code"):
    sns.catplot(data=df, x=x_label, y=y_label, hue=hue)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


# In[ ]:


def show_pairplot(df,hue="Transition_code"):
    sns.pairplot(df,hue=hue)
    plt.show()


# ## Outliers

# In[ ]:


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


# In[ ]:


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

# In[ ]:


def target_encoder(df, target="Transition"):
    le_make = LabelEncoder()
    df[f"{target}_code"] = le_make.fit_transform(df[target])
    return le_make

def target_decoder(le_make, preds):
    return le_make.inverse_transform(preds)


# In[ ]:


le_make_train = target_encoder(train_df)
le_make_control = target_encoder(control_df)


# ## Basic Exploration

# In[ ]:


def main_exploration(df):
    print(df.shape) 


# In[ ]:


main_exploration(control_df)
main_exploration(train_df)
main_exploration(test_df)


# In[ ]:


def numerical_exploration(df):
    # age exploration
    show_histogram(df["Age"],title="Histogram")
    print(df["Age"].describe())
    detect_outliers(df,"Age")


# In[ ]:


#numerical_exploration(control_df)
#numerical_exploration(train_df)


# In[ ]:


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


# In[ ]:


#categorical_exploration(train_df)


# In[ ]:


show_catplot(train_df, "Age", "Transition", hue="Sex")
show_heatmap(train_df[["Age","Transition_code","Sex"]])


# In[ ]:


sns.heatmap(train_df.isnull(), yticklabels=False, cbar=False, cmap="viridis")


# ## Diagnostic Exploration

# In[ ]:


diagnostics_configs_columns = ["diagnostics_Configuration_Settings","diagnostics_Configuration_EnabledImageTypes"]
def diagnostics_configs(df):
    for col in diagnostics_configs_columns:
        print(len(df[col].unique()))


# In[ ]:


diagnostics_configs(control_df)
diagnostics_configs(train_df)


# In[ ]:


diagnostics_versions_columns = ["diagnostics_Versions_PyRadiomics","diagnostics_Versions_Numpy","diagnostics_Versions_SimpleITK","diagnostics_Versions_PyWavelet","diagnostics_Versions_Python"] 
def diagnostics_versions_explorer(df):
    for column in diagnostics_versions_columns:
        print(column,": ")
        values = df[column].unique()
        print(values)


# In[ ]:


diagnostics_versions_explorer(control_df)
diagnostics_versions_explorer(train_df)


# In[ ]:


diagnostics_image_columns = ["diagnostics_Image-original_Mean","diagnostics_Image-original_Minimum","diagnostics_Image-original_Maximum"]
def diagnostics_image_explorer(df):

    for column in diagnostics_image_columns:
        show_histogram(df=df[column],title=column)

    explore_outliers(df,diagnostics_image_columns)

    print(df[diagnostics_image_columns].describe())
    print(df[diagnostics_image_columns].info())
    


# In[ ]:


diagnostics_image_explorer(control_df)
diagnostics_image_explorer(train_df)


# In[ ]:


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


# In[ ]:


diagnostics_mask_explorer(control_df)
diagnostics_mask_explorer(train_df)


# In[ ]:


diagnostics = ["diagnostics_Mask-original_Spacing","diagnostics_Mask-original_Size","diagnostics_Mask-original_BoundingBox","diagnostics_Mask-original_VoxelNum","diagnostics_Mask-original_VolumeNum","diagnostics_Mask-original_CenterOfMassIndex","diagnostics_Mask-original_CenterOfMass","diagnostics_Image-original_Spacing","diagnostics_Image-original_Size","diagnostics_Image-original_Mean","diagnostics_Image-original_Maximum"]
def masks_images_correlation(df):
    show_heatmap(df=df[diagnostics])   


# In[ ]:


masks_images_correlation(control_df)


# ## Drop Unnecessary Columns

# In[ ]:


unnecessary_columns = diagnostics_versions_columns + diagnostics_configs_columns +["diagnostics_Image-original_Dimensionality","diagnostics_Image-original_Minimum","diagnostics_Image-original_Size","diagnostics_Mask-original_Spacing","diagnostics_Image-original_Spacing","diagnostics_Mask-original_Size","diagnostics_Image-original_Hash","diagnostics_Mask-original_Hash","ID","Image","Mask",'diagnostics_Mask-original_CenterOfMassIndex']

unnecessary_df = pd.DataFrame()
for col in unnecessary_columns+["Transition"]:
    le_make = LabelEncoder()
    unnecessary_df[f"{col}_code"] = le_make.fit_transform(train_df[col])

show_heatmap(unnecessary_df)


# In[ ]:


control_df = control_df.drop(columns=unnecessary_columns,axis=1,errors="ignore")
train_df = train_df.drop(columns=unnecessary_columns,axis=1,errors="ignore")
test_df = test_df.drop(columns=unnecessary_columns,axis=1,errors="ignore")


# In[ ]:


main_exploration(train_df)


# ## Top Correlations Function
# Esta função devolve as colunas mais/menos correlacionadas com a feature target desejada

# In[ ]:


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

# In[ ]:


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

# In[ ]:


def non_numeric_exploration(df):
    non_numeric_columns = train_df.select_dtypes(exclude=['int64', 'float64']).columns
    return non_numeric_columns


# In[ ]:


non_numeric_columns = non_numeric_exploration(train_df)
print(train_df[non_numeric_columns].head())


# ## Non Numerical Columns PreProcessing

# In[ ]:


# Separar a coluna de BoundingBox em várias colunas
train_df[['x_min', 'y_min', 'largura', 'altura', 'profundidade', 'extra']] = train_df['diagnostics_Mask-original_BoundingBox'].str.strip('()').str.split(',', expand=True).astype(float)


# Separar a coluna de CenterOfMassIndex em várias colunas
train_df[['x_center', 'y_center', 'z_center']] = train_df['diagnostics_Mask-original_CenterOfMass'].str.strip('()').str.split(',', expand=True).astype(float)


# In[ ]:


# Separar a coluna de BoundingBox em várias colunas
test_df[['x_min', 'y_min', 'largura', 'altura', 'profundidade', 'extra']] = test_df['diagnostics_Mask-original_BoundingBox'].str.strip('()').str.split(',', expand=True).astype(float)


# Separar a coluna de CenterOfMassIndex em várias colunas
test_df[['x_center', 'y_center', 'z_center']] = test_df['diagnostics_Mask-original_CenterOfMass'].str.strip('()').str.split(',', expand=True).astype(float)


# In[ ]:


# Separar a coluna de BoundingBox em várias colunas
control_df[['x_min', 'y_min', 'largura', 'altura', 'profundidade', 'extra']] = control_df['diagnostics_Mask-original_BoundingBox'].str.strip('()').str.split(',', expand=True).astype(float)


# Separar a coluna de CenterOfMassIndex em várias colunas
control_df[['x_center', 'y_center', 'z_center']] = control_df['diagnostics_Mask-original_CenterOfMass'].str.strip('()').str.split(',', expand=True).astype(float)


# In[ ]:


train_df = train_df.drop(['diagnostics_Mask-original_BoundingBox', 'diagnostics_Mask-original_CenterOfMass'], axis=1, errors="ignore")
test_df = test_df.drop(['diagnostics_Mask-original_BoundingBox', 'diagnostics_Mask-original_CenterOfMass'], axis=1, errors="ignore")
control_df = control_df.drop(['diagnostics_Mask-original_BoundingBox', 'diagnostics_Mask-original_CenterOfMass'], axis=1, errors="ignore")


# In[ ]:


new_numeric_columns = ['x_min', 'y_min', 'largura', 'altura', 'profundidade', 'extra','x_center', 'y_center', 'z_center',"Transition_code"]
show_heatmap(train_df[new_numeric_columns])


# ## Numeric Diagnostics Corr

# In[ ]:


diagnostics_corr = top_correlations(train_df, starts_with="diagnostics")
show_heatmap(train_df[diagnostics_corr])


# ## Radiomics

# In[ ]:


rad_corr = top_correlations(train_df,starts_with="lbp",number=20)
show_heatmap(train_df[rad_corr])


# ## Top Correlations
# Retorna as X colunas com melhor correlação com o nosso target

# In[ ]:


top_features = top_correlations(train_df)
#show_heatmap(train_df[top_features])


# # Data Processing

# ## Data Scaler
# Padroniza os dados

# In[ ]:


from sklearn.preprocessing import StandardScaler

def data_scaler(df):
    scaler_df = df.drop(columns=["Transition","Transition_code"],errors="ignore")
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(scaler_df),columns=scaler_df.columns)
    return df_scaled


# In[ ]:


scaled_train_df = data_scaler(train_df)
scaled_control_df = data_scaler(control_df)
scaled_test_df = data_scaler(test_df)


# ## Data Normalizer
# Normalizar dados

# In[ ]:


from sklearn.preprocessing import MinMaxScaler

def data_normalizer(df):
    scaler_df = df.drop(columns=["Transition","Transition_code"],errors="ignore")
    
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(scaler_df), columns=scaler_df.columns)
    
    return df_normalized


# In[ ]:


normalized_train_df = data_normalizer(train_df)
normalized_control_df = data_normalizer(control_df)
normalized_test_df = data_normalizer(test_df)
target = "Transition_code"


# In[ ]:


corr_df = scaled_train_df.copy()
corr_df.loc[:,"Transition_code"] = train_df["Transition_code"].values


# ## Correlation Analisys

# In[ ]:


corr_threshold = 0.01
def apply_correlation(df,threshold):
    df = df.drop(columns=["Transition"],errors="ignore")
    correlation = df.corr()[target].abs().sort_values(ascending=False)
    important_features = correlation[correlation > threshold].index.tolist()
    
    if target in important_features:
        important_features.remove(target)

    return important_features


# In[ ]:


corr_important_features = apply_correlation(corr_df,corr_threshold)


# In[ ]:


corr_train_df = scaled_train_df[corr_important_features].copy()
corr_control_df = scaled_control_df[corr_important_features].copy()
corr_test_df = scaled_test_df[corr_important_features].copy()


# In[ ]:


main_exploration(corr_train_df)
main_exploration(corr_control_df)
main_exploration(corr_test_df)


# In[ ]:


print("Important: ",len(corr_important_features))


# ## Add Transition_code to DataSets

# In[ ]:


corr_train_df.loc[:,"Transition_code"] = train_df["Transition_code"].values
corr_control_df.loc[:,"Transition_code"] = control_df["Transition_code"].values


# In[ ]:


show_boxplot(corr_train_df)


# # Testing Phase

# ## ML Models

# In[ ]:


from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from xgboost import XGBClassifier


# In[ ]:


def define_X_y(train_df, test_df = pd.DataFrame()):
    if test_df.empty:
        X = train_df.drop(columns=["Transition_code","Transition"],errors="ignore")
        y = train_df["Transition_code"]

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=27)

        return x_train, x_test, y_train, y_test

    else:
        x_train = train_df.drop("Transition_code",axis=1,errors="ignore")
        y_train = train_df["Transition_code"]
        x_test = test_df

        return x_train, x_test, y_train, None


# In[ ]:


grid_search_mode = False

random_forest_params = {'n_estimators': 200, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': None}
xgb_params = {'subsample': 1.0, 'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'colsample_bytree': 1.0}
gradient_params = {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_depth': 3, 'learning_rate': 0.1}

results = {}


# In[ ]:


x_train, x_test, y_train, y_test = define_X_y(corr_train_df,corr_test_df)

main_exploration(x_train)
main_exploration(x_test)


# ### Random Forest w/ GridSearch Cross Validation

# In[ ]:


def random_forest(x_train, y_train, best_params=None, mode=False):
    random_forest = RandomForestClassifier(random_state=27, class_weight='balanced', max_samples=0.8,n_jobs=-1)

    if best_params is not None:
        random_forest.set_params(**best_params)
    if mode:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt']
        }
        
        grid_search_random = RandomizedSearchCV(
            estimator=random_forest,
            param_distributions=param_grid,
            n_iter=60,
            cv=5,
            scoring='f1_macro',
            n_jobs=-1,
            random_state=27
        )

        grid_search_random.fit(x_train, y_train)
        best_params = grid_search_random.best_params_
        print(f"Melhores parâmetros: {best_params}")
        
        random_forest = grid_search_random.best_estimator_

    random_forest.fit(x_train, y_train)


    return random_forest, best_params


# In[ ]:


random_forest_model, random_forest_params = random_forest(x_train, y_train,best_params=random_forest_params,mode=grid_search_mode)


# In[ ]:


random_forest_pred = random_forest_model.predict(x_test)


# ### XGBoost w/ GridSearch Cross Validation

# In[ ]:


def xgboost_model(x_train, y_train, best_params=None, mode=False):
    xgb_model = XGBClassifier(eval_metric="mlogloss", random_state=27,n_jobs=-1)

    if best_params is not None:
        xgb_model.set_params(**best_params)
    if mode:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }

        grid_search_random = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_grid,
            n_iter=20,
            cv=4,
            scoring='f1_macro',
            n_jobs=-1,
            random_state=27
        )

        grid_search_random.fit(x_train, y_train)
        best_params = grid_search_random.best_params_
        print(f"Melhores parâmetros: {best_params}")
        
        xgb_model = grid_search_random.best_estimator_

    xgb_model.fit(x_train, y_train)
    return xgb_model, best_params


# In[ ]:


xgb_model,xgb_params = xgboost_model(x_train, y_train,best_params=xgb_params,mode=grid_search_mode)


# In[ ]:


xgb_pred = xgb_model.predict(x_test)


# ### Gradient Boost w/ GridSearch Cross Validation

# In[ ]:


def gradient_boosting(x_train, y_train, best_params=None, mode=False):
    gradient_boosting = GradientBoostingClassifier(random_state=27)

    if best_params is not None:
        gradient_boosting.set_params(**best_params)
            
    if mode:
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }

        grid_search_random = RandomizedSearchCV(
            estimator=gradient_boosting,
            param_distributions=param_grid,
            n_iter=20,
            cv=4,
            scoring='f1_macro',
            n_jobs=-1,
            random_state=27
        )
        
        grid_search_random.fit(x_train, y_train)
        best_params = grid_search_random.best_params_
        print(f"Melhores parâmetros: {best_params}")
        
        gradient_boosting = grid_search_random.best_estimator_

    gradient_boosting.fit(x_train, y_train)
    return gradient_boosting, best_params


# In[ ]:


gradient_model,gradient_params = gradient_boosting(x_train, y_train,best_params=gradient_params,mode=grid_search_mode)


# In[ ]:


gradient_pred = gradient_model.predict(x_test)


# In[ ]:


f1_macro_score_rf = f1_score(y_test, random_forest_pred, average="macro")
results["RandomForest"] = f1_macro_score_rf
f1_macro_score_xgb = f1_score(y_test, xgb_pred, average="macro")
results["XGBoost"] = f1_macro_score_xgb
f1_macro_score_gradient = f1_score(y_test, gradient_pred, average="macro")
results["GradientBoost"] = f1_macro_score_gradient

models_score = plt.figure(figsize=(6,3))

mod = list(results.keys())
f1 = list(results.values())

plt.bar(mod,f1, color = "lightblue", width = 0.5)

plt.xlabel("Model")
plt.ylabel("Macro F1")
plt.title("Models Macro F1 Comparison")
plt.show()


# ## Ensemble Learning (RandomForest w/ XGBoost)

# ### Voting Classifier

# In[ ]:


calibrated_rf = CalibratedClassifierCV(random_forest_model, method='sigmoid', cv=4)
calibrated_xgb = CalibratedClassifierCV(xgb_model, method='sigmoid', cv=4)
calibrated_gradient = CalibratedClassifierCV(gradient_model, method='sigmoid', cv=4)


# In[ ]:


def ensemble_voting_classifier(x_train, y_train):
    ensemble_model_v = VotingClassifier(
        estimators=[
            ("random_forest", calibrated_rf),
            ("xgboost", calibrated_xgb),
            ("gradientboost", calibrated_gradient),
        ],
        voting="soft",
        n_jobs=-1
    )
    ensemble_model_v.fit(x_train, y_train)
    return ensemble_model_v


# In[ ]:


ensemble_voting_model = ensemble_voting_classifier(x_train,y_train)


# In[ ]:


ensemble_voting_pred = ensemble_voting_model.predict(x_test)


# ### Stacking Classifier

# In[ ]:


def ensemble_stacking_classifier(x_train, y_train):
    final_model = RandomForestClassifier(n_estimators=100, random_state=27, n_jobs=-1)
    
    ensemble_model_s = StackingClassifier(
        estimators=[
            ("random_forest", calibrated_rf),
            ("xgboost", calibrated_xgb),
            ("gradientboost", calibrated_gradient)
        ],
        final_estimator=final_model,
        cv=3,
        n_jobs=-1
    )
    ensemble_model_s.fit(x_train, y_train)
    return ensemble_model_s


# In[ ]:


ensemble_stacking_model = ensemble_stacking_classifier(x_train,y_train)


# In[ ]:


ensemble_stacking_pred = ensemble_stacking_model.predict(x_test)


# ### Models Comparison

# In[ ]:


f1_macro_score_voting_ensemble = f1_score(y_test, ensemble_voting_pred, average="macro")
results["VotingEnsemble"] = f1_macro_score_voting_ensemble
f1_macro_score_stacking_ensemble = f1_score(y_test, ensemble_stacking_pred, average="macro")
results["StackingEnsemble"] = f1_macro_score_stacking_ensemble

models_score = plt.figure(figsize=(6,3))

mod = list(results.keys())
f1 = list(results.values())

plt.bar(mod,f1, color = "lightblue", width = 0.5)

plt.xlabel("Model")
plt.ylabel("Macro F1")
plt.xticks(rotation=15)
plt.title("Models Macro F1 Comparison")
plt.show()

print(f"F1 Macro Score na melhor combinação de parametros para RandomForest: {f1_macro_score_rf:.2f}")
print(f"F1 Macro Score na melhor combinação de parametros para XGBoost: {f1_macro_score_xgb:.2f}")
print(f"F1 Macro Score na melhor combinação de parametros para GradientBoost: {f1_macro_score_gradient:.2f}")
print(f"F1 Macro Score na melhor combinação de parametros para VotingEnsemble: {f1_macro_score_voting_ensemble:.2f}")
print(f"F1 Macro Score na melhor combinação de parametros para StackingEnsemble: {f1_macro_score_stacking_ensemble:.2f}")


# ## Save Model's Info

# In[ ]:


results_df = pd.read_csv("../results.csv")

new_entry = {
    "scaler": True,
    "abs": True,
    "grid_search": grid_search_mode,
    "corr_threshold": corr_threshold,
    "xgb_threshold": np.nan,
    "pca": False,
    "n_features": len(corr_train_df.columns),
    "dataset": "corr_xgb_train_df",
    "best_score": max(results.values()),
    "predict_score": np.nan,
    "control_score": np.nan,
    "score": results,
    "random_forest_params": random_forest_params,
    "xgb_params": xgb_params,
    "gradient_params": gradient_params,
    "e2": np.nan,
    "e3": np.nan,
    "e4": np.nan,
    "e5": np.nan,
    "e6": np.nan,
}


# In[ ]:


new_line = pd.DataFrame([new_entry])
results_df = pd.concat([results_df, new_line], ignore_index=True)


# In[ ]:


results_df.to_csv("../results.csv",index=False)
print("Results Updated!")


# ## Write Predicts to CSV

# In[ ]:


def preds_to_csv(preds, df=dummy_df):
    if len(preds) == 100:
        y_pred_original = target_decoder(le_make_train, preds)
        
        df["Result"] = y_pred_original
        
        df.to_csv("../Dataset/dummy_submission.csv", index=False)

        print("CSV updated!\n", y_pred_original)
    else:
        print("Invalid input!")


# In[ ]:


preds_to_csv(ensemble_stacking_pred)


# ## Use Past Data To Build a Model

# In[ ]:


def load_line(index, path="../results.csv"):
    df = pd.read_csv(path)

    line = linha = df.iloc[index]

    dict_line = {}

    for feature,value in line.items():
        try: 
            dict_line[feature] = ast.literal_eval(value) if isinstance(value,str) and value.startswith("{") else value
        except (ValueError, SyntaxError):
            dict_line[feature] = value
    
    return dict_line


# In[ ]:


def get_load_preds(x_train,y_train,rf_params,xgb_params,gra_params,model):
    random_forest_model, random_forest_params = random_forest(x_train, y_train,best_params=rf_params,mode=False)
    print("RandomF >")
    if model == "RandomForest":
        return random_forest_model
        
    xgb_model,xgb_params = xgboost_model(x_train, y_train,best_params=xgb_params,mode=False)
    print("XGB >")
    if model == "XGBoost":
        return xgb_model
    
    gradient_model,gradient_params = gradient_boosting(x_train, y_train,best_params=gra_params,mode=False)
    print("GradientB >")
    if model == "GradientBoost":
        return gradient_model
        
    ensemble_voting_model = ensemble_voting_classifier(x_train,y_train)
    print("VotingEns >")
    if model == "VotingEnsemble":
        return ensemble_voting_model

    ensemble_stacking_model = ensemble_stacking_classifier(x_train,y_train)
    print("StackingEns >")

    return ensemble_stacking_model


# In[ ]:


load_dict = load_line(3)
load_columns = apply_correlation(corr_df,load_dict["corr_threshold"])
load_train_df = scaled_train_df[load_columns].copy()
load_control_df = scaled_control_df[load_columns].copy()
load_test_df = scaled_test_df[load_columns]
load_train_df.loc[:,"Transition_code"] = train_df["Transition_code"].values
load_control_df.loc[:,"Transition_code"] = control_df["Transition_code"].values
x_load_train, x_load_test, y_load_train, y_load_test = define_X_y(load_train_df,load_test_df)
load_random_forest_params = load_dict["random_forest_params"]
load_xgb_params = load_dict["xgb_params"]
load_gradient_params = load_dict["gradient_params"]
load_best_model = max(load_dict["score"],key=load_dict["score"].get)

print(load_best_model)


# In[ ]:





# In[ ]:


model = get_load_preds(x_load_train,y_load_train,load_random_forest_params,load_xgb_params,load_gradient_params,load_best_model)
preds = model.predict(x_load_test)


# In[ ]:


preds_to_csv(preds)

