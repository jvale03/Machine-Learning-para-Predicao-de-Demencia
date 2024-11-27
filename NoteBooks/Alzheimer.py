#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ast
import shap
import pickle
import os
import math


# ## sklearn

# In[2]:


from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_val_score, RepeatedStratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, make_scorer, confusion_matrix, roc_curve, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.inspection import permutation_importance
from scipy.stats import skew, kurtosis


# ## Models

# In[3]:


from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization
from sklearn.svm import SVC


# # Load CSVs

# In[4]:


control_df = pd.read_csv("../Dataset/train_radiomics_occipital_CONTROL.csv")
train_df = pd.read_csv("../Dataset/train_radiomics_hipocamp.csv")
test_df = pd.read_csv("../Dataset/test_radiomics_hipocamp.csv")
dummy_df = pd.read_csv("../Dataset/dummy_submission.csv")


# # Save & Load Data

# In[5]:


uni_path = "../DataSaver/"

def save_stuff(data,path):
    file_path = os.path.join(uni_path, path)
    
    with open(file_path,"wb") as file:
        pickle.dump(data,file)

def load_stuff(path):
    file_path = os.path.join(uni_path,path)

    with open(file_path,"rb") as file:
        data = pickle.load(file)

    return data


# # Data Exploration

# ## Category Encoder and Decoder

# In[6]:


def target_encoder(df, target="Transition"):
    le_make = LabelEncoder()
    df[f"{target}_code"] = le_make.fit_transform(df[target])
    return le_make

def target_decoder(le_make, preds):
    return le_make.inverse_transform(preds)


# In[7]:


le_make_train = target_encoder(train_df)
le_make_control = target_encoder(control_df)


# In[8]:


# Obtenção da distribuição e contagem de cada classe
target_distribution = train_df['Transition_code'].value_counts(normalize=True).sort_index()
target_counts = train_df['Transition_code'].value_counts().sort_index()

plt.figure(figsize=(6, 3.5))
ax = sns.barplot(x=target_distribution.index, y=target_distribution.values)
plt.title("Distribuição da variável alvo (Transition_code)")
plt.xlabel("Classes")
plt.ylabel("Frequência")

le_make = target_encoder(train_df) 
decoded_labels = le_make.inverse_transform(target_distribution.index)

for index, value in enumerate(target_distribution.index):
    ax.text(index, target_distribution[value] + 0.01, f'{target_counts[value]} ({decoded_labels[index]})', ha='center')

plt.show()


# ## MatPlots

# In[9]:


def show_histogram(df,title="histogram"):
    plt.figure(figsize=(13,8))
    plt.subplots_adjust(bottom=0.17)
    plt.title(title)
    sns.histplot(df)
    plt.show()


# In[10]:


def show_pie(df,title="pie"):
    labels = df.unique().tolist()
    counts = df.value_counts()
    sizes = [counts[var_cat] for var_cat in labels]
    _, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct="%1.1f%%",shadow=True)
    ax1.axis("equal")
    plt.title(title)
    plt.show()


# In[11]:


def show_boxplot(df,title="boxplot"):
    plt.figure(figsize=(13,8))
    plt.subplots_adjust(bottom=0.17)
    df.boxplot()
    plt.xticks(rotation=15)
    plt.title(title)
    plt.show()


# In[12]:


def show_heatmap(df,title="correlation heatmap"):
    df = df.select_dtypes(include="number")
    plt.figure(figsize=(6,3.5))
    plt.subplots_adjust(bottom=0.25,left=0.22,right=0.95)
    plt.xticks(rotation=15)
    plt.title(title)
    sns.heatmap(df.corr(),annot=True,cmap="coolwarm",linewidths=0.5)
    plt.show()


# In[13]:


def show_jointplot(df,x_label,y_label,title="jointplot",hue="Transition_code"):
    sns.jointplot(data=df,x=x_label,y=y_label,hue=hue)
    plt.show()


# In[14]:


def show_catplot(df, x_label, y_label, title="catplot", hue="Transition_code", height=3, aspect=1.5):
    sns.catplot(data=df, x=x_label, y=y_label, hue=hue, height=height, aspect=aspect)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


# In[15]:


def show_pairplot(df,hue="Transition_code"):
    sns.pairplot(df,hue=hue)
    plt.show()


# ## Basic Exploration

# In[16]:


def main_exploration(df):
    print(df.shape) 


# In[17]:


def numerical_exploration(df):
    # age exploration
    show_histogram(df["Age"],title="Histogram")
    print(df["Age"].describe())


# In[18]:


train_df.describe()


# In[19]:


numerical_exploration(train_df)


# In[20]:


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


# In[21]:


show_catplot(train_df, "Age", "Transition", hue="Sex")
show_heatmap(train_df[["Age","Transition_code","Sex"]])


# In[22]:


diagnostics_configs_columns = ["diagnostics_Configuration_Settings","diagnostics_Configuration_EnabledImageTypes"]
def diagnostics_configs(df):
    for col in diagnostics_configs_columns:
        print(len(df[col].unique()))


# In[23]:


diagnostics_configs(train_df)


# In[24]:


diagnostics_versions_columns = ["diagnostics_Versions_PyRadiomics","diagnostics_Versions_Numpy","diagnostics_Versions_SimpleITK","diagnostics_Versions_PyWavelet","diagnostics_Versions_Python"] 
def diagnostics_versions_explorer(df):
    for column in diagnostics_versions_columns:
        print(column,": ")
        values = df[column].unique()
        print(values)


# In[25]:


diagnostics_versions_explorer(train_df)


# In[26]:


diagnostics_versions_columns = ["diagnostics_Versions_PyRadiomics","diagnostics_Versions_Numpy","diagnostics_Versions_SimpleITK","diagnostics_Versions_PyWavelet","diagnostics_Versions_Python"] 


# In[27]:


diagnostics_configs_columns = ["diagnostics_Configuration_Settings","diagnostics_Configuration_EnabledImageTypes"]


# In[28]:


unnecessary_columns = diagnostics_versions_columns + diagnostics_configs_columns +["diagnostics_Image-original_Dimensionality","diagnostics_Image-original_Minimum","diagnostics_Image-original_Size","diagnostics_Mask-original_Spacing","diagnostics_Image-original_Spacing","diagnostics_Mask-original_Size","diagnostics_Image-original_Hash","diagnostics_Mask-original_Hash","ID","Image","Mask",'diagnostics_Mask-original_CenterOfMassIndex','diagnostics_Versions_PyRadiomics', 'diagnostics_Versions_Numpy', 'diagnostics_Versions_SimpleITK', 'diagnostics_Versions_PyWavelet', 'diagnostics_Versions_Python', 'diagnostics_Configuration_Settings', 'diagnostics_Configuration_EnabledImageTypes']


# In[29]:


unnecessary_df = pd.DataFrame()
for col in unnecessary_columns+["Transition"]:
    le_make = LabelEncoder()
    unnecessary_df[f"{col}_code"] = le_make.fit_transform(train_df[col])

show_heatmap(unnecessary_df)


# In[30]:


colunas_nao_numericas = train_df.select_dtypes(exclude=['number']).columns

print(train_df[colunas_nao_numericas].head())


# ## Correlations

# In[31]:


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


# In[32]:


rad_corr = top_correlations(train_df,starts_with="lbp",number=20)
show_heatmap(train_df[rad_corr])


# # Data Processing

# ## Drop Unnecessary Columns

# In[33]:


control_df = control_df.drop(columns=unnecessary_columns,axis=1,errors="ignore")
train_df = train_df.drop(columns=unnecessary_columns,axis=1,errors="ignore")
test_df = test_df.drop(columns=unnecessary_columns,axis=1,errors="ignore")


# In[34]:


train_df[['diagnostics_Mask-original_BoundingBox']]


# ## Non Numerical Columns

# In[35]:


# Separar a coluna de BoundingBox em várias colunas
train_df[['x_min', 'y_min', 'largura', 'altura', 'profundidade', 'extra']] = train_df['diagnostics_Mask-original_BoundingBox'].str.strip('()').str.split(',', expand=True).astype(float)

# Separar a coluna de CenterOfMassIndex em várias colunas
train_df[['x_center', 'y_center', 'z_center']] = train_df['diagnostics_Mask-original_CenterOfMass'].str.strip('()').str.split(',', expand=True).astype(float)


# In[36]:


# Separar a coluna de BoundingBox em várias colunas
test_df[['x_min', 'y_min', 'largura', 'altura', 'profundidade', 'extra']] = test_df['diagnostics_Mask-original_BoundingBox'].str.strip('()').str.split(',', expand=True).astype(float)

# Separar a coluna de CenterOfMassIndex em várias colunas
test_df[['x_center', 'y_center', 'z_center']] = test_df['diagnostics_Mask-original_CenterOfMass'].str.strip('()').str.split(',', expand=True).astype(float)


# In[37]:


# Separar a coluna de BoundingBox em várias colunas
control_df[['x_min', 'y_min', 'largura', 'altura', 'profundidade', 'extra']] = control_df['diagnostics_Mask-original_BoundingBox'].str.strip('()').str.split(',', expand=True).astype(float)

# Separar a coluna de CenterOfMassIndex em várias colunas
control_df[['x_center', 'y_center', 'z_center']] = control_df['diagnostics_Mask-original_CenterOfMass'].str.strip('()').str.split(',', expand=True).astype(float)


# In[38]:


train_df = train_df.drop(['diagnostics_Mask-original_BoundingBox', 'diagnostics_Mask-original_CenterOfMass'], axis=1, errors="ignore")
test_df = test_df.drop(['diagnostaics_Mask-original_BoundingBox', 'diagnostics_Mask-original_CenterOfMass'], axis=1, errors="ignore")
control_df = control_df.drop(['diagnostics_Mask-original_BoundingBox', 'diagnostics_Mask-original_CenterOfMass'], axis=1, errors="ignore")


# In[39]:


main_exploration(train_df)


# In[40]:


train_df = train_df.select_dtypes(include=['number'])
control_df = control_df.select_dtypes(include=['number'])
test_df = test_df.select_dtypes(include=['number'])


# ## Data Scaler

# In[41]:


from sklearn.preprocessing import StandardScaler

def data_scaler(df):
    scaler_df = df.drop(columns=["Transition","Transition_code"],errors="ignore")
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(scaler_df),columns=scaler_df.columns)
    return df_scaled


# In[42]:


scaled_train_df = data_scaler(train_df)
scaled_control_df = data_scaler(control_df)
scaled_test_df = data_scaler(test_df)

scaled_train_df["Transition_code"] = train_df["Transition_code"].values
scaled_control_df["Transition_code"] = train_df["Transition_code"].values

scaled_train_df.shape


# ## Correlation Analisys

# In[43]:


corr_df = scaled_train_df.copy()
corr_df.loc[:,"Transition_code"] = train_df["Transition_code"].values
target = "Transition_code"


# In[44]:


corr_threshold_target = 0.01
corr_threshold_features = 0.95

def apply_correlation(df, threshold_target, threshold_features):
    # Remove a coluna "Transition" se existir
    df = df.drop(columns=["Transition"], errors="ignore")
    
    # Calcula a correlação com o target
    correlation_with_target = df.corr()[target].abs()
    important_features = correlation_with_target[correlation_with_target > threshold_target].index.tolist()
    
    if target in important_features:
        important_features.remove(target)
    
    # Remove features que são muito correlacionadas entre si
    correlation_matrix = df[important_features].corr().abs()
    selected_features = important_features.copy()
    
    for i, feature in enumerate(important_features):
        for other_feature in important_features[i+1:]:
            if correlation_matrix.loc[feature, other_feature] > threshold_features:
                if other_feature in selected_features:
                    selected_features.remove(other_feature)
    
    return selected_features


# In[45]:


important_features = apply_correlation(scaled_train_df, corr_threshold_target, corr_threshold_features)
print(len(important_features))


# In[46]:


corr_train_df = scaled_train_df[important_features]
corr_control_df = scaled_control_df[important_features]
corr_test_df = scaled_test_df[important_features]
corr_train_df.shape


# In[47]:


corr_train_df["Transition_code"] = train_df["Transition_code"].values
corr_control_df["Transition_code"] = train_df["Transition_code"].values


# In[48]:


main_exploration(corr_train_df)
main_exploration(corr_control_df)
main_exploration(corr_test_df)


# # Testing Phase

# In[49]:


def define_X_y(train_df, test_df = pd.DataFrame(),random_state=27):
    if test_df.empty:
        X = train_df.drop(columns=["Transition_code","Transition"],errors="ignore")
        y = train_df["Transition_code"]

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y,random_state=random_state)

        return x_train, x_test, y_train, y_test

    else:
        x_train = train_df.drop("Transition_code",axis=1,errors="ignore")
        y_train = train_df["Transition_code"]
        x_test = test_df

        return x_train, x_test, y_train, None


# In[50]:


results = {}
x_train, x_test, y_train, y_test = define_X_y(corr_train_df,corr_test_df)
main_exploration(x_train)
main_exploration(x_test)

scorer = make_scorer(f1_score, average='macro')


# ## Params

# In[51]:


total_samples = len(x_train)

class_counts = np.bincount(y_train)

scale_pos_weight = total_samples / class_counts

weights = {i:w for i,w in enumerate(scale_pos_weight)}

sample_class_weights = {i: total_samples / count for i,count in enumerate(class_counts)}
sample_weights = np.array([sample_class_weights[label] for label in y_train])


# ### Grid Params

# In[52]:


param_grid_rf = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 5],
    'bootstrap': [True, False],
    'class_weight': ['balanced', 'balanced_subsample']
}


param_grid_xgb = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [3, 5, 10],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'min_child_weight': [2, 3, 5],
    'objective': ['multi:softprob'],
    'alpha': [0.01, 0.05, 0.1, 1, 10],
    'lambda': [0.5, 1, 5, 10, 100]
}


param_grid_light = {
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [20, 30, 40],
    'max_depth': [3, 5, 10, 15],
    'min_data_in_leaf': [10, 20, 30],
    'bagging_fraction': [0.5, 0.7, 0.8, 1.0],
    'objective': ['multiclassova'],
}


param_grid_svm = {
    'C': [0.1, 0.5, 1, 2, 5, 7, 10],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['rbf'],
    'class_weight': ['balanced'],
}

param_grid_log = {
    'penalty': ['l1', 'l2', 'elasticnet', None],
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga'],
    'max_iter': [100, 200, 500, 1000],
    'class_weight': ['balanced', None],
    'multi_class': ['ovr', 'multinomial'],
    'tol': [1e-4, 1e-3]
}


# ### Bayes Params

# In[53]:


param_baye_rf = {
    'n_estimators': (100, 300),
    'max_depth': (5, 30),
    'min_samples_split': (2, 10),
    'min_samples_leaf': (1, 4),
    'bootstrap': (0, 1),
    'class_weight': (0, 1),
}


param_baye_xgb = {
    'learning_rate': (0.01, 0.15),
    'n_estimators': (100, 300),
    'max_depth': (5, 20),
    'subsample': (0.7, 1.0),
    'colsample_bytree': (0.7, 1.0),
    'min_child_weight': (1, 15),
}

param_baye_light = {
    'learning_rate': (0.01, 0.2),
    'num_leaves': (20, 50),
    'max_depth': (5, 15),
    'min_data_in_leaf': (10, 30),
    'bagging_fraction': (0.5, 1.0)
}


param_baye_svm = {
    'C': (1, 100),
    'gamma': (0.01, 1),
    'kernel': (0, 1),
    'class_weight': (0, 1),
    'degree': (2, 4),
    'tol': (1e-4, 1e-3)
}

param_baye_log = {
    'C': (0.01, 10),
    'max_iter': (300, 1500),
    'solver': (0,2)  # Selecione os solvers compatíveis com os dados.
}


# ## Models

# ### SVM

# In[54]:


def svm_train_model(x_train,y_train):
    model = SVC(random_state=27,class_weight="balanced", probability=True)
    model.fit(x_train,y_train)

    return model

# Grid Model
def svm_grid_train_model(x_train,y_train, param_grid=param_grid_svm):
    model = SVC(random_state=27)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=27)

    grid_search = RandomizedSearchCV(model, param_grid, cv=cv, n_iter=300,random_state=27, scoring=scorer, n_jobs=-1, verbose=1)
    grid_search.fit(x_train,y_train)
    print(grid_search.best_params_)

    return grid_search.best_estimator_

def objective_svm(C, gamma, kernel, class_weight, degree, tol):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=27)
    
    params = {
        'C': C,
        'gamma': 'scale' if round(gamma) == 0 else gamma,  # 'scale' ou um valor float de gamma
        'kernel': 'linear' if round(kernel) == 0 else 'rbf',  # 'linear' ou 'rbf'
        'class_weight': None if round(class_weight) == 0 else 'balanced',
        'degree': int(degree) if round(kernel) == 1 else 3,  # Degree apenas para kernel polinomial
        'tol': tol,
        'random_state': 27
    }
    
    model = SVC(**params)
    score = cross_val_score(model, x_train, y_train, cv=cv, scoring='accuracy').mean()  # Troque "scorer" por "accuracy"
    return score

def svm_baye_train_model(x_train, y_train, param_baye=param_baye_svm):
    svm_bo = BayesianOptimization(
        f=objective_svm,
        pbounds=param_baye,
        random_state=27,
    )
    
    svm_bo.maximize(init_points=7, n_iter=25)

    best_params = svm_bo.max['params']

    gamma = 'scale' if round(best_params["gamma"]) == 0 else best_params["gamma"]
    kernel = 'linear' if round(best_params["kernel"]) == 0 else 'rbf'
    best_params_updated = {
        'C': best_params['C'],
        'gamma': gamma,
        'kernel': kernel,
        'class_weight': None if round(best_params['class_weight']) == 0 else 'balanced',
        'degree': int(best_params['degree']) if kernel == 'rbf' else 3,  # Degree só para kernel 'rbf'
        'tol': best_params['tol'],
        'random_state': 27
    }

    print(best_params_updated)

    model = SVC(**best_params_updated)
    model.fit(x_train, y_train)

    svm_results = pd.DataFrame(svm_bo.res)
    svm_results.sort_values(by="target", ascending=False, inplace=True)

    return model, svm_results, svm_bo.max


# ### RandomForest

# In[55]:


# Basic Model
def random_forest_model(x_train,y_train):
    model = RandomForestClassifier(random_state=27,class_weight= 'balanced_subsample')
    model.fit(x_train,y_train)
    
    return model

# Grid Model
def random_forest_grid_model(x_train,y_train, param_grid_rf=param_grid_rf):
    model = RandomForestClassifier(random_state=27)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=27)
    
    grid_search = RandomizedSearchCV(model, param_grid_rf, cv=cv, n_iter=300,random_state=27, scoring=scorer, n_jobs=-1, verbose=1)
    grid_search.fit(x_train,y_train)
    print(grid_search.best_params_)

    return grid_search.best_estimator_

#Bayes Model
def objective_random_forest(n_estimators, max_depth, min_samples_split, min_samples_leaf,bootstrap,class_weight):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=27)
    
    params = {
        'n_estimators': int(n_estimators),
        'max_depth': int(max_depth),
        'min_samples_split': int(min_samples_split),
        'min_samples_leaf': int(min_samples_leaf),
        'bootstrap': bool(round(bootstrap)),
        'class_weight': None if round(class_weight) == 0 else 'balanced',
        'random_state': 27
    }
    
    model = RandomForestClassifier(**params)
    score = cross_val_score(model, x_train, y_train, cv=cv, scoring=scorer).mean()
    return score

def random_forest_baye_model(x_train,y_train, param_baye=param_baye_rf):
    rf_bo = BayesianOptimization(
        f=objective_random_forest,
        pbounds=param_baye,
        random_state=27,
    )
    
    rf_bo.maximize(init_points=7, n_iter=25)

    best_params = rf_bo.max['params']
    bootstrap = bool(round(best_params["bootstrap"]))
    best_params_updated = {
        'n_estimators': int(best_params['n_estimators']),
        'max_depth': int(best_params['max_depth']),
        'min_samples_split': int(best_params['min_samples_split']),
        'min_samples_leaf': int(best_params['min_samples_leaf']),
        'bootstrap': bootstrap,
        'class_weight': None if round(best_params['class_weight']) == 0 else 'balanced',
        'random_state': 27
    }

    print(best_params_updated)

    model = RandomForestClassifier(**best_params_updated)
    model.fit(x_train,y_train)

    rf_results = pd.DataFrame(rf_bo.res)
    rf_results.sort_values(by="target",ascending=False,inplace=True)

    return model, rf_results, rf_bo.max


# ### XGBoost

# In[56]:


# Basic Model
def xgboost_model(x_train,y_train):
    model = XGBClassifier(random_state=27,objective= 'multi:softprob')
    model.fit(x_train,y_train, sample_weight=sample_weights)

    return model

# Grid Model
def xgboost_grid_model(x_train,y_train, param_grid_xgb=param_grid_xgb):
    model = XGBClassifier(random_state=27)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=27)

    grid_search = RandomizedSearchCV(model, param_grid_xgb, cv=cv, n_iter=300,random_state=27, scoring=scorer, n_jobs=-1, verbose=1)
    grid_search.fit(x_train,y_train)
    print(grid_search.best_params_)
    

    return grid_search.best_estimator_

# Baye Model
def objective_xgboost(learning_rate, n_estimators, max_depth, subsample, colsample_bytree,min_child_weight):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=27)
    
    params = {
        'learning_rate': learning_rate,
        'n_estimators': int(n_estimators),
        'max_depth': int(max_depth),
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'min_child_weight': min_child_weight,
        'random_state': 27
    }
    
    model = XGBClassifier(**params)
    score = cross_val_score(model, x_train, y_train, cv=cv, scoring=scorer).mean()
    return score
    
def xgboost_baye_model(x_train,y_train, param_baye=param_baye_xgb):
    xgb_bo = BayesianOptimization(
    f=objective_xgboost,
    pbounds=param_baye,
    random_state=27,
    )
    
    xgb_bo.maximize(init_points=7, n_iter=25)

    best_params = xgb_bo.max['params']
    best_params_updated = {
        'learning_rate': best_params['learning_rate'],
        'n_estimators': int(best_params['n_estimators']),
        'max_depth': int(best_params['max_depth']), 
        'subsample': best_params['subsample'],
        'colsample_bytree': best_params['colsample_bytree'],
        'min_child_weight': best_params['min_child_weight'],
        'random_state': 27
    }

    print(best_params_updated)
    
    model = XGBClassifier(**best_params_updated)
    model.fit(x_train,y_train, sample_weight=sample_weights)

    xgb_results = pd.DataFrame(xgb_bo.res)
    xgb_results.sort_values(by="target",ascending=False,inplace=True)

    return model, xgb_results, xgb_bo.max


# ### LightGBM

# In[57]:


# Basic Model
def light_boost_model(x_train,y_train):
    model =  lgb.LGBMClassifier(verbose=-1,objective='multiclassova',class_weight=weights)
    model.fit(x_train,y_train)

    return model

# Grid Model
def light_grid_train_model(x_train,y_train, param_grid=param_grid_light):
    model = lgb.LGBMClassifier(verbose=-1,random_state=27,class_weight=weights)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=27)

    grid_search = RandomizedSearchCV(model, param_grid_light, cv=cv, n_iter=300,random_state=27, scoring=scorer, n_jobs=-1, verbose=1)
    grid_search.fit(x_train,y_train)
    print(grid_search.best_params_)

    return grid_search.best_estimator_

# Bayes Model
def objective_light_boost(learning_rate, num_leaves, max_depth, min_data_in_leaf, bagging_fraction):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=27)
    
    params = {
        'learning_rate': learning_rate,
        'num_leaves': int(num_leaves),
        'max_depth': int(max_depth),
        'min_data_in_leaf': int(min_data_in_leaf),
        'bagging_fraction': bagging_fraction,
        'boosting_type': 'gbdt',
        'objective': 'multiclassova', 
        'num_class': 5,
        'is_unbalance': True,
        'n_jobs': -1,
        'random_state': 27,
        'class_weight': weights,
        'verbose': -1
    }
    
    model = lgb.LGBMClassifier(**params)
    score = cross_val_score(model, x_train, y_train, cv=cv, scoring=scorer).mean()
    return score


def light_baye_train_model(x_train, y_train, param_baye=param_baye_light):
    light_bo = BayesianOptimization(
        f=objective_light_boost,
        pbounds=param_baye,
        random_state=27,
    )
    
    light_bo.maximize(init_points=7, n_iter=20)

    best_params = light_bo.max['params']
    best_params_updated = {
        'learning_rate': best_params['learning_rate'],
        'num_leaves': int(best_params['num_leaves']),
        'max_depth': int(best_params['max_depth']),
        'min_data_in_leaf': int(best_params['min_data_in_leaf']),
        'bagging_fraction': best_params['bagging_fraction'],
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 5,
        'is_unbalance': True,
        'n_jobs': -1,
        'random_state': 27,
        'verbose': -1
    }

    print(best_params_updated)
    
    model = lgb.LGBMClassifier(**best_params_updated)
    model.fit(x_train, y_train)

    light_results = pd.DataFrame(light_bo.res)
    light_results.sort_values(by="target", ascending=False, inplace=True)

    return model, light_results, light_bo.max


# ### Logistic Regression

# In[58]:


# Basic Model
def log_reg_model(x_train,y_train):
    model = LogisticRegression(max_iter=1000,random_state=27)
    model.fit(x_train,y_train)
    
    return model

# Grid Model
def log_grid_train_model(x_train,y_train, param_grid=param_grid_log):
    model = LogisticRegression(random_state=27, verbose=0)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=27)

    grid_search = RandomizedSearchCV(model, param_grid, cv=cv, n_iter=300,random_state=27, scoring=scorer, n_jobs=-1, verbose=1)
    grid_search.fit(x_train,y_train)
    print(grid_search.best_params_)

    return grid_search.best_estimator_

# Função de objetivo para otimização bayesiana
def objective_logistic_regression(C, max_iter, solver):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=27)

    solvers = ['lbfgs', 'liblinear', 'saga']
    solver = solvers[int(round(solver))]  # Converte o índice numérico de volta para string


    params = {
        'C': C,
        'max_iter': int(max_iter),
        'solver': solver,
        'random_state': 27,
        'verbose': 0
    }

    model = LogisticRegression(**params)
    score = cross_val_score(model, x_train, y_train, cv=cv, scoring=scorer).mean()
    return score

# Função principal para regressão logística com BayesOpt
def log_baye_train_model(x_train, y_train, param_baye=param_baye_log):
    log_bo = BayesianOptimization(
        f=objective_logistic_regression,
        pbounds=param_baye,
        random_state=27,
    )
    
    log_bo.maximize(init_points=5, n_iter=25)

    best_params = log_bo.max['params']

    # Mapeando o índice numérico de volta para o solver real
    solvers = ['lbfgs', 'liblinear', 'saga']
    best_solver = solvers[int(round(best_params['solver']))]  # Converte índice para string

    best_params_updated = {
        'C': best_params['C'],
        'max_iter': int(best_params['max_iter']),
        'solver': best_solver,  # Usa o nome do solver mapeado
        'random_state': 27,
        'verbose': 0
    }

    print(best_params_updated)

    model = LogisticRegression(**best_params_updated)
    model.fit(x_train, y_train)

    # Criar DataFrame com resultados da otimização
    log_results = pd.DataFrame(log_bo.res)
    log_results.sort_values(by="target", ascending=False, inplace=True)

    return model, log_results, log_bo.max


# ### GradientBoost

# In[61]:


# Basic Model
def gradient_boost_model(x_train, y_train):
    model = GradientBoostingClassifier(random_state=27)
    model.fit(x_train,y_train)
    
    return model

# Grid Model
def gradient_grid_model(x_train,y_train, param_grid_gb):
    model = GradientBoostingClassifier(random_state=27)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=27)

    grid_search_gb = GridSearchCV(model, param_grid_gb, cv=cv, scoring=scorer, n_jobs=-1, verbose=1)
    grid_search_gb.fit(x_train,y_train)
    print(grid_search_gb.best_params_)

    return grid_search_gb.best_estimator_


# Baye Model
def objective_gradient_boost(learning_rate, n_estimators, max_depth, subsample,min_samples_split,min_samples_leaf):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=27)
    
    params = {
        'learning_rate': learning_rate,
        'n_estimators': int(n_estimators),
        'max_depth': int(max_depth),
        'subsample': subsample,
        'min_samples_split': int(min_samples_split),
        'min_samples_leaf': int(min_samples_leaf),
        'random_state': 27
    }
    
    model = GradientBoostingClassifier(**params)
    score = cross_val_score(model, x_train, y_train, cv=cv, scoring=scorer).mean()
    return score


def gradient_baye_model(x_train,y_train, param_baye):
    gb_bo = BayesianOptimization(
        f=objective_gradient_boost,
        pbounds=param_baye,
        random_state=27,
    )
    
    gb_bo.maximize(init_points=7, n_iter=70)

    best_params = gb_bo.max['params']
    best_params_updated = {
        'learning_rate': best_params['learning_rate'],
        'n_estimators': int(best_params['n_estimators']),
        'max_depth': int(best_params['max_depth']),  
        'subsample': best_params['subsample'],
        'min_samples_split': int(best_params['min_samples_split']), 
        'min_samples_leaf': int(best_params['min_samples_leaf']), 
        'random_state': 27 
    }

    print(best_params_updated)
    
    model = GradientBoostingClassifier(**best_params_updated)
    model.fit(x_train,y_train)

    gb_results = pd.DataFrame(gb_bo.res)
    gb_results.sort_values(by="target",ascending=False,inplace=True)

    return model, gb_results, gb_bo.max


# ### CatBoosting

# In[62]:


# Basic Model
def cat_boost_model(x_train,y_train):
    model = CatBoostClassifier(verbose=False, task_type="GPU")
    model.fit(x_train,y_train)

    return model

# Bayes Model
def objective_cat_boost(learning_rate, iterations, depth, l2_leaf_reg, border_count):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=27)
    
    params = {
        'learning_rate': learning_rate,
        'iterations': int(iterations),
        'depth': int(depth),
        'l2_leaf_reg': l2_leaf_reg,
        'border_count': int(border_count),
        'task_type': 'GPU',
        'random_seed': 27,
        'verbose': 0
    }
    
    model = CatBoostClassifier(**params)
    score = cross_val_score(model, x_train, y_train, cv=cv, scoring=scorer).mean()
    return score


def cat_baye_model(x_train, y_train, param_baye):
    cat_bo = BayesianOptimization(
        f=objective_cat_boost,
        pbounds=param_baye,
        random_state=27,
    )
    
    cat_bo.maximize(init_points=5, n_iter=20)


    best_params = cat_bo.max['params']
    best_params_updated = {
        'learning_rate': best_params['learning_rate'],
        'iterations': int(best_params['iterations']),
        'depth': int(best_params['depth']),
        'l2_leaf_reg': best_params['l2_leaf_reg'],
        'border_count': int(best_params['border_count']),
        'task_type': 'GPU',
        'random_seed': 27,
        'verbose': 0
    }

    print(best_params_updated)
    

    model = CatBoostClassifier(**best_params_updated)
    model.fit(x_train, y_train)


    cat_results = pd.DataFrame(cat_bo.res)
    cat_results.sort_values(by="target", ascending=False, inplace=True)

    return model, cat_results, cat_bo.max


# ### Voting Ensemble

# In[63]:


def voting_ensemble(x_train,y_train,estimators):
    model = VotingClassifier(estimators=estimators, voting="hard")
    model.fit(x_train,y_train)
    
    return model


# ### Stacking Ensemble

# In[64]:


def stacking_ensemble(x_train,y_train,estimators):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=27)
    
    model = StackingClassifier(
        estimators=estimators, 
        final_estimator=LogisticRegression(verbose=0,random_state=27), 
        cv=cv, 
        n_jobs=-1,
    )
    
    model.fit(x_train,y_train)
    
    return model


# ## Models Applier

# In[65]:


def apply_basic_models(x_train,y_train,x_test,y_test,n_repeats=4, title="Models Macro F1 Comparison", rf=1, xgb=1, gradient=0, cat=0, log=1, light=1,svm=1):
        
    if rf:
        rf_model = random_forest_model(x_train,y_train)
        results["RandomForest"] = [rf_model,None]
    else:
        rf_model = None

    if xgb:
        xgb_model = xgboost_model(x_train,y_train)
        results["XGBoost"] = [xgb_model,None]
    else:
        xgb_model = None
        
    if gradient:
        gradient_model = gradient_boost_model(x_train,y_train)
        results["GradientBoost"] = [gradient_model,None]
    else:
        gradient_model = None
        
    if cat:
        cat_model = cat_boost_model(x_train,y_train)
        results["CatBoost"] = [cat_model,None]
    else:
        cat_model = None
        
    if log:
        log_model = log_reg_model(x_train,y_train)
        results["Logistic"] = [log_model,None]
    else:
        log_model = None      

    if light:
        light_model = light_boost_model(x_train,y_train)
        results["LightGBM"] = [light_model,None]
    else:
        light_model = None  

    if svm:
        svm_model = svm_train_model(x_train,y_train)
        results["SVM"] = [svm_model,None]
    else:
        svm_model = None 

    models_comparison(results,title,x_train=x_train,y_train=y_train,n_repeats=n_repeats)

    return rf_model, xgb_model, gradient_model, cat_model, log_model, light_model, svm_model


# ## Models Comparison

# In[66]:


def models_comparison(results, title, x_train, y_train,n_repeats=2):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=27)
    
    for result in results:
        if results[result][1] is None:
            # Calcular F1 Score usando cross-validation
            f1_scores = cross_val_score(
                results[result][0], x_train, y_train, cv=cv, scoring=make_scorer(f1_score, average="macro")
            )
            results[result][1] = round(f1_scores.mean(),4)
            print(f"F1 Macro Score em {result}: {results[result][1]} ± {round(f1_scores.std(),3)}")
        
        else:
            print(f"F1 Macro Score em {result}: {results[result][1]}")
        
    
    # Criar gráfico
    models_score = plt.figure(figsize=(6, 3))

    mod = list(results.keys())
    f1 = list([score[1] for score in results.values()])
    
    plt.bar(mod, f1, color="lightblue", width=0.5)
    plt.xlabel("Modelo")
    plt.ylabel("Macro F1")
    plt.xticks(rotation=15)
    plt.title(title)
    plt.show()


# ## MultiClass Analysis

# In[67]:


from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def class_accuracy_cv(model, x_train, y_train):
    # Obtém previsões para todos os folds de validação cruzada
    y_pred = cross_val_predict(model, x_train, y_train, cv=5)  # Aqui 'cv=5' é o número de folds, pode ser ajustado
    
    # Calcula a matriz de confusão
    conf_matrix = confusion_matrix(y_train, y_pred)
    
    # Acurácia por classe
    class_accuracies = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    class_accuracies_percentage = class_accuracies * 100
    total_per_class = np.sum(conf_matrix, axis=1) 
    correct_per_class = np.diag(conf_matrix)
    
    # Plot de acurácia por classe
    plt.figure(figsize=(7, 4))
    classes = np.unique(y_train)
    plt.bar(classes, class_accuracies_percentage, color='skyblue', alpha=0.8)
    plt.xlabel("Classes")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy per Class (Cross-Validation)")
    plt.ylim(0, 100)
    
    # Adiciona os valores das contagens no gráfico
    for i, v in enumerate(class_accuracies_percentage):
        text = f"{correct_per_class[i]}/{total_per_class[i]} ({v:.1f}%)"
        plt.text(classes[i], v + 2, text, ha='center', fontsize=10)
    
    plt.show()
    
    # Plot da matriz de confusão
    plt.figure(figsize=(7, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True Values")
    plt.title("Confusion Matrix (Cross-Validation)")
    plt.show()


# ## ROC & AUC Analysis

# In[68]:


def roc_auc_mm_cv(models, X_train, y_train, cv=5):
    # Binariza o target para multiclasse
    y_train_bin = label_binarize(y_train, classes=[0, 1, 2, 3, 4])
    n_classes = y_train_bin.shape[1]

    # Define o número de linhas e colunas para a grade de subgráficos
    n_cols = 2  # Número fixo de colunas
    n_rows = math.ceil(n_classes / n_cols)  # Calcula o número de linhas necessárias

    # Configura os subgráficos e aumenta o tamanho da figura
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    axs = axs.flatten()  # Achata a matriz de axs para facilitar o acesso

    colors = ['blue', 'green', 'red', 'purple', 'orange']  # Cores para cada modelo

    # Dicionário para armazenar as AUCs médias por classe para cada modelo
    aucs_per_class = {name: [] for name in models.keys()}

    # Cross-validation e cálculo das AUCs
    for classe in range(n_classes):
        ax = axs[classe]  # Acessa o subplot correspondente à classe

        for i, (name, model) in enumerate(models.items()):
            # Obtém as previsões e probabilidades usando cross-validation
            y_score = cross_val_predict(model, X_train, y_train, cv=cv, method='predict_proba')

            # Calcula a curva ROC e AUC para a classe atual e o modelo atual
            fpr, tpr, _ = roc_curve(y_train_bin[:, classe], y_score[:, classe])
            auc = roc_auc_score(y_train_bin[:, classe], y_score[:, classe])

            # Adiciona a AUC à lista de AUCs por classe
            aucs_per_class[name].append(auc)

            # Plota a curva ROC para o modelo na classe atual
            ax.plot(fpr, tpr, color=colors[i], linestyle='--', label=f'{name} (AUC = {auc:.2f})')

        # Linha de referência (modelo aleatório)
        ax.plot([0, 1], [0, 1], 'k--', label='Aleatório (AUC = 0.5)')
        ax.set_title(f'Classe {classe}')
        ax.set_xlabel('Falsos Positivos (FPR)')

        # Apenas o primeiro gráfico precisa do rótulo do eixo Y
        if classe % n_cols == 0:
            ax.set_ylabel('Verdadeiros Positivos (TPR)')

        # Adiciona a legenda em cada subgráfico
        ax.legend(loc='lower right')

    # Remove subgráficos extras, caso o número de classes não preencha todos os subgráficos
    for i in range(n_classes, len(axs)):
        fig.delaxes(axs[i])

    # Título principal
    plt.suptitle('Curvas ROC Multiclasse Comparativas (One-vs-Rest) - Cross-Validation')
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Ajusta o layout para incluir o título
    plt.show()

    # Calcula o MM (média das AUCs médias por classe)
    auc_means = {name: np.mean(aucs) for name, aucs in aucs_per_class.items()}
    mm = np.mean(list(auc_means.values()))  # Média das médias

    # Exibe o MM para todos os modelos
    print("Média das AUCs para cada modelo:")
    for name, mean_auc in auc_means.items():
        print(f'{name}: {mean_auc:.2f}')
    
    print(f'Média geral das AUCs (MM): {mm:.2f}')
    return mm


# In[69]:


def bayes_visualization(params,bayes_results,best_hyperparameters):
    param_names = list(params.keys())

    cols = 3
    rows = math.ceil(len(param_names)/cols)
    
    fig, axes = plt.subplots(rows,cols,figsize=(4*cols,4*rows))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    for i, param in enumerate(param_names):
        if param != "target":
            ax = axes[i//3, i%3]
            ax.plot(bayes_results['params'].apply(lambda x: x[param]),
                bayes_results['target'], 'bo-', lw=1, markersize=4)
            ax.set_title(f'Optimization of {param}')
            ax.set_xlabel(param)
            ax.set_ylabel('F1 Macro Score')
     
            best_value = best_hyperparameters['params'][param]
            ax.plot(best_value, best_hyperparameters['target'], 'yo', markersize=6)
            
    for i in range(len(param_names), 6):
        fig.delaxes(axes.flatten()[i])
 
    plt.show()


# # Models Tester

# In[ ]:


rf_model,xgb_model,gradient_model,cat_model, log_model,light_model, svm_model = apply_basic_models(x_train,y_train,x_test,y_test,n_repeats=3)


# In[ ]:


stacking_model = stacking_ensemble(x_train,y_train,[("rf",rf_model),("xgb",xgb_model),("light",light_model)])
results["StackingBasicSVM"] = [stacking_model,None]
stacking_model_svm = stacking_ensemble(x_train,y_train,[("svm",svm_model),("rf",rf_model),("xgb",xgb_model),("light",light_model)])
results["StackingBasicSVM"] = [stacking_model_svm,None]
stacking_model_log = stacking_ensemble(x_train,y_train,[("log",log_model),("svm",svm_model),("rf",rf_model),("xgb",xgb_model),("light",light_model)])
results["StackingBasicLog"] = [stacking_model_log,None]

models_comparison(results,"Ensemble",x_train=x_train,y_train=y_train,n_repeats=3)


# In[ ]:


class_accuracy_cv(stacking_model_log,x_train,y_train)


# ## Bayesian Optimization

# In[ ]:


results = {}

rf_baye_model, rf_baye_results, best_params = random_forest_baye_model(x_train,y_train)
results["RandomForestBaye"] = [rf_baye_model,None]
bayes_visualization(param_baye_rf, rf_baye_results, best_params)
models_comparison(results, "BayeSearch",x_train=x_train,y_train=y_train)

xgb_baye_model, xgb_baye_results, best_params = xgboost_baye_model(x_train,y_train)
results["XGBoostBaye"] = [xgb_baye_model,None]
bayes_visualization(param_baye_xgb, xgb_baye_results, best_params)
models_comparison(results, "BayeSearch",x_train=x_train,y_train=y_train)

light_baye_model, light_baye_results, best_params = light_baye_train_model(x_train,y_train)
results["LightBoostBaye"] = [light_baye_model,None]
bayes_visualization(param_baye_light, light_baye_results, best_params)
models_comparison(results, "BayeSearch",x_train=x_train,y_train=y_train)

svm_baye_model, svm_baye_results, best_params = svm_baye_train_model(x_train,y_train)
results["SVMBaye"] = [svm_baye_model,None]
bayes_visualization(param_baye_svm, svm_baye_results, best_params)
models_comparison(results, "BayeSearch",x_train=x_train,y_train=y_train)

log_baye_model, log_baye_results, best_params = log_baye_train_model(x_train,y_train)
results["LogBaye"] = [log_baye_model,None]
models_comparison(results, "BayeSearch",x_train=x_train,y_train=y_train)


# In[70]:


"""
save_stuff(rf_baye_model, f"Models/rf_baye_model_{str(round(results['RandomForestBaye'][1], 2)).replace('.', '_')}.pkl")
save_stuff(xgb_baye_model, f"Models/xgb_baye_model_{str(round(results['XGBoostBaye'][1], 2)).replace('.', '_')}.pkl")
save_stuff(light_baye_model, f"Models/light_baye_model_{str(round(results['LightBoostBaye'][1], 2)).replace('.', '_')}.pkl")
save_stuff(svm_baye_model, f"Models/svm_baye_model_{str(round(results['SVMBaye'][1], 2)).replace('.', '_')}.pkl")
save_stuff(log_baye_model, f"Models/log_baye_model_{str(round(results['LogBaye'][1], 2)).replace('.', '_')}.pkl")
"""

rf_baye_model = load_stuff(f"Models/rf_baye_model_0_35.pkl")
xgb_baye_model = load_stuff(f"Models/xgb_baye_model_0_35.pkl")
light_baye_model = load_stuff(f"Models/light_baye_model_0_38.pkl")
svm_baye_model = load_stuff(f"Models/svm_baye_model_0_31.pkl")
log_baye_model = load_stuff(f"Models/log_baye_model_0_32.pkl")


# In[71]:


rf_baye_model = RandomForestClassifier(**rf_baye_model.get_params())
xgb_baye_model = XGBClassifier(**xgb_baye_model.get_params())
light_baye_model = lgb.LGBMClassifier(**light_baye_model.get_params())
svm_baye_model = SVC(**svm_baye_model.get_params())
log_baye_model = LogisticRegression(**log_baye_model.get_params())


# In[ ]:


stacking_model_log = stacking_ensemble(x_train,y_train,[("log",log_baye_model),("svm",svm_baye_model),("rf",rf_baye_model),("xgb",xgb_baye_model),("light",light_baye_model)])
results["StackingBayeLog"] = [stacking_model_log,None]

models_comparison(results,"BayeEnsemble",x_train=x_train,y_train=y_train,n_repeats=4)


# In[ ]:


class_accuracy_cv(stacking_model_log,x_train,y_train)


# # SHAP Analysis

# In[72]:


X_shap_init = corr_train_df.drop("Transition_code", axis=1)  # Features
y_shap_init = corr_train_df["Transition_code"]

X_shap_init.shape


# In[ ]:


rf_baye_model.fit(X_shap_init, y_shap_init)
xgb_baye_model.fit(X_shap_init, y_shap_init)
light_baye_model.fit(X_shap_init, y_shap_init)
log_baye_model.fit(X_shap_init, y_shap_init)

explainer = shap.TreeExplainer(rf_baye_model,X_shap_init)
shap_values_rf = explainer(X_shap_init)
explainer = shap.TreeExplainer(xgb_baye_model,X_shap_init)
shap_values_xgb = explainer(X_shap_init)
explainer = shap.TreeExplainer(light_baye_model,X_shap_init)
shap_values_light = explainer(X_shap_init)
explainer = shap.LinearExplainer(log_baye_model,X_shap_init)
shap_values_log = explainer(X_shap_init)


# In[74]:


"""
save_stuff(shap_values_rf,"SHAP_Values/shap_values_rf.pkl")
save_stuff(shap_values_xgb,"SHAP_Values/shap_values_xgb.pkl")
save_stuff(shap_values_light,"SHAP_Values/shap_values_light.pkl")
save_stuff(shap_values_log,"SHAP_Values/shap_values_log.pkl")
"""
shap_values_rf = load_stuff("SHAP_Values/shap_values_rf.pkl")
shap_values_xgb = load_stuff("SHAP_Values/shap_values_xgb.pkl")
shap_values_light = load_stuff("SHAP_Values/shap_values_light.pkl")
shap_values_log = load_stuff("SHAP_Values/shap_values_log.pkl")


# ## Deep SHAP Analysis

# ### Methods

# In[75]:


def get_global_shap_info_df(shap_values_list, X):
    all_shap_values = []

    for shap_values in shap_values_list:
        if hasattr(shap_values, 'values'):
            shap_array = shap_values.values
        else:
            shap_array = shap_values 
        
        all_shap_values.append(shap_array)
    
    # Concatena todos os SHAP values dos modelos
    all_shap_values = np.concatenate(all_shap_values, axis=0)  # Concatena ao longo do eixo 0 (modelos)
    
    abs_shap_values = np.abs(all_shap_values)
    
    # Calcula a média global dos SHAP values
    feature_shap_mean = np.mean(abs_shap_values, axis=(0, 2))  # Média global dos SHAP values absolutos
    feature_shap_max = np.max(abs_shap_values, axis=(0, 2))  # Máximo global dos SHAP values absolutos
    feature_shap_std = np.std(abs_shap_values, axis=(0, 2))  # Desvio padrão global dos SHAP values absolutos
    feature_shap_positive_ratio = np.mean(all_shap_values > 0, axis=(0, 2))  # Proporção de SHAP values positivos
    feature_shap_negative_ratio = np.mean(all_shap_values < 0, axis=(0, 2))  # Proporção de SHAP values negativos
    
    # Cria o DataFrame com as importâncias e métricas adicionais
    feature_shap_importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": feature_shap_mean,
        "importance_max": feature_shap_max,
        "importance_std": feature_shap_std,
        "positive_ratio": feature_shap_positive_ratio,
        "negative_ratio": feature_shap_negative_ratio
    }).sort_values(by="importance_mean", ascending=False)  # Ordena pelo valor médio global dos SHAP

    return feature_shap_importance_df


# In[76]:


def get_shap_info_df(shap_values, X):
    # Verifica se shap_values é um objeto shap.Explanation ou um ndarray
    if hasattr(shap_values, 'values'):
        shap_array = shap_values.values
    else:
        shap_array = shap_values 
    
    abs_shap_values = np.abs(shap_array)
    
    feature_shap_mean = np.mean(abs_shap_values, axis=(0, 2))  # Média dos SHAP values absolutos
    feature_shap_max = np.max(abs_shap_values, axis=(0, 2))  # Máximo dos SHAP values absolutos
    feature_shap_std = np.std(abs_shap_values, axis=(0, 2))  # Desvio padrão dos SHAP values absolutos
    feature_shap_positive_ratio = np.mean(shap_array > 0, axis=(0, 2))
    feature_shap_negative_ratio = np.mean(shap_array < 0, axis=(0, 2))

    
    # Cria o DataFrame com as importâncias e métricas adicionais
    feature_shap_importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": feature_shap_mean,
        "importance_max": feature_shap_max,
        "importance_std": feature_shap_std,
        "positive_ratio": feature_shap_positive_ratio,
        "negative_ratio": feature_shap_negative_ratio
    }).sort_values(by="importance_mean", ascending=False)  # Ordena pelo valor máximo de SHAP

    return feature_shap_importance_df


# In[77]:


def get_shap_info_per_class(shap_values, X):
    # Verifica se shap_values é um objeto shap.Explanation ou um ndarray
    if hasattr(shap_values, 'values'):
        shap_array = shap_values.values
    else:
        shap_array = shap_values 

    abs_shap_values = np.abs(shap_array)  # Valores absolutos dos SHAP values
    n_classes = shap_array.shape[2]  # Número de classes
    
    # Normaliza os valores absolutos pelo máximo global de cada classe
    max_global_per_class = np.max(abs_shap_values, axis=(0, 1))  # Máximo global por classe
    
    for class_idx in range(n_classes):
        # Extrai SHAP values para a classe atual
        class_abs_shap_values = abs_shap_values[:, :, class_idx]
        class_shap_values = shap_array[:, :, class_idx]
        
        # Calcula métricas normalizadas
        feature_shap_mean = np.mean(class_abs_shap_values, axis=0)  # Média
        feature_shap_max = np.max(class_abs_shap_values, axis=0)  # Máximo
        feature_shap_std = np.std(class_abs_shap_values, axis=0)  # Desvio padrão
        feature_shap_positive_ratio = np.mean(class_shap_values > 0, axis=0)  # Razão de valores positivos
        feature_shap_negative_ratio = np.mean(class_shap_values < 0, axis=0)  # Razão de valores negativos

        # Cria DataFrame com as informações
        class_df = pd.DataFrame({
            "feature": X.columns,
            "importance_mean": feature_shap_mean,
            "importance_max": feature_shap_max,
            "importance_std": feature_shap_std,
            "positive_ratio": feature_shap_positive_ratio,
            "negative_ratio": feature_shap_negative_ratio
        }).sort_values(by="importance_mean", ascending=False)  # Ordena pelo valor médio

        # Salva em CSV
        class_df.to_csv(f"../Dataset/SHAP_Values/class{class_idx}_shap_values.csv", index=False)


# In[78]:


def new_shap_values(shap_values,discard_features):
    shap_array = shap_values.values
    feature_names = shap_values.feature_names
    
    features_a_remover = discard_features if isinstance(discard_features, list) else [discard_features]
    features_a_remover = set(features_a_remover) & set(list(feature_names))
    indices_a_remover = [feature_names.index(feature) for feature in features_a_remover]

    shap_values_filtrados = np.delete(shap_array, indices_a_remover, axis=1)
    
    features_restantes = [f for i, f in enumerate(feature_names) if i not in indices_a_remover]
    shap_values_filtrados = shap.Explanation(values=shap_values_filtrados, 
                                             base_values=shap_values_xgb.base_values, 
                                             feature_names=features_restantes)

    return shap_values_filtrados


# In[79]:


def clean_shap_df(df,discard):
    shap_importances_df_cleaned = df[~df['feature'].isin(discard)]
    shap_importances_df_cleaned.to_csv("../Dataset/SHAP_Values/global_shap_values.csv",index=False)
    return shap_importances_df_cleaned


# In[80]:


def sorted_shap_values(shap_values,X_shap, class_index):
    if hasattr(shap_values, 'values'):
        shap_array = shap_values.values
    else:
        shap_array = shap_values 

    shap_class_values = shap_array[:, :, class_index]

    mean_shap_values = np.mean(np.abs(shap_class_values), axis=0)

    shap_df = pd.DataFrame(mean_shap_values, index=X_shap.columns, columns=["mean_shap_value"])

    sorted_features = shap_df.sort_values(by="mean_shap_value", ascending=False)

    return sorted_features


# In[81]:


def shap_values_df_analysis(df):
    
    fig, axes = plt.subplots(4, 2, figsize=(14, 15))
    fig.suptitle("Distribuição dos Valores de Importância das Features", fontsize=16)
    
    importance_columns = df.drop(columns=["feature"]).columns
    
    for idx, col in enumerate(importance_columns):
        ax = axes[idx // 2, idx % 2]
        ax.hist(df[col], bins=30, color='teal', edgecolor='black', alpha=0.7)
        ax.set_title(f'Distribuição de {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequência')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# In[82]:


def show_shap_importance_summary_plot(shap_values,X_shap, classe):
    sorted_shap_values_df = sorted_shap_values(shap_values,X_shap,classe)
    sorted_columns = sorted_shap_values_df.index
    
    n_features = len(sorted_columns)
    n_features_per_plot = 10
    
    feature_indices = [X_shap.columns.get_loc(feature) for feature in sorted_columns]

    for i in range(0, n_features, n_features_per_plot):
        selected_columns = feature_indices[i:i + n_features_per_plot]

        shap.summary_plot(shap_values[:, selected_columns, classe])
        
        plt.show()


# In[83]:


def show_shap_importance_heatmap(shap_values,X_shap,classe):
    sorted_shap_values_df = sorted_shap_values(shap_values,X_shap,classe)
    sorted_columns = sorted_shap_values_df.index
    
    n_features = len(sorted_columns)
    n_features_per_plot = 10
    
    feature_indices = [X_shap.columns.get_loc(feature) for feature in sorted_columns]

    for i in range(0, n_features, n_features_per_plot):
        selected_columns = feature_indices[i:i + n_features_per_plot]
        
        shap.plots.heatmap(shap_values[:, selected_columns, classe])
        
        plt.show()


# ### Global
# 1. remover features que tenham `SHAP Value` = 0 para a média de todos os modelos.
# 2. caso seja insuficiente, aplicar um threshold muito baixo para remover as insignificantes
# 3. analisar independentemente os SHAP Values de cada modelo
# 4. tentar utilizar apenas as features importantes para cada modelo e depois fazer ensemble (cada modelo usa um dataset)

# #### RF

# In[84]:


shap_importances_df_rf = get_shap_info_df(shap_values_rf,X_shap_init)
shap_importances_df_rf.to_csv("../Dataset/SHAP_Values/global_shap_values_rf.csv",index=False)


# In[85]:


shap_importances_df_rf.describe()


# In[86]:


discard_features_rf = shap_importances_df_rf[
        (shap_importances_df_rf["importance_mean"] <= 0.00025)
    ]["feature"].tolist()
print(len(discard_features_rf))


# In[87]:


x_train_rf = corr_train_df.drop(columns=discard_features_rf)
x_test_rf = corr_test_df.drop(columns=discard_features_rf)
x_train_rf = x_train_rf.drop(columns=["Transition_code"])

x_train_rf.shape


# In[88]:


rf_shap_model = random_forest_model(x_train_rf,y_train)
results["rf_shap_model"] = [rf_shap_model,None]


# In[89]:


models_comparison(results,"Ensemble",x_train=x_train_rf,y_train=y_train,n_repeats=5)
results={}


# #### XGB

# In[90]:


shap_importances_df_xgb = get_shap_info_df(shap_values_xgb,X_shap_init)
shap_importances_df_xgb.to_csv("../Dataset/SHAP_Values/global_shap_values_xgb.csv",index=False)


# In[91]:


shap_importances_df_xgb.describe()


# In[92]:


discard_features_xgb = shap_importances_df_xgb[
        (shap_importances_df_xgb["importance_mean"] <= 0.004)
    ]["feature"].tolist()
print(len(discard_features_xgb))


# In[93]:


x_train_xgb = corr_train_df.drop(columns=discard_features_xgb)
x_test_xgb = corr_test_df.drop(columns=discard_features_xgb)
x_train_xgb = x_train_xgb.drop(columns=["Transition_code"])
x_train_xgb.shape


# In[94]:


xgb_shap_model = xgboost_model(x_train_xgb,y_train)
results["xgb_shap_model"] = [xgb_shap_model,None]


# In[95]:


models_comparison(results,"Ensemble",x_train=x_train_xgb,y_train=y_train,n_repeats=5)
results={}


# #### LGB

# In[96]:


shap_importances_df_light = get_shap_info_df(shap_values_light,X_shap_init)
shap_importances_df_light.to_csv("../Dataset/SHAP_Values/global_shap_values_light.csv",index=False)


# In[97]:


shap_importances_df_light.describe()


# In[98]:


discard_features_light = shap_importances_df_light[
        (shap_importances_df_light["importance_mean"] <= 0.005)
    ]["feature"].tolist()
print(len(discard_features_light))


# In[99]:


x_train_light = corr_train_df.drop(columns=discard_features_light)
x_test_light = corr_test_df.drop(columns=discard_features_light)
x_train_light = x_train_light.drop(columns=["Transition_code"])
x_train_light.shape


# In[100]:


light_shap_model = light_boost_model(x_train_light,y_train)
results["light_shap_model"] = [light_shap_model,None]


# In[101]:


models_comparison(results,"Ensemble",x_train=x_train_light,y_train=y_train,n_repeats=5)
results={}


# #### Log

# In[102]:


shap_importances_df_log = get_shap_info_df(shap_values_log,X_shap_init)
shap_importances_df_log.to_csv("../Dataset/SHAP_Values/global_shap_values_log.csv",index=False)


# In[103]:


discard_features_log = shap_importances_df_log[
        (shap_importances_df_log["importance_mean"] <= 0.03)
    ]["feature"].tolist()
print(len(discard_features_log))


# In[104]:


shap_importances_df_log.describe()


# In[105]:


x_train_log = corr_train_df.drop(columns=discard_features_log)
x_test_log = corr_test_df.drop(columns=discard_features_log)
x_train_log = x_train_log.drop(columns=["Transition_code"])
x_train_log.shape


# In[106]:


log_shap_model = log_reg_model(x_train_log,y_train)
results["log_shap_model"] = [log_shap_model,None]


# In[107]:


models_comparison(results,"Ensemble",x_train=x_train_light,y_train=y_train,n_repeats=5)
results={}


# #### SVM

# In[108]:


shap_importances_df_global = get_global_shap_info_df([shap_values_rf,shap_values_xgb,shap_values_light,shap_values_log],X_shap_init)
shap_importances_df_global.to_csv("../Dataset/SHAP_Values/global_shap_values.csv",index=False)


# In[109]:


shap_importances_df_global.describe()


# In[110]:


discard_features_global = shap_importances_df_global[
        (shap_importances_df_global["importance_mean"] <= 0.02)
    ]["feature"].tolist()
print(len(discard_features_global))


# In[111]:


x_train_svm = corr_train_df.drop(columns=discard_features_global)
x_test_svm = corr_test_df.drop(columns=discard_features_global)
x_train_svm = x_train_svm.drop(columns=["Transition_code"])

x_train_svm.shape


# In[112]:


svm_shap_model = svm_train_model(x_train_svm,y_train)
results["svm_shap_model"] = [svm_shap_model,None]


# In[113]:


models_comparison(results,"Ensemble",x_train=x_train_svm,y_train=y_train,n_repeats=5)
results={}


# #### Ensemble Manual

# In[114]:


preds_rf = rf_shap_model.predict_proba(x_test_rf)
preds_xgb = xgb_shap_model.predict_proba(x_test_xgb)
preds_light = light_shap_model.predict_proba(x_test_light)
preds_log = log_shap_model.predict_proba(x_test_log)
preds_svm = svm_shap_model.predict_proba(x_test_svm)


# In[115]:


stacked_models = np.hstack([preds_rf,preds_xgb,preds_light,preds_log,preds_svm])


# In[118]:


meta_model = LogisticRegression(random_state=27)
final_preds = meta_model.predict(stacked_models)


# ### MultiClass

# In[ ]:


get_shap_info_per_class(shap_values,X_shap)


# #### Class 0

# In[ ]:


class0_df = pd.read_csv("../Dataset/SHAP_Values/class0_shap_values.csv")
class0_df.describe()


# In[ ]:


class0_discard_features = class0_df[
        (class0_df["importance_mean"] <= 0.00) | 
        ((class0_df["negative_ratio"] - class0_df["positive_ratio"] > 0.5) & 
        (class0_df["importance_mean"] < 0.02))
    ]["feature"].tolist()
print(len(class0_discard_features))


# #### Class 1

# In[ ]:


class1_df = pd.read_csv("../Dataset/SHAP_Values/class1_shap_values.csv")
class1_df.describe()


# In[ ]:


class1_discard_features = class1_df[
        (class1_df["importance_mean"] <= 0.0) | 
        ((class1_df["negative_ratio"] - class1_df["positive_ratio"] > 0.5) & 
        (class1_df["importance_mean"] < 0.02))
    ]["feature"].tolist()
print(len(class1_discard_features))


# #### Class 2

# In[ ]:


class2_df = pd.read_csv("../Dataset/SHAP_Values/class2_shap_values.csv")
class2_df.describe()


# In[ ]:


class2_discard_features = class2_df[
        (class2_df["importance_mean"] <= 0.0) | 
        ((class2_df["negative_ratio"] - class2_df["positive_ratio"] > 0.5) & 
        (class2_df["importance_mean"] < 0.02))
    ]["feature"].tolist()
print(len(class2_discard_features))


# #### Class 3

# In[ ]:


class3_df = pd.read_csv("../Dataset/SHAP_Values/class3_shap_values.csv")
class3_df.describe()


# In[ ]:


class3_discard_features = class3_df[
        (class3_df["importance_mean"] <= 0.0) | 
        ((class3_df["negative_ratio"] - class0_df["positive_ratio"] > 0.5) & 
        (class3_df["importance_mean"] < 0.02))
    ]["feature"].tolist()
print(len(class3_discard_features))


# #### Class 4

# In[ ]:


class4_df = pd.read_csv("../Dataset/SHAP_Values/class4_shap_values.csv")
class4_df.describe()


# In[ ]:


class4_discard_features = class4_df[
        (class4_df["importance_mean"] <= 0.0) | 
        ((class4_df["negative_ratio"] - class0_df["positive_ratio"] > 0.5) & 
        (class4_df["importance_mean"] < 0.02))
    ]["feature"].tolist()
print(len(class4_discard_features))


# In[ ]:


# Lista das features a descartar para cada classe
class_discard_features = [class0_discard_features, class1_discard_features, class2_discard_features, class3_discard_features, class4_discard_features]

# Contar quantas classes descartam cada feature
feature_counts = {}
for discard_list in class_discard_features:
    for feature in discard_list:
        if feature not in feature_counts:
            feature_counts[feature] = 0
        feature_counts[feature] += 1

# Selecionar as features que são descartadas em 3 ou mais classes
combined_classes = [feature for feature, count in feature_counts.items() if count > 3]

# Verificar quantas features serão descartadas
print(len(combined_classes))


# In[ ]:


shap_train_df = shap_train_df.drop(columns=combined_classes)
shap_control_df = shap_control_df.drop(columns=combined_classes)
shap_test_df = shap_test_df.drop(columns=combined_classes)
shap_train_df.to_csv("../Dataset/train_df_without_shap_low_values.csv",index=False)
shap_importances_df_cleaned = clean_shap_df(shap_importances_df_cleaned,combined_classes)


# In[ ]:


shap_values = new_shap_values(shap_values,combined_classes)
X_shap = shap_train_df.drop("Transition_code",axis=1)


# In[ ]:


shap_importances_df_cleaned.describe()


# #### SHAP Plots

# In[ ]:


show_shap_importance_summary_plot(shap_values,X_shap, 0)
show_shap_importance_heatmap(shap_values,X_shap, 0)


# In[ ]:


show_shap_importance_summary_plot(shap_values,X_shap, 1)
show_shap_importance_heatmap(shap_values,X_shap, 1)


# In[ ]:


show_shap_importance_summary_plot(shap_values,X_shap, 2)
show_shap_importance_heatmap(shap_values,X_shap, 2)


# In[ ]:


show_shap_importance_summary_plot(shap_values,X_shap, 3)
show_shap_importance_heatmap(shap_values,X_shap, 3)


# In[ ]:


show_shap_importance_summary_plot(shap_values,X_shap, 4)
show_shap_importance_heatmap(shap_values,X_shap, 4)


# # Models Tester

# In[ ]:


results = {}
x_train, x_test, y_train, y_test = define_X_y(shap_train_df,shap_test_df,random_state=12)
main_exploration(x_train)
main_exploration(x_test)


# ## Basic Models

# In[ ]:


rf_model, xgb_model, gradient_model, cat_model, log_model, light_model,svm_model = apply_basic_models(x_train,y_train,x_test,y_test,n_repeats=3)


# In[ ]:


#stacking_model = stacking_ensemble(x_train,y_train,[("rf",rf_model),("xgb",xgb_model),("light",light_model)])
#results["StackingBasic"] = [stacking_model,None]
#stacking_model_svm = stacking_ensemble(x_train,y_train,[("svm",svm_model),("rf",rf_model),("xgb",xgb_model),("light",light_model)])
#results["StackingBasicSVM"] = [stacking_model_svm,None]
stacking_model_log = stacking_ensemble(x_train,y_train,[("log",log_model),("svm",svm_model),("rf",rf_model),("xgb",xgb_model),("light",light_model)])
results["StackingBasicLog"] = [stacking_model_log,None]
models_comparison(results,"Ensemble",x_train=x_train,y_train=y_train,n_repeats=2)

#stacking_model_log = stacking_ensemble(x_train,y_train,[("log",log_baye_model),("svm",svm_baye_model),("rf",rf_baye_model),("xgb",xgb_baye_model),("light",light_baye_model)])
#results["StackingBayeLog"] = [stacking_model_log,None]


# In[ ]:


class_accuracy_cv(stacking_model_log,x_train,y_train)
class_accuracy_cv(log_model,x_train,y_train)
class_accuracy_cv(svm_model,x_train,y_train)
class_accuracy_cv(rf_model,x_train,y_train)
class_accuracy_cv(xgb_model,x_train,y_train)
class_accuracy_cv(light_model,x_train,y_train)


# In[ ]:


roc_auc(models,x_train, y_train, x_test, y_test)
models = {"RF": rf_model,"XGB": xgb_model,"Light":light_model,"Stacking": stacking_model_svm}


# ## GridSearch Tuning

# In[ ]:


results = {}
svm_grid_model = svm_grid_train_model(x_train,y_train)
results["SVMGrid"] = [svm_grid_model,None]
models_comparison(results, "GridSearch",x_train=x_train,y_train=y_train)

log_grid_model = log_grid_train_model(x_train,y_train)
results["LogGrid"] = [log_grid_model,None]
models_comparison(results, "GridSearch",x_train=x_train,y_train=y_train)

rf_grid_model = random_forest_grid_model(x_train,y_train)
results["RandomForestGrid"] = [rf_grid_model,None]
models_comparison(results, "GridSearch",x_train=x_train,y_train=y_train)

xgb_grid_model = xgboost_grid_model(x_train,y_train)
results["XGBoostGrid"] = [xgb_grid_model,None]
models_comparison(results, "GridSearch",x_train=x_train,y_train=y_train)

light_grid_model = light_grid_train_model(x_train,y_train)
results["lightGrid"] = [light_grid_model,None]
models_comparison(results, "GridSearch",x_train=x_train,y_train=y_train)


# In[ ]:


rf_grid_model = RandomForestClassifier(**rf_grid_model.get_params())
xgb_grid_model = XGBClassifier(**xgb_grid_model.get_params())
light_grid_model = lgb.LGBMClassifier(**light_grid_model.get_params())
svm_grid_model = SVC(**svm_grid_model.get_params())
log_grid_model = LogisticRegression(**log_grid_model.get_params())

stacking_model_log = stacking_ensemble(x_train,y_train,estimators=[("log",log_grid_model),("svm",svm_grid_model),("rf",rf_grid_model),("xgb",xgb_grid_model),("light",light_grid_model)])

results["StackingGridLog"] = [stacking_model_log,None]

models_comparison(results,"Grid Ensemble",x_train=x_train,y_train=y_train,n_repeats=5)


# In[ ]:


class_accuracy_cv(log_grid_model,x_train,y_train)
class_accuracy_cv(svm_grid_model,x_train,y_train)
class_accuracy_cv(rf_grid_model,x_train,y_train)
class_accuracy_cv(xgb_grid_model,x_train,y_train)
class_accuracy_cv(light_grid_model,x_train,y_train)
class_accuracy_cv(stacking_model_log,x_train,y_train)


# In[ ]:


models = {"RF": rf_model,"XGB": xgb_model,"Light":light_model,"Stacking": stacking_model_svm}
roc_auc(models,x_train, y_train, x_test, y_test)


# ## Ensemble with Best Models

# In[ ]:


rf_best_model = RandomForestClassifier(**rf_baye_model.get_params())
xgb_best_model = XGBClassifier(**xgb_grid_model.get_params())
light_best_model = lgb.LGBMClassifier(**light_grid_model.get_params())
svm_best_model = SVC(**svm_grid_model.get_params())

stacking_model = stacking_ensemble(x_train,y_train,estimators=[("rf",rf_best_model),("xgb",xgb_best_model),("light",light_best_model)])
stacking_model_svm = stacking_ensemble(x_train,y_train,estimators=[("svm",svm_best_model),("rf",rf_best_model),("xgb",xgb_best_model),("light",light_best_model)])

results["StackingGrid"] = [stacking_model,None]
results["StackingGridSVM"] = [stacking_model_svm,None]

models_comparison(results,"Grid Ensemble",x_train=x_train,y_train=y_train,n_repeats=10)


# In[ ]:


class_accuracy(stacking_model,x_test,y_test)


# # Get Preds

# In[ ]:


results = {}
x_train_final, x_test_final, y_train_final, y_test_final = define_X_y(shap_train_df,shap_test_df,random_state=20)
main_exploration(x_train_final)
main_exploration(x_test_final)


# In[ ]:


rf_params = rf_model.get_params()
xgb_params = xgb_model.get_params()
light_params = light_model.get_params()
svm_params = svm_model.get_params()


# In[ ]:


rf_grid_params = rf_grid_model.get_params()
xgb_grid_params = xgb_grid_model.get_params()
light_grid_params = light_grid_model.get_params()
svm_grid_params = svm_grid_model.get_params()
log_grid_model = log_grid_model.get_params()


# In[ ]:


rf_preds_model = RandomForestClassifier(**rf_grid_params)
xgb_preds_model = XGBClassifier(**xgb_grid_params)
light_preds_model = lgb.LGBMClassifier(**light_grid_params)
svm_preds_model = SVC(**svm_grid_params)
log_grid_model = LogisticRegression(**log_grid_model)


# In[ ]:


#stacking_model = stacking_ensemble(x_train_final,y_train_final,estimators=[("rf",rf_preds_model),("xgb",xgb_preds_model),("light",light_preds_model)])
#stacking_model_svm = stacking_ensemble(x_train_final,y_train_final,estimators=[("svm",svm_preds_model),("rf",rf_preds_model),("xgb",xgb_preds_model),("light",light_preds_model)])
stacking_model_log = stacking_ensemble(x_train_final,y_train_final,estimators=[("log",log_grid_model),("svm",svm_preds_model),("rf",rf_preds_model),("xgb",xgb_preds_model),("light",light_preds_model)])


# ## Final Test

# In[ ]:


def final_test_cv(model, x_train, y_train, n_repeats=5, n_splits=5,random_state=27):
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    
    score_mean = cross_val_score(model, x_train, y_train, cv=cv, scoring='f1_macro').mean()
    
    print(f"F1 Score mean Stacking: {score_mean}")


# In[ ]:


final_test_cv(stacking_model_log,x_train_final,y_train_final,n_repeats=5,n_splits=5,random_state=12)
final_test_cv(stacking_model_log,x_train_final,y_train_final,n_repeats=5,n_splits=5,random_state=332)
final_test_cv(stacking_model_log,x_train_final,y_train_final,n_repeats=5,n_splits=5,random_state=3)
final_test_cv(stacking_model_log,x_train_final,y_train_final,n_repeats=5,n_splits=5,random_state=331231)


# ## Preds to CSV

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


preds_to_csv(stacking_model_svm.predict(x_test_final))


# In[ ]:


save_stuff(stacking_model_log,"Models/stacking_reduction_dos_shap_globais_thre_0_02.pkl")


# In[ ]:


stacking_model

