#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ast
import shap

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV


# In[2]:


control_df = pd.read_csv("../Dataset/train_radiomics_occipital_CONTROL.csv")
train_df = pd.read_csv("../Dataset/train_radiomics_hipocamp.csv")
test_df = pd.read_csv("../Dataset/test_radiomics_hipocamp.csv")
dummy_df = pd.read_csv("../Dataset/dummy_submission.csv")


# # Data Exploration

# ## Category Encoder and Decoder

# In[3]:


def target_encoder(df, target="Transition"):
    le_make = LabelEncoder()
    df[f"{target}_code"] = le_make.fit_transform(df[target])
    return le_make

def target_decoder(le_make, preds):
    return le_make.inverse_transform(preds)


# In[4]:


le_make_train = target_encoder(train_df)
le_make_control = target_encoder(control_df)


# ## MatPlots

# In[5]:


def show_histogram(df,title="histogram"):
    plt.figure(figsize=(13,8))
    plt.subplots_adjust(bottom=0.17)
    plt.title(title)
    sns.histplot(df)
    plt.show()


# In[6]:


def show_pie(df,title="pie"):
    labels = df.unique().tolist()
    counts = df.value_counts()
    sizes = [counts[var_cat] for var_cat in labels]
    _, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct="%1.1f%%",shadow=True)
    ax1.axis("equal")
    plt.title(title)
    plt.show()


# In[7]:


def show_boxplot(df,title="boxplot"):
    plt.figure(figsize=(13,8))
    plt.subplots_adjust(bottom=0.17)
    df.boxplot()
    plt.xticks(rotation=15)
    plt.title(title)
    plt.show()


# In[8]:


def show_heatmap(df,title="correlation heatmap"):
    df = df.select_dtypes(include="number")
    plt.figure(figsize=(13,8))
    plt.subplots_adjust(bottom=0.25,left=0.22,right=0.95)
    plt.xticks(rotation=15)
    plt.title(title)
    sns.heatmap(df.corr(),annot=True,cmap="coolwarm",linewidths=0.5)
    plt.show()


# In[9]:


def show_jointplot(df,x_label,y_label,title="jointplot",hue="Transition_code"):
    sns.jointplot(data=df,x=x_label,y=y_label,hue=hue)
    plt.show()


# In[10]:


def show_catplot(df, x_label, y_label, title="catplot", hue="Transition_code"):
    sns.catplot(data=df, x=x_label, y=y_label, hue=hue)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


# In[11]:


def show_pairplot(df,hue="Transition_code"):
    sns.pairplot(df,hue=hue)
    plt.show()


# ## Basic Exploration

# In[12]:


def main_exploration(df):
    print(df.shape) 


# In[13]:


def numerical_exploration(df):
    # age exploration
    show_histogram(df["Age"],title="Histogram")
    print(df["Age"].describe())


# In[14]:


numerical_exploration(train_df)


# In[15]:


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


# In[16]:


show_catplot(train_df, "Age", "Transition", hue="Sex")
show_heatmap(train_df[["Age","Transition_code","Sex"]])


# In[17]:


sns.heatmap(train_df.isnull(), yticklabels=False, cbar=False, cmap="viridis")


# In[18]:


diagnostics_configs_columns = ["diagnostics_Configuration_Settings","diagnostics_Configuration_EnabledImageTypes"]
def diagnostics_configs(df):
    for col in diagnostics_configs_columns:
        print(len(df[col].unique()))


# In[19]:


diagnostics_configs(train_df)


# In[20]:


diagnostics_versions_columns = ["diagnostics_Versions_PyRadiomics","diagnostics_Versions_Numpy","diagnostics_Versions_SimpleITK","diagnostics_Versions_PyWavelet","diagnostics_Versions_Python"] 
def diagnostics_versions_explorer(df):
    for column in diagnostics_versions_columns:
        print(column,": ")
        values = df[column].unique()
        print(values)


# In[21]:


diagnostics_versions_explorer(train_df)


# In[22]:


diagnostics_versions_columns = ["diagnostics_Versions_PyRadiomics","diagnostics_Versions_Numpy","diagnostics_Versions_SimpleITK","diagnostics_Versions_PyWavelet","diagnostics_Versions_Python"] 


# In[23]:


diagnostics_configs_columns = ["diagnostics_Configuration_Settings","diagnostics_Configuration_EnabledImageTypes"]


# In[24]:


unnecessary_columns = diagnostics_versions_columns + diagnostics_configs_columns +["diagnostics_Image-original_Dimensionality","diagnostics_Image-original_Minimum","diagnostics_Image-original_Size","diagnostics_Mask-original_Spacing","diagnostics_Image-original_Spacing","diagnostics_Mask-original_Size","diagnostics_Image-original_Hash","diagnostics_Mask-original_Hash","ID","Image","Mask",'diagnostics_Mask-original_CenterOfMassIndex']


# In[25]:


unnecessary_df = pd.DataFrame()
for col in unnecessary_columns+["Transition"]:
    le_make = LabelEncoder()
    unnecessary_df[f"{col}_code"] = le_make.fit_transform(train_df[col])

show_heatmap(unnecessary_df)


# ## Correlations

# In[26]:


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


# In[27]:


rad_corr = top_correlations(train_df,starts_with="lbp",number=20)
show_heatmap(train_df[rad_corr])


# # Data Processing

# ## Drop Unnecessary Columns

# In[28]:


control_df = control_df.drop(columns=unnecessary_columns,axis=1,errors="ignore")
train_df = train_df.drop(columns=unnecessary_columns,axis=1,errors="ignore")
test_df = test_df.drop(columns=unnecessary_columns,axis=1,errors="ignore")


# ## Nunique Columns

# In[29]:


nunique_columns = train_df.columns[train_df.nunique() == 1].tolist()
train_df = train_df.drop(columns=nunique_columns, errors="ignore")
test_df = test_df.drop(columns=nunique_columns, errors="ignore")
control_df = control_df.drop(columns=nunique_columns, errors="ignore")


# ## Non Numerical Columns

# In[30]:


# Separar a coluna de BoundingBox em várias colunas
train_df[['x_min', 'y_min', 'largura', 'altura', 'profundidade', 'extra']] = train_df['diagnostics_Mask-original_BoundingBox'].str.strip('()').str.split(',', expand=True).astype(float)

# Separar a coluna de CenterOfMassIndex em várias colunas
train_df[['x_center', 'y_center', 'z_center']] = train_df['diagnostics_Mask-original_CenterOfMass'].str.strip('()').str.split(',', expand=True).astype(float)


# In[31]:


# Separar a coluna de BoundingBox em várias colunas
test_df[['x_min', 'y_min', 'largura', 'altura', 'profundidade', 'extra']] = test_df['diagnostics_Mask-original_BoundingBox'].str.strip('()').str.split(',', expand=True).astype(float)

# Separar a coluna de CenterOfMassIndex em várias colunas
test_df[['x_center', 'y_center', 'z_center']] = test_df['diagnostics_Mask-original_CenterOfMass'].str.strip('()').str.split(',', expand=True).astype(float)


# In[32]:


# Separar a coluna de BoundingBox em várias colunas
control_df[['x_min', 'y_min', 'largura', 'altura', 'profundidade', 'extra']] = control_df['diagnostics_Mask-original_BoundingBox'].str.strip('()').str.split(',', expand=True).astype(float)

# Separar a coluna de CenterOfMassIndex em várias colunas
control_df[['x_center', 'y_center', 'z_center']] = control_df['diagnostics_Mask-original_CenterOfMass'].str.strip('()').str.split(',', expand=True).astype(float)


# In[33]:


train_df = train_df.drop(['diagnostics_Mask-original_BoundingBox', 'diagnostics_Mask-original_CenterOfMass'], axis=1, errors="ignore")
test_df = test_df.drop(['diagnostaics_Mask-original_BoundingBox', 'diagnostics_Mask-original_CenterOfMass'], axis=1, errors="ignore")
control_df = control_df.drop(['diagnostics_Mask-original_BoundingBox', 'diagnostics_Mask-original_CenterOfMass'], axis=1, errors="ignore")


# In[34]:


main_exploration(train_df)


# In[35]:


train_df = train_df.select_dtypes(include=['number'])
control_df = control_df.select_dtypes(include=['number'])
test_df = test_df.select_dtypes(include=['number'])


# ## Data Scaler

# In[36]:


from sklearn.preprocessing import StandardScaler

def data_scaler(df):
    scaler_df = df.drop(columns=["Transition","Transition_code"],errors="ignore")
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(scaler_df),columns=scaler_df.columns)
    return df_scaled


# In[37]:


scaled_train_df = data_scaler(train_df)
scaled_control_df = data_scaler(control_df)
scaled_test_df = data_scaler(test_df)

scaled_train_df["Transition_code"] = train_df["Transition_code"].values
scaled_control_df["Transition_code"] = train_df["Transition_code"].values


# ## Correlation Analisys

# In[38]:


corr_df = scaled_train_df.copy()
corr_df.loc[:,"Transition_code"] = train_df["Transition_code"].values
target = "Transition_code"


# In[39]:


corr_threshold = 0
def apply_correlation(df,threshold):
    df = df.drop(columns=["Transition"],errors="ignore")
    correlation = df.corr()[target].abs().sort_values(ascending=False)
    important_features = correlation[correlation > threshold].index.tolist()
    
    if target in important_features:
        important_features.remove(target)

    return important_features


# In[40]:


important_features = apply_correlation(scaled_train_df, corr_threshold)


# In[41]:


corr_train_df = scaled_train_df[important_features]
corr_control_df = scaled_control_df[important_features]
corr_test_df = scaled_test_df[important_features]


# In[42]:


corr_train_df["Transition_code"] = train_df["Transition_code"].values
corr_control_df["Transition_code"] = train_df["Transition_code"].values


# In[179]:


main_exploration(corr_train_df)
main_exploration(corr_control_df)
main_exploration(corr_test_df)


# # Testing Phase

# In[44]:


from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from xgboost import XGBClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression


# In[45]:


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


# In[180]:


results = {}
x_train, x_test, y_train, y_test = define_X_y(corr_train_df)
main_exploration(x_train)
main_exploration(x_test)


# ## Models

# ### RandomForest

# In[47]:


def random_forest_model(x_train,y_train):
    model = RandomForestClassifier(random_state=27)
    model.fit(x_train,y_train)
    
    return model


# ### XGBoost

# In[48]:


def xgboost_model(x_train,y_train):
    model = XGBClassifier(random_state=27)
    model.fit(x_train,y_train)

    return model


# ### GradientBoost

# In[49]:


def gradient_model(x_train, y_train):
    model = GradientBoostingClassifier(random_state=27)
    model.fit(x_train,y_train)
    
    return model


# ### Logistic Regression L2

# In[50]:


def log_reg_model(x_train,y_train):
    model = LogisticRegression(penalty='l2', C=1.0, solver='liblinear',random_state=27)
    model.fit(x_train,y_train)
    
    return model


# ## Models Applier

# In[218]:


def apply_models(x_train,y_train,x_test,y_test, title="Models Macro F1 Comparison"):
    rf_model = random_forest_model(x_train,y_train)
    results["RandomForest"] = [rf_model,None]

    xgb_model = xgboost_model(x_train,y_train)
    results["XGBoost"] = [xgb_model,None]

    gradient_models = gradient_model(x_train,y_train)
    results["Gradient"] = [gradient_models,None]

    log_model = log_reg_model(x_train,y_train)
    results["LogRegression"] = [log_model,None]

    models_comparison(results,title,x_test=x_test,y_test=y_test)

    return rf_model,xgb_model,gradient_models, log_model


# ## Models Comparison

# In[223]:


def models_comparison(results,title,x_test=x_test,y_test=y_test):
    for result in results:
        if results[result][1] == None:
            preds = results[result][0].predict(x_test)
            results[result][1] = f1_score(y_test, preds, average="macro")
        print(f"F1 Macro Score em {result}: {results[result][1]}")
    
    models_score = plt.figure(figsize=(6,3))

    mod = list(results.keys())
    f1 = list([score[1] for score in results.values()])
    
    plt.bar(mod,f1, color = "lightblue", width = 0.5)
    
    plt.xlabel("Model")
    plt.ylabel("Macro F1")
    plt.xticks(rotation=15)
    plt.title(title)
    plt.show()


# ## Models Tester

# In[53]:


rf_model,xgb_model,gradient_models, log_model = apply_models(x_train,y_train,x_test,y_test)


# ## Feature Importance Analysis

# In[54]:


from sklearn.inspection import permutation_importance


# ### Permutation Importance

# In[57]:


#print("rf...")
#pi_rf_result = permutation_importance(rf_model,x_test,y_test,n_repeats=10,random_state=27,n_jobs=-1)
#print("xgb...")
#pi_xgb_result = permutation_importance(xgb_model,x_test,y_test,n_repeats=10,random_state=27,n_jobs=-1)
#print("gradient...")
#pi_gradient_result = permutation_importance(gradient_model,x_test,y_test,n_repeats=10,random_state=27,n_jobs=-1)

pi_rf_result = load_stuff("Permutations/pi_rf_result.pkl")
pi_xgb_result = load_stuff("Permutations/pi_xgb_result.pkl")
pi_gradient_result = load_stuff("Permutations/pi_gradient_result.pkl")


# In[181]:


x_train, x_test, y_train, y_test = define_X_y(corr_train_df)
main_exploration(x_train)
main_exploration(x_test)


# In[209]:


def show_negative_perm_importance(result, x_test,equal=False):
    negative_importances = pd.Series(result.importances_mean, index=x_test.columns)
    if equal == False:
        negative_importances = negative_importances[negative_importances < 0].sort_values()
    else: 
        negative_importances = negative_importances[negative_importances <= 0].sort_values()
    
    if not negative_importances.empty and len(negative_importances) < 500:
        fig, ax = plt.subplots(figsize=(10, 6))
        negative_importances.plot.bar(ax=ax)
        ax.set_title("Negative Permutation Importance")
        ax.set_ylabel("Mean Accuracy Increase")
        plt.show()
    else:
        print("Não é possível fazer plot!")

    return negative_importances.index.tolist()

def show_positive_perm_importance(result, x_test):
    positive_importances = pd.Series(result.importances_mean, index=x_test.columns)
    positive_importances = positive_importances[positive_importances > 0].sort_values()
    
    if not positive_importances.empty and len(positive_importances) < 500:
        fig, ax = plt.subplots(figsize=(10, 6))
        positive_importances.plot.bar(ax=ax)
        ax.set_title("Positive Permutation Importance")
        ax.set_ylabel("Mean Accuracy Increase")
        plt.show()
    else:
        print("Não é possível fazer plot!")

    return positive_importances.index.tolist()


# ## Remove Negative Importances

# In[183]:


negative_columns_rf = show_negative_perm_importance(pi_rf_result, x_test)
print(len(negative_columns_rf))


# In[184]:


negative_columns_xgb = show_negative_perm_importance(pi_xgb_result, x_test)
print(len(negative_columns_xgb))


# In[185]:


negative_columns_gradient = show_negative_perm_importance(pi_gradient_result, x_test)
print(len(negative_columns_gradient))


# In[186]:


negative_columns = negative_columns_gradient + negative_columns_xgb + negative_columns_rf
negative_columns = list(dict.fromkeys(negative_columns))
print(len(negative_columns))


# ## Models Tester

# In[194]:


without_neg_imp_train_df = corr_train_df.drop(columns=negative_columns)
without_neg_imp_control_df = corr_control_df.drop(columns=negative_columns)
without_neg_imp_test_df = corr_test_df.drop(columns=negative_columns)
results = {}
x_train, x_test, y_train, y_test = define_X_y(without_neg_imp_train_df)
main_exploration(x_train)
main_exploration(x_test)


# In[83]:


rf_model,xgb_model,gradient_models, log_model = apply_models(x_train,y_train,x_test,y_test)


# ## Remove Null Importances

# In[204]:


x_train, x_test, y_train, y_test = define_X_y(corr_train_df)
main_exploration(x_train)
main_exploration(x_test)


# In[189]:


null_columns_rf = show_negative_perm_importance(pi_rf_result, x_test,equal=True)
print(len(null_columns_rf))


# In[211]:


null_columns_xgb = show_negative_perm_importance(pi_xgb_result, x_test,equal=True)
positive_columns_xgb = show_positive_perm_importance(pi_xgb_result, x_test)
print(len(null_columns_xgb))
print(len(positive_columns_xgb))


# In[191]:


null_columns_gradient = show_negative_perm_importance(pi_gradient_result, x_test,equal=True)
print(len(null_columns_gradient))


# In[192]:


null_columns = set(null_columns_rf) & set(null_columns_xgb) & set (null_columns_gradient) 
print(len(null_columns))


# ## Models Tester

# In[276]:


without_null_imp_train_df = corr_train_df.drop(columns=null_columns_xgb)
without_null_imp_control_df = corr_control_df.drop(columns=null_columns_xgb)
without_null_imp_test_df = corr_test_df.drop(columns=null_columns_xgb)
results = {}
x_train, x_test, y_train, y_test = define_X_y(without_null_imp_train_df)
main_exploration(x_train)
main_exploration(x_test)


# In[278]:


rf_model,xgb_model,gradient_models, log_model = apply_models(x_train,y_train,x_test,y_test)


# In[150]:


important_features = apply_correlation(without_null_imp_train_df,0.1)
show_heatmap(without_null_imp_train_df[important_features])


# ## SHAP Analysis

# ### Global

# In[270]:


X_shap = without_null_imp_train_df.drop("Transition_code",axis=1)

explainer = shap.Explainer(xgb_model,X_shap)
shap_values = explainer(X_shap)


# In[271]:


n_features = shap_values.shape[1]
n_features_per_plot = 10


# In[272]:


for i in range(0, n_features, n_features_per_plot):
    selected_shap_values = shap_values[:, i:i + n_features_per_plot, 1]
    
    shap.plots.heatmap(selected_shap_values)
    
    plt.show()


# In[273]:


for i in range(0, n_features, n_features_per_plot):
    selected_shap_values = shap_values[:, i:i + n_features_per_plot, 1]
    
    shap.summary_plot(selected_shap_values, feature_names=positive_columns_xgb[i:i + n_features_per_plot], show=False)
    
    plt.show()


# ### Local

# In[290]:


preds = xgb_model.predict(x_test)
errors = preds != y_test
error_index = errors[errors].index.to_numpy()


# In[294]:





# ## Models Tester

# In[263]:


try_features = ["original_glszm_SizeZoneNonUniformityNormalized","wavelet-LLL_glrlm_RunLengthNonUniformityNormalized","wavelet-LHL_firstorder_InterquartileRange"]
# melhor ate agora: ["original_glszm_SizeZoneNonUniformityNormalized","wavelet-LLL_glrlm_RunLengthNonUniformityNormalized","wavelet-LHL_firstorder_InterquartileRange"]
# ["wavelet-LLL_glrlm_RunLengthNonUniformityNormalized","wavelet-LHL_firstorder_InterquartileRange"]
# "wavelet-LLL_glrlm_RunLengthNonUniformityNormalized"


# In[264]:


shap_train_df = without_null_imp_train_df.drop(columns=try_features,errors="ignore")
shap_control_df = without_null_imp_control_df.drop(columns=[])
shap_test_df = without_null_imp_test_df.drop(columns=[])
results = {}
x_train, x_test, y_train, y_test = define_X_y(shap_train_df)
main_exploration(x_train)
main_exploration(x_test)


# In[265]:


apply_models(x_train,y_train,x_test,y_test)


# # Preds to CSV

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


preds_to_csv(gradient_model.predict(x_test))


# # Save & Load Data

# In[56]:


import pickle
import os

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


# In[ ]:


save_stuff(pi_rf_result,"Permutations/pi_rf_result.pkl")
save_stuff(pi_xgb_result,"Permutations/pi_xgb_result.pkl")
save_stuff(pi_gradient_result,"Permutations/pi_gradient_result.pkl")


# In[ ]:




