import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

main_df = pd.read_csv("../Dataset/train_radiomics_occipital_CONTROL.csv")

def main_info(df=main_df):
    print(df.info())


print(main_df.head())