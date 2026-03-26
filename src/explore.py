import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_and_split_data

_, _, _, _, df = load_and_split_data()

def run_eda(df):
    # Summary stats
    print(df.describe())
    
    # Null check
    print(df.isnull().sum())

    # Heatmap
    plt.figure(figsize=(10,8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.show()

    # Scatter
    plt.scatter(df["MedInc"], df["PRICE"])
    plt.xlabel("Median Income")
    plt.ylabel("House Price")
    plt.show()

    # Histogram
    sns.histplot(df["PRICE"], bins=30, kde=True)
    plt.show()

if __name__ == "__main__":
    run_eda(df)