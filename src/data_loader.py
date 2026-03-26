import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

def load_and_split_data(test_size=0.2, random_state=42):
    # Fetch raw data
    data = fetch_california_housing()
    
    # Convert to DataFrame
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['PRICE'] = data.target
    
    # Split into Features (X) and Target (y)
    X = df.drop("PRICE", axis=1)
    y = df["PRICE"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test, df