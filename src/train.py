import pandas as pd
from sklearn.linear_model import LinearRegression
from data_loader import load_and_split_data
import joblib

# Load the split data
X_train, X_test, y_train, y_test, _ = load_and_split_data()

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    model = train_model(X_train, y_train)
    
    # Print Coefficients
    coefficients = pd.DataFrame({
        "Feature": X_train.columns,
        "Weight": model.coef_
    })
    print(coefficients)
    print("Intercept:", model.intercept_)
    
    # Save the model for the test script
    joblib.dump(model, 'linear_model.pkl')
    print("Model saved as linear_model.pkl")