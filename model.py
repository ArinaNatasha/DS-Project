import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def model_data(result_boxcox_normalized):
    # Import Train Dataset
    X_train_resampled_boxcox_normalized = pd.read_csv("trainData.csv")
    y_train_resampled = pd.read_csv("trainTarget.csv")

    # Decision Tree
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train_resampled_boxcox_normalized, y_train_resampled)

    # Make predictions on the input data
    dt_pred = dt_model.predict(result_boxcox_normalized)

    return dt_pred