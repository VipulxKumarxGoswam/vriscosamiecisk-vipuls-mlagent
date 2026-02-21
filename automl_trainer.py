import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score


def train_automl(csv_path):

    os.makedirs("model", exist_ok=True)

    # Load Dataset
    df = pd.read_csv(csv_path)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # encoding categorical columns
    X = pd.get_dummies(X)
    feature_columns = list(X.columns)
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {}

    # Detect problem type
    if y.dtype == "object" or len(y.unique()) < 20:

        # Classification Models

        logistic = LogisticRegression(max_iter=1000)
        logistic.fit(X_train, y_train)

        pred1 = logistic.predict(X_test)
        score1 = accuracy_score(y_test, pred1)

        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)

        pred2 = rf.predict(X_test)
        score2 = accuracy_score(y_test, pred2)

        models["LogisticRegression"] = (logistic, score1)
        models["RandomForest"] = (rf, score2)

    else:

        # Regression Models

        linear = LinearRegression()
        linear.fit(X_train, y_train)

        pred = linear.predict(X_test)
        score = r2_score(y_test, pred)

        models["LinearRegression"] = (linear, score)

    # Find Best Model

    best_name = None
    best_model = None
    best_score = -999

    all_scores = {}

    for name, (model, score) in models.items():

        print(name, "Score:", score)

        all_scores[name] = float(score)

        if score > best_score:

            best_score = score
            best_model = model
            best_name = name

    # Save Files

    joblib.dump(best_model, "model/final_model.pkl")

    joblib.dump(feature_columns,"model/feature_columns.pkl")

    joblib.dump(best_score,
                "model/best_score.pkl")

    joblib.dump(best_name,
                "model/best_model_name.pkl")

    #  NEW SAVE
    joblib.dump(all_scores,
                "model/all_scores.pkl")

    print(" BEST MODEL SAVED :", best_name)