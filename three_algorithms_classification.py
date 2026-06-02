import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Import three different algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def load_dataset(path="Iris.csv"):
    df = pd.read_csv(path)
    df = df.drop(columns=["Id"], errors="ignore")
    X = df.drop(columns=["Species"])
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["Species"])
    return X, y, label_encoder.classes_


def evaluate_model(name, model, X_train, X_test, y_train, y_test, target_names):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"================ {name} ================")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print()


def main():
    print("Loading dataset from Iris.csv...")
    X, y, target_names = load_dataset("Iris.csv")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}\n")
    
    algorithms = {
        "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }
    
    for name, model in algorithms.items():
        evaluate_model(name, model, X_train, X_test, y_train, y_test, target_names)


if __name__ == "__main__":
    main()
