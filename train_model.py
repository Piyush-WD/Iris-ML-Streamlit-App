import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the built-in Iris dataset (no external dataset download required).
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=150,
    random_state=42
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(f"Test accuracy: {accuracy_score(y_test, predictions):.3f}")
print(classification_report(y_test, predictions, target_names=iris.target_names))

joblib.dump(model, "model.joblib")
print("Saved trained model to model.joblib")
