import pandas as pd
import matplotlib.pyplot as plt

# DATASET
from sklearn.datasets import load_iris

# SPLIT DATA
from sklearn.model_selection import train_test_split

# ALGORITHMS
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# EVALUATION
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc

# ROC SUPPORT
from sklearn.preprocessing import label_binarize


# =====================================================
# LOAD DATASET
# =====================================================

print("Loading Dataset...\n")

iris = load_iris()

X = iris.data
y = iris.target

# SHOW DATA
df = pd.DataFrame(X, columns=iris.feature_names)

print(df.head())

print("\nFlower Names:")
print(iris.target_names)


# =====================================================
# SPLIT DATA
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

print("\nTraining Data:", len(X_train))
print("Testing Data:", len(X_test))


# =====================================================
# LOGISTIC REGRESSION
# =====================================================

print("\n===== LOGISTIC REGRESSION =====")

lr_model = LogisticRegression(max_iter=200)

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_pred)

print("Accuracy:", lr_accuracy)


# =====================================================
# DECISION TREE
# =====================================================

print("\n===== DECISION TREE =====")

dt_model = DecisionTreeClassifier()

dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_pred)

print("Accuracy:", dt_accuracy)


# =====================================================
# RANDOM FOREST
# =====================================================

print("\n===== RANDOM FOREST =====")

rf_model = RandomForestClassifier()

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)

print("Accuracy:", rf_accuracy)


# =====================================================
# CONFUSION MATRIX
# =====================================================

print("\nShowing Confusion Matrix...")

cm = confusion_matrix(y_test, rf_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

display.plot()

plt.title("Confusion Matrix")

plt.show()


# =====================================================
# ROC CURVE
# =====================================================

print("\nShowing ROC Curve...")

# Convert to binary
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

# Prediction probabilities
y_score = rf_model.predict_proba(X_test)

# ROC
fpr, tpr, threshold = roc_curve(
    y_test_bin[:, 0],
    y_score[:, 0]
)

roc_auc = auc(fpr, tpr)

# PLOT ROC
plt.plot(fpr, tpr, label="AUC = %0.2f" % roc_auc)

plt.plot([0, 1], [0, 1], 'r--')

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()


# =====================================================
# FINAL RESULT
# =====================================================

print("\n==============================")
print("FINAL ACCURACY")
print("==============================")

print("Logistic Regression :", lr_accuracy)

print("Decision Tree       :", dt_accuracy)

print("Random Forest       :", rf_accuracy)

print("\nProject Completed Successfully")