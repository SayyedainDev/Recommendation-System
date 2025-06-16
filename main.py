import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, accuracy_score, f1_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# ------------------- 1. Load and Clean Data -------------------
data = pd.read_csv('pet_adoption_data.csv')

print("Initial shape:", data.shape)
print("Columns in dataset:", list(data.columns))

# Drop duplicates
data.drop_duplicates(inplace=True)

# Fill missing values
data.fillna(method='ffill', inplace=True)

# ------------------- 2. Outlier Handling -------------------
def handle_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] < lower, lower, df[col])
        df[col] = np.where(df[col] > upper, upper, df[col])
    return df

numerical_cols = ['AgeMonths', 'WeightKg', 'TimeInShelterDays', 'AdoptionFee']
numerical_cols = [col for col in numerical_cols if col in data.columns]
data = handle_outliers(data, numerical_cols)

# ------------------- 4. EDA -------------------
print("\n--- Descriptive Statistics ---")
print(data.describe())

# Histograms
for col in numerical_cols:
    data[col].hist(bins=20)
    plt.title(f"Distribution of {col}")
    plt.show()

# Boxplots
for col in numerical_cols:
    sns.boxplot(x=data[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# Correlation Heatmap
sns.heatmap(data[numerical_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# ------------------- 5. Preprocessing -------------------
target = 'AdoptionLikelihood'
categorical_cols = ['PetType', 'Breed', 'Color', 'Size', 'Vaccinated', 'HealthCondition', 'PreviousOwner']
categorical_cols = [col for col in categorical_cols if col in data.columns]

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded = encoder.fit_transform(data[categorical_cols])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))

scaler = MinMaxScaler()
scaled_num = scaler.fit_transform(data[numerical_cols])
scaled_num_df = pd.DataFrame(scaled_num, columns=numerical_cols)

features_df = pd.concat([scaled_num_df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
X = features_df
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------- 6. Train Models -------------------
models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'KNN': KNeighborsClassifier(),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'SVM': SVC(probability=True)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    results[name] = {'model': model, 'accuracy': acc, 'f1': f1}

    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name}")

plt.title("ROC Curves")
plt.legend()
plt.show()

# Bar chart comparison
df_results = pd.DataFrame({k: {'Accuracy': v['accuracy'], 'F1': v['f1']} for k, v in results.items()}).T
print("\n--- Model Comparison ---")
print(df_results)
df_results.plot(kind='bar', figsize=(10, 5))
plt.title("Model Accuracy & F1 Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.show()

# ------------------- 7. Save Best Model -------------------
best_model_name = df_results['F1'].idxmax()
best_model = results[best_model_name]['model']
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(encoder, 'encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(categorical_cols, 'categorical_columns.pkl')
joblib.dump(numerical_cols, 'numerical_columns.pkl')

print(f"✅ Saved best model ({best_model_name}) to 'best_model.pkl'")
print("✅ Saved encoder, scaler, and column names (categorical & numerical)")