import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("D:/LPU/Offer Letter/SCT/SCT_DS_3.csv")  # Replace with actual path

print("Dataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

label_cols = ['job', 'marital', 'education', 'default', 'housing',
              'loan', 'contact', 'month', 'poutcome', 'deposit']

le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col])

X = df.drop('deposit', axis=1)
y = df['deposit']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

importance = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
importance.plot(kind='bar', color='teal')
plt.title("Feature Importance")
plt.ylabel("Importance Score")
plt.show()
 # This will open the decision tree visualization in your default viewer