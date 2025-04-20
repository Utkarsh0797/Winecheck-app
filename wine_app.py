import pandas as pd

# ✅ Step 1: Load dataset and SKIP the first row manually if it contains headers
df = pd.read_csv('wine.data.csv', skiprows=1, header=None)

# ✅ Step 2: Assign proper column names (Wine_Class is the first column)
df.columns = [
    'Wine_Class', 'Alcohol', 'Malic_Acid', 'Ash', 'Alcalinity_of_Ash', 'Magnesium',
    'Total_Phenols', 'Flavanoids', 'Nonflavanoid_Phenols', 'Proanthocyanins',
    'Color_Intensity', 'Hue', 'OD280/OD315_of_Diluted_Wines', 'Proline'
]

# ✅ Step 3: Convert all to numeric (safe check)
df = df.apply(pd.to_numeric, errors='coerce')

# ✅ Step 4: Drop any row that still has NaN values
df.dropna(inplace=True)

# ✅ Step 5: Split into features and label
X = df.drop('Wine_Class', axis=1)
y = df['Wine_Class']

# ✅ Step 6: Split dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Step 7: Train KNN model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# ✅ Step 8: Evaluate accuracy
from sklearn.metrics import accuracy_score
y_pred = knn.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
