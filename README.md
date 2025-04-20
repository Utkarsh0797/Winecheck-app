#wine_app.py
import pandas as pd


df = pd.read_csv('wine.data.csv', skiprows=1, header=None)

df.columns = [
    'Wine_Class', 'Alcohol', 'Malic_Acid', 'Ash', 'Alcalinity_of_Ash', 'Magnesium',
    'Total_Phenols', 'Flavanoids', 'Nonflavanoid_Phenols', 'Proanthocyanins',
    'Color_Intensity', 'Hue', 'OD280/OD315_of_Diluted_Wines', 'Proline'
]

df = df.apply(pd.to_numeric, errors='coerce')


df.dropna(inplace=True)

X = df.drop('Wine_Class', axis=1)
y = df['Wine_Class']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)


from sklearn.metrics import accuracy_score
y_pred = knn.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
