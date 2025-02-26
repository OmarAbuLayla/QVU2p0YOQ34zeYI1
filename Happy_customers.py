import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load and prepare data
df = pd.read_csv("ACME-HappinessSurvey2020.csv")
df.rename(columns={
    'Y': 'Happy',
    'X1': 'On_time',
    'X2': 'Items_ok',
    'X3': 'Ordered_all',
    'X4': 'Good_price',
    'X5': 'Satisfied',
    'X6': 'app_easy'
}, inplace=True)

y = df['Happy']
X = df.drop(columns=['Happy'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)



# Compare accuracies
print("Random Forest Accuracy:", rf_accuracy)
