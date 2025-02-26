import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

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

# Split data without scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Expanded hyperparameter grid
param_grid = {
    'n_estimators': [200, 400, 600],
    'max_depth': [None, 20, 40],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced', None],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy']
}

# Grid search with increased CV folds
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=10,  # More folds for better generalization
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

# Evaluate best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
optimized_accuracy = accuracy_score(y_test, y_pred)

print("Best Parameters:", grid_search.best_params_)
print("Optimized Accuracy:", optimized_accuracy)