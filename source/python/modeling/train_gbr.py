import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from constants import root_path, df, X, y

# Grid Search and Cross Validate It
n_estimators = [100]
max_leaf_nodes = [2,3,4,6,8,10]
learning_rate = [0.1]
min_samples_leaf = [1]
subsample = [0.1, 0.2, 0.25, 0.5, 0.75, 1.0]

gbtree = GradientBoostingRegressor()
pipeline = Pipeline([('standardize', StandardScaler()),
                     ('gbr', gbtree)])

param_grid = dict(gbr__n_estimators=n_estimators, gbr__max_leaf_nodes=max_leaf_nodes,
                  gbr__learning_rate=learning_rate, gbr__subsample=subsample,
                  gbr__min_samples_leaf=min_samples_leaf)
metrics = ['neg_mean_squared_error']

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
grid_gbr = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring=metrics,
                        cv=kfold.split(X, df.iloc[:, 2].values), 
                        return_train_score=True, refit='neg_mean_squared_error')
results_gbr = grid_gbr.fit(X, y)

# Save model and results
df_results = pd.DataFrame(results_gbr.cv_results_)
df_results.to_csv(root_path / 'results' / 'results_gbr_metrics.csv', index=False)

joblib.dump(grid_gbr.best_estimator_, root_path / 'models' / 'grid_gbr.pkl')