import joblib
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from constants import root_path, df, X, y


# Grid Search and Cross Validate It
n_estimators = [100]
max_depth = [2,3,4,6,8,10]
learning_rate = [0.2]
subsample = [0.9]
colsample_bytree = [1.0]
lambda_lst=[0, 1.0, 2.0]

xgbtree = XGBRegressor()
pipeline = Pipeline([('standardize', StandardScaler()),
                     ('xgb', xgbtree)])

param_grid = dict(xgb__n_estimators=n_estimators, xgb__max_depth=max_depth,
                  xgb__learning_rate=learning_rate, xgb__subsample=subsample,
                  xgb__colsample_bytree=colsample_bytree, xgb__lambda=lambda_lst)
metrics = ['neg_mean_squared_error']
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
grid_xgb = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring=metrics,
                        cv=kfold.split(X, df.iloc[:, 2].values),
                        return_train_score=True, refit='neg_mean_squared_error')
results_xgb = grid_xgb.fit(X, y)

# Save model and results
df_results = pd.DataFrame(results_xgb.cv_results_)
df_results.to_csv(root_path / 'results' / 'results_xgb_metrics.csv', index=False)

joblib.dump(grid_xgb.best_estimator_, root_path / 'models' / 'grid_xgb.pkl')