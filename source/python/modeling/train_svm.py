import joblib
import pandas as pd
from pathlib import Path
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process.kernels import RBF
from constants import root_path, df, X, y

# Grid Search and Cross Validate It
C_lst = [100, 200, 300, 400, 500]
rbf_lst = [RBF(0.25), RBF(0.5), RBF(1), RBF(2), RBF(4)]
svr = SVR(gamma='scale')

pipeline = Pipeline([('standardize', StandardScaler()),
                     ('svr', svr)])

param_grid = dict(svr__C=C_lst, svr__kernel=rbf_lst)

metrics = ['neg_mean_squared_error']
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
grid_svm = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring=metrics,
                        cv=kfold.split(X, df.iloc[:, 2].values), 
                        return_train_score=True, refit='neg_mean_squared_error')
results_svm = grid_svm.fit(X, y)

# Save model and results
df_results = pd.DataFrame(results_svm.cv_results_)
df_results.to_csv(root_path / 'results' / 'results_svm_metrics.csv', index=False)

joblib.dump(grid_svm.best_estimator_, root_path / 'models' / 'grid_svm.pkl')
