import pandas as pd
import os
from project1_utils import Project1Utils   # CHANGED

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "../../../data/project1/wine-quality-red.csv")

    df = pd.read_csv(file_path)

    target_col = df.columns[-1]
    feature_cols = df.columns[:-1]

    X = df[feature_cols]
    y = df[target_col]

    utils = Project1Utils(X, y)

    utils.run_linear_regressions()
    utils.run_ridge_regressions(alpha=0.5)
    utils.run_lasso_regressions(alpha=0.5)
    utils.run_transformed_regressions()
    utils.run_sym_ridge_regression()

if __name__ == "__main__":
    main()