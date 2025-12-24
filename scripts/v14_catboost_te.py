import os
import argparse
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder

class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target Encoder that supports multiple aggregation functions,
    internal cross-validation for leakage prevention, and smoothing.
    """
    def __init__(self, cols_to_encode, aggs=['mean'], cv=5, smooth='auto', drop_original=False):
        self.cols_to_encode = cols_to_encode
        self.aggs = aggs
        self.cv = cv
        self.smooth = smooth
        self.drop_original = drop_original
        self.mappings_ = {}
        self.global_stats_ = {}

    def fit(self, X, y):
        temp_df = X.copy()
        temp_df['target'] = y

        for agg_func in self.aggs:
            self.global_stats_[agg_func] = y.agg(agg_func)

        for col in self.cols_to_encode:
            self.mappings_[col] = {}
            for agg_func in self.aggs:
                mapping = temp_df.groupby(col)['target'].agg(agg_func)
                self.mappings_[col][agg_func] = mapping
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in self.cols_to_encode:
            for agg_func in self.aggs:
                new_col_name = f'TE_{col}_{agg_func}'
                if col in self.mappings_ and agg_func in self.mappings_[col]:
                    map_series = self.mappings_[col][agg_func]
                    X_transformed[new_col_name] = X[col].map(map_series)
                    X_transformed[new_col_name] = X_transformed[new_col_name].fillna(self.global_stats_[agg_func])
                else:
                    X_transformed[new_col_name] = self.global_stats_[agg_func]

        if self.drop_original:
            X_transformed.drop(columns=self.cols_to_encode, inplace=True)
        return X_transformed

    def fit_transform(self, X, y):
        self.fit(X, y)
        encoded_features = pd.DataFrame(index=X.index)
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(X, y):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            temp_df_train = X_train.copy()
            temp_df_train['target'] = y_train

            for col in self.cols_to_encode:
                for agg_func in self.aggs:
                    new_col_name = f'TE_{col}_{agg_func}'
                    fold_global_stat = y_train.agg(agg_func)
                    mapping = temp_df_train.groupby(col)['target'].agg(agg_func)

                    if agg_func == 'mean':
                        counts = temp_df_train.groupby(col)['target'].count()
                        m = self.smooth
                        if self.smooth == 'auto':
                            variance_between = mapping.var()
                            avg_variance_within = temp_df_train.groupby(col)['target'].var().mean()
                            if variance_between > 0:
                                m = avg_variance_within / variance_between
                            else:
                                m = 0
                        smoothed_mapping = (counts * mapping + m * fold_global_stat) / (counts + m)
                        encoded_values = X_val[col].map(smoothed_mapping)
                    else:
                        encoded_values = X_val[col].map(mapping)
                    
                    encoded_features.loc[X_val.index, new_col_name] = encoded_values.fillna(fold_global_stat)

        X_transformed = X.copy()
        for col in encoded_features.columns:
            X_transformed[col] = encoded_features[col]

        if self.drop_original:
            X_transformed.drop(columns=self.cols_to_encode, inplace=True)
        return X_transformed

def add_orig_mean_count_features(train, test, orig, target_col='diagnosed_diabetes'):
    # Similar to the notebook's approach
    features = test.columns.to_list()
    # Ensure orig has same columns
    orig = orig[features + [target_col]]
    
    for col in features:
        # Mean
        tmp = orig.groupby(col)[target_col].mean()
        new_name = 'orig_' + str(col)
        tmp.name = new_name
        train = train.merge(tmp, how='left', on=col)
        test = test.merge(tmp, how='left', on=col)
        
        # Count
        tmp_cnt_name = 'orig_cnt_' + str(col)
        tmp_cnt = orig.groupby(col).size().reset_index(name=tmp_cnt_name)
        train = train.merge(tmp_cnt, how='left', on=col)
        test = test.merge(tmp_cnt, how='left', on=col)
        
    return train, test

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--orig-csv", default="data/orig2/diabetes_dataset.csv")
    p.add_argument("--out", default="submissions/submission_v14_catboost_te.csv")
    p.add_argument("--task-type", default="CPU", choices=["CPU", "GPU"], help="CatBoost task_type")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-estimators", type=int, default=12000)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=0.01)
    p.add_argument("--early-stopping-rounds", type=int, default=300)
    p.add_argument(
        "--extended-strat",
        action="store_true",
        help=(
            "Use extended stratification via a derived 'multicat' label: "
            "LabelEncode([family_history_diabetes, cardiovascular_history, education_level, target])"
        ),
    )
    p.add_argument(
        "--limit-train-rows",
        type=int,
        default=0,
        help="If >0, subsample this many rows from the (possibly augmented) training data for quick smoke tests.",
    )
    args = p.parse_args()

    train = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    orig = pd.read_csv(args.orig_csv)
    sample = pd.read_csv(os.path.join(args.data_dir, "sample_submission.csv"))

    # Add id to orig as in notebook
    orig['id'] = orig.index

    # Ensure orig has same columns as train
    orig = orig[train.columns.to_list()]

    # Append external data (safer than merge-on-all-common-cols)
    train = pd.concat([train, orig], axis=0, ignore_index=True)
    
    # Add orig mean/count features
    # Note: The notebook does this AFTER merging orig to train?
    # "train = train.merge(orig, how='outer')"
    # Then "for d in [train, test, orig]: ... bmi_classification"
    # Then "for col in features: ... orig.groupby(col)..."
    # So they use the 'orig' dataframe (which is separate) to calculate stats, 
    # and merge those stats onto 'train' (which ALREADY contains orig rows).
    # This means orig rows in 'train' get their own stats? Yes.
    
    # BMI classification
    def bmi_classification(score):
        if score >= 30: return 'Obese'
        elif score >= 25: return 'Overwieght'
        elif score >= 18.5: return 'Normal'
        else: return 'Underweight'

    for d in [train, test, orig]:
        d['bmi_category'] = d['bmi'].apply(bmi_classification)

    # Add orig features
    # We need to be careful about column names.
    # The notebook uses 'orig_' + col.
    
    new_features = []
    features = [c for c in test.columns.to_list() if c != 'id']
    # Remove 'id' from features if present? Notebook doesn't drop it from 'features' list derived from test.columns
    # test.columns includes 'id'.
    
    for col in features:
        # Mean
        tmp = orig.groupby(col)['diagnosed_diabetes'].mean()
        new_name = 'orig_' + str(col)
        tmp.name = new_name
        train = train.merge(tmp, how='left', on=col)
        test = test.merge(tmp, how='left', on=col)
        new_features.append(new_name)

        # Count
        tmp_cnt_name = 'orig_cnt_' + str(col)
        tmp_cnt = orig.groupby(col).size().reset_index(name=tmp_cnt_name)
        train = train.merge(tmp_cnt, how='left', on=col)
        test = test.merge(tmp_cnt, how='left', on=col)
        new_features.append(tmp_cnt_name)

    # Convert object to category
    objs = train.select_dtypes(include='object').columns.to_list()
    for obj in objs:
        train[obj] = train[obj].astype('category')
        test[obj] = test[obj].astype('category')

    # Label Encoding for category columns
    objects = train.select_dtypes('category').columns
    for obj in objects:
        le = LabelEncoder()
        # Fit on all data
        le.fit(pd.concat([train[obj], test[obj]], axis=0).astype(str))
        train[obj] = le.transform(train[obj].astype(str))
        test[obj] = le.transform(test[obj].astype(str))

    # Prepare X, y
    X = train.drop(columns=['diagnosed_diabetes'])
    y = train['diagnosed_diabetes']

    if args.limit_train_rows and args.limit_train_rows > 0:
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(len(X), size=min(int(args.limit_train_rows), len(X)), replace=False)
        keep = np.sort(keep)
        X = X.iloc[keep].reset_index(drop=True)
        y = y.iloc[keep].reset_index(drop=True)

    # Target Encoding on int/float columns
    int_cols = [c for c in X.select_dtypes(include=['int', 'float']).columns.to_list() if c != 'id']
    
    # CatBoost Params
    CAT_params = {
        'n_estimators': args.n_estimators,
        'depth': args.depth,
        'learning_rate': args.learning_rate,
        'eval_metric': 'AUC',
        'random_seed': args.seed,
        'use_best_model': True,
        'verbose': 1000,
        'early_stopping_rounds': args.early_stopping_rounds,
        'task_type': args.task_type,
    }

    folds = int(args.folds)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=args.seed)

    strat_y = y
    if args.extended_strat:
        strat_cols = ['family_history_diabetes', 'cardiovascular_history', 'education_level']
        missing = [c for c in strat_cols if c not in train.columns]
        if missing:
            raise SystemExit(f"Missing columns required for --extended-strat: {missing}")
        multicat = LabelEncoder().fit_transform(
            pd.concat([train.loc[: len(y) - 1, strat_cols].reset_index(drop=True), y.reset_index(drop=True)], axis=1)
            .astype(str)
            .agg('_'.join, axis=1)
        )
        strat_y = pd.Series(multicat)

    cat_test_pred = np.zeros(len(test))
    cat_auc_scores = []

    for i, (train_index, test_index) in enumerate(skf.split(X, strat_y), 1):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        TE = TargetEncoder(cols_to_encode=int_cols, cv=5, smooth='auto', aggs=['mean'], drop_original=False)
        X_train = TE.fit_transform(X_train, y_train)
        X_test = TE.transform(X_test)
        test_fold = TE.transform(test) # Transform test for this fold

        cat_model = CatBoostClassifier(**CAT_params)
        cat_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=1000)
        
        cat_test_pred += cat_model.predict_proba(test_fold)[:,1] / folds
        
        cat_y_pred = cat_model.predict_proba(X_test)[:,1]
        score = roc_auc_score(y_test, cat_y_pred)
        cat_auc_scores.append(score)
        print(f'Fold {i} CatBoost AUC score: {score:.5f}')

    print(f"Mean AUC: {np.mean(cat_auc_scores):.5f}")

    submission = pd.DataFrame({
        'id': sample['id'],
        'diagnosed_diabetes': cat_test_pred
    })
    submission.to_csv(args.out, index=False)
    print(f"Saved {args.out}")

if __name__ == "__main__":
    main()
