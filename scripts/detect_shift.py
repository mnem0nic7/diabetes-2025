
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

def detect_shift():
    print("Loading data...")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    # Add origin label
    train['is_test'] = 0
    test['is_test'] = 1

    # Drop target from train if present using defaults from existing code knowledge
    if 'diagnosed_diabetes' in train.columns:
        train = train.drop(columns=['diagnosed_diabetes'])

    # Combine
    print(f"Train shape: {train.shape}, Test shape: {test.shape}")
    df = pd.concat([train, test], axis=0).reset_index(drop=True)
    
    # Identify features
    drop_cols = ['id', 'is_test']
    features = [c for c in df.columns if c not in drop_cols]
    
    # Handle categoricals for LGBM
    cat_cols = []
    for c in features:
        if df[c].dtype == 'object':
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c].astype(str))
            cat_cols.append(c)
    
    X = df[features]
    y = df['is_test']
    
    print(f"Running adversarial validation on {len(features)} features...")
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(df))
    feature_importances = pd.DataFrame(index=features)
    feature_importances['gain'] = 0.0
    
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        
        clf = lgb.LGBMClassifier(
            objective='binary',
            metric='auc',
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
        
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
        oof_preds[va_idx] = clf.predict_proba(X_va)[:, 1]
        
        feature_importances['gain'] += clf.feature_importances_ / 5
        print(f"Fold {fold} AUC: {roc_auc_score(y_va, oof_preds[va_idx]):.4f}")
        
    overall_auc = roc_auc_score(y, oof_preds)
    print(f"\nOverall AUC: {overall_auc:.4f}")
    
    if overall_auc > 0.6:
        print("\nPossible covariate shift detected!")
        print("Top 10 features drifting:")
        print(feature_importances.sort_values('gain', ascending=False).head(10))
    else:
        print("\nNo significant shift detected (AUC close to 0.5).")

if __name__ == "__main__":
    detect_shift()
