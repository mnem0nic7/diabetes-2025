
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import gc

def train_weighted_model():
    print("Loading data...")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    
    # Save target and ids
    target_col = 'diagnosed_diabetes'
    id_col = 'id'
    
    y = train[target_col]
    train_ids = train[id_col]
    test_ids = test[id_col]
    
    # --- Step 1: Adversarial Validation / Weight Calculation ---
    print("\n--- Step 1: Calculating Adversarial Weights ---")
    
    # Prepare data for adversarial validation
    train_adv = train.drop(columns=[target_col, id_col]).copy()
    test_adv = test.drop(columns=[id_col]).copy()
    
    train_adv['is_test'] = 0
    test_adv['is_test'] = 1
    
    adv_data = pd.concat([train_adv, test_adv], axis=0).reset_index(drop=True)
    y_adv = adv_data['is_test']
    X_adv = adv_data.drop(columns=['is_test'])
    
    # Encode categoricals
    cat_cols = []
    for c in X_adv.columns:
        if X_adv[c].dtype == 'object':
            le = LabelEncoder()
            X_adv[c] = le.fit_transform(X_adv[c].astype(str))
            cat_cols.append(c)
            
    # Train proxy model to predict is_test
    # We use the whole dataset for this to get weights for the train set
    adv_model = lgb.LGBMClassifier(
        objective='binary',
        n_estimators=100,
        random_state=42,
        n_jobs=4,
        verbosity=-1
    )
    
    adv_model.fit(X_adv, y_adv)
    
    # Get probabilities for the TRAINING set only
    # We need to know P(is_test | x) for x in train
    X_train_encoded = X_adv.iloc[:len(train)]
    p_test = adv_model.predict_proba(X_train_encoded)[:, 1]
    
    # Calculate weights: w = p(test)/p(train) approx p_test / (1 - p_test)
    # Clip probabilities to avoid division by zero or extreme weights
    p_test = np.clip(p_test, 1e-5, 1 - 1e-5)
    weights = p_test / (1 - p_test)
    
    # Clip weights reasonably
    weights = np.clip(weights, 0, 10.0)
    weights = weights / weights.mean() # Normalize
    
    print(f"Weights calculated. Mean: {weights.mean():.4f}, Max: {weights.max():.4f}, Min: {weights.min():.4f}")
    
    # Cleanup to save memory
    del train_adv, test_adv, adv_data, X_adv, y_adv, adv_model, X_train_encoded
    gc.collect()
    
    # --- Step 2: Weighted Training ---
    print("\n--- Step 2: Training Weighted Model ---")
    
    # Prepare main training data
    X = train.drop(columns=[target_col, id_col])
    X_test = test.drop(columns=[id_col])
    
    del train, test
    gc.collect()
    
    # Encode main data (re-using simple label encoding for consistency)
    for c in X.columns:
        if X[c].dtype == 'object':
            combined = pd.concat([X[c], X_test[c]], axis=0).astype(str)
            le = LabelEncoder()
            le.fit(combined)
            X[c] = le.transform(X[c].astype(str))
            X_test[c] = le.transform(X_test[c].astype(str))
            del combined
            gc.collect()
            
    # Stratified K-Fold
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    # Convert to numpy to save pandas overhead during indexing loop
    X_vals = X.values
    y_vals = y.values
    X_test_vals = X_test.values
    
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_va = X_vals[tr_idx], X_vals[va_idx]
        y_tr, y_va = y_vals[tr_idx], y_vals[va_idx]
        w_tr = weights[tr_idx] # Apply weights to training set only
        
        clf = lgb.LGBMClassifier(
            objective='binary',
            metric='auc',
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=4,
            verbosity=-1
        )
        
        # Fit with sample weights
        clf.fit(
            X_tr, y_tr, 
            sample_weight=w_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(stopping_rounds=50)]
        )
        
        oof_preds[va_idx] = clf.predict_proba(X_va)[:, 1]
        test_preds += clf.predict_proba(X_test)[:, 1] / n_splits
        
        auc = roc_auc_score(y_va, oof_preds[va_idx])
        print(f"Fold {fold} AUC: {auc:.5f}")
        
    overall_auc = roc_auc_score(y, oof_preds)
    print(f"\nOverall Weighted CV AUC: {overall_auc:.5f}")
    
    # --- Step 3: Submission ---
    sub = pd.DataFrame({
        'id': test_ids,
        'diagnosed_diabetes': test_preds
    })
    
    out_file = "submission_weighted_lgbm.csv"
    sub.to_csv(out_file, index=False)
    print(f"Submission saved to {out_file}")

if __name__ == "__main__":
    train_weighted_model()
