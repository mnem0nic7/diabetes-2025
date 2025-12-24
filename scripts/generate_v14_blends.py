import pandas as pd

def blend(file1, file2, w1, w2, output_name):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    # Ensure IDs match
    assert (df1['id'] == df2['id']).all()
    
    df1['diagnosed_diabetes'] = df1['diagnosed_diabetes'] * w1 + df2['diagnosed_diabetes'] * w2
    df1.to_csv(output_name, index=False)
    print(f"Created {output_name}")

catboost_file = 'submission_v14_catboost_te.csv'
best_lgb_file = 'submission_v8_cutoff_boost_w15_origenc.csv'

# 50-50 Blend
blend(catboost_file, best_lgb_file, 0.5, 0.5, 'submission_v14_blend_cat_w15_50_50.csv')

# 30-70 Blend (30% CatBoost, 70% LGBM)
blend(catboost_file, best_lgb_file, 0.3, 0.7, 'submission_v14_blend_cat_w15_30_70.csv')

# 70-30 Blend (70% CatBoost, 30% LGBM)
blend(catboost_file, best_lgb_file, 0.7, 0.3, 'submission_v14_blend_cat_w15_70_30.csv')
