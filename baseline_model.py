import warnings
warnings.filterwarnings('ignore')

from auto_eda import smart_eda
from data_fetcher import get_price_data
from indicators import add_technical_indicators
from feature_engineering import add_custom_features, get_final_features
from label_generator import create_labels

from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,classification_report

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier

def prepare_data(stock_name='RELIANCE', n_days=2,run_eda=False): #(default)
    df_20y, _ = get_price_data(stock_name)
    df_20y = add_technical_indicators(df_20y)
    df_20y = add_custom_features(df_20y)
    df_20y = create_labels(df_20y, n_days=n_days)

    X = get_final_features(df_20y)
    y = df_20y.loc[X.index, 'Label']

    if run_eda:
    # Combine X and y into one DataFrame for EDA with label
        df_eda = X.copy()
        df_eda['Label'] = y.values
        smart_eda(df_eda, label_col_name='Label')



    return train_test_split(X, y, test_size=0.15, shuffle=False)

def evaluate_model(name,model, X_train_scaled,y_train,X_test_scaled,y_test):
    model.fit(X_train_scaled,y_train)
    #  Precision–Recall Trade-off Optimize (Threshold tuning)
    y_probs = model.predict_proba(X_test_scaled)[:,1]  # Probability of class True
    y_pred = (y_probs >= 0.45).astype(int)        # Default 0.5 hai, try 0.6 or 0.7
    # y_pred = model.predict(X_test)

    scores = {'Model': name,
             'Accuracy': accuracy_score(y_test,y_pred),
             'Precision': precision_score(y_test, y_pred),
             'Recall': recall_score(y_test,y_pred),
             'F1 score': f1_score(y_test,y_pred)}

    return scores

def run_all(X_train_scaled,X_test_scaled,y_train,y_test):
    models = []

    cv = TimeSeriesSplit(n_splits=5)
    rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]}
    
    rf = GridSearchCV(RandomForestClassifier(random_state=42,class_weight='balanced'), rf_params ,scoring='f1', cv=cv)
    models.append(("Random Forest", rf))

    xgb_params = {
    'n_estimators': [100, 200,300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.001],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1]}
    
    neg = sum(y_train == 0)
    pos = sum(y_train == 1)
    scale_pos_weight = neg / pos
    xgb = GridSearchCV(XGBClassifier(use_label_encoder=False,scale_pos_weight=scale_pos_weight, eval_metric ='logloss', random_state=42), xgb_params,scoring='f1', cv=cv)
    models.append(('XGBoost', xgb))


    gb_params = {
    'n_estimators': [100, 150,300],
    'learning_rate': [0.01, 0.05, 0.001],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0]}
    
    gb = GridSearchCV(GradientBoostingClassifier(random_state=42), gb_params, cv=cv,scoring='f1')
    models.append(("Gradient Boosting", gb))

    ada_params = {
    'n_estimators': [50, 100, 150,300],
    'learning_rate': [0.001, 0.1, 0.5, 1.0]}
    
    ada = GridSearchCV(AdaBoostClassifier(random_state=42), ada_params, cv=cv,scoring='f1')
    models.append(("AdaBoost", ada))

    results = []

    for name, model in models:
        print(f"Training and Evaluating ------>>>>>>>> {name}.....\n")
        score = evaluate_model(name, model,X_train_scaled, y_train, X_test_scaled, y_test)
        print(f"Best params combinations for {name} -->>>>>>>>> {model.best_params_} ---> Best score for this comnination is {model.best_score_}")
        results.append(score)

    return results,X_test_scaled,y_test

def train_baseline_model(stock_name='RELIANCE', n_days=2):
    """
    Train the baseline model (price-based) and return the best model and test set.
    """
    X_train, X_test, y_train, y_test = prepare_data(stock_name=stock_name, n_days=1, run_eda=False)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    
    best_model = XGBClassifier(
        use_label_encoder=False,
        scale_pos_weight=(sum(y_train == 0) / sum(y_train == 1)),
        eval_metric='logloss',
        random_state=42
    )

    best_model.fit(X_train_scaled, y_train)
    y_pred = best_model.predict(X_test_scaled)

    # final test report
    print("Baseline Model (Price Only) Evaluation:")
    print(classification_report(y_test, y_pred))

    return best_model, X_test_scaled, y_test

def train_all_models(stock_name='RELIANCE', run_eda=False):
    
    X_train, X_test, y_train, y_test = prepare_data(stock_name=stock_name, n_days=5, run_eda=run_eda)
    
    print("y_train and y_test proportion of labels")
    print(y_train.value_counts(normalize=True))
    print(y_test.value_counts(normalize=True))

    # scaling by standardscaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results,X_test_scaled,y_test = run_all(X_train_scaled, X_test_scaled, y_train, y_test)

    import pandas as pd
    df_results = pd.DataFrame(results)
    print("\nModel Comparison (by Accuracy):\n", df_results.sort_values(by='Accuracy', ascending=False))
    print("\nModel Comparison (by F1 Score):\n", df_results.sort_values(by='F1 score', ascending=False))

    return df_results


# if __name__ == "__main__":
#     stock = input("Enter stock name (e.g., RELIANCE, INFY): ").strip().upper()


#     # Prepare for training
#     X_train, X_test, y_train, y_test = prepare_data(stock_name=stock, n_days=5,run_eda=True)
#     print("y_train and y_test proportion of labels")
#     print(y_train.value_counts(normalize=True))
#     print(y_test.value_counts(normalize=True))

    
#     # Apply StandardScaler
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
    
#     results = run_all(X_train_scaled, X_test_scaled, y_train, y_test)

#     import pandas as pd
#     df_results = pd.DataFrame(results)
#     print("\nModel Comparison (by Accuracy):\n", df_results.sort_values(by='Accuracy', ascending=False))
#     print("\nModel Comparison (by F1 Score):\n", df_results.sort_values(by='F1 score', ascending=False))

    
