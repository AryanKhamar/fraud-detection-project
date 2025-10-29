from sklearn.metrics import classification_report, precision_recall_curve, auc
import xgboost as xgb

def evaluate_model(model, X_test, y_test, xgb_model=False):
    if xgb_model:
        dtest = xgb.DMatrix(X_test)
        y_proba = model.predict(dtest)
        y_pred = (y_proba > 0.5).astype(int)
    else:
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba > 0.5).astype(int)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    print(f"PR AUC: {pr_auc:.4f}")
    return pr_auc