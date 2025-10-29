from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

def train_logistic(X_train, y_train):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipe.fit(X_train, y_train)
    return pipe

def train_xgboost(X_train, y_train):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
        "max_depth": 6,
        "eta": 0.1,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "seed": 42,
    }
    bst = xgb.train(params, dtrain, num_boost_round=300)
    bst.save_model("xgb_model.json")
    return bst