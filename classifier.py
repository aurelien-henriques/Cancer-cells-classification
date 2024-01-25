from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import StratifiedKFold

def Preprocess(X):
    X_ar = X.toarray()
    X_pre = np.log1p(X_ar)
    #X_pre = X_pre/np.sum(X_pre,axis=1,keepdims=True)
    return X_pre

class LassoFeatureSelector(TransformerMixin, BaseEstimator):
    def __init__(self, alpha=0.017):
        self.alpha = alpha
        self.lasso = Lasso(alpha=alpha)
        self.scaler = StandardScaler()
        self.selected_features = None

    def fit(self, X, y=None):
        X_scaled = self.scaler.fit_transform(X)
        self.lasso.fit(X_scaled, y)
        self.selected_features = [i for i, coef in enumerate(self.lasso.coef_) if coef != 0]
        return self

    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        return X_scaled[:, self.selected_features]

class Classifier:
    def __init__(self, lasso_alpha=0.017):
        self.base_models = [
            ('rf', make_pipeline(LassoFeatureSelector(alpha=lasso_alpha), 
                                 StandardScaler(), 
                                 RandomForestClassifier(n_estimators=256,criterion='entropy', random_state=30))),

            ('mlp', make_pipeline(LassoFeatureSelector(alpha=lasso_alpha), 
                                  StandardScaler(), 
                                  MLPClassifier(hidden_layer_sizes=(256,), max_iter=1000,solver='adam',alpha=0.01, random_state=30)))
        ]
        self.meta_model = MLPClassifier(hidden_layer_sizes=(256,),alpha=0.01,solver='adam', max_iter=1000, random_state=30)
        self.stacking_model = StackingClassifier(estimators=self.base_models, final_estimator=self.meta_model, cv=StratifiedKFold(n_splits=5))

    def fit(self, X_train, y_train):
        X_train_pre = Preprocess(X_train)
        self.stacking_model.fit(X_train_pre, y_train)

    def predict_proba(self, X_test):
        X_test_pre = Preprocess(X_test)
        return self.stacking_model.predict_proba(X_test_pre)