from typing import Callable, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC



class ModelManager:
    """
    Flexible sklearn pipeline manager with multiple back-ends.
    All models expose predict_proba via native method or calibration.
    """

    def __init__(
        self,
        predictor_cols: List[str],
        model_name: str = "logreg",
    ) -> None:
        self.predictor_cols = predictor_cols
        self.numeric_cols = predictor_cols[:-2]  
        self.model_name = model_name
      
    def _build_model(self, name: str):
        if name == "logreg":
            return LogisticRegression(
                max_iter=5000,
                solver="saga",
                C=0.5,
                class_weight=self.class_weight,
                random_state=self.random_state,
            )
        if name == "sgdlog":
            base = SGDClassifier(
                loss="log_loss",
                penalty="l2",
                class_weight=self.class_weight,
                random_state=self.random_state,
            )
            return CalibratedClassifierCV(base, cv=3)
        if name == "rf":
            return RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                n_jobs=-1,
                class_weight=self.class_weight,
                random_state=self.random_state,
            )
        if name == "hgb":
            return HistGradientBoostingClassifier(
                learning_rate=0.1,
                max_depth=None,
                max_bins=255,
                random_state=self.random_state,
            )
        if name == "linsvc":
            svc = LinearSVC(
                class_weight=self.class_weight,
                random_state=self.random_state,
            )
            # Calibrate to get predict_proba
            return CalibratedClassifierCV(svc, cv=3)

        raise ValueError(f"Unknown estimator name: {name}")

    def _build_pipeline(self) -> Pipeline:
        steps: List[Tuple[str, object]] = []
        # add code here
        return Pipeline(steps)

    def train(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]
    ) -> float:
        """train the model, return test accuracy"""

        stratify = y if isinstance(y, (pd.Series, np.ndarray)) else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=stratify
        )
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X_train, y_train)
        y_pred = self.pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        self.last_test_accuracy = float(acc)
        return self.last_test_accuracy

    def predict(self, X_one: Union[pd.DataFrame, np.ndarray]) -> float:
        
        if self.pipeline is None:
            raise RuntimeError("Model is not trained.")
        
        proba = self.pipeline.predict_proba(X_one)[0][1]
        return float(proba)
