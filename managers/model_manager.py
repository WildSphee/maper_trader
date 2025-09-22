from typing import List, Tuple, Union, Optional, Dict
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, SGDClassifier, LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer

class ModelManager:
    """
    Dual-head manager: classifier for direction, regressor for next-bar return,
    plus quantile regressors for uncertainty.
    """

    def __init__(
        self,
        predictor_cols: List[str],
        cls_name: str = "hgb",
        reg_name: str = "hgb",
        class_weight: Union[str, dict, None] = "balanced",
        random_state: int = 42,
        q_low: float = 0.10,
        q_high: float = 0.90,
    ) -> None:
        self.predictor_cols = predictor_cols
        self.cls_name = cls_name
        self.reg_name = reg_name
        self.class_weight = class_weight
        self.random_state = random_state
        self.q_low, self.q_high = float(q_low), float(q_high)

        self.cls_pipe: Optional[Pipeline] = None
        self.reg_pipe: Optional[Pipeline] = None
        self.qlo_pipe: Optional[Pipeline] = None
        self.qhi_pipe: Optional[Pipeline] = None

        self.last_cls_acc: Optional[float] = None
        self.last_reg_mape: Optional[float] = None  # on returns, “MAPE” = mean abs pct err of exp(ret)-1
        self.last_reg_mae: Optional[float] = None

    # ---------- blocks ----------
    def _needs_scaler(self, name: str) -> bool:
        return name in {"logreg", "sgdlog", "linsvc", "linreg"}

    def _selector(self):
        return FunctionTransformer(
            lambda X: X[self.predictor_cols].to_numpy() if isinstance(X, pd.DataFrame) else X,
            feature_names_out="one-to-one",
        )

    # ----- classifiers -----
    def _build_classifier(self, name: str):
        if name == "logreg":
            return LogisticRegression(max_iter=5000, solver="saga", C=0.5,
                                      class_weight=self.class_weight, random_state=self.random_state)
        if name == "sgdlog":
            base = SGDClassifier(loss="log_loss", penalty="l2",
                                 class_weight=self.class_weight, random_state=self.random_state)
            return CalibratedClassifierCV(base, cv=3)
        if name == "rf":
            return RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                         class_weight=self.class_weight, random_state=self.random_state)
        if name == "hgb":
            return HistGradientBoostingClassifier(random_state=self.random_state)
        if name == "linsvc":
            svc = LinearSVC(class_weight=self.class_weight, random_state=self.random_state)
            return CalibratedClassifierCV(svc, cv=3)
        raise ValueError(f"Unknown classifier: {name}")

    # ----- regressors -----
    def _build_regressor(self, name: str):
        if name == "linreg":
            return LinearRegression()
        if name == "rf":
            return RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=self.random_state)
        if name == "hgb":
            # squared_error is fine for returns
            from sklearn.ensemble import HistGradientBoostingRegressor
            return HistGradientBoostingRegressor(random_state=self.random_state)
        raise ValueError(f"Unknown regressor: {name}")

    def _build_quantile(self, q: float):
        # Fast, robust quantile via GradientBoostingRegressor(loss="quantile")
        return GradientBoostingRegressor(loss="quantile", alpha=q, random_state=self.random_state)

    def _pipe(self, estimator, needs_scaler: bool) -> Pipeline:
        steps: List[Tuple[str, object]] = [
            ("select", self._selector()),
            ("impute", SimpleImputer(strategy="median")),
        ]
        if needs_scaler:
            steps.append(("scale", StandardScaler()))
        steps.append(("model", estimator))
        return Pipeline(steps)

    # ---------- training ----------
    def train_dual(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y_cls: Union[pd.Series, np.ndarray],
        y_reg: Union[pd.Series, np.ndarray],
        test_size: float = 0.2,
    ) -> Dict[str, float]:
        X_tr, X_te, yc_tr, yc_te = train_test_split(
            X, y_cls, test_size=test_size, random_state=self.random_state, stratify=y_cls
        )
        # same split for regression to keep alignment
        _, _, yr_tr, yr_te = train_test_split(
            X, y_reg, test_size=test_size, random_state=self.random_state, stratify=None
        )

        # classifier
        self.cls_pipe = self._pipe(self._build_classifier(self.cls_name), self._needs_scaler(self.cls_name))
        self.cls_pipe.fit(X_tr, yc_tr)
        cls_acc = accuracy_score(yc_te, self.cls_pipe.predict(X_te))

        # regressor (expected return)
        self.reg_pipe = self._pipe(self._build_regressor(self.reg_name), self._needs_scaler(self.reg_name))
        self.reg_pipe.fit(X_tr, yr_tr)
        pred_ret = self.reg_pipe.predict(X_te)
        mae = mean_absolute_error(yr_te, pred_ret)
        # Convert to “MAPE” on price change: compare exp(ret)-1 vs exp(true)-1
        mape_like = np.mean(
            np.abs((np.expm1(pred_ret) - np.expm1(yr_te)) / (np.expm1(yr_te) + 1e-12))
        )

        # quantiles
        self.qlo_pipe = self._pipe(self._build_quantile(self.q_low), self._needs_scaler("linreg"))
        self.qhi_pipe = self._pipe(self._build_quantile(self.q_high), self._needs_scaler("linreg"))
        self.qlo_pipe.fit(X_tr, yr_tr)
        self.qhi_pipe.fit(X_tr, yr_tr)

        self.last_cls_acc = float(cls_acc)
        self.last_reg_mae = float(mae)
        self.last_reg_mape = float(mape_like)
        return {"cls_acc": self.last_cls_acc, "reg_mae": self.last_reg_mae, "reg_mape_like": self.last_reg_mape}

    # ---------- inference (trading outputs) ----------
    def _ensure_trained(self):
        if any(p is None for p in [self.cls_pipe, self.reg_pipe, self.qlo_pipe, self.qhi_pipe]):
            raise RuntimeError("Models not trained. Call train_dual() first.")

    def trading_outputs(
        self,
        X_row: Union[pd.DataFrame, np.ndarray],
        current_price: float,
        fee_bps: float = 2.0,        # per side, bps
        slippage_bps: float = 1.0,   # per side, bps
        kelly_cap: float = 0.2,      # cap max fraction
    ) -> Dict[str, float]:
        """
        Returns a compact dict of trading-ready metrics for one latest row.
        """
        self._ensure_trained()
        # select single row
        x = X_row.iloc[-1:] if isinstance(X_row, pd.DataFrame) else X_row[-1:].reshape(1, -1)

        p_up = float(self.cls_pipe.predict_proba(x)[0][1])
        ret_mean = float(self.reg_pipe.predict(x)[0])         # E[log return]
        ret_qlo = float(self.qlo_pipe.predict(x)[0])          # low quantile log ret
        ret_qhi = float(self.qhi_pipe.predict(x)[0])          # high quantile log ret

        # convert to simple returns
        r_mean = float(np.expm1(ret_mean))
        r_lo   = float(np.expm1(ret_qlo))
        r_hi   = float(np.expm1(ret_qhi))

        # price forecasts
        px_exp = current_price * (1.0 + r_mean)
        px_lo  = current_price * (1.0 + r_lo)
        px_hi  = current_price * (1.0 + r_hi)

        # costs
        roundtrip_cost = (fee_bps + slippage_bps) * 2.0 / 10_000.0

        # downside (VaR-ish at (1-q_high) level, using lower quantile)
        var_pct = min(0.0, r_lo)  # a negative number (loss percent)

        # suggested SL/TP from quantiles
        stop_loss = px_lo
        take_profit = px_hi

        # Kelly fraction (approx): p = p_up, payoff ~ r_hi/abs(r_lo)
        # Safeguards against divide-by-zero / signs
        gain = max(1e-6, r_hi)
        loss = max(1e-6, -min(0.0, r_lo))
        edge = p_up * gain - (1 - p_up) * loss
        if gain > 0:
            kelly = edge / gain
        else:
            kelly = 0.0
        kelly = float(np.clip(kelly, 0.0, kelly_cap))

        # expected net return after costs (one-way)
        exp_net = r_mean - roundtrip_cost / 2.0

        return {
            "prob_up": p_up,
            "expected_return": r_mean,
            "expected_price": px_exp,
            "ret_q10": r_lo,
            "ret_q90": r_hi,
            "price_q10": px_lo,
            "price_q90": px_hi,
            "var_like": var_pct,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "kelly_fraction": kelly,
            "expected_return_net_cost": exp_net,
            "roundtrip_cost": roundtrip_cost,
        }
