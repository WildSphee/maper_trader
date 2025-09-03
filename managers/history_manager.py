from typing import Tuple

import numpy as np
import pandas as pd
import talib
from binance.client import Client

INTERVAL_TO_MS = {
    Client.KLINE_INTERVAL_1MINUTE: 60_000,
    Client.KLINE_INTERVAL_3MINUTE: 3 * 60_000,
    Client.KLINE_INTERVAL_5MINUTE: 5 * 60_000,
    Client.KLINE_INTERVAL_15MINUTE: 15 * 60_000,
    Client.KLINE_INTERVAL_30MINUTE: 30 * 60_000,
    Client.KLINE_INTERVAL_1HOUR: 60 * 60_000,
    Client.KLINE_INTERVAL_2HOUR: 2 * 60 * 60_000,
    Client.KLINE_INTERVAL_4HOUR: 4 * 60 * 60_000,
    Client.KLINE_INTERVAL_6HOUR: 6 * 60 * 60_000,
    Client.KLINE_INTERVAL_8HOUR: 8 * 60 * 60_000,
    Client.KLINE_INTERVAL_12HOUR: 12 * 60 * 60_000,
    Client.KLINE_INTERVAL_1DAY: 24 * 60 * 60_000,
}


class HistoryManager:
    """
    Manages historical and latest market data, indicators, and features.
    """

    def __init__(
        self,
        client: Client,
        symbol: str,
        interval: str,
        start_str: str,
        timelag: int = 20,
    ) -> None:
        self.client = client
        self.symbol = symbol
        self.interval = interval
        self.start_str = start_str
        self.timelag = timelag

        if interval not in INTERVAL_TO_MS:
            raise ValueError(f"Unsupported interval: {interval}")

        self.predictor_cols = [
            "EMA",
            "CMO",
            "MINUSDM",
            "PLUSDM",
            "CLOSE",
            "CLOSEL1",
            "CLOSEL2",
            "CLOSEL3",
            "PATT_3OUT",
            "PATT_CMB",
            "RSI",
            "MACD",
            "MACD_SIGNAL",
            "MACD_HIST",
            "ADX",
            "ATR",
            "NATR",
            "BB_UPPER",
            "BB_MIDDLE",
            "BB_LOWER",
            "BB_WIDTH",
            "OBV",
            "MFI",
            "AD",
            "ADOSC",
            "STOCH_K",
            "STOCH_D",
            "TRIX",
            "ROC",
        ]


    def update_with_latest(self, limit: int = 2) -> None:
        """
        Pulls the most recent klines (default 2 for safety), appends any new CLOSED bars,
        then recomputes indicators/features.
        """
        recent = self.client.get_klines(
            symbol=self.symbol, interval=self.interval, limit=limit
        )

        df = self.df_ohlcv.copy()

        # --- basic series ---
        close = df["close"].to_numpy(dtype=float)
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        openp = df["open"].to_numpy(dtype=float)
        vol = df["volume"].to_numpy(dtype=float)

        # --- your existing indicators ---
        ema = talib.EMA(close, timeperiod=self.timelag)
        cmo = talib.CMO(close, timeperiod=self.timelag)
        minusdm = talib.MINUS_DM(high, low, timeperiod=self.timelag)
        plusdm = talib.PLUS_DM(high, low, timeperiod=self.timelag)

        patt_3out_raw = talib.CDL3OUTSIDE(openp, high, low, close)
        patt_cmb_raw = talib.CDLCLOSINGMARUBOZU(openp, high, low, close)

        # --- lags ---
        closel1 = df["close"].shift(1)
        closel2 = df["close"].shift(2)
        closel3 = df["close"].shift(3)

        # --- NEW: TA-Lib indicators (periods mostly = timelag) ---
        # Momentum
        rsi = talib.RSI(close, timeperiod=self.timelag)
        macd, macd_sig, macd_hist = talib.MACD(
            close, fastperiod=12, slowperiod=26, signalperiod=9
        )
        adx = talib.ADX(high, low, close, timeperiod=self.timelag)
        trix = talib.TRIX(close, timeperiod=self.timelag)  # triple smoothed ROC
        roc = talib.ROC(close, timeperiod=self.timelag)  # simple ROC

        # Volatility
        atr = talib.ATR(high, low, close, timeperiod=self.timelag)
        natr = talib.NATR(
            high, low, close, timeperiod=self.timelag
        )  # normalized ATR (%)
        bb_upper, bb_middle, bb_lower = talib.BBANDS(
            close, timeperiod=self.timelag, nbdevup=2, nbdevdn=2, matype=0
        )
        bb_width = np.divide(
            bb_upper - bb_lower,
            bb_middle,
            out=np.full_like(bb_middle, np.nan, dtype=float),
            where=(bb_middle != 0) & ~np.isnan(bb_middle),
        )

        # Volume/flow
        obv = talib.OBV(close, vol)
        mfi = talib.MFI(high, low, close, vol, timeperiod=self.timelag)
        ad = talib.AD(high, low, close, vol)  # Chaikin Acc/Dist line
        adosc = talib.ADOSC(high, low, close, vol, fastperiod=3, slowperiod=10)

        # Stochastics (defaults are classic 14,3,3; here tie to timelag)
        # If timelag < 3, clip to safe minimums to avoid warnings
        k_period = max(int(self.timelag), 5)
        d_period = 3
        stoch_k, stoch_d = talib.STOCH(
            high,
            low,
            close,
            fastk_period=k_period,
            slowk_period=d_period,
            slowk_matype=0,
            slowd_period=d_period,
            slowd_matype=0,
        )

        # --- assemble feature table ---
        feat = pd.DataFrame(
            {
                "EMA": ema,
                "CMO": cmo,
                "MINUSDM": minusdm,
                "PLUSDM": plusdm,
                "CLOSE": df["close"],
                "CLOSEL1": closel1,
                "CLOSEL2": closel2,
                "CLOSEL3": closel3,
                "PATT_3OUT": patt_3out,
                "PATT_CMB": patt_cmb,
                "RSI": rsi,
                "MACD": macd,
                "MACD_SIGNAL": macd_sig,
                "MACD_HIST": macd_hist,
                "ADX": adx,
                "ATR": atr,
                "NATR": natr,
                "BB_UPPER": bb_upper,
                "BB_MIDDLE": bb_middle,
                "BB_LOWER": bb_lower,
                "BB_WIDTH": bb_width,
                "OBV": obv,
                "MFI": mfi,
                "AD": ad,
                "ADOSC": adosc,
                "STOCH_K": stoch_k,
                "STOCH_D": stoch_d,
                "TRIX": trix,
                "ROC": roc,
            },
            index=df.index,
        )

        up_down = (df["close"].shift(-1) > df["close"]).astype("float64")
        feat["UP_DOWN"] = up_down

        feat = feat.replace([np.inf, -np.inf], np.nan).dropna()
        feat = feat.astype({"UP_DOWN": "int64"})

        self.df_features = feat

    @property
    def last_closed_open_time_ms(self) -> int:
        """
        Returns the open_time ms of the last CLOSED bar in df_features.
        (df_features has dropped the last unlabeled row, so last row is safely closed)
        """
        # map index back to base ohlcv
        last_idx = self.df_features.index[-1]
        return int(self.df_ohlcv.loc[last_idx, "open_time"])

    @property
    def next_bar_close_time_ms(self) -> int:
        """
        Compute the expected close time of the next bar after the last CLOSED bar in features.
        """
        last_open_ms = self.last_closed_open_time_ms()
        return last_open_ms + self.interval_ms

    @property
    def dataset(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Returns (X, y) for model training.
        """
        X = self.df_features[self.predictor_cols].copy()
        y = self.df_features["UP_DOWN"].copy()
        return X, y

