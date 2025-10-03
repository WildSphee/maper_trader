from typing import Tuple
import numpy as np
import pandas as pd
import talib
from binance.client import Client

from features.fear_index import get_fng_features
from features.on_chain_data import get_btc_onchain_smoothed

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

BINANCE_TO_STRIV = {
    Client.KLINE_INTERVAL_1HOUR: "1h",
    Client.KLINE_INTERVAL_4HOUR: "4h",
    Client.KLINE_INTERVAL_1DAY: "1d",
}

SUPPORTED_IMPORT_INTERVALS = set(BINANCE_TO_STRIV.keys())


def _bin_pattern_to_binary(arr: np.ndarray) -> np.ndarray:
    # {-100, 0, +100} map to {0,1}
    return (np.asarray(arr, dtype=float) != 0).astype(float)

def _klines_to_df(klines: list) -> pd.DataFrame:
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(klines, columns=cols)
    df = df[["open_time","open","high","low","close","volume","close_time"]].copy()
    df["open"]   = df["open"].astype(float)
    df["high"]   = df["high"].astype(float)
    df["low"]    = df["low"].astype(float)
    df["close"]  = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df["open_time"]  = df["open_time"].astype("int64")
    df["close_time"] = df["close_time"].astype("int64")
    df.set_index("open_time", inplace=True)
    df.sort_index(inplace=True)
    return df


class HistoryManager:
    """OHLCV + TA + Fear&Greed + On-chain (all aligned, leakage-safe)"""

    def __init__(
        self,
        client: Client,
        symbol: str,
        interval: str,               # expect Client.KLINE_INTERVAL_1HOUR / 4HOUR / 1DAY
        start_str: str,              # e.g., "1 Jan, 2023"
        timelag: int = 20,
        fng_lookback_days: int = 3,
        onchain_lookback_days: int = 3,
        include_fng: bool = True,
        include_onchain: bool = True,
    ) -> None:
        if interval not in INTERVAL_TO_MS:
            raise ValueError(f"Unsupported interval: {interval}")
        if interval not in SUPPORTED_IMPORT_INTERVALS:
            raise ValueError("This manager supports only 1h / 4h / 1d for imported features.")

        self.client = client
        self.symbol = symbol
        self.interval = interval
        self.start_str = start_str
        self.timelag = int(timelag)
        self.interval_ms = INTERVAL_TO_MS[interval]
        self.fng_lookback_days = int(fng_lookback_days)
        self.onchain_lookback_days = int(onchain_lookback_days)
        self.include_fng = include_fng
        self.include_onchain = include_onchain

        # Predictor columns (we'll append the imported ones dynamically)
        self.predictor_cols = [
            "EMA","MINUSDM","PLUSDM","CLOSE","CLOSEL1","CLOSEL2",
            "PATT_3OUT","PATT_CMB","RSI","MACD","MACD_SIGNAL","MACD_HIST","ADX",
            "ATR","NATR","BB_UPPER","BB_MIDDLE","BB_LOWER","BB_WIDTH","OBV","MFI",
            "AD","ADOSC","STOCH_K","STOCH_D",
        ]  # type: ignore

        raw = self.client.get_historical_klines(
            symbol=self.symbol, interval=self.interval, start_str=self.start_str
        )
        self.df_ohlcv = _klines_to_df(raw)
        self.df_features = pd.DataFrame()
        self._recompute_features()

    # -------- core feature recompute --------
    def _recompute_features(self) -> None:
        df = self.df_ohlcv
        if len(df) < max(self.timelag, 30):
            self.df_features = pd.DataFrame()
            return

        close = df["close"].to_numpy(dtype=float)
        high  = df["high"].to_numpy(dtype=float)
        low   = df["low"].to_numpy(dtype=float)
        openp = df["open"].to_numpy(dtype=float)
        vol   = df["volume"].to_numpy(dtype=float)

        ema     = talib.EMA(close, timeperiod=self.timelag)
        minusdm = talib.MINUS_DM(high, low, timeperiod=self.timelag)
        plusdm  = talib.PLUS_DM(high, low, timeperiod=self.timelag)
        patt_3out = _bin_pattern_to_binary(talib.CDL3OUTSIDE(openp, high, low, close))
        patt_cmb  = _bin_pattern_to_binary(talib.CDLCLOSINGMARUBOZU(openp, high, low, close))
        closel1 = df["close"].shift(1)
        closel2 = df["close"].shift(2)
        rsi   = talib.RSI(close, timeperiod=self.timelag)
        macd, macd_sig, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        adx   = talib.ADX(high, low, close, timeperiod=self.timelag)
        atr   = talib.ATR(high, low, close, timeperiod=self.timelag)
        natr  = talib.NATR(high, low, close, timeperiod=self.timelag)
        bb_u, bb_m, bb_l = talib.BBANDS(close, timeperiod=self.timelag, nbdevup=2, nbdevdn=2, matype=0)
        bb_w = np.divide(
            bb_u - bb_l, bb_m,
            out=np.full_like(bb_m, np.nan, dtype=float),
            where=(bb_m != 0) & ~np.isnan(bb_m),
        )
        obv = talib.OBV(close, vol)
        mfi = talib.MFI(high, low, close, vol, timeperiod=self.timelag)
        ad  = talib.AD(high, low, close, vol)
        adosc = talib.ADOSC(high, low, close, vol, fastperiod=3, slowperiod=10)
        k_period = max(self.timelag, 5); d_period = 3
        stoch_k, stoch_d = talib.STOCH(
            high, low, close,
            fastk_period=k_period,
            slowk_period=d_period, slowk_matype=0,
            slowd_period=d_period, slowd_matype=0,
        )

        feat = pd.DataFrame(
            {
                "EMA": ema, "MINUSDM": minusdm, "PLUSDM": plusdm,
                "CLOSE": df["close"], "CLOSEL1": closel1, "CLOSEL2": closel2,
                "PATT_3OUT": patt_3out, "PATT_CMB": patt_cmb,
                "RSI": rsi, "MACD": macd, "MACD_SIGNAL": macd_sig, "MACD_HIST": macd_hist,
                "ADX": adx, "ATR": atr, "NATR": natr,
                "BB_UPPER": bb_u, "BB_MIDDLE": bb_m, "BB_LOWER": bb_l, "BB_WIDTH": bb_w,
                "OBV": obv, "MFI": mfi, "AD": ad, "ADOSC": adosc,
                "STOCH_K": stoch_k, "STOCH_D": stoch_d,
            },
            index=df.index,
        )

        # ===== merge imported features on the same time grid =====
        # Convert ms index -> UTC datetime for joining
        feat_dt = feat.copy()
        feat_dt.index = pd.to_datetime(feat_dt.index, unit="ms", utc=True)

        interval_str = BINANCE_TO_STRIV[self.interval]
        start_utc = feat_dt.index[0]
        end_utc   = feat_dt.index[-1]

        if self.include_fng:
            fng = get_fng_features(
                start=start_utc.isoformat(),
                end=end_utc.isoformat(),
                interval=interval_str,
                lookback_days=self.fng_lookback_days,
            )
            # keep only the leakage-safe columns
            use_cols = [c for c in ["fng_ordinal_smooth","fng_ordinal_smooth_int"] if c in fng.columns]
            feat_dt = feat_dt.join(fng[use_cols], how="left")
            for c in use_cols:
                if c not in self.predictor_cols:
                    self.predictor_cols.append(c)

        if self.include_onchain:
            onch = get_btc_onchain_smoothed(
                start=start_utc.isoformat(),
                end=end_utc.isoformat(),
                interval=interval_str,
                lookback_days=self.onchain_lookback_days,
                include_price=False,
            )
            # only *_smooth columns (avoid raw/price)
            oc_cols = [c for c in onch.columns if c.endswith("_smooth")]
            feat_dt = feat_dt.join(onch[oc_cols], how="left")
            for c in oc_cols:
                if c not in self.predictor_cols:
                    self.predictor_cols.append(c)

        # Restore ms index
        feat_dt.index = (feat_dt.index.view("int64") // 1_000_000)
        feat_dt.index.name = "open_time"

        # Targets
        up_down = (df["close"].shift(-1) > df["close"]).astype("float64")
        ret_next = np.log(df["close"].shift(-1)) - np.log(df["close"])
        feat_dt["UP_DOWN"] = up_down
        feat_dt["RET_NEXT"] = ret_next

        # Clean
        feat_dt.replace([np.inf, -np.inf], np.nan, inplace=True)
        feat_dt.dropna(inplace=True)
        feat_dt = feat_dt.astype({"UP_DOWN": "int64"})

        self.df_features = feat_dt

    # -------- live update + dataset accessors --------
    def update_with_latest(self, limit: int = 3) -> None:
        recent = self.client.get_klines(symbol=self.symbol, interval=self.interval, limit=limit)
        df_recent = _klines_to_df(recent)

        if self.df_ohlcv.empty:
            self.df_ohlcv = df_recent
        else:
            last_known_open = self.df_ohlcv.index[-1]
            to_append = df_recent[df_recent.index > last_known_open]
            if not to_append.empty:
                self.df_ohlcv = pd.concat([self.df_ohlcv, to_append]).sort_index()
                self.df_ohlcv = self.df_ohlcv[~self.df_ohlcv.index.duplicated(keep="last")]

        self._recompute_features()

    @property
    def last_closed_open_time_ms(self) -> int:
        if self.df_features.empty:
            raise RuntimeError("No features yet.")
        return int(self.df_features.index[-1])

    @property
    def next_bar_close_time_ms(self) -> int:
        return self.last_closed_open_time_ms + self.interval_ms

    @property
    def dataset(self) -> Tuple[pd.DataFrame, pd.Series]:
        if self.df_features.empty:
            raise RuntimeError("No features yet.")
        X = self.df_features[self.predictor_cols].copy()
        y = self.df_features["UP_DOWN"].copy()
        return X, y

    @property
    def dataset_dual(self):
        if self.df_features.empty:
            raise RuntimeError("No features yet.")
        X = self.df_features[self.predictor_cols].copy()
        y_cls = self.df_features["UP_DOWN"].astype(int).copy()
        y_reg = self.df_features["RET_NEXT"].copy()
        return X, y_cls, y_reg

    def export_features_csv(self, path: str) -> None:
        if self.df_features.empty:
            raise RuntimeError("No features to export yet.")
        self.df_features.to_csv(path, index=True)

    def export_ohlcv_csv(self, path: str) -> None:
        if self.df_ohlcv.empty:
            raise RuntimeError("No OHLCV data to export yet.")
        self.df_ohlcv.to_csv(path, index=True)