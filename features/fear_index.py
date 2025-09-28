import requests
import pandas as pd

def fetch_fear_and_greed_history(limit=0, fmt="json", date_format=""):
    """
    Fetches full historical Fear & Greed Index data from Alternative.me API.
    """
    url = "https://api.alternative.me/fng/"
    params = {
        "limit": limit,           # 0 = all data
        "format": fmt,            # "json" or "csv"
        "date_format": date_format  # "", "us", "cn", "kr", or "world"
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()

    if fmt == "json":
        data = resp.json()
        records = data.get("data", [])
        df = pd.DataFrame(records)
        # Convert types
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
        df["fng"] = pd.to_numeric(df["value"], errors="coerce")
        df["fng_label"] = df["value_classification"]
        df = df[["timestamp", "fng", "fng_label"]].sort_values("timestamp").reset_index(drop=True)
    else:
        # For CSV output, let pandas parse directly
        df = pd.read_csv(pd.compat.StringIO(resp.text), parse_dates=["timestamp"])
    return df

# Fetch the full history
# df_fng = fetch_fear_and_greed_history(limit=0, fmt="json")
# df_fng
