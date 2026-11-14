"""株価データの取得とセクター単位への集計（Streamlit版・HTML版で共通）"""
import pandas as pd
import yfinance as yf


def load_prices(tickers, period: str = "2y") -> dict:
    tickers = list(tickers)
    raw = yf.download(tickers, period=period, group_by="ticker",
                      auto_adjust=False, progress=False, threads=True)
    out = {}
    for t in tickers:
        try:
            df = raw[t].dropna(how="all")
        except KeyError:
            continue
        if not df.empty:
            out[t] = df
    return out


def tickers_for(sectors: dict) -> list:
    return sorted({cfg["price_ticker"] for cfg in sectors.values()}
                  | {m for cfg in sectors.values() for m in cfg.get("members", [])})


def build_sector_data(sectors: dict, prices: dict, volume_mode: str = "構成銘柄の売買代金合計") -> dict:
    """セクターごとに (OHLC, 売買代金) を作る"""
    data = {}
    for name, cfg in sectors.items():
        etf = prices.get(cfg["price_ticker"])
        if etf is None or len(etf) < 80:
            continue
        if cfg.get("volume") == "shares":
            # 指数: 出来高（株数）をそのまま使う。0 は「配信なし」とみなして欠損扱い
            turnover = etf["Volume"].where(etf["Volume"] > 0)
        elif cfg.get("volume") == "turnover" or not cfg.get("members"):
            turnover = (etf["Close"] * etf["Volume"]).where(etf["Volume"] > 0)
        elif volume_mode == "構成銘柄の売買代金合計":
            parts = [prices[m]["Close"] * prices[m]["Volume"] for m in cfg["members"] if m in prices]
            turnover = pd.concat(parts, axis=1).sum(axis=1, min_count=1) if parts else etf["Close"] * etf["Volume"]
        else:
            turnover = (etf["Close"] * etf["Volume"]).where(etf["Volume"] > 0)
        data[name] = (etf, turnover)
    return data
