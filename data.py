"""株価データの取得とセクター単位への集計（Streamlit版・HTML版で共通）

【株価の扱い】
  Yahoo Finance の Close は「株式分割・併合は調整済み、配当は未調整」の終値です。
  （Adj Close は配当も調整済み。）テクニカル指標・RS・騰落率はすべて Close で計算します。
  auto_adjust=False で取得しているので、Adj Close も列として手元に残ります。
【データ品質】
  取得失敗・欠損・更新が止まっている銘柄は、推測値で埋めず除外し、レポートに一覧を出します。
"""
import time

import pandas as pd
import yfinance as yf

# 直近日がこの日数より古い銘柄は「更新停止（上場廃止など）」として分析から外す
STALE_DAYS = 7


ISSUES = {}  # ティッカー → データ不良のメモ（レポートに表示する）
SPLIT_RATIOS = (2, 3, 4, 5, 10, 20, 100)


def _near(r, k, tol=0.04):
    return abs(r * k - 1) <= tol or abs(r / k - 1) <= tol


def sanitize(df: pd.DataFrame, max_len: int = 10):
    """Yahoo側の分割調整の一時的な不整合を検出して取り除く。

    終値が 1/k（または k倍）に一気に変わり、max_len営業日以内に k倍（1/k）で元に戻る場合は、
    その区間を「データ不良」とみなして行ごと削除する（休場日と同じ扱い。値の補正・補間はしない）。
    元に戻らない変化（本物の分割・併合のうち調整が反映されていない可能性）は触らず、issues に記録する。
    戻り値: (整えたdf, 問題のメモlist)
    """
    c = df["Close"]
    r = (c / c.shift(1)).to_numpy()
    idx = df.index
    bad, issues, i = set(), [], 1
    while i < len(r):
        k = next((k for k in SPLIT_RATIOS if r[i] == r[i] and (_near(r[i], k) and abs(r[i] - 1) > 0.4)), None)
        if k is None:
            i += 1
            continue
        down = r[i] < 1
        j = next((j for j in range(i + 1, min(i + 1 + max_len, len(r)))
                  if r[j] == r[j] and (abs(r[j] * r[i] - 1) <= 0.08)), None)
        if j is not None:
            bad.update(range(i, j))
            issues.append(f"{idx[i].strftime('%Y-%m-%d')}〜{idx[j - 1].strftime('%Y-%m-%d')}：価格が1/{k}（またはk倍）になり後日戻る不整合のため除外")
            i = j + 1
        else:
            issues.append(f"{idx[i].strftime('%Y-%m-%d')}：終値が{'1/' if down else ''}{k}{'' if down else '倍'}規模で急変（分割・併合の調整漏れの可能性。値は未修正）")
            i += 1
    if bad:
        df = df.iloc[[n for n in range(len(df)) if n not in bad]]
    return df, issues


def _download(tickers, period):
    return yf.download(tickers, period=period, group_by="ticker",
                       auto_adjust=False, progress=False, threads=True)


def load_prices_ex(tickers, period: str = "2y", retries: int = 3):
    """(価格dict, 取得失敗ティッカーのlist) を返す。失敗分は間隔を空けて再取得する"""
    tickers = list(dict.fromkeys(tickers))
    out, todo = {}, tickers
    for attempt in range(retries):
        if not todo:
            break
        try:
            raw = _download(todo, period)
        except Exception as e:  # ネットワーク・API エラー
            print(f"  取得エラー（{attempt + 1}/{retries}回目）: {e}")
            time.sleep(3 * (attempt + 1))
            continue
        still = []
        for t in todo:
            try:
                df = raw[t] if len(todo) > 1 else raw
                df = df.dropna(how="all")
            except KeyError:
                still.append(t)
                continue
            if df.empty:
                still.append(t)
            else:
                df, iss = sanitize(df)
                if iss:
                    ISSUES[t] = iss
                out[t] = df
        todo = still
        if todo:
            time.sleep(2)
    return out, todo


def load_prices(tickers, period: str = "2y") -> dict:
    """従来互換（Streamlit版が使う）"""
    return load_prices_ex(tickers, period)[0]


def drop_stale(prices: dict, ref_ticker: str, max_days: int = STALE_DAYS):
    """基準ティッカーの最新日より max_days 以上古い銘柄を除外。(残り, 除外list) を返す"""
    if ref_ticker not in prices:
        return prices, []
    ref = prices[ref_ticker].index.max()
    keep, stale = {}, []
    for t, df in prices.items():
        if (ref - df.index.max()).days > max_days:
            stale.append(t)
        else:
            keep[t] = df
    return keep, stale


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
