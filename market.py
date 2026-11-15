"""市場分析：市場スコア（0〜100）と相場環境の4分類。日本・米国・主要国を切り替えて使う。

配点（初期仮説）  トレンド35 / 騰落銘柄数・市場の広がり25 / ボラティリティ20 / モメンタム20
すべて「未来のデータを使わない」日次の時系列として計算するので、履歴もそのまま出せる。

市場ごとのデータの違い（画面にも表示する）
  * 日本：TOPIX は連動ETF（1306）で代用。日経VIは取得できず「データなし」→実現ボラで代用。
  * 米国：VIX（^VIX）を使える。
  * 広がり（騰落銘柄数・25日線上比率）は、代表銘柄を用意した市場（日本・米国）のみ。
    それ以外の市場は「データなし」とし、残り3要素の配点で0〜100に再計算する（精度は下がる）。
  * 騰落銘柄数などは全上場銘柄ではなく代表銘柄だけの参考値。
"""
import numpy as np
import pandas as pd

from indicators import macd
from scoring import clip01, num, prank, weighted_score

WEIGHTS = {"trend": 35, "breadth": 25, "vol": 20, "momentum": 20}
LABELS = {"trend": "トレンド", "breadth": "騰落銘柄数・市場の広がり",
          "vol": "ボラティリティ", "momentum": "市場全体のモメンタム"}
REGIMES = ["強い上昇相場", "緩やかな上昇相場", "下降相場", "急落・高ボラティリティ相場"]
MIN_COVERAGE = 0.7   # 4要素のうち使えた配点がこの割合以上の日だけ採点（広がり無し=0.75でOK）


def breadth_frames(stocks: dict, cal: pd.Index) -> dict:
    """銘柄ごとに「今日上昇/下落」「5日前比プラス」「25日線の上」を 1/0/NaN で持つ表を作る"""
    up1, dn1, up5, ab25, turn = {}, {}, {}, {}, {}
    for t, df in stocks.items():
        c = df["Close"]
        d = c.diff()
        r5 = c.pct_change(5)
        ma = c.rolling(25).mean()
        up1[t] = (d > 0).astype(float).where(d.notna())
        dn1[t] = (d < 0).astype(float).where(d.notna())
        up5[t] = (r5 > 0).astype(float).where(r5.notna())
        ab25[t] = (c > ma).astype(float).where(ma.notna())
        turn[t] = (c * df["Volume"]).where(df["Volume"] > 0)
    f = lambda m: pd.DataFrame(m).reindex(cal)
    return {"up1": f(up1), "dn1": f(dn1), "up5": f(up5), "ab25": f(ab25), "turn": f(turn)}


def _index_block(df: pd.DataFrame) -> dict:
    c = df["Close"].dropna()
    ma25, ma75, ma200 = (c.rolling(n).mean() for n in (25, 75, 200))
    _, _, hist = macd(c)
    binary = pd.concat([(c > ma25), (c > ma75), (c > ma200), (ma25 > ma75), (ma25 > ma25.shift(5))],
                       axis=1).astype(float).mean(axis=1).where(ma200.notna())
    dev_rank = prank(c / ma75 - 1)
    trend = 0.5 * binary + 0.5 * dev_rank
    mom = pd.concat([prank(c.pct_change(20)), prank(c.pct_change(60)), prank(hist / c)], axis=1).mean(axis=1, skipna=False)
    return {"c": c, "ma25": ma25, "ma75": ma75, "ma200": ma200, "hist": hist, "trend": trend, "mom": mom}


def build_market(prices: dict, cfg: dict, universe_tickers=None, history_days: int = 60) -> dict:
    """1市場分の市場スコア等を作る。
    cfg: {"label", "indices":[(名前, ticker),...]（先頭が基準指数）, "vi":(名前, ticker)|None}"""
    idx = [(nm, t) for nm, t in cfg["indices"] if t in prices]
    if not idx:
        return {"available": False, "label": cfg["label"], "error": f"{cfg['label']}の指数データを取得できませんでした"}
    pname, ptick = idx[0]
    blocks = {t: _index_block(prices[t]) for _, t in idx}
    n = blocks[ptick]
    cal = n["c"].index
    c = n["c"]

    # ---- トレンド（35）：位置関係5項目 ＋ 75日線乖離の順位。指数の平均 ----
    trend = pd.concat([b["trend"].reindex(cal) for b in blocks.values()], axis=1).mean(axis=1)

    # ---- 広がり（25）：代表銘柄がある市場のみ ----
    stocks = {t: prices[t] for t in (universe_tickers or []) if t in prices}
    if stocks:
        bf = breadth_frames(stocks, cal)
        nvalid = bf["ab25"].notna().sum(axis=1)
        adv, dec = bf["up1"].sum(axis=1, min_count=1), bf["dn1"].sum(axis=1, min_count=1)
        adr25 = adv.rolling(25).sum() / dec.rolling(25).sum().replace(0, np.nan) * 100
        above25 = bf["ab25"].sum(axis=1, min_count=1) / nvalid.replace(0, np.nan)
        breadth = 0.5 * above25 + 0.5 * prank(adr25)
        turn_sum = bf["turn"].sum(axis=1, min_count=1)
    else:
        nvalid = pd.Series(0, index=cal)
        adv = dec = adr25 = above25 = turn_sum = pd.Series(np.nan, index=cal)
        breadth = pd.Series(np.nan, index=cal)

    # ---- ボラティリティ（20）：VIXがあれば使う。無ければ実現ボラの順位＋高値からの下落 ----
    rv20 = np.log(c).diff().rolling(20).std() * np.sqrt(252) * 100
    rv_rank = prank(rv20)
    dd60 = c / c.rolling(60).max() - 1
    dd_score = clip01(1 + dd60 / 0.10)   # 高値から-10%で0点
    vi_name, vi_t = cfg.get("vi") or (None, None)
    vi_ok = vi_t is not None and vi_t in prices
    if vi_ok:
        vi = prices[vi_t]["Close"].dropna().reindex(cal).ffill()
        vol = 0.5 * (1 - prank(vi)) + 0.25 * (1 - rv_rank) + 0.25 * dd_score
    else:
        vi = pd.Series(np.nan, index=cal)
        vol = 0.6 * (1 - rv_rank) + 0.4 * dd_score

    # ---- モメンタム（20） ----
    momentum = pd.concat([b["mom"].reindex(cal) for b in blocks.values()], axis=1).mean(axis=1, skipna=False)

    parts = {"trend": trend, "breadth": breadth, "vol": vol, "momentum": momentum}
    score, coverage = weighted_score(parts, WEIGHTS)
    score = score.where(coverage >= MIN_COVERAGE)

    # ---- 相場環境の分類（判定順に上から。基準指数で判定） ----
    ret5 = c.pct_change(5)
    crash = ((ret5 <= -0.04) | (dd60 <= -0.08)) & (rv_rank >= 0.7)
    strong = (score >= 65) & (c > n["ma25"]) & (n["ma25"] > n["ma75"])
    gentle = c > n["ma75"]
    regime = pd.Series(np.select([crash, strong, gentle], [3, 0, 1], default=2), index=cal).where(score.notna())
    if not score.notna().any():
        return {"available": False, "label": cfg["label"], "error": "履歴が足りずスコアを計算できません"}

    last = score.dropna().index[-1]
    i = cal.get_loc(last)
    pick = lambda s, nd=2: num(s.iloc[i], nd)

    def idx_row(label, ticker):
        b = blocks[ticker]
        cc = b["c"].reindex(cal).ffill()
        g = lambda s: s.reindex(cal).iloc[i]
        return {"name": label, "ticker": ticker, "close": num(cc.iloc[i]),
                "chg1": num((cc.iloc[i] / cc.iloc[i - 1] - 1) * 100),
                "ma25": num(g(b["ma25"])), "ma75": num(g(b["ma75"])), "ma200": num(g(b["ma200"])),
                "dev25": num((cc.iloc[i] / g(b["ma25"]) - 1) * 100), "dev75": num((cc.iloc[i] / g(b["ma75"]) - 1) * 100),
                "dev200": num((cc.iloc[i] / g(b["ma200"]) - 1) * 100)}
    indices = [idx_row(nm, t) for nm, t in idx]

    pv = prices[ptick]["Volume"].where(prices[ptick]["Volume"] > 0).reindex(cal)
    vr = pv / pv.shift(1).rolling(20, min_periods=10).mean()
    tr = turn_sum / turn_sum.shift(1).rolling(20, min_periods=10).mean()
    has_b = bool(stocks) and bool(nvalid.iloc[i] > 0)

    stats = {
        "adv": int(adv.iloc[i]) if pd.notna(adv.iloc[i]) else None,
        "dec": int(dec.iloc[i]) if pd.notna(dec.iloc[i]) else None,
        "n": int(nvalid.iloc[i]),
        "adr25": pick(adr25, 1),
        "above25": num(above25.iloc[i] * 100, 1),
        "rv20": pick(rv20, 1), "rvRank": num(rv_rank.iloc[i] * 100, 0),
        "dd60": num(dd60.iloc[i] * 100, 1), "ret5": num(ret5.iloc[i] * 100, 1),
        "roc20": num(c.pct_change(20).iloc[i] * 100, 1), "roc60": num(c.pct_change(60).iloc[i] * 100, 1),
        "macdHistPct": num(n["hist"].reindex(cal).iloc[i] / c.iloc[i] * 100, 2),
        "primaryVolRatio": pick(vr), "turnRatio": pick(tr),
        "vi": pick(vi, 1), "viRank": num(prank(vi).iloc[i] * 100, 0) if vi_ok else None,
    }

    names = "・".join(nm for nm, _ in idx)
    n_idx = len(idx)
    basis = {
        "trend": f"{names}：①終値>25日線 ②>75日線 ③>200日線 ④25日線>75日線 ⑤25日線が5日前より上、の5項目の平均（50%）"
                 f"＋75日線乖離の過去250日順位（50%）" + ("を指数間で平均" if n_idx > 1 else ""),
        "breadth": (f"代表{stats['n']}社の25日線上比率（50%）＋25日騰落レシオの過去250日順位（50%）。全上場銘柄ではありません"
                    if has_b else "この市場は代表銘柄を用意していないため「データなし」。残り3要素の配点で0〜100に再計算しています（精度は下がります）"),
        "vol": (f"{vi_name}の過去250日順位を反転（50%）＋{pname}の実現ボラ20日の順位を反転（25%）＋60日高値からの下落率（-10%で0点）（25%）"
                if vi_ok else f"{pname}の実現ボラ20日（年率）の過去250日順位を反転（60%）＋60日高値からの下落率（-10%で0点）（40%）。"
                              "ボラティリティ指数はデータなし"),
        "momentum": "20日騰落率・60日騰落率・MACDヒストグラム/株価の、各過去250日順位の平均" + ("。指数間で平均" if n_idx > 1 else ""),
    }
    comps = []
    for k, w in WEIGHTS.items():
        v = parts[k].iloc[i]
        comps.append({"key": k, "label": LABELS[k], "max": w, "ratio": num(v, 3),
                      "points": num(v * w, 1) if pd.notna(v) else None, "basis": basis[k]})

    h = score.dropna().tail(history_days)
    history = {"dates": [d.strftime("%m/%d") for d in h.index],
               "score": [num(v, 0) for v in h],
               "regime": [int(regime.get(d)) for d in h.index]}

    return {
        "available": True, "label": cfg["label"], "primaryName": pname, "viName": vi_name if vi_ok else None,
        "date": last.strftime("%Y-%m-%d"),
        "score": num(score.iloc[i], 0), "regime": REGIMES[int(regime.iloc[i])], "regimeIdx": int(regime.iloc[i]),
        "components": comps, "indices": indices, "stats": stats, "history": history, "hasBreadth": has_b,
        "regimeRule": ("判定順：①急落・高ボラ＝（5日騰落≦-4% または 60日高値から-8%以下）かつ 実現ボラ順位≧70% "
                       f"／②強い上昇＝スコア≧65 かつ 終値>25日線>75日線 ／③緩やかな上昇＝終値>75日線 ／④それ以外＝下降（基準指数：{pname}）"),
        "notes": cfg.get("notes", []) + ([] if vi_ok else ["ボラティリティ指数：データなし（実現ボラで代用）"])
                 + ([] if has_b else ["騰落銘柄数・騰落レシオ・25日線上比率：データなし"]),
    }


def build_world(prices: dict, long_prices: dict, world: list, ma_targets: dict) -> dict:
    """世界の主要指数の表と、期間切替チャート用データ（全市場で共通）。
    world: [(名前, ticker, 地域)]。ma_targets: 市場キー → 基準指数ticker（その移動平均線をチャートに出す）"""
    lp = long_prices or {}
    close_of = lambda t: (lp[t] if t in lp else prices[t])["Close"].dropna()

    rows, series = [], []
    for name, t, region in world:
        if t not in prices or len(prices[t]["Close"].dropna()) < 30:
            rows.append({"name": name, "ticker": t, "available": False})
            continue
        oc = prices[t]["Close"].dropna()
        m25, m75, m200 = (oc.rolling(k).mean() for k in (25, 75, 200))
        pr = lambda k: num((oc.iloc[-1] / oc.iloc[-1 - k] - 1) * 100, 1) if len(oc) > k else None
        dv = lambda m: num((oc.iloc[-1] / m.iloc[-1] - 1) * 100, 1) if pd.notna(m.iloc[-1]) else None
        rows.append({"name": name, "ticker": t, "region": region, "available": True, "date": oc.index[-1].strftime("%m/%d"),
                     "close": num(oc.iloc[-1]), "chg1": pr(1), "chg5": pr(5), "chg20": pr(20), "chg60": pr(60),
                     "dev25": dv(m25), "dev75": dv(m75), "dev200": dv(m200)})
        series.append((t, name, region, close_of(t)))
    if not series:
        return {"rows": rows, "chart": {"periods": []}}

    # 期間切替：年初来・1年は日足、3年・5年・10年は週足（金曜終値）
    last_day = max(s[3].index[-1] for s in series)
    periods = {"ytd": ("年初来", pd.Timestamp(year=last_day.year, month=1, day=1), False),
               "1y": ("1年", last_day - pd.DateOffset(years=1), False),
               "3y": ("3年（週足）", last_day - pd.DateOffset(years=3), True),
               "5y": ("5年（週足）", last_day - pd.DateOffset(years=5), True),
               "10y": ("10年（週足）", last_day - pd.DateOffset(years=10), True)}
    mas = {mk: {k: close_of(t).rolling(w).mean() for k, w in (("ma25", 25), ("ma75", 75), ("ma200", 200))}
           for mk, t in ma_targets.items() if t in prices}
    out = []
    for pk, (plabel, start, weekly) in periods.items():
        cut = lambda s: (s[s.index >= start].resample("W-FRI").last().dropna() if weekly else s[s.index >= start])
        cs = [(key, name, region, cut(s)) for key, name, region, s in series]
        grid = pd.DatetimeIndex(sorted(set().union(*[set(x[3].index) for x in cs])))
        if len(grid) < 5:
            continue
        ma = {}
        for mk, d in mas.items():
            ma[mk] = {}
            for k, m in d.items():
                ms = m[m.index >= start]
                ms = ms.resample("W-FRI").last() if weekly else ms
                ma[mk][k] = [num(v) for v in ms.reindex(grid)]
        out.append({"key": pk, "label": plabel, "weekly": weekly, "dates": [d.strftime("%Y-%m-%d") for d in grid],
                    "series": [{"key": key, "name": name, "group": region, "c": [num(v) for v in s.reindex(grid)]}
                               for key, name, region, s in cs],
                    "ma": ma})
    return {"rows": rows, "chart": {"periods": out}}
