"""バックテスト（検証）：スコアが高いほど将来リターンが良かったかを、過去データで確かめる。

★ 検証するのは「画面に出しているスコアそのもの」。stocks.py と同じ計算（features / components / W_*）を
  過去の各日に当てはめる。配点（W_STOCK など）を変えれば、検証結果もそれに従って変わる。

★ 先読みを避けるルール
  - 各日のスコアは、その日の終値までのデータだけで計算されている（時系列計算。未来は使わない）。
  - 約定は「シグナルが出た翌営業日の終値」。そこから 5/10/20 営業日後の終値で決済する。
      将来リターン = 終値[t+1+N] ÷ 終値[t+1] − 1
  - 累積リターンのグラフも、保有銘柄は2営業日前のスコアで決め、翌日終値約定として計算する。
  - 順位系の要素（相対強度・セクター内順位）は、同じ日付の銘柄間だけで比べる。

★ 売買コスト：片道 手数料0.10% + スリッページ0.05% = 往復0.30% を差し引く（COST_*で変更可）。

★ 限界（画面にも表示する）
  - 銘柄は「現在の代表銘柄」。過去時点の構成銘柄は取得できないため、生存者バイアスがある
    （上場廃止・低迷で外れた銘柄が含まれず、成績は実際より良く出やすい）。
  - 総合スコアの「市場」「セクター」部分は過去日の再現をしていないため、検証していない
    （個別銘柄スコア・エントリー・リスク・3分類のみ）。
  - 毎日のシグナルは期間が重なるため、観測数は独立ではない（統計的な確からしさは目安）。
"""
import numpy as np
import pandas as pd

from stocks import (L_ENTRY, L_RISK, L_STOCK, MIN_COV, TYPES, W_ENTRY, W_RISK, W_STOCK,
                    components, features, sel_scores, heat_levels, HEAT_ITEMS, CANDLE_KIND)
from indicators import candle_patterns
from scoring import weighted_score

HORIZONS = (5, 10, 20)
LAG = 1                       # 約定＝シグナルの翌営業日終値
COST_FEE = 0.0010             # 片道の手数料
COST_SLIP = 0.0005            # 片道のスリッページ
COST_SIDE = COST_FEE + COST_SLIP
COST_RT = 2 * COST_SIDE       # 往復
MIN_ROWS = 260                # これ未満の履歴しかない銘柄は検証から除く
YEAR_DAYS = 245

STOCK_EDGES = [(0, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 101)]
ENTRY_EDGES = [(0, 30), (30, 50), (50, 70), (70, 101)]
RISK_EDGES = [(0, 40), (40, 50), (50, 60), (60, 70), (70, 101)]
SEL_EDGES = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 101)]


def _r(v, nd=2):
    if v is None:
        return None
    v = float(v)
    return None if (np.isnan(v) or np.isinf(v)) else round(v, nd)


# ---------- 過去の各日のスコア（銘柄×日付のパネル） ----------

def build_panels(prices: dict, universe: dict, topix: pd.Series):
    cal = topix.dropna().index
    tp = topix.reindex(cal)
    sector_of = {t: s for s, ts in universe.items() for t in ts}
    feats, comps, used = {}, {}, []
    for t in sector_of:
        df = prices.get(t)
        if df is None or len(df.dropna(subset=["Close"])) < MIN_ROWS:
            continue
        f = features(df)
        feats[t], comps[t] = f, components(f)
        used.append(t)

    blend = {}
    for t in used:
        c = feats[t]["c"].reindex(cal)
        blend[t] = sum(w * (c.pct_change(k) - tp.pct_change(k)) for k, w in ((5, 0.25), (20, 0.40), (60, 0.35)))
    B = pd.DataFrame(blend)
    rs_rank = B.rank(axis=1, pct=True)
    sec_rank = pd.DataFrame(np.nan, index=B.index, columns=B.columns)
    by_sec = {}
    for t in used:
        by_sec.setdefault(sector_of[t], []).append(t)
    for s, ts in by_sec.items():
        if len(ts) >= 4:
            sec_rank[ts] = B[ts].rank(axis=1, pct=True)

    names = ["score", "entry", "risk", "trend", "momentum", "volume", "rs", "secrel",
             "pullback", "macd", "breakout", "rsi", "atr", "drawdown", "stability", "t1", "t2", "t3", "C"] \
        + ["h_" + k for k, _, _ in HEAT_ITEMS] + ["c_" + k for k in CANDLE_KIND]
    P = {k: {} for k in names}
    for t in used:
        f, cp = feats[t], comps[t]
        c = f["c"]
        idx = c.index
        parts = dict(cp["stock"])
        parts["rs"] = rs_rank[t].reindex(idx)
        parts["secrel"] = sec_rank[t].reindex(idx)
        s_score, s_cov = weighted_score(parts, W_STOCK)
        s_score = s_score.where(s_cov >= MIN_COV)
        e_score, _ = weighted_score(cp["entry"], W_ENTRY)
        risk_parts = dict(cp["risk"])
        risk_parts["concentration"] = pd.Series(np.nan, index=idx)
        r_score, _ = weighted_score(risk_parts, W_RISK)

        # 3分類のルール（stocks.py の analyze と同じ。tests で最新日の一致を確認する）
        ex = cp["extra"]
        ma25, ma75, hist, rsi = f["ma25"], f["ma75"], f["hist"], f["rsi"]
        vol_brk3 = (ex["brk"] & (ex["vd"] >= 1.3)).rolling(3).max().fillna(0) > 0
        t1 = ((c > ma25) & (ma25 > ma75) & (ma75 > ma75.shift(5)) & (f["adx"] >= 20) & (f["pdi"] > f["mdi"])
              & (ex["dev25"] <= 0.10) & (s_score >= 55))
        rs_r = parts["rs"]
        t2 = ((vol_brk3 | (ex["gc3"] & (c > ma25))) & (c > ma75) & (rs_r.isna() | (rs_r >= 0.4)) & (ex["dev25"] <= 0.12))
        t3 = ((ex["dd60"] <= -0.10) & (rsi.rolling(10).min() <= 35) & (rsi >= 40) & (rsi > rsi.shift(3))
              & (hist > hist.shift(1)) & (c > f["ma5"]))

        P["score"][t], P["entry"][t], P["risk"][t] = s_score, e_score, r_score
        for k in ("trend", "momentum", "volume", "rs", "secrel"):
            P[k][t] = parts[k]
        for k in ("pullback", "macd", "breakout", "rsi"):
            P[k][t] = cp["entry"][k]
        for k in ("atr", "drawdown", "stability"):
            P[k][t] = cp["risk"][k]
        P["t1"][t], P["t2"][t], P["t3"][t] = t1, t2, t3
        for k, v in heat_levels(f, ex).items():
            P["h_" + k][t] = v
        pats = candle_patterns(f["o"], f["h"], f["l"], f["c"])
        for k in CANDLE_KIND:
            P["c_" + k][t] = pats[k]
        P["C"][t] = c

    out = {}
    for k, d in P.items():
        df = pd.DataFrame(d).reindex(cal)
        out[k] = df.fillna(False).astype(bool) if k in ("t1", "t2", "t3") or k.startswith("c_") else df
    lev = [out["h_" + k] for k, _, _ in HEAT_ITEMS]
    out["os"] = sum((x == -1).astype(int) for x in lev)      # 売られすぎが点灯した指標の数
    out["ob"] = sum((x == 1).astype(int) for x in lev)       # 買われすぎが点灯した指標の数
    sel, selc, _ = sel_scores(out["C"])          # 選別スコア（検証版）と、その3要素
    out["sel"] = sel
    out["sel_mom"], out["sel_rev20"], out["sel_dev75"] = selc["mom"], selc["rev20"], selc["dev75"]
    return out, cal


# ---------- 将来リターンと統計 ----------

def _fwd(C: pd.DataFrame, n: int):
    return C.shift(-(LAG + n)) / C.shift(-LAG) - 1


def _stat_h(mask, F, uni, bms):
    mk = mask & np.isfinite(F)
    k = int(mk.sum())
    d = {"n": k}
    if k == 0:
        return d
    g = F[mk]
    x = g - COST_RT
    win, loss = x[x > 0], x[x <= 0]
    uni_b = np.broadcast_to(uni[:, None], F.shape)[mk]
    d.update({
        "gross": _r(g.mean() * 100), "mean": _r(x.mean() * 100), "median": _r(np.median(x) * 100),
        "win": _r((x > 0).mean() * 100, 1), "avgWin": _r(win.mean() * 100) if win.size else None,
        "avgLoss": _r(loss.mean() * 100) if loss.size else None,
        "rr": _r(win.mean() / abs(loss.mean())) if win.size and loss.size and loss.mean() != 0 else None,
        "p5": _r(np.percentile(x, 5) * 100), "sd": _r(x.std() * 100),
        "exUni": _r((x - uni_b).mean() * 100),
    })
    for nm, arr in bms.items():
        b = np.broadcast_to(arr[:, None], F.shape)[mk]
        ok = np.isfinite(b)
        d["ex" + nm] = _r((x - b)[ok].mean() * 100) if ok.any() else None
    return d


def _curve_returns(mask: pd.DataFrame, R: pd.DataFrame, cost: float = COST_SIDE, with_turn: bool = False):
    """銘柄を毎日等金額で保有した場合の日次リターン（コスト差引後）。保有は2営業日前のシグナルで決める"""
    pos = mask.shift(LAG + 1, fill_value=False)
    valid = pos & R.notna()
    cnt = valid.sum(axis=1)
    w = valid.astype(float).div(cnt.where(cnt > 0), axis=0).fillna(0.0)
    gross = (w * R.fillna(0.0)).sum(axis=1)
    turn = w.diff().abs().sum(axis=1).fillna(0.0)
    net = gross - turn * cost
    return (net, turn) if with_turn else net


def _hold_rule(S: pd.DataFrame, enter: float, exit_: float) -> pd.DataFrame:
    """ヒステリシス：enter以上で買い、exit_未満（またはデータなし）で売り、その間は前日の状態を保つ（売買の頻度を抑える）"""
    a = S.to_numpy()
    st = np.zeros(a.shape, dtype=bool)
    prev = np.zeros(a.shape[1], dtype=bool)
    for i in range(a.shape[0]):
        x = a[i]
        buy = x >= enter
        sell = ~(x >= exit_)
        prev = np.where(buy, True, np.where(sell, False, prev))
        st[i] = prev
    return pd.DataFrame(st, index=S.index, columns=S.columns)


def _perf(r: pd.Series):
    r = r.dropna()
    if r.empty:
        return {}
    eq = (1 + r).cumprod()
    total = eq.iloc[-1] - 1
    years = len(r) / YEAR_DAYS
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else None
    vol = r.std() * np.sqrt(YEAR_DAYS)
    mdd = (eq / eq.cummax() - 1).min()
    return {"total": _r(total * 100, 1), "cagr": _r(cagr * 100, 1) if cagr is not None else None,
            "vol": _r(vol * 100, 1), "mdd": _r(mdd * 100, 1),
            "ret_dd": _r(cagr / abs(mdd), 2) if cagr is not None and mdd < 0 else None,
            "sharpe": _r(r.mean() * YEAR_DAYS / vol, 2) if vol > 0 else None}


def _ic_series(X: pd.DataFrame, F: pd.DataFrame):
    Xm, Fm = X.where(F.notna()), F.where(X.notna())
    cnt = Xm.notna().sum(axis=1)
    ic = Xm.rank(axis=1).corrwith(Fm.rank(axis=1), axis=1)
    return ic[cnt >= 30].dropna()


def _ic(X: pd.DataFrame, F: pd.DataFrame):
    ic = _ic_series(X, F)
    if len(ic) < 30:
        return None
    return {"mean": _r(ic.mean(), 3), "pos": _r((ic > 0).mean() * 100, 0), "n": int(len(ic)),
            "t": _r(ic.mean() / ic.std() * np.sqrt(len(ic)), 1) if ic.std() > 0 else None}


# ---------- 配点の見直し検証（前半／後半に分けて、配点を変えても結果が安定するかを見る） ----------

REVIEW_SETS = [
    ("現在の配点（トレンド30・モメンタム25・相対強度20・出来高15・セクター内10）", dict(trend=30, momentum=25, rs=20, volume=15, secrel=10)),
    ("均等配点（各20）", dict(trend=20, momentum=20, rs=20, volume=20, secrel=20)),
    ("トレンドだけ", dict(trend=1)),
    ("モメンタムだけ", dict(momentum=1)),
    ("相対強度だけ", dict(rs=1)),
    ("出来高だけ", dict(volume=1)),
    ("セクター内の強さだけ", dict(secrel=1)),
    ("エントリー：押し目＋RSI", dict(pullback=1, rsi=1)),
    ("エントリー：MACD＋上抜け", dict(macd=1, breakout=1)),
    ("リスク（安全度）の3要素", dict(atr=30, drawdown=25, stability=25)),
    ("【新】選別スコア：長期モメンタム＋20日の押し＋75日線の押し（各1/3）", dict(sel_mom=1, sel_rev20=1, sel_dev75=1)),
    ("【新】長期モメンタムだけ（12か月−1か月）", dict(sel_mom=1)),
    ("【新】20日の押しだけ", dict(sel_rev20=1)),
    ("【新】75日線の押しだけ", dict(sel_dev75=1)),
]


def _combo(P: dict, w: dict, need: float = 0.7):
    num = sum(v * P[k].fillna(0) for k, v in w.items())
    den = sum(v * P[k].notna() for k, v in w.items())
    return (num / den.where(den > 0)).where(den / sum(w.values()) >= need) * 100


def _spread(X: pd.DataFrame, F: pd.DataFrame, lo, hi):
    """その日の順位で、上位20%と下位20%の銘柄の将来リターンの差（%）。期間 [lo, hi) の平均"""
    r = X.rank(axis=1, pct=True)
    d = (F.where(r >= 0.8).mean(axis=1) - F.where(r <= 0.2).mean(axis=1)).dropna()
    d = d[(d.index >= lo) & (d.index < hi)]
    return _r(d.mean() * 100) if len(d) else None


def build_review(P: dict, F20: pd.DataFrame, first, last):
    idx = list(P["score"].index)
    i0 = idx.index(first)
    mid = idx[i0 + (len(idx) - i0) // 2]
    F = F20 - COST_RT
    rows = []
    for label, w in REVIEW_SETS:
        X = _combo(P, w)
        ic = _ic_series(X, F)
        ic = ic[ic.index >= first]
        rows.append({"label": label,
                     "ic1": _r(ic[ic.index < mid].mean(), 3), "ic2": _r(ic[ic.index >= mid].mean(), 3),
                     "sp1": _spread(X, F, first, mid), "sp2": _spread(X, F, mid, last + pd.Timedelta(days=1))})
    return {"split": mid.strftime("%Y-%m-%d"), "start": first.strftime("%Y-%m-%d"), "end": last.strftime("%Y-%m-%d"), "rows": rows}

def build_regime(P: dict, F20: pd.DataFrame, tp: pd.Series, first):
    """市場局面別（TOPIXが200日線の上／下）の、スコアと将来リターンの順位相関IC。局面は、その日までの価格だけで判定する"""
    up = (tp > tp.rolling(200).mean()).reindex(P["score"].index)
    valid = tp.rolling(200).mean().reindex(P["score"].index).notna()
    F = F20 - COST_RT
    rows = []
    for key, label in (("score", "現行の個別銘柄スコア"), ("sel", "選別スコア（検証版）")):
        ic = _ic_series(P[key], F)
        ic = ic[(ic.index >= first)]
        u = up.reindex(ic.index).fillna(False) & valid.reindex(ic.index).fillna(False)
        d = (~up.reindex(ic.index).fillna(False)) & valid.reindex(ic.index).fillna(False)
        rows.append({"label": label, "icUp": _r(ic[u].mean(), 3), "nUp": int(u.sum()), "icDown": _r(ic[d].mean(), 3), "nDown": int(d.sum())})
    return rows

def build_backtest(prices: dict, universe: dict, topix: pd.Series, n225: pd.Series | None):
    P, cal = build_panels(prices, universe, topix)
    S, C = P["score"], P["C"]
    if S.notna().sum(axis=1).max() < 30:
        raise ValueError("検証に使える銘柄が少なすぎます")
    first = S.index[(S.notna().sum(axis=1) >= 30).to_numpy().argmax()]
    FWD = {n: _fwd(C, n) for n in HORIZONS}
    UNI = {n: FWD[n].mean(axis=1).to_numpy() for n in HORIZONS}
    tp = topix.reindex(cal).ffill()
    bm_close = {"Topix": tp}
    if n225 is not None:
        bm_close["N225"] = n225.reindex(cal).ffill()
    BM = {n: {k: (v.shift(-(LAG + n)) / v.shift(-LAG) - 1).to_numpy() for k, v in bm_close.items()} for n in HORIZONS}
    R = C.pct_change()
    valid_dates = (S.index >= first)
    vd = pd.Series(valid_dates, index=S.index)

    def stats(mask: pd.DataFrame):
        mask = mask.where(vd, False, axis=0).astype(bool)
        m = mask.to_numpy()
        d = {"obs": int(m.sum())}
        for n in HORIZONS:
            d[f"h{n}"] = _stat_h(m, FWD[n].to_numpy(), UNI[n], {k: v for k, v in BM[n].items()})
        cr = _curve_returns(mask, R, cost=0.0).loc[first:]   # 最大下落率の目安：コスト抜き・毎日入れ替え
        d["curve"] = _perf(cr)
        return d

    def bands(score, edges):
        rows = []
        for lo, hi in edges:
            m = (score >= lo) & (score < hi)
            d = stats(m)
            d["lo"], d["hi"] = lo, min(hi, 100)
            rows.append(d)
        return rows

    stock_b, entry_b, risk_b = bands(S, STOCK_EDGES), bands(P["entry"], ENTRY_EDGES), bands(P["risk"], RISK_EDGES)
    sel_b = bands(P["sel"], SEL_EDGES)
    types = []
    for k in (1, 2, 3):
        d = stats(P[f"t{k}"])
        d["id"], d["label"] = k, TYPES[k]
        types.append(d)

    # ---- 過熱・売られすぎ、ローソク足シグナル ----
    heat_rows = []
    for n in (2, 3, 4):
        d = stats(P["os"] >= n); d.update(label=f"売られすぎが{n}個以上点灯", group="売られすぎ"); heat_rows.append(d)
    for n in (2, 3, 4):
        d = stats(P["ob"] >= n); d.update(label=f"買われすぎが{n}個以上点灯", group="買われすぎ"); heat_rows.append(d)
    for k, label, _ in HEAT_ITEMS:
        d = stats(P["h_" + k] == -1); d.update(label=f"{label}：売られすぎ", group="単独（売られすぎ）"); heat_rows.append(d)
        d = stats(P["h_" + k] == 1); d.update(label=f"{label}：買われすぎ", group="単独（買われすぎ）"); heat_rows.append(d)
    candle_rows = []
    for k, kind in CANDLE_KIND.items():
        d = stats(P["c_" + k]); d.update(label=k + ("（買い系）" if kind > 0 else "（売り系）"), group="買い系" if kind > 0 else "売り系"); candle_rows.append(d)

    # ---- 累積リターン ----
    # 運用ルール：買い基準と売り基準を分けて（ヒステリシス）入れ替えを抑える。毎日入れ替えると売買コストが重い
    low = _hold_rule(-S, -35, -55)        # 低スコア（35未満で保有、55以上で売り）。参考用
    strat = {"銘柄スコア：70以上で買い・50未満で売り": _hold_rule(S, 70, 50),
             "銘柄スコア：60以上で買い・45未満で売り": _hold_rule(S, 60, 45),
             "選別スコア（検証版）：70以上で買い・50未満で売り": _hold_rule(P["sel"], 70, 50),
             "参考：低スコア（35未満で買い・55以上で売り）": low,
             "代表銘柄の平均（全銘柄を等金額）": S.notna()}
    rets, turns = {}, {}
    for k, m in strat.items():
        rr, tt = _curve_returns(m, R, with_turn=True)
        rets[k], turns[k] = rr.loc[first:], tt.loc[first:]
    start = list(rets.values())[0].index
    rets["TOPIX（ETF 1306）"] = tp.pct_change().reindex(start).fillna(0.0)
    if n225 is not None:
        rets["日経平均"] = bm_close["N225"].pct_change().reindex(start).fillna(0.0)
    curves = {"dates": [d.strftime("%Y-%m-%d") for d in start], "series": {}, "perf": {}}
    for k, r in rets.items():
        curves["series"][k] = [_r(v, 2) for v in (100 * (1 + r).cumprod())]
        curves["perf"][k] = _perf(r)
        if k in turns:   # 1日あたりの入れ替え量（片道）と、年換算のコスト負担
            curves["perf"][k]["turn"] = _r(turns[k].mean() * 100, 1)
            curves["perf"][k]["costYr"] = _r(turns[k].mean() * COST_SIDE * YEAR_DAYS * 100, 1)

    # ---- 要素別の有効性（順位相関IC：その日の銘柄間で、スコアが高い銘柄ほど将来リターンが高かったか） ----
    items = [("score", "個別銘柄スコア（総合）", "stock")] + [(k, L_STOCK[k], "stock") for k in W_STOCK] \
        + [("entry", "エントリータイミング（総合）", "entry")] + [(k, L_ENTRY[k], "entry") for k in W_ENTRY] \
        + [("risk", "リスクスコア（高=安全）", "risk")] + [(k, L_RISK[k], "risk") for k in W_RISK if k != "concentration"]
    ic_rows = []
    for k, label, kind in items:
        row = {"key": k, "label": label, "kind": kind, "total": k in ("score", "entry", "risk"),
               "w": (W_STOCK if kind == "stock" else W_ENTRY if kind == "entry" else W_RISK).get(k)}
        for n in (5, 20):
            row[f"ic{n}"] = _ic(P[k], FWD[n])
        ic_rows.append(row)

    # ---- 年別の安定性（20日後・コスト差引後の平均、%） ----
    F20 = FWD[20]
    years = []
    for y in sorted(set(S.index[valid_dates].year)):
        sel = (S.index.year == y) & valid_dates
        if not sel.any():
            continue
        f = F20.to_numpy()[sel]
        def mean_where(m):
            mm = m.to_numpy()[sel] & np.isfinite(f)
            return (_r((f[mm] - COST_RT).mean() * 100), int(mm.sum())) if mm.any() else (None, 0)
        hi, lo, al = mean_where(S >= 60), mean_where(S < 40), mean_where(S.notna())
        years.append({"year": int(y), "hi": hi[0], "hiN": hi[1], "lo": lo[0], "loN": lo[1], "all": al[0], "allN": al[1]})

    last = S.index.max()
    return {
        "period": {"start": first.strftime("%Y-%m-%d"), "end": last.strftime("%Y-%m-%d"), "days": int(valid_dates.sum()),
                   "stocks": int((S.notna().sum() > 0).sum())},
        "assume": {"entry": "シグナル翌営業日の終値で約定", "exit": "約定の5/10/20営業日後の終値で決済",
                   "feeSide": COST_FEE * 100, "slipSide": COST_SLIP * 100, "roundTrip": COST_RT * 100},
        "weights": {"stock": W_STOCK, "entry": W_ENTRY, "risk": W_RISK},
        "stockBands": stock_b, "selBands": sel_b, "heatRows": heat_rows, "candleRows": candle_rows, "entryBands": entry_b, "riskBands": risk_b, "types": types,
        "curves": curves, "ic": ic_rows, "years": years,
        "review": build_review(P, FWD[20], first, last),
        "regime": build_regime(P, FWD[20], tp, first),
    }
