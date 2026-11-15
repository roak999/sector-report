"""セクター分析の拡張：新セクタースコア（0〜100）・TOPIX比の相対強度・ランキング変動・ローテーション。

既存の -100〜+100 の総合スコア（indicators.py）はそのまま残し、これは並列で出す「新スコア」。

配点（初期仮説）  TOPIXに対する相対強度30 / セクター内上昇銘柄比率25 / 移動平均・トレンド20 / 出来高15 / MACD・モメンタム10
  * 相対強度  : 5・20・60日の「セクター騰落率 − TOPIX騰落率」を 0.25/0.40/0.35 で合成し、
                その日の17業種の中での順位（0〜1）。
  * 上昇銘柄比率: 代表銘柄の「25日線より上の比率」(50%) と「5日前より上がっている比率」(50%)。
  * 移動平均  : 既存の移動平均スコア（-2〜+2）を 0〜1 に換算。
  * 出来高    : 直近5日平均/その前20日平均 の自分の過去250日順位。株価が5日で上昇なら高いほど加点、
                下落なら高いほど減点（出来高を伴う下落を強いとは評価しない）。
  * MACD      : 既存MACDスコアを0〜1に換算(50%) ＋ ヒストグラム/株価の過去250日順位(50%)。
  欠損（出来高が取れない等）の要素は除外し、残りの重みで 0〜100 に再計算する。
"""
import numpy as np
import pandas as pd

from indicators import macd, score_macd, score_ma
from scoring import num, prank, weighted_score
from market import breadth_frames

WEIGHTS = {"rs": 30, "breadth": 25, "trend": 20, "volume": 15, "macd": 10}
LABELS = {"rs": "TOPIXに対する相対強度", "breadth": "セクター内の上昇銘柄比率",
          "trend": "移動平均線・トレンド", "volume": "出来高", "macd": "MACD・モメンタム"}


def build_sector_plus(sector_data: dict, prices: dict, universe: dict, topix: pd.Series,
                      history_days: int = 60) -> dict:
    cal = topix.dropna().index
    names = [n for n in sector_data if n in universe]
    closes = pd.DataFrame({n: sector_data[n][0]["Close"] for n in names}).reindex(cal)
    tp = topix.reindex(cal)

    # ---- 相対強度 ----
    ret = {k: closes.pct_change(k) for k in (5, 20, 60)}
    exc = {k: ret[k].sub(tp.pct_change(k), axis=0) for k in (5, 20, 60)}
    rs_raw = 0.25 * exc[5] + 0.40 * exc[20] + 0.35 * exc[60]
    rs_comp = rs_raw.rank(axis=1, pct=True)

    # ---- 各業種の部品 ----
    comp_frames = {k: {} for k in WEIGHTS}
    extra = {}
    for n in names:
        ohlc, turnover = sector_data[n]
        c = ohlc["Close"]
        # 上昇銘柄比率（代表銘柄）
        members = [t for t in universe[n] if t in prices]
        stocks = {t: prices[t] for t in members}
        bf = breadth_frames(stocks, cal) if stocks else None
        if bf is not None and len(stocks) > 0:
            nv = bf["ab25"].notna().sum(axis=1).replace(0, np.nan)
            above = bf["ab25"].sum(axis=1, min_count=1) / nv
            up5 = bf["up5"].sum(axis=1, min_count=1) / bf["up5"].notna().sum(axis=1).replace(0, np.nan)
            adv1 = bf["up1"].sum(axis=1, min_count=1) / bf["up1"].notna().sum(axis=1).replace(0, np.nan)
            breadth = 0.5 * above + 0.5 * up5
            extra[n] = {"above": above, "adv1": adv1, "n": nv}
        else:
            breadth = pd.Series(np.nan, index=cal)
            extra[n] = {"above": breadth, "adv1": breadth, "n": breadth}
        comp_frames["breadth"][n] = breadth

        comp_frames["trend"][n] = ((score_ma(c) + 2) / 4).reindex(cal)

        t = turnover.reindex(c.index)
        vr = t.rolling(5, min_periods=3).mean() / t.shift(5).rolling(20, min_periods=10).mean()
        sign = np.sign(c.pct_change(5))
        vcomp = 0.5 + sign * (prank(vr) - 0.5)
        comp_frames["volume"][n] = vcomp.reindex(cal)
        extra[n]["volInc"] = (vr - 1).reindex(cal)

        _, _, hist = macd(c)
        comp_frames["macd"][n] = (0.5 * (score_macd(c) + 2) / 4 + 0.5 * prank(hist / c)).reindex(cal)
    comp_frames["rs"] = {n: rs_comp[n] for n in names}

    # ---- 新セクタースコア（日次） ----
    score_cols, cover_cols = {}, {}
    for n in names:
        parts = {k: comp_frames[k][n] for k in WEIGHTS}
        s, cov = weighted_score(parts, WEIGHTS)
        score_cols[n], cover_cols[n] = s.where(cov >= 0.5), cov
    score = pd.DataFrame(score_cols)
    ranks = score.rank(axis=1, ascending=False, method="min")

    valid_days = score.dropna(how="all").index
    last = valid_days[-1]
    i = cal.get_loc(last)

    def rank_at(n, back):
        j = i - back
        return None if j < 0 or pd.isna(ranks[n].iloc[j]) else int(ranks[n].iloc[j])

    # ---- ローテーション（RRG風）：RS線とその25日平均からの乖離、およびその5日変化 ----
    rs_line = closes.div(tp, axis=0)
    rs_dev = (rs_line / rs_line.rolling(25).mean() - 1) * 100
    rs_mom = rs_dev - rs_dev.shift(5)

    rows = []
    for n in names:
        if pd.isna(score[n].iloc[i]):
            continue
        comps = []
        for k, w in WEIGHTS.items():
            v = comp_frames[k][n].iloc[i]
            comps.append({"key": k, "label": LABELS[k], "max": w, "ratio": num(v, 3),
                          "points": num(v * w, 1) if pd.notna(v) else None})
        e = extra[n]
        r_now = int(ranks[n].iloc[i])
        r1, r5, r20 = rank_at(n, 1), rank_at(n, 5), rank_at(n, 20)
        trail = [{"x": num(rs_dev[n].iloc[j], 2), "y": num(rs_mom[n].iloc[j], 2)}
                 for j in range(i - 5, i + 1) if j >= 0 and pd.notna(rs_dev[n].iloc[j]) and pd.notna(rs_mom[n].iloc[j])]
        rows.append({
            "name": n, "score": num(score[n].iloc[i], 0), "rank": r_now,
            "rank1": r1, "rank5": r5, "rank20": r20,
            "d1": None if r1 is None else r1 - r_now, "d5": None if r5 is None else r5 - r_now,
            "d20": None if r20 is None else r20 - r_now,
            "ret5": num(ret[5][n].iloc[i] * 100, 1), "ret20": num(ret[20][n].iloc[i] * 100, 1),
            "ret60": num(ret[60][n].iloc[i] * 100, 1),
            "rs5": num(exc[5][n].iloc[i] * 100, 1), "rs20": num(exc[20][n].iloc[i] * 100, 1),
            "rs60": num(exc[60][n].iloc[i] * 100, 1),
            "adv1": num(e["adv1"].iloc[i] * 100, 0), "above25": num(e["above"].iloc[i] * 100, 0),
            "members": None if pd.isna(e["n"].iloc[i]) else int(e["n"].iloc[i]),
            "volInc": num(e["volInc"].iloc[i] * 100, 0),
            "coverage": num(cover_cols[n].iloc[i], 2),
            "comps": comps, "trail": trail,
        })
    rows.sort(key=lambda r: r["rank"])

    h = score.loc[valid_days].tail(history_days)
    hr = ranks.loc[h.index]
    history = {"dates": [d.strftime("%m/%d") for d in h.index],
               "rows": [{"name": r["name"], "score": [num(v, 0) for v in h[r["name"]]],
                         "rank": [None if pd.isna(v) else int(v) for v in hr[r["name"]]]} for r in rows]}
    return {"date": last.strftime("%Y-%m-%d"), "rows": rows, "history": history,
            "weights": WEIGHTS, "labels": LABELS,
            "rrgNote": "横軸＝TOPIXに対するRS線の25日平均からの乖離(%)、縦軸＝その5日変化。"
                       "右上=先行（強く加速）/右下=失速/左下=劣後/左上=改善（弱いが回復中）"}
