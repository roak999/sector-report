"""個別銘柄の分析：指標、各スコア（0〜100）、スクリーニング分類、売買の参考値。

すべて日次の時系列として計算し（未来のデータは使わない）、画面には最新日の値を出す。
将来のバックテストで同じ計算を過去日にも使えるよう、スコアは時系列のまま作ってから最新行を取る。

スコア設計（初期仮説。配点は scoring の方針どおりバックテストで見直す）
  個別銘柄スコア 100点 = トレンド30 + モメンタム25 + 相対強度20 + 出来高15 + セクター内の強さ10
  総合評価 = 市場スコア×0.20 + セクタースコア×0.30 + 個別銘柄スコア×0.50（3つは独立して表示）
  エントリータイミング 100点 = 25日線付近の押し目30 + MACD改善/GC25 + 出来高を伴う上抜け25 + RSIの適正水準20
  リスク（高いほど安全）100点 = ATRによる変動リスク30 + 高値からの下落率25 + トレンドの安定性25 + セクター集中リスク20
    ※セクター集中リスクは保有銘柄が決まって初めて計算できるため、ここでは「データなし」とし、
      残りの配点で再計算する（ポートフォリオ画面で反映予定）。
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from indicators import CANDLE_HELP, candle_patterns, macd
from names import NAMES
from scoring import clip01, num, prank, weighted_score

W_STOCK = {"trend": 30, "momentum": 25, "rs": 20, "volume": 15, "secrel": 10}
L_STOCK = {"trend": "トレンド", "momentum": "モメンタム", "rs": "相対強度（TOPIX比）", "volume": "出来高", "secrel": "セクター内での強さ"}
W_ENTRY = {"pullback": 30, "macd": 25, "breakout": 25, "rsi": 20}
L_ENTRY = {"pullback": "25日線付近での押し目", "macd": "MACDの改善・GC", "breakout": "出来高を伴う上抜け", "rsi": "RSIの適正水準"}
W_RISK = {"atr": 30, "drawdown": 25, "stability": 25, "concentration": 20}
L_RISK = {"atr": "ATRによる変動リスク", "drawdown": "高値からの下落率", "stability": "トレンドの安定性", "concentration": "セクター集中リスク"}
W_TOTAL = {"market": 0.20, "sector": 0.30, "stock": 0.50}
MIN_COV = 0.7
TYPES = {1: "上昇トレンド継続", 2: "上昇初動・ブレイクアウト候補", 3: "下落からの反転候補"}


# ---------- 登録銘柄 ----------

def load_watchlist(path: Path):
    """watchlist.txt を読む。戻り値: [(ticker, 名前|None, セクター|None)]"""
    out = []
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = [p.strip() for p in re.split(r"[,\t，、]", line) if p.strip()]
        code = parts[0].upper().replace(".T", "")
        if not re.fullmatch(r"[0-9]{3}[0-9A-Z]", code):
            print(f"watchlist: 証券コードとして読めない行をスキップ: {raw!r}")
            continue
        out.append((code + ".T", parts[1] if len(parts) > 1 else None, parts[2] if len(parts) > 2 else None))
    return out


# ---------- 指標 ----------

def _wilder(s, n):
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def features(df: pd.DataFrame, n: int = 14) -> dict:
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    v = df["Volume"].where(df["Volume"] > 0)
    d = c.diff()
    au, ad = _wilder(d.clip(lower=0), n), _wilder((-d).clip(lower=0), n)
    rsi = 100 - 100 / (1 + au / ad.replace(0, np.nan))
    rsi = rsi.where(ad != 0, 100.0).where(au.notna())
    up, dn = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=c.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=c.index)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = _wilder(tr, n)
    pdi, mdi = 100 * _wilder(pdm, n) / atr, 100 * _wilder(mdm, n) / atr
    adx = _wilder(100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan), n)
    ma = {k: c.rolling(k).mean() for k in (5, 25, 75, 200)}
    sd20 = c.rolling(20).std(ddof=0)
    line, sig, hist = macd(c)
    return {"o": o, "h": h, "l": l, "c": c, "v": v, "rsi": rsi, "adx": adx, "pdi": pdi, "mdi": mdi, "atr": atr,
            "ma5": ma[5], "ma25": ma[25], "ma75": ma[75], "ma200": ma[200],
            "bbu": c.rolling(20).mean() + 2 * sd20, "bbl": c.rolling(20).mean() - 2 * sd20,
            "macd": line, "signal": sig, "hist": hist}


def components(f: dict) -> dict:
    """銘柄単体で決まる要素（0〜1の時系列）。相対強度・セクター内は後で横断的に付ける"""
    c, v, hist = f["c"], f["v"], f["hist"]
    ma25, ma75, ma200 = f["ma25"], f["ma75"], f["ma200"]

    # --- 個別銘柄スコアの要素 ---
    binary = pd.concat([c > ma25, c > ma75, c > ma200, ma25 > ma75, ma75 > ma200, ma25 > ma25.shift(5)],
                       axis=1).astype(float).mean(axis=1).where(ma200.notna())
    adx_dir = (clip01((f["adx"] - 15) / 20) * (f["pdi"] > f["mdi"])).where(f["adx"].notna())
    trend = 0.5 * binary + 0.3 * prank(c / ma75 - 1) + 0.2 * adx_dir
    momentum = (0.4 * prank(c.pct_change(20)) + 0.3 * prank(c.pct_change(60)) + 0.3 * prank(hist / c))
    vr5 = v.rolling(5, min_periods=3).mean() / v.shift(5).rolling(20, min_periods=10).mean()
    volume = 0.5 + np.sign(c.pct_change(5)) * (prank(vr5) - 0.5)

    # --- エントリータイミング ---
    dev25 = c / ma25 - 1
    uptrend = ((c > ma75) & (ma25 > ma75)).astype(float)
    pullback = clip01(1 - dev25.abs() / 0.06) * (0.4 + 0.6 * uptrend)
    above = hist > 0
    gc = above & ~above.shift(1, fill_value=False)
    gc3 = gc.rolling(3).max().fillna(0) > 0
    rising = hist > hist.shift(1)
    macd_s = pd.Series(np.select([gc3, above & rising, ~above & rising, above & ~rising], [1.0, 0.7, 0.5, 0.3], default=0.0), index=c.index)
    prior_hi = c.shift(1).rolling(20).max()
    brk = c > prior_hi
    vd = v / v.shift(1).rolling(20, min_periods=10).mean()
    volx = clip01((vd - 1) / 1.0)
    breakout = (brk.astype(float) * (0.4 + 0.6 * volx.fillna(0)) + (~brk).astype(float) * 0.3 * volx.fillna(0) * (c > c.shift(1)))
    r = f["rsi"]
    rsi_s = pd.Series(np.select([r <= 30, r < 45, r <= 65, r < 80], [0.0, (r - 30) / 15, 1.0, (80 - r) / 15], default=0.0), index=c.index).where(r.notna())
    entry = {"pullback": pullback.where(ma75.notna()), "macd": macd_s.where(hist.notna()), "breakout": breakout.where(prior_hi.notna()), "rsi": rsi_s}

    # --- リスク（高いほど安全） ---
    atrp = f["atr"] / c
    atr_s = 0.5 * clip01((0.05 - atrp) / 0.04) + 0.5 * (1 - prank(atrp))
    dd60, dd250 = c / c.rolling(60).max() - 1, c / c.rolling(250, min_periods=120).max() - 1
    dd_s = 0.5 * clip01(1 + dd60 / 0.15) + 0.5 * clip01(1 + dd250 / 0.30)
    lc = np.log(c)
    r2 = lc.rolling(60).corr(pd.Series(np.arange(len(c)), index=c.index)) ** 2
    jumpy = (c.pct_change().abs() > 0.03).astype(float).rolling(60).mean()
    stab = 0.6 * r2 + 0.4 * (1 - clip01(jumpy / 0.15))
    risk = {"atr": atr_s, "drawdown": dd_s, "stability": stab}

    extra = {"vr5": vr5, "vd": vd, "dev25": dev25, "dd60": dd60, "dd250": dd250, "atrp": atrp, "gc3": gc3, "brk": brk}
    return {"stock": {"trend": trend, "momentum": momentum, "volume": volume}, "entry": entry, "risk": risk, "extra": extra}


# ---------- 選別スコア（検証版）：10年の検証で、前半・後半の両方で順位相関がプラスだった3要素 ----------
# 「長期（12か月−直近1か月）は強い」×「直近は押している（20日・75日線に対して）」銘柄を高く評価する。
# 現行の個別銘柄スコア（トレンド・モメンタム等）は10年検証で高スコアが有利とは言えなかったため、別スコアとして並べる。
L_SEL = {"mom": "長期モメンタム（12か月−直近1か月の騰落率）", "rev20": "20日の押し（20日騰落率が低いほど高い）", "dev75": "75日線からの押し（75日線乖離が低いほど高い）"}


def sel_scores(CP: pd.DataFrame):
    """CP: 日付×銘柄の終値（取引日カレンダーに揃え、補完しない）。戻り値: (0〜100のスコア, 要素のパーセンタイル順位, 生の値)
    すべてその日までのデータと、同じ日の銘柄間の順位だけで計算する（先読みなし）"""
    raw = {"mom": CP.shift(20) / CP.shift(250) - 1, "rev20": CP / CP.shift(20) - 1, "dev75": CP / CP.rolling(75).mean() - 1}
    comp = {"mom": raw["mom"].rank(axis=1, pct=True), "rev20": (-raw["rev20"]).rank(axis=1, pct=True),
            "dev75": (-raw["dev75"]).rank(axis=1, pct=True)}
    ok = comp["mom"].notna() & comp["rev20"].notna() & comp["dev75"].notna()
    score = ((comp["mom"] + comp["rev20"] + comp["dev75"]) / 3 * 100).where(ok)
    return score, comp, raw

# ---------- 過熱・売られすぎ（参考）と ローソク足シグナル ----------
# 各指標を「買われすぎ(+1)／中立(0)／売られすぎ(-1)」に分け、点灯した数を数える。基準値は初期の仮説で、
# 検証画面で「点灯数が多いほど、その後どうなったか」を確かめる（売買の合図ではなく、過熱感を見る参考情報）。
HEAT_ITEMS = [
    ("psy", "サイコロジカル（12日）", "12日中9日以上上昇＝買われすぎ／3日以下＝売られすぎ"),
    ("rsi", "RSI（14日）", "70以上＝買われすぎ／30以下＝売られすぎ"),
    ("dev25", "25日線からの乖離", "自分の過去250日で上位10%＝買われすぎ／下位10%＝売られすぎ"),
    ("bb", "ボリンジャーバンド（20日±2σ）", "上限を超える＝買われすぎ／下限を割る＝売られすぎ"),
    ("macd", "MACD（株価比）", "自分の過去250日で上位10%＝買われすぎ／下位10%＝売られすぎ"),
    ("vol", "出来高を伴う急変", "出来高が20日平均の2倍以上で、当日+3%以上＝過熱／−3%以下＝投げ売り（売られすぎ）"),
]
CANDLE_KIND = {"陽の包み足": 1, "赤三兵": 1, "ハンマー": 1, "陰の包み足": -1, "黒三兵": -1, "上ヒゲ": -1}


def heat_levels(f: dict, extra: dict) -> dict:
    """名前 → 日次の水準（+1 買われすぎ / 0 中立 / -1 売られすぎ / 欠損）。その日までのデータだけで計算する"""
    c = f["c"]
    lv = lambda hi, lo, base: pd.Series(np.where(hi, 1.0, np.where(lo, -1.0, 0.0)), index=c.index).where(base.notna())
    up_ratio = (c.diff() > 0).astype(float).rolling(12).mean().where(c.diff().notna())
    p_dev, mm = prank(extra["dev25"]), f["macd"] / c
    p_macd = prank(mm)
    bw = (f["bbu"] - f["bbl"]).replace(0, np.nan)
    pb = (c - f["bbl"]) / bw
    ret1, vd = c.pct_change(), extra["vd"]
    return {
        "psy": lv(up_ratio >= 9 / 12 - 1e-9, up_ratio <= 3 / 12 + 1e-9, up_ratio),
        "rsi": lv(f["rsi"] >= 70, f["rsi"] <= 30, f["rsi"]),
        "dev25": lv(p_dev >= 0.9, p_dev <= 0.1, p_dev),
        "bb": lv(pb >= 1.0, pb <= 0.0, pb),
        "macd": lv(p_macd >= 0.9, p_macd <= 0.1, p_macd),
        "vol": lv((vd >= 2) & (ret1 >= 0.03), (vd >= 2) & (ret1 <= -0.03), vd),
    }


def heat_row(f: dict, extra: dict, i: int) -> dict:
    """最新日の過熱・売られすぎの表示用データ"""
    L = heat_levels(f, extra)
    c, r = f["c"], f["rsi"]
    up12 = int((c.diff() > 0).astype(float).rolling(12).sum().iloc[i]) if i >= 12 else None
    pb = ((c - f["bbl"]) / (f["bbu"] - f["bbl"]).replace(0, np.nan)).iloc[i]
    pdv, pmc = prank(extra["dev25"]).iloc[i], prank(f["macd"] / c).iloc[i]
    vals = {
        "psy": None if up12 is None else f"12日中{up12}日上昇",
        "rsi": None if pd.isna(r.iloc[i]) else f"{r.iloc[i]:.0f}",
        "dev25": None if pd.isna(extra["dev25"].iloc[i]) else f"{extra['dev25'].iloc[i] * 100:+.1f}%（過去250日で上位{(1 - pdv) * 100:.0f}%）" if pd.notna(pdv) else f"{extra['dev25'].iloc[i] * 100:+.1f}%",
        "bb": None if pd.isna(pb) else f"位置 {pb * 100:.0f}%（0=下限・100=上限）",
        "macd": None if pd.isna(pmc) else f"過去250日で上位{(1 - pmc) * 100:.0f}%",
        "vol": None if pd.isna(extra["vd"].iloc[i]) else f"出来高{extra['vd'].iloc[i]:.1f}倍・当日{(c.pct_change().iloc[i]) * 100:+.1f}%",
    }
    items, os_, ob = [], 0, 0
    for k, label, rule in HEAT_ITEMS:
        v = L[k].iloc[i]
        lev = None if pd.isna(v) else int(v)
        os_ += lev == -1
        ob += lev == 1
        items.append({"key": k, "label": label, "rule": rule, "value": vals[k], "level": lev})
    return {"items": items, "os": int(os_), "ob": int(ob)}


def candle_recent(f: dict, i: int, n: int = 5) -> list:
    """直近n日に出たローソク足パターン（新しい順）。{name, daysAgo, kind(+1買い系/-1売り系), help}"""
    pats = candle_patterns(f["o"], f["h"], f["l"], f["c"])
    out = []
    for back in range(n):
        j = i - back
        if j < 0:
            break
        for name, ser in pats.items():
            if bool(ser.iloc[j]):
                out.append({"name": name, "daysAgo": back, "kind": CANDLE_KIND[name], "help": CANDLE_HELP[name]})
    return out

# ---------- 判定の文言 ----------

def grade(x):
    if x is None or pd.isna(x):
        return "データなし"
    return "強い" if x >= 0.7 else "やや強い" if x >= 0.55 else "中立" if x >= 0.45 else "やや弱い" if x >= 0.3 else "弱い"


def entry_grade(s):
    return "データなし" if s is None else "好機（条件が揃っている）" if s >= 70 else "まずまず" if s >= 50 else "様子見" if s >= 30 else "不向き（条件が揃っていない）"


def macd_text(f, i):
    h = f["hist"]
    ab = h > 0
    cross = (ab != ab.shift(1)) & h.shift(1).notna()
    pos = np.where(cross.to_numpy(), np.arange(len(h)), np.nan)
    last = pd.Series(pos, index=h.index).ffill().iloc[i]
    if not pd.isna(last) and i - last < 3:
        return ("GC" if ab.iloc[i] else "DC") + ("当日" if i - last == 0 else f"{int(i - last)}日前")
    return ("ヒストグラム＋拡大" if h.iloc[i] > h.iloc[i - 1] else "ヒストグラム＋縮小") if ab.iloc[i] else \
           ("ヒストグラム－縮小" if h.iloc[i] > h.iloc[i - 1] else "ヒストグラム－拡大")


def levels(f, i):
    """支持線・抵抗線・ATRに基づく『参考値』。売買価格を断定するものではない"""
    c, atr = f["c"].iloc[i], f["atr"].iloc[i]
    if pd.isna(atr):
        return None
    low20 = f["l"].iloc[max(0, i - 19):i + 1].min()
    high60 = f["h"].iloc[max(0, i - 59):i + 1].max()
    ma25 = f["ma25"].iloc[i]
    stop_atr, stop_sup = c - 2 * atr, low20
    stop = max(stop_atr, stop_sup)             # 浅いほうを参考損切りにする（下の支持線を割ったら見直し）
    resist = high60 if high60 > c + 0.5 * atr else None
    target = resist if resist is not None else c + 2 * atr
    risk, reward = c - stop, target - c
    return {"close": num(c), "atr": num(atr), "atrPct": num(atr / c * 100, 2),
            "stopAtr": num(stop_atr), "support20": num(low20), "ma25": num(ma25), "stop": num(stop),
            "resistance60": num(high60), "target": num(target),
            "targetKind": "直近60日高値（抵抗線）" if resist is not None else "高値圏のため 終値＋2ATR（参考）",
            "rr": num(reward / risk, 2) if risk > 0 else None,
            "note": "支持線・抵抗線・ATRから機械的に出した参考値です。売買価格を断定するものではありません。"}


# ---------- 本体 ----------

def analyze(prices: dict, targets: list, topix: pd.Series, sector_scores: dict, market_score, detail_days: int = 250):
    """targets: [(ticker, name, sector)]。戻り値: (一覧rows, 詳細dict, 取得できなかったticker)"""
    cal = topix.dropna().index
    tp = topix.reindex(cal)
    feats, comps, used, missing = {}, {}, [], []
    for t, name, sector in targets:
        df = prices.get(t)
        if df is None or len(df.dropna(subset=["Close"])) < 120:
            missing.append(t)
            continue
        f = features(df)
        feats[t], comps[t] = f, components(f)
        used.append((t, name, sector))

    # 横断：TOPIX比の合成超過収益 → 全体順位（相対強度）、同セクター内順位（セクター内の強さ）
    blend = {}
    for t, _, _ in used:
        c = feats[t]["c"].reindex(cal)
        blend[t] = sum(w * (c.pct_change(k) - tp.pct_change(k)) for k, w in ((5, 0.25), (20, 0.40), (60, 0.35)))
    B = pd.DataFrame(blend)
    rs_rank = B.rank(axis=1, pct=True)
    sec_rank = pd.DataFrame(index=B.index, columns=B.columns, dtype=float)
    by_sec = {}
    for t, _, s in used:
        by_sec.setdefault(s, []).append(t)
    for s, ts in by_sec.items():
        if s and s != "未分類" and len(ts) >= 4:      # 4社未満のセクターは比較しない（データなし）
            sec_rank[ts] = B[ts].rank(axis=1, pct=True)

    CP = pd.DataFrame({t: feats[t]["c"].reindex(cal) for t, _, _ in used})
    SEL, SELC, SELR = sel_scores(CP)

    rows, detail = [], {}
    for t, name, sector in used:
        f, cp = feats[t], comps[t]
        c = f["c"]
        idx = c.index
        i = len(idx) - 1
        d = idx[i]
        j = cal.get_indexer([d])[0]
        parts = dict(cp["stock"])
        parts["rs"] = rs_rank[t].reindex(idx) if t in rs_rank else pd.Series(np.nan, index=idx)
        parts["secrel"] = sec_rank[t].reindex(idx)
        s_score, s_cov = weighted_score(parts, W_STOCK)
        s_score = s_score.where(s_cov >= MIN_COV)
        e_score, _ = weighted_score(cp["entry"], W_ENTRY)
        risk_parts = dict(cp["risk"]); risk_parts["concentration"] = pd.Series(np.nan, index=idx)
        r_score, r_cov = weighted_score(risk_parts, W_RISK)

        sv = s_score.iloc[i]
        if pd.isna(sv):
            continue
        sec_sc = sector_scores.get(sector)
        wsum, tot = W_TOTAL["stock"], W_TOTAL["stock"] * sv
        if market_score is not None:
            wsum += W_TOTAL["market"]; tot += W_TOTAL["market"] * market_score
        if sec_sc is not None:
            wsum += W_TOTAL["sector"]; tot += W_TOTAL["sector"] * sec_sc
        total = tot / wsum

        ex, lv = cp["extra"], None
        rsi_now, adx_now = f["rsi"].iloc[i], f["adx"].iloc[i]
        ma25, ma75 = f["ma25"].iloc[i], f["ma75"].iloc[i]
        rs_now = B[t].iloc[j] if (t in B and j >= 0) else np.nan
        # --- 3分類（ルールは画面にも表示） ---
        hist = f["hist"]
        vol_brk3 = ((ex["brk"] & (ex["vd"] >= 1.3)).rolling(3).max().fillna(0) > 0).iloc[i]
        type1 = bool(c.iloc[i] > ma25 > ma75 and ma75 > f["ma75"].iloc[i - 5] and adx_now >= 20 and f["pdi"].iloc[i] > f["mdi"].iloc[i]
                     and ex["dev25"].iloc[i] <= 0.10 and sv >= 55)
        rs_r = parts["rs"].iloc[i]
        type2 = bool((vol_brk3 or (ex["gc3"].iloc[i] and c.iloc[i] > ma25)) and c.iloc[i] > ma75
                     and (pd.isna(rs_r) or rs_r >= 0.4) and ex["dev25"].iloc[i] <= 0.12)
        rsi_min10 = f["rsi"].iloc[i - 9:i + 1].min()
        type3 = bool(ex["dd60"].iloc[i] <= -0.10 and rsi_min10 <= 35 and rsi_now >= 40 and rsi_now > f["rsi"].iloc[i - 3]
                     and hist.iloc[i] > hist.iloc[i - 1] and c.iloc[i] > f["ma5"].iloc[i])
        tags = [k for k, ok in ((1, type1), (2, type2), (3, type3)) if ok]

        def comp_list(parts_, weights, labels, idx_i=i):
            out = []
            for k, w in weights.items():
                v = parts_[k].iloc[idx_i] if k in parts_ else np.nan
                out.append({"key": k, "label": labels[k], "max": w, "ratio": num(v, 3),
                            "points": num(v * w, 1) if pd.notna(v) else None})
            return out
        stock_c, entry_c, risk_c = comp_list(parts, W_STOCK, L_STOCK), comp_list(cp["entry"], W_ENTRY, L_ENTRY), comp_list(risk_parts, W_RISK, L_RISK)
        lv = levels(f, i)
        ret = lambda k: num((c.iloc[i] / c.iloc[i - k] - 1) * 100, 1) if i >= k else None
        row = {
            "code": t.replace(".T", ""), "name": name or t.replace(".T", ""), "sector": sector or "未分類", "date": d.strftime("%m/%d"),
            "price": num(c.iloc[i]), "chg1": ret(1), "ret5": ret(5), "ret20": ret(20), "ret60": ret(60),
            "total": num(total, 0), "score": num(sv, 0), "sectorScore": sec_sc, "marketScore": market_score,
            "rs20": num((c.pct_change(20).iloc[i] - tp.pct_change(20).reindex(idx).iloc[i]) * 100, 1),
            "rsRank": num(rs_r * 100, 0) if pd.notna(rs_r) else None,
            "volInc": num((ex["vr5"].iloc[i] - 1) * 100, 0), "volDay": num(ex["vd"].iloc[i], 2),
            "rsi": num(rsi_now, 0), "adx": num(adx_now, 0), "macd": macd_text(f, i),
            "macdUp": bool(hist.iloc[i] > hist.iloc[i - 1]), "macdPos": bool(hist.iloc[i] > 0),
            "above25": bool(c.iloc[i] > ma25), "above75": bool(c.iloc[i] > ma75), "ma25gt75": bool(ma25 > ma75),
            "dev25": num(ex["dev25"].iloc[i] * 100, 1), "atrPct": num(ex["atrp"].iloc[i] * 100, 2),
            "entry": num(e_score.iloc[i], 0), "risk": num(r_score.iloc[i], 0), "riskCov": num(r_cov.iloc[i], 2),
            "stop": lv["stop"] if lv else None, "target": lv["target"] if lv else None, "rr": lv["rr"] if lv else None,
            "sel": num(SEL[t].iloc[j], 0) if j >= 0 else None,
            "selComps": [{"key": k, "label": L_SEL[k], "ratio": num(SELC[k][t].iloc[j], 3) if j >= 0 else None,
                          "raw": num(SELR[k][t].iloc[j] * 100, 1) if j >= 0 and pd.notna(SELR[k][t].iloc[j]) else None} for k in L_SEL],
            "heat": heat_row(f, ex, i), "candles": candle_recent(f, i),
            "tags": tags, "comps": stock_c, "entryComps": entry_c, "riskComps": risk_c,
            "inUniverse": True,
        }
        rows.append(row)

        # --- 個別銘柄の詳細（別JSONに出す） ---
        k0 = max(0, len(idx) - detail_days)
        sl = slice(k0, None)
        ix = idx[sl]
        rs_line = (c / tp.reindex(idx).ffill()).iloc[sl]
        rs_line = rs_line / rs_line.dropna().iloc[0] * 100 if rs_line.notna().any() else rs_line
        ser = lambda s: [num(v, 3 if s.abs().max() < 10 else 2) if pd.notna(v) else None for v in s.iloc[sl]]
        detail[row["code"]] = {
            "code": row["code"], "name": row["name"], "sector": row["sector"], "date": d.strftime("%Y-%m-%d"),
            "dates": [x.strftime("%Y-%m-%d") for x in ix],
            "o": [num(v) for v in f["o"].iloc[sl]], "h": [num(v) for v in f["h"].iloc[sl]],
            "l": [num(v) for v in f["l"].iloc[sl]], "c": [num(v) for v in c.iloc[sl]],
            "v": [num(v / 1e3, 0) if pd.notna(v) else None for v in f["v"].iloc[sl]],
            **{k: [num(v) for v in f[k].iloc[sl]] for k in ("ma5", "ma25", "ma75", "ma200", "bbu", "bbl", "atr")},
            "macd": ser(f["macd"]), "signal": ser(f["signal"]), "hist": ser(f["hist"]),
            "rsi": ser(f["rsi"]), "adx": ser(f["adx"]), "pdi": ser(f["pdi"]), "mdi": ser(f["mdi"]),
            "rs": [num(v) for v in rs_line],
            "analysis": {
                "trend": grade(parts["trend"].iloc[i]), "momentum": grade(parts["momentum"].iloc[i]),
                "volume": grade(parts["volume"].iloc[i]), "rs": grade(parts["rs"].iloc[i]),
                "entry": entry_grade(row["entry"]), "levels": lv,
                "volDetail": f"5日平均÷前20日平均＝{(ex['vr5'].iloc[i]):.2f}倍／当日出来高÷20日平均＝{ex['vd'].iloc[i]:.2f}倍",
                "rsDetail": f"TOPIX比 5日 {num((c.pct_change(5).iloc[i] - tp.pct_change(5).reindex(idx).iloc[i]) * 100, 1)}% / "
                            f"20日 {row['rs20']}% / 60日 {num((c.pct_change(60).iloc[i] - tp.pct_change(60).reindex(idx).iloc[i]) * 100, 1)}%",
            },
        }

    # セクター内順位（個別銘柄スコア順）
    by = {}
    for r in rows:
        by.setdefault(r["sector"], []).append(r)
    for s, rs_ in by.items():
        rs_.sort(key=lambda r: -(r["score"] or 0))
        for k, r in enumerate(rs_, 1):
            r["secRank"], r["secN"] = k, len(rs_)
            detail[r["code"]]["analysis"]["secRank"] = f"{r['sector']} {len(rs_)}銘柄中 {k}位" + ("（分析対象の銘柄内の順位）" if s != "未分類" else "（セクター未分類のため比較なし）")
    rows.sort(key=lambda r: -(r["total"] or 0))
    return rows, detail, missing
