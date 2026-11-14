"""テクニカル指標の計算と採点（各指標 -2〜+2）。

すべて日次の時系列で計算するので、「今日の点数」だけでなく
過去の点数の推移（＝あとでバックテストに使える）も出せる。
"""
import numpy as np
import pandas as pd

INDICATORS = ["サイコロ", "出来高", "MACD", "ローソク足", "移動平均"]

DEFAULT_WEIGHTS = {name: 1.0 for name in INDICATORS}


# ---------- 個別指標 ----------

def score_psychological(close: pd.Series, n: int = 12) -> pd.Series:
    """サイコロジカルライン: 直近n日のうち上昇した日の割合。
    25%以下=売られすぎ(+2) / 33%以下(+1) / 67%以上(-1) / 75%以上=過熱(-2)"""
    up = (close.diff() > 0).astype(float)
    ratio = up.rolling(n).mean()
    s = pd.Series(0, index=close.index, dtype=float)
    s[ratio <= 4 / 12] = 1
    s[ratio <= 3 / 12] = 2
    s[ratio >= 8 / 12] = -1
    s[ratio >= 9 / 12] = -2
    s[ratio.isna()] = np.nan
    return s


def score_volume(close: pd.Series, turnover: pd.Series, n: int = 20) -> pd.Series:
    """出来高（売買代金）変化: 当日 ÷ 過去n日平均。
    1.5倍以上で上昇=+2、1.2倍以上で上昇=+1（下落なら符号反転）"""
    avg = turnover.shift(1).rolling(n).mean()
    ratio = turnover / avg
    direction = np.sign(close.diff())
    s = pd.Series(0, index=close.index, dtype=float)
    s[ratio >= 1.2] = 1
    s[ratio >= 1.5] = 2
    s = s * direction + 0.0  # -0 を 0 にそろえる
    s[ratio.isna()] = np.nan
    return s


def macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    line = ema_f - ema_s
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def score_macd(close: pd.Series, cross_window: int = 3) -> pd.Series:
    """直近cross_window日以内のゴールデンクロス=+2 / デッドクロス=-2。
    クロスが無ければ ヒストグラムがプラスで拡大=+1 / マイナスで拡大=-1"""
    _, _, hist = macd(close)
    above = hist > 0
    golden = above & ~above.shift(1, fill_value=False)
    dead = ~above & above.shift(1, fill_value=False)
    recent_golden = golden.rolling(cross_window).max().astype(bool)
    recent_dead = dead.rolling(cross_window).max().astype(bool)
    growing = hist.diff()
    s = pd.Series(0, index=close.index, dtype=float)
    s[(hist > 0) & (growing > 0)] = 1
    s[(hist < 0) & (growing < 0)] = -1
    s[recent_golden] = 2
    s[recent_dead] = -2
    s.iloc[:35] = np.nan  # 計算が安定するまでは採点しない
    return s


def candle_patterns(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series) -> dict:
    """ローソク足パターンの判定結果（名前 → 日ごとの True/False）"""
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    upper = h - pd.concat([o, c], axis=1).max(axis=1)
    lower = pd.concat([o, c], axis=1).min(axis=1) - l
    bull = c > o
    bear = c < o
    po, pc = o.shift(1), c.shift(1)

    bull_engulf = bull & (pc < po) & (o <= pc) & (c >= po)
    bear_engulf = bear & (pc > po) & (o >= pc) & (c <= po)

    three_white = bull & bull.shift(1, fill_value=False) & bull.shift(2, fill_value=False) \
        & (c > c.shift(1)) & (c.shift(1) > c.shift(2))
    three_black = bear & bear.shift(1, fill_value=False) & bear.shift(2, fill_value=False) \
        & (c < c.shift(1)) & (c.shift(1) < c.shift(2))

    hammer = (lower >= 2 * body) & (upper <= body) & (lower / rng >= 0.6)
    shooting = (upper >= 2 * body) & (lower <= body) & (upper / rng >= 0.6)

    # 並び順＝表示の優先順（強いパターンを先に）
    return {
        "陽の包み足": bull_engulf, "赤三兵": three_white,
        "陰の包み足": bear_engulf, "黒三兵": three_black,
        "ハンマー": hammer, "上ヒゲ": shooting,
    }


def score_candles(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    """ローソク足パターン。
    +2: 陽の包み足 / 赤三兵      +1: 下ヒゲの長いハンマー
    -2: 陰の包み足 / 黒三兵      -1: 上ヒゲの長い線"""
    p = candle_patterns(o, h, l, c)
    s = pd.Series(0, index=c.index, dtype=float)
    s[p["ハンマー"]] = 1
    s[p["上ヒゲ"]] = -1
    s[p["赤三兵"] | p["陽の包み足"]] = 2
    s[p["黒三兵"] | p["陰の包み足"]] = -2
    return s


def score_ma(close: pd.Series, short=25, long=75, slope_days=5) -> pd.Series:
    """25日線が75日線より上で上向き、かつ株価が25日線より上=+2 / 25日線が上=+1（逆は符号反転）"""
    ma_s = close.rolling(short).mean()
    ma_l = close.rolling(long).mean()
    rising = ma_s > ma_s.shift(slope_days)
    falling = ma_s < ma_s.shift(slope_days)
    s = pd.Series(0, index=close.index, dtype=float)
    s[ma_s > ma_l] = 1
    s[ma_s < ma_l] = -1
    s[(ma_s > ma_l) & rising & (close > ma_s)] = 2
    s[(ma_s < ma_l) & falling & (close < ma_s)] = -2
    s[ma_l.isna()] = np.nan
    return s


# ---------- 総合スコア ----------

def score_all(ohlc: pd.DataFrame, turnover: pd.Series) -> pd.DataFrame:
    """1セクター分の日次スコア表（列=各指標）を返す"""
    o, h, l, c = ohlc["Open"], ohlc["High"], ohlc["Low"], ohlc["Close"]
    return pd.DataFrame({
        "サイコロ": score_psychological(c),
        "出来高": score_volume(c, turnover.reindex(c.index)),
        "MACD": score_macd(c),
        "ローソク足": score_candles(o, h, l, c),
        "移動平均": score_ma(c),
    })


def describe_all(ohlc: pd.DataFrame, turnover: pd.Series) -> pd.DataFrame:
    """ヒートマップのセルに表示する「実際の数値」（日ごと・文字列）

    サイコロ  : 直近12日の 何勝何敗（横ばいは「分」）
    出来高    : 20日平均比 何倍 ＋ 当日の株価の向き（↑↓）
    MACD      : 3日以内のクロスは「GC/DC 何日前」、それ以外はヒストグラムの向きと大きさ（株価比%）
    ローソク足: 出たパターン名。無ければ陽線/陰線と実体の大きさ（始値比%）
    移動平均  : 25日線と75日線の位置関係（25>75 など）、25日線の向き、株価の25日線乖離率
    """
    o, h, l, c = ohlc["Open"], ohlc["High"], ohlc["Low"], ohlc["Close"]
    idx = c.index
    t = turnover.reindex(idx)

    # サイコロ: 何勝何敗
    d = c.diff()
    win = (d > 0).astype(float).rolling(12).sum()
    lose = (d < 0).astype(float).rolling(12).sum()
    draw = 12 - win - lose
    psy = pd.Series([
        "" if pd.isna(w) else f"{w:.0f}勝{x:.0f}敗" + (f"{dr:.0f}分" if dr > 0 else "")
        for w, x, dr in zip(win, lose, draw)], index=idx)

    # 出来高: 20日平均比
    ratio = t / t.shift(1).rolling(20).mean()
    arrow = np.where(d > 0, "↑", np.where(d < 0, "↓", "→"))
    vol = pd.Series(["" if pd.isna(r) else f"{r:.2f}倍{a}" for r, a in zip(ratio, arrow)], index=idx)

    # MACD: クロスからの日数、またはヒストグラム
    _, _, hist = macd(c)
    above = hist > 0
    cross = (above != above.shift(1)) & above.shift(1).notna()
    cross.iloc[0] = False
    last_cross_pos = pd.Series(np.where(cross, np.arange(len(idx)), np.nan), index=idx).ffill()
    days_since = np.arange(len(idx)) - last_cross_pos
    hist_pct = hist / c * 100
    growing = hist.abs() > hist.abs().shift(1)
    macd_txt = []
    for i in range(len(idx)):
        if i < 35:
            macd_txt.append("")
        elif not pd.isna(days_since.iloc[i]) and days_since.iloc[i] < 3:
            n = int(days_since.iloc[i])
            macd_txt.append(("GC" if above.iloc[i] else "DC") + ("当日" if n == 0 else f"{n}日前"))
        else:
            macd_txt.append(f"{round(hist_pct.iloc[i], 2) + 0.0:+.2f}%" + ("拡大" if growing.iloc[i] else "縮小"))
    macd_s = pd.Series(macd_txt, index=idx)

    # ローソク足: パターン名 or 陽線/陰線と騰落率
    pats = candle_patterns(o, h, l, c)
    chg = (c / o - 1) * 100  # 実体の大きさ（始値比）。陽線なら＋、陰線なら−になる
    candle_txt = []
    for i in range(len(idx)):
        name = next((k for k, v in pats.items() if bool(v.iloc[i])), None)
        if name:
            candle_txt.append(name)
        elif pd.isna(chg.iloc[i]):
            candle_txt.append("")
        else:
            v = round(chg.iloc[i], 1) + 0.0
            body = "陽線" if v > 0 else ("陰線" if v < 0 else "十字")
            candle_txt.append(body if v == 0 else f"{body}{v:+.1f}%")
    candle_s = pd.Series(candle_txt, index=idx)

    # 移動平均: 25日線 vs 75日線、向き、乖離率
    ma25, ma75 = c.rolling(25).mean(), c.rolling(75).mean()
    slope = np.where(ma25 > ma25.shift(5), "↑", np.where(ma25 < ma25.shift(5), "↓", "→"))
    dev = (c / ma25 - 1) * 100
    ma_s = pd.Series([
        "" if pd.isna(l75) else f"{'25>75' if s25 > l75 else '25<75'}{sl} {round(dv, 1) + 0.0:+.1f}%"
        for s25, l75, sl, dv in zip(ma25, ma75, slope, dev)], index=idx)

    return pd.DataFrame({"サイコロ": psy, "出来高": vol, "MACD": macd_s,
                         "ローソク足": candle_s, "移動平均": ma_s})


def composite(scores: pd.DataFrame, weights: dict, trend_damp: float = 0.5) -> pd.Series:
    """重み付き合計を -100〜+100 に換算。

    逆張りと順張りの矛盾対策: 移動平均が ±2（強いトレンド）のとき、
    サイコロがトレンドと逆向き（上昇トレンド中の「過熱」など）なら
    サイコロの重みを trend_damp 倍に下げる。
    """
    w = pd.DataFrame({k: weights.get(k, 0.0) for k in scores.columns}, index=scores.index)
    strong = scores["移動平均"].abs() == 2
    opposite = np.sign(scores["サイコロ"]) == -np.sign(scores["移動平均"])
    w.loc[strong & opposite, "サイコロ"] *= trend_damp
    num = (scores * w).sum(axis=1, min_count=1)
    den = (2 * w.where(scores.notna(), 0)).sum(axis=1)
    out = (num / den.replace(0, np.nan) * 100).round(0)
    return out
