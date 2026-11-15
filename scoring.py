"""スコア計算の共通部品（0〜100点に統一）。

方針
  * 単純な閾値ではなく、「自分自身の過去250営業日の中での順位（パーセンタイル）」で標準化する
    （prank）。市場ごと・銘柄ごとにボラティリティの水準が違っても同じ物差しで比べられる。
  * 同じ情報を複数の指標で二重に数えないよう、各スコアの構成要素は役割を分けている
    （例：トレンド＝位置関係、モメンタム＝変化の勢い、ボラティリティ＝ブレの大きさ）。
  * 配点は初期仮説。重みは各ビルダーの WEIGHTS で一元管理し、バックテストで見直せる。
"""
import numpy as np
import pandas as pd

RANK_WINDOW = 250
RANK_MIN = 120


def prank(s: pd.Series, win: int = RANK_WINDOW, minp: int = RANK_MIN) -> pd.Series:
    """直近 win 日の中での今日の値の順位（0〜1）。履歴が minp 日未満なら欠損。"""
    return s.rolling(win, min_periods=minp).rank(pct=True)


def clip01(x):
    return x.clip(0, 1) if hasattr(x, "clip") else max(0.0, min(1.0, x))


def weighted_score(parts: dict, weights: dict):
    """parts: 名前→0〜1の値(Series/float)。欠損の要素は除いて重みを再配分し、0〜100に換算。
    戻り値: (総合0〜100, 実際に使えた重みの割合0〜1)"""
    names = list(weights)
    df = pd.concat({k: pd.Series(parts[k]) if not isinstance(parts[k], pd.Series) else parts[k]
                    for k in names}, axis=1)
    w = pd.Series(weights, dtype=float)
    valid = df.notna()
    num = (df.fillna(0) * w).sum(axis=1)
    den = (valid * w).sum(axis=1)
    score = (num / den.replace(0, np.nan) * 100)
    return score, den / w.sum()


def num(v, nd=2):
    if v is None:
        return None
    try:
        if pd.isna(v) or np.isinf(v):
            return None
    except TypeError:
        return None
    return round(float(v), nd)
