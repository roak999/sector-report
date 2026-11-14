"""セクター別テクニカル総合スコア（MVP：日本17業種）

起動:  streamlit run app.py
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import data
from indicators import DEFAULT_WEIGHTS, INDICATORS, composite, describe_all, macd, score_all
from sectors import REGIONS

st.set_page_config(page_title="セクター別テクニカルスコア", layout="wide")

# 赤＝売り寄り、黄＝中立、緑＝買い寄り
COLORSCALE = [[0.0, "#c0392b"], [0.25, "#f1948a"], [0.5, "#f7f7f7"], [0.75, "#82e0aa"], [1.0, "#1e8449"]]


# ---------------- データ取得 ----------------

@st.cache_data(ttl=60 * 60, show_spinner="株価データを取得中…")
def load_prices(tickers: tuple, period: str = "2y") -> dict:
    return data.load_prices(tickers, period)


@st.cache_data(ttl=60 * 60)
def compute_scores(_sector_data: dict, weights: tuple, cache_key: str):
    w = dict(weights)
    per_sector, desc, comp = {}, {}, {}
    for name, (ohlc, turnover) in _sector_data.items():
        sc = score_all(ohlc, turnover)
        per_sector[name] = sc
        desc[name] = describe_all(ohlc, turnover)
        comp[name] = composite(sc, w)
    return per_sector, desc, pd.DataFrame(comp)


# ---------------- サイドバー ----------------

st.sidebar.title("設定")
region = st.sidebar.selectbox("地域", list(REGIONS.keys()))
volume_mode = st.sidebar.radio("出来高に使うデータ", ["構成銘柄の売買代金合計", "ETF自体の出来高"],
                               help="業種別ETFは売買が少ないので、構成銘柄の合計がおすすめ")
st.sidebar.subheader("重み")
weights = {k: st.sidebar.slider(k, 0.0, 3.0, DEFAULT_WEIGHTS[k], 0.5) for k in INDICATORS}
history_days = st.sidebar.slider("推移の表示日数", 10, 120, 40, 5)
if st.sidebar.button("データを再取得"):
    st.cache_data.clear()

# ---------------- 計算 ----------------

sectors = REGIONS[region]
tickers = data.tickers_for(sectors)
prices = load_prices(tuple(tickers))
sector_data = data.build_sector_data(sectors, prices, volume_mode)
if not sector_data:
    st.error("株価データを取得できませんでした。ネット接続か銘柄コードを確認してください。")
    st.stop()

missing = [n for n in sectors if n not in sector_data]
if missing:
    st.warning("データ不足で除外: " + "、".join(missing))

key = f"{region}|{volume_mode}|{max(df.index.max() for df, _ in sector_data.values())}"
per_sector, descs, comp_hist = compute_scores(sector_data, tuple(sorted(weights.items())), key)
comp_hist = comp_hist.dropna(how="all")
as_of = comp_hist.index.max()

# ---------------- 画面 ----------------

st.title("セクター別テクニカル総合スコア")
st.caption(f"{region}｜基準日 {as_of:%Y-%m-%d}｜総合スコア −100（売り寄り）〜 +100（買い寄り）")

tab1, tab2, tab3 = st.tabs(["📊 今日のシグナル", "🗓 総合スコアの推移", "🔍 セクター詳細"])

# --- タブ1: 今日のヒートマップ ---
# セルの色＝点数（-2〜+2）、セルの文字＝実際の数値
HEADERS = {"総合": "総合", "サイコロ": "サイコロ<br>(12日勝敗)", "出来高": "出来高<br>(20日平均比)",
           "MACD": "MACD<br>(クロス/ヒスト)", "ローソク足": "ローソク足<br>(パターン)",
           "移動平均": "移動平均<br>(25日線乖離)"}

with tab1:
    rows, texts = [], []
    for name, sc in per_sector.items():
        last = sc.loc[:as_of].iloc[-1]
        total = comp_hist[name].loc[:as_of].iloc[-1]
        rows.append({"セクター": name, "総合": total, **last.to_dict()})
        texts.append({"セクター": name, "総合": f"{total + 0:+.0f}",
                      **descs[name].loc[:as_of].iloc[-1].to_dict()})
    today = pd.DataFrame(rows).set_index("セクター").sort_values("総合", ascending=False)
    text = pd.DataFrame(texts).set_index("セクター").loc[today.index]

    cols = ["総合"] + INDICATORS
    z = today[cols].copy()
    z[INDICATORS] = z[INDICATORS] * 50  # 総合(±100)と各指標(±2)の色の尺度をそろえる
    score_txt = today[cols].apply(lambda col: col.map(lambda v: "-" if pd.isna(v) else f"{v + 0:+.0f}"))

    fig = go.Figure(go.Heatmap(
        z=z.values, x=[HEADERS[c] for c in cols], y=today.index,
        text=text[cols].values, customdata=score_txt.values, texttemplate="%{text}",
        textfont=dict(size=13), colorscale=COLORSCALE, zmin=-100, zmax=100, showscale=False,
        xgap=2, ygap=2, hovertemplate="%{y}｜%{x}<br>%{text}（点数 %{customdata}）<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(side="top")
    fig.update_layout(height=40 * len(today) + 100, margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("色＝点数（濃い緑 +2 〜 濃い赤 −2）。"
               "出来高の↑↓は当日の株価の向き。MACDのGC/DCはゴールデン/デッドクロス、%はヒストグラムの株価比。"
               "移動平均は「25日線と75日線の位置・25日線の向き・株価の25日線からの乖離率」。")

    c1, c2 = st.columns(2)
    c1.markdown("**買い寄り上位**  \n" + "  \n".join(f"{n}（{v:+.0f}）" for n, v in today["総合"].head(3).items()))
    c2.markdown("**売り寄り上位**  \n" + "  \n".join(f"{n}（{v:+.0f}）" for n, v in today["総合"].tail(3)[::-1].items()))

# --- タブ2: 推移ヒートマップ ---
with tab2:
    h = comp_hist.tail(history_days)
    order = h.iloc[-1].sort_values(ascending=False).index
    h = h[order]
    fig2 = go.Figure(go.Heatmap(
        z=h.T.values, x=h.index.strftime("%m/%d"), y=order,
        colorscale=COLORSCALE, zmin=-100, zmax=100, xgap=1, ygap=1,
        colorbar=dict(title="総合"), hovertemplate="%{y}｜%{x}: %{z:+.0f}<extra></extra>"))
    fig2.update_yaxes(autorange="reversed")
    fig2.update_layout(height=40 * len(order) + 80, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig2, width="stretch")
    st.caption("右端が最新日。色が赤→緑に変わったところが、買いシグナルが強まったタイミングです。")

# --- タブ3: 詳細チャート ---
with tab3:
    name = st.selectbox("セクター", list(today.index))
    ohlc, turnover = sector_data[name]
    d = ohlc.tail(180)
    c = ohlc["Close"]
    ma25, ma75 = c.rolling(25).mean().loc[d.index], c.rolling(75).mean().loc[d.index]
    line, sig, hist = (s.loc[d.index] for s in macd(c))

    fig3 = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2],
                         vertical_spacing=0.03)
    fig3.add_trace(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
                                  name="価格", increasing_line_color="#c0392b",
                                  decreasing_line_color="#1f618d"), 1, 1)
    fig3.add_trace(go.Scatter(x=d.index, y=ma25, name="25日線", line=dict(width=1.2, color="#7d3c98")), 1, 1)
    fig3.add_trace(go.Scatter(x=d.index, y=ma75, name="75日線", line=dict(width=1.2, color="#ca6f1e")), 1, 1)
    fig3.add_trace(go.Bar(x=d.index, y=turnover.reindex(d.index) / 1e8, name="売買代金(億円)",
                          marker_color="#99a3a4"), 2, 1)
    fig3.add_trace(go.Bar(x=d.index, y=hist, name="MACDヒスト",
                          marker_color=["#1e8449" if v > 0 else "#c0392b" for v in hist]), 3, 1)
    fig3.add_trace(go.Scatter(x=d.index, y=line, name="MACD", line=dict(width=1, color="#2874a6")), 3, 1)
    fig3.add_trace(go.Scatter(x=d.index, y=sig, name="シグナル", line=dict(width=1, color="#ca6f1e")), 3, 1)
    fig3.update_layout(height=720, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=20, b=10),
                       legend=dict(orientation="h", y=1.02))
    fig3.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    st.plotly_chart(fig3, width="stretch")

    st.markdown("**直近10日の内訳**（色＝点数）")
    sc10 = per_sector[name].tail(10).copy()
    sc10.insert(0, "総合", comp_hist[name].reindex(sc10.index) / 50)  # ±100 → ±2 に換算して色付け
    tx10 = descs[name].reindex(sc10.index).copy()
    tx10.insert(0, "総合", comp_hist[name].reindex(sc10.index).map(lambda v: "-" if pd.isna(v) else f"{v + 0:+.0f}"))
    sc10.index = tx10.index = sc10.index.strftime("%Y-%m-%d")
    sc10, tx10 = sc10.iloc[::-1], tx10.iloc[::-1]

    def cell_color(v):
        if pd.isna(v):
            return ""
        v = max(-2, min(2, v))
        stops = {-2: "#c0392b", -1: "#f1948a", 0: "#f7f7f7", 1: "#82e0aa", 2: "#1e8449"}
        bg = stops[int(round(v))]
        fg = "white" if abs(round(v)) == 2 else "black"
        return f"background-color: {bg}; color: {fg}"

    st.dataframe(tx10.style.apply(lambda _: sc10.map(cell_color), axis=None), width="stretch")

st.caption("※ テクニカル指標による参考情報です。投資判断はご自身の責任で行ってください。")
