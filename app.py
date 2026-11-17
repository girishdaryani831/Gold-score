import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Gold Score", page_icon="🪙", layout="centered")


@st.cache_data(ttl=300)  # reuse data for 5 minutes, then fetch fresh prices
def get_ohlcv(ticker):
    df = yf.download(ticker, period="2y", interval="1d",
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(subset=["Close"])


def soft(x):
    """Soft scaling: moderate moves score moderately, only extreme moves
    approach -1 or +1 (replaces the old hard cut-off)."""
    return float(np.tanh(x))


def rsi(s, p=14):
    d = s.diff()
    up = d.clip(lower=0).rolling(p).mean()
    dn = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - 100 / (1 + up / dn)


st.title("🪙 Gold Score")
st.caption("Research tool only, not financial advice.")

timeframe = st.radio("Timeframe", ["Short", "Medium", "Long"],
                     index=1, horizontal=True)
WEIGHTS = {
    "Short": {"Technicals": 0.45, "Volume & flows": 0.35, "Macro proxies": 0.20},
    "Medium": {"Technicals": 0.35, "Volume & flows": 0.25, "Macro proxies": 0.40},
    "Long": {"Technicals": 0.20, "Volume & flows": 0.15, "Macro proxies": 0.65},
}[timeframe]

if st.button("Refresh data"):
    st.cache_data.clear()

try:
    gold_df = get_ohlcv("GC=F")       # gold futures
    usd = get_ohlcv("DX-Y.NYB")["Close"]   # US dollar index
    y10 = get_ohlcv("^TNX")["Close"]       # US 10-year yield (proxy)
except Exception as e:
    st.error(f"Could not load market data: {e}")
    st.stop()

gold = gold_df["Close"]
if len(gold) < 220 or len(usd) < 70 or len(y10) < 70:
    st.error("Not enough data came back from Yahoo Finance. Try Refresh in a minute.")
    st.stop()

# ---------- Pillar 1: technicals ----------
sma50 = gold.rolling(50).mean().iloc[-1]
sma200 = gold.rolling(200).mean().iloc[-1]
last = gold.iloc[-1]
rsi_now = rsi(gold).iloc[-1]
mom60 = gold.pct_change(60).iloc[-1]

tech = {
    "Price vs 200-day average": soft((last / sma200 - 1) / 0.10),
    "50-day vs 200-day average": soft((sma50 / sma200 - 1) / 0.06),
    # RSI above 70 = overbought (bearish), below 30 = oversold (bullish)
    "RSI (14)": soft((50 - rsi_now) / 15) if (rsi_now > 70 or rsi_now < 30)
                else soft((rsi_now - 50) / 25),
    "60-day momentum": soft(mom60 / 0.12),
}

# ---------- Pillar 2: volume & flows ----------
vol = {}
try:
    d = gold_df[gold_df["Volume"] > 0].dropna(subset=["Volume"])
    if len(d) >= 120:
        ret20 = d["Close"].pct_change(20).iloc[-1]
        rel_vol = (d["Volume"].rolling(20).mean().iloc[-1]
                   / d["Volume"].rolling(100).mean().iloc[-1])
        # a move on rising volume counts for more than a move on thin volume
        vol["20-day move, volume-weighted"] = soft(
            ret20 / 0.04 * float(np.clip(rel_vol, 0.6, 1.5)))
        # on-balance volume: share of recent volume on up days vs down days
        direction = np.sign(d["Close"].diff())
        flow = ((direction * d["Volume"]).tail(20).sum()
                / d["Volume"].tail(20).sum())
        vol["On-balance volume flow (20d)"] = soft(flow / 0.25)
except Exception:
    pass

try:
    g = get_ohlcv("GLD")  # largest gold ETF: money-flow proxy for ETF demand
    g = g[g["Volume"] > 0]
    if len(g) >= 40:
        hl = (g["High"] - g["Low"]).replace(0, np.nan)
        mfm = ((g["Close"] - g["Low"]) - (g["High"] - g["Close"])) / hl
        cmf = ((mfm * g["Volume"]).tail(20).sum()
               / g["Volume"].tail(20).sum())
        vol["GLD ETF money flow (Chaikin, 20d)"] = soft(cmf / 0.15)
except Exception:
    pass

# ---------- Pillar 3: macro proxies (gold moves inversely to both) ----------
macro = {
    "US dollar trend (60d, inverse)": soft(-usd.pct_change(60).iloc[-1] / 0.05),
    "10-year yield trend (60d, inverse)": soft(-y10.pct_change(60).iloc[-1] / 0.15),
}

pillars = {"Technicals": tech, "Volume & flows": vol, "Macro proxies": macro}
pillars = {n: v for n, v in pillars.items() if v}   # drop a pillar with no data

w = {n: WEIGHTS[n] for n in pillars}
total_w = sum(w.values())
w = {n: x / total_w for n, x in w.items()}          # re-normalise to 100%

pillar_scores = {n: float(np.mean(list(v.values()))) for n, v in pillars.items()}
overall = sum(w[n] * pillar_scores[n] for n in pillars)
bullish_pct = 50 * (overall + 1)

all_scores = [s for v in pillars.values() for s in v.values()]
agree = float(np.mean([np.sign(s) == np.sign(overall) for s in all_scores]))
confidence = "High" if agree >= 0.75 else "Medium" if agree >= 0.5 else "Low"

# ---------- Display ----------
c1, c2, c3 = st.columns(3)
c1.metric("Bullish", f"{bullish_pct:.0f}%")
c2.metric("Confidence", confidence)
c3.metric("Gold price", f"{last:,.0f}")
st.progress(int(round(bullish_pct)))
st.caption(f"{agree * 100:.0f}% of indicators agree with the overall direction. "
           f"Data as of {gold.index[-1].date()}.")

strong = [s for s in pillar_scores.values() if abs(s) > 0.15]
if any(s > 0 for s in strong) and any(s < 0 for s in strong):
    st.warning("Mixed signals: the pillars disagree. Read the reasons below "
               "before drawing conclusions.")

st.subheader("Pillars")
st.bar_chart(pd.Series({n: 50 * (s + 1) for n, s in pillar_scores.items()},
                       name="Bullish %"))
st.caption(f"This score covers {len(pillars)} of 4 pillars. "
           "Sentiment and risk is not built yet.")

st.subheader("Why")
for name, inds in pillars.items():
    with st.expander(f"{name}: {50 * (pillar_scores[name] + 1):.0f}% bullish "
                     f"(weight {w[name] * 100:.0f}%)"):
        for k, v in inds.items():
            lean = "bullish" if v > 0.15 else "bearish" if v < -0.15 else "neutral"
            st.write(f"**{k}**: {v:+.2f} ({lean})")

st.subheader("Gold price with 50 and 200-day averages")
chart = pd.DataFrame({
    "Gold": gold,
    "50-day": gold.rolling(50).mean(),
    "200-day": gold.rolling(200).mean(),
}).dropna()
st.line_chart(chart)

with st.expander("How this score is calculated"):
    st.write(
        "Each indicator becomes a score between -1 (bearish) and +1 (bullish) "
        "using soft scaling, so moderate moves score moderately. Scores are "
        "averaged inside each pillar, then combined with the timeframe weights. "
        "Bullish % = 50 x (score + 1). Confidence reflects how many indicators "
        "agree. The ETF money-flow indicator is a proxy from GLD price and "
        "volume, not official ETF inflow data. Weights and scaling are first "
        "guesses and have not been backtested yet."
    )
