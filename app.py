import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Gold Score", page_icon="🪙", layout="centered")


@st.cache_data(ttl=300)  # reuse data for 5 minutes, then fetch fresh prices
def get(ticker):
    df = yf.download(ticker, period="2y", interval="1d",
                     progress=False, auto_adjust=True)
    s = df["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s.dropna()


def clip(x):
    return float(np.clip(x, -1, 1))


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
    "Short": {"Technicals": 0.7, "Macro proxies": 0.3},
    "Medium": {"Technicals": 0.5, "Macro proxies": 0.5},
    "Long": {"Technicals": 0.3, "Macro proxies": 0.7},
}[timeframe]

if st.button("Refresh data"):
    st.cache_data.clear()

try:
    gold = get("GC=F")       # gold futures
    usd = get("DX-Y.NYB")    # US dollar index
    y10 = get("^TNX")        # US 10-year yield (proxy until FRED real yields)
except Exception as e:
    st.error(f"Could not load market data: {e}")
    st.stop()

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
    "Price vs 200-day average": clip((last / sma200 - 1) / 0.10),
    "50-day vs 200-day average": clip((sma50 / sma200 - 1) / 0.05),
    "RSI (14)": clip((50 - rsi_now) / 20) if (rsi_now > 70 or rsi_now < 30)
                else clip((rsi_now - 50) / 40),
    "60-day momentum": clip(mom60 / 0.10),
}

# ---------- Pillar 2: macro proxies (gold moves inversely to both) ----------
macro = {
    "US dollar trend (60d, inverse)": clip(-usd.pct_change(60).iloc[-1] / 0.04),
    "10-year yield trend (60d, inverse)": clip(-y10.pct_change(60).iloc[-1] / 0.10),
}

pillars = {"Technicals": tech, "Macro proxies": macro}
pillar_scores = {n: float(np.mean(list(v.values()))) for n, v in pillars.items()}
overall = sum(WEIGHTS[n] * pillar_scores[n] for n in pillars)
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

st.subheader("Pillars")
st.bar_chart(pd.Series({n: 50 * (s + 1) for n, s in pillar_scores.items()},
                       name="Bullish %"))
st.caption("Volume/flows and sentiment pillars are not built yet, "
           "so this score covers 2 of 4 pillars.")

st.subheader("Why")
for name, inds in pillars.items():
    with st.expander(f"{name}: {50 * (pillar_scores[name] + 1):.0f}% bullish"):
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
        "Each indicator becomes a score from -1 (bearish) to +1 (bullish). "
        "Scores are averaged inside each pillar, then combined using the "
        "timeframe weights. Bullish % = 50 x (score + 1). Confidence reflects "
        "how many indicators agree. Weights and scaling are first guesses and "
        "have not been backtested yet."
    )
