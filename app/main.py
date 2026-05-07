from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
import pandas as pd
import numpy as np
import logging
import os
import sys
import yfinance as yf
import os
import sys

# Add project root to sys.path so we can import src
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data_loader import DataLoader
from src.feature_engineer import FeatureEngineer
from src.regime_detector import RegimeDetector
from src.portfolio_optimizer import PortfolioOptimizer
from src.config import HMM_MODEL_PATH, ETF_TICKERS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Market Regime API")

# Allow CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global caches so we don't re-download memory on every request
_CACHE: Dict[str, Any] = {
    "live_data": None,
    "last_fetched": None
}

def get_live_market_data():
    """Fetches the latest data through the pipeline."""
    # We load standard pipeline
    loader = DataLoader()
    
    try:
        combined = loader.get_combined_data()
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        raise HTTPException(status_code=500, detail="Failed to load market data")

    # If no FRED key, mock macro data exactly as we did in backtester
    if "vix" not in combined.columns or combined["vix"].isna().all():
        logger.warning("Mocking macro data because FRED key is missing")
        n = len(combined)
        rng = np.random.default_rng(42)
        combined["vix"] = rng.uniform(10, 40, n)
        combined["yield_10y"] = rng.uniform(1.5, 4.5, n)
        combined["yield_2y"] = rng.uniform(0.5, 4.0, n)
        combined["cpi"] = 272.0 + np.cumsum(rng.uniform(0.0, 0.3, n))
        combined["unemployment"] = rng.uniform(3.5, 10.0, n)
        combined["yield_spread"] = combined["yield_10y"] - combined["yield_2y"]

    fe = FeatureEngineer()
    
    try:
        X_raw = fe.compute_features(combined)
        X_norm = fe.fit_transform(X_raw)
    except Exception as e:
        logger.error(f"Failed feature engineering: {e}")
        raise HTTPException(status_code=500, detail="Failed feature engineering")
        
    prices = loader.fetch_etf_data()
    prices.columns = [c.lower() for c in prices.columns]
    prices_aligned = prices.loc[X_norm.index]
    valid_tickers = [c for c in prices_aligned.columns if str(c).upper() in ETF_TICKERS]
    prices_aligned = prices_aligned[valid_tickers]
    
    return prices_aligned, X_norm, combined

@app.get("/api/live-status")
def get_live_status():
    """
    Returns the CURRENT market regime and today's optimal target portfolio.
    This effectively runs our model for 'live execution'.
    """
    if not os.path.exists(HMM_MODEL_PATH):
        raise HTTPException(status_code=500, detail="HMM Model not trained.")
        
    prices, X_norm, combined = get_live_market_data()
    
    # 1. Predict Regime
    rd = RegimeDetector.load(HMM_MODEL_PATH)
    regimes = rd.predict(X_norm)
    current_regime = regimes.iloc[-1]
    
    # 2. Predict Probabilities (Confidence)
    try:
        proba_series = rd.predict_proba(X_norm).iloc[-1]
        probabilities = proba_series.to_dict()
        
        # 3. Reconcile smoothed regime with current probabilities
        # Logic: If the smoothed regime has very low confidence (< 10%) but another regime 
        # is dominant (> 50%), we switch the reported label to the dominant one 
        # for dashboard consistency.
        dominant_regime = proba_series.idxmax()
        if probabilities.get(current_regime, 0) < 0.1 and probabilities.get(dominant_regime, 0) > 0.5:
            logger.info(f"Reconciling regime: smoothed={current_regime} -> raw={dominant_regime}")
            current_regime = dominant_regime

    except Exception as e:
        logger.warning(f"Failed to compute probabilities: {e}")
        probabilities = {current_regime: 1.0}

    # 4. Get Target Portfolio Allocation
    po = PortfolioOptimizer()
    
    # We want today's optimal weights. Get the recent price window for the lookback period.
    window_prices = prices.iloc[-po.lookback_days:]
    current_weights = po.optimize(window_prices, current_regime)
    
    # Prepare historical chart data (e.g. last 30 days)
    last_30_regimes = regimes.tail(30).to_frame(name="regime")
    last_30_prices = prices['spy'].tail(30)
    
    # Combine and format
    history_df = last_30_regimes.join(last_30_prices)
    history_df = history_df.reset_index()
    history_df['date'] = history_df['date'].dt.strftime('%Y-%m-%d')
    history = history_df.to_dict(orient="records")
    
    # Clean weights to remove negligible ones (like 1e-12)
    cleaned_weights = {k: round(v, 4) for k, v in current_weights.items() if v > 0.001}
    
    # 4. Predict Individual Asset Regimes
    asset_regimes = {}
    from src.config import MODELS_DIR
    for ticker in cleaned_weights.keys():
        ticker_lower = ticker.lower()
        model_path = os.path.join(MODELS_DIR, "hmms", f"hmm_{ticker}.pkl")
        if os.path.exists(model_path) and ticker_lower in combined.columns:
            try:
                fe_asset = FeatureEngineer(equity_col=ticker_lower)
                X_raw_asset = fe_asset.compute_features(combined)
                X_norm_asset = fe_asset.fit_transform(X_raw_asset)
                
                rd_asset = RegimeDetector.load(model_path)
                asset_reg_series = rd_asset.predict(X_norm_asset)
                asset_regimes[ticker] = asset_reg_series.iloc[-1]
            except Exception as e:
                logger.warning(f"Failed to detect regime for {ticker}: {e}")
                asset_regimes[ticker] = "Unknown"
        else:
            asset_regimes[ticker] = "Unknown"
            
    # Get latest exact prices to prove to user it's live data
    latest_prices = prices.iloc[-1].round(2).to_dict()
    
    # Fetch USD/INR exchange rate
    try:
        usd_inr_rate = yf.Ticker("INR=X").fast_info.last_price
    except Exception as e:
        logger.warning(f"Failed to fetch INR rate: {e}")
        usd_inr_rate = 83.50 # Fallback
        
    return {
        "date": str(regimes.index[-1].date()),
        "regime": current_regime,
        "probabilities": probabilities,
        "target_weights": cleaned_weights,
        "latest_prices": latest_prices,
        "historical_regimes_30d": history,
        "asset_regimes": asset_regimes,
        "usd_inr_rate": usd_inr_rate
    }

@app.get("/api/global-vitals")
def get_global_vitals():
    """
    Returns high-level macro 'health' indicators for the dashboard sidebar.
    XIV was delisted in 2018; replaced vol_ratio with VXX 5d/30d vol ratio as fear proxy.
    """
    try:
        # XIV is delisted — only fetch live tickers
        tickers = ["^VIX", "^TNX", "^IRX", "SPY", "VXX"]
        data = yf.download(tickers, period="2mo", progress=False)["Close"].dropna(how="all")

        vix = float(data["^VIX"].dropna().iloc[-1])
        yield_spread = float(data["^TNX"].dropna().iloc[-1]) - float(data["^IRX"].dropna().iloc[-1])

        spy = data["SPY"].dropna()
        delta = spy.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        last_gain = float(gain.iloc[-1])
        last_loss = float(loss.iloc[-1])
        rsi = 100 - (100 / (1 + last_gain / last_loss)) if last_loss != 0 else 50.0

        mom_1m = float((spy.iloc[-1] / spy.iloc[-21]) - 1)

        # Vol ratio: VXX 5-day realised vol / 30-day realised vol (short-term fear vs baseline)
        vxx = data["VXX"].dropna()
        vxx_rets = vxx.pct_change().dropna()
        vol_5d  = float(vxx_rets.tail(5).std()  * (252 ** 0.5))
        vol_30d = float(vxx_rets.tail(30).std() * (252 ** 0.5))
        vol_ratio = round(vol_5d / vol_30d, 2) if vol_30d > 0 else 1.0

        return {
            "vix": round(vix, 2),
            "yield_spread": round(yield_spread, 3),
            "spy_rsi": round(rsi, 2),
            "spy_mom_1m": round(mom_1m * 100, 2),
            "vol_ratio": vol_ratio,
        }
    except Exception as e:
        logger.error(f"Vitals failure: {e}")
        return {"vix": 20.0, "yield_spread": 0.5, "spy_rsi": 50.0, "spy_mom_1m": 0.0, "vol_ratio": 1.0}


@app.get("/api/asset-details/{ticker}")
def get_asset_details(ticker: str):
    """
    Returns the algorithmic reason and live news for a specific asset.
    """
    ticker_upper = ticker.upper()
    ticker_lower = ticker.lower()
    
    # 1. Generate Algorithmic Reason
    reason = f"No algorithmic data available for {ticker_upper}."
    try:
        _, _, combined = get_live_market_data()
        
        if ticker_lower in combined.columns:
            prices_asset = combined[ticker_lower]
            
            # Simple direct calculation for 'Reason' (more robust than scaled features)
            # 1-month momentum (21 trading days)
            mom_1m = (prices_asset.iloc[-1] / prices_asset.iloc[-21]) - 1
            
            # 3-month realized volatility (63 trading days)
            returns = np.log(prices_asset / prices_asset.shift(1))
            vol_3m = returns.tail(63).std() * np.sqrt(252)
            
            trend = "positive" if mom_1m > 0 else "negative"
            vol_level = "high" if vol_3m > 0.20 else "low"
            
            reason = f"{ticker_upper} exhibits a {trend} 1-month momentum ({mom_1m*100:.1f}%) paired with {vol_level} trailing three-month volatility ({vol_3m*100:.1f}%)."
        else:
            reason = f"Ticker {ticker_upper} not found in the primary dataset."
    except Exception as e:
        logger.warning(f"Failed to generate reason for {ticker}: {e}")

    # 2. Fetch Live News
    news_items = []
    try:
        raw_news = yf.Ticker(ticker_upper).news
        for item in raw_news[:3]:
            content = item.get('content', {})
            title = content.get('title', 'Headline Unavailable')
            url = content.get('clickThroughUrl', {}).get('url', '')
            publisher = content.get('provider', {}).get('displayName', 'Yahoo Finance')
            pub_date = content.get('pubDate', '')
            news_items.append({
                "title": title,
                "url": url,
                "publisher": publisher,
                "pubDate": pub_date
            })
    except Exception as e:
        logger.warning(f"Failed to fetch news for {ticker}: {e}")
        
    return {
        "ticker": ticker_upper,
        "reason": reason,
        "news": news_items
    }

@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "FastAPI is running the ML Pipeline"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
