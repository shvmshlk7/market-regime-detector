import os
import sys
import logging
import joblib

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data_loader import DataLoader
from src.feature_engineer import FeatureEngineer
from src.regime_detector import RegimeDetector
from src.config import ETF_TICKERS, MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HMMS_DIR = os.path.join(MODELS_DIR, "hmms")

def main():
    os.makedirs(HMMS_DIR, exist_ok=True)
    
    loader = DataLoader()
    logger.info("Loading combined data for all tickers...")
    combined = loader.get_combined_data()
    
    if "vix" not in combined.columns or combined["vix"].isna().all():
        logger.warning("Mocking macro data because FRED key is missing")
        import numpy as np
        n = len(combined)
        rng = np.random.default_rng(42)
        combined["vix"] = rng.uniform(10, 40, n)
        combined["yield_10y"] = rng.uniform(1.5, 4.5, n)
        combined["yield_2y"] = rng.uniform(0.5, 4.0, n)
        combined["cpi"] = 272.0 + np.cumsum(rng.uniform(0.0, 0.3, n))
        combined["unemployment"] = rng.uniform(3.5, 10.0, n)
        combined["yield_spread"] = combined["yield_10y"] - combined["yield_2y"]

    for ticker in ETF_TICKERS:
        ticker_lower = ticker.lower()
        if ticker_lower not in combined.columns:
            logger.error(f"Ticker {ticker} not found in dataset. Skipping.")
            continue
            
        logger.info(f"\n{'='*40}\nTraining HMM for {ticker}\n{'='*40}")
        
        try:
            fe = FeatureEngineer(equity_col=ticker_lower)
            X_raw = fe.compute_features(combined)
            X_norm = fe.fit_transform(X_raw)
            
            rd = RegimeDetector()
            rd.fit(X_norm)
            
            save_path = os.path.join(HMMS_DIR, f"hmm_{ticker}.pkl")
            rd.save(save_path)
            logger.info(f"Successfully saved {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to train HMM for {ticker}: {e}")

if __name__ == "__main__":
    main()
