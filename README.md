# 📈 Market Regime Detector + Portfolio Optimizer

A production-grade quantitative finance system that uses a **Gaussian Hidden Markov Model (HMM)** to detect live market regimes (Bull / Bear / Sideways) and automatically constructs an optimal portfolio using **Mean-Variance Optimization (Max Sharpe)**.

## 🖥️ Live Dashboard

The **COMMAND DECK** frontend provides a real-time terminal-style dashboard built with Next.js showing:
- Live regime detection with confidence probabilities
- Optimal portfolio target weights (10 ETFs)
- 30-day regime progression matrix
- Per-asset regime classification
- Macro vitals (VIX, Yield Spread, SPY RSI)
- Live news feed per asset
- USD / INR currency toggle

---

## 🏗️ Architecture

```
market-regime-detector/
├── app/               # FastAPI backend
│   └── main.py        # REST API endpoints
├── src/               # Core ML pipeline
│   ├── data_loader.py       # yfinance + FRED data fetching
│   ├── feature_engineer.py  # 15-feature extraction & normalization
│   ├── regime_detector.py   # Gaussian HMM (hmmlearn)
│   ├── portfolio_optimizer.py # Mean-Variance / Hedge optimizer
│   └── config.py            # Global settings
├── models/            # Trained HMM pkl files (11 models)
├── frontend/          # Next.js 16 dashboard (COMMAND DECK)
│   └── src/app/
│       ├── page.tsx   # Main dashboard component
│       └── globals.css
├── tests/             # Pytest test suite
├── scripts/           # Training & backtest scripts
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Clone & Setup Backend

```bash
git clone https://github.com/YOUR_USERNAME/market-regime-detector.git
cd market-regime-detector

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your FRED API key (optional — mocked if missing)
```

### 3. Run the Backend

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

API will be available at: `http://127.0.0.1:8000`

### 4. Run the Frontend

```bash
cd frontend
npm install
npm run dev
```

Dashboard at: `http://localhost:3000`

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Backend health check |
| GET | `/api/live-status` | Current regime + portfolio weights |
| GET | `/api/global-vitals` | VIX, yield spread, SPY RSI |
| GET | `/api/asset-details/{ticker}` | Algo reasoning + live news |

---

## 🧠 ML Pipeline

| Stage | Detail |
|-------|--------|
| **Data** | 10 ETFs via yfinance + FRED macro indicators |
| **Features** | 15 normalized features (volatility, momentum, macro, cross-asset) |
| **Model** | Gaussian HMM, 3 regimes, full covariance, 200 iterations |
| **Optimizer** | PyPortfolioOpt Max Sharpe / Hedge allocation |
| **Regimes** | Bull · Bear · Sideways |

---

## 📦 Tech Stack

**Backend:** Python · FastAPI · hmmlearn · PyPortfolioOpt · yfinance · pandas  
**Frontend:** Next.js 16 · TypeScript · Recharts · GSAP · Tailwind CSS  
**ML:** Gaussian HMM · StandardScaler · Mean-Variance Optimization

---

## ⚙️ Environment Variables

Copy `.env.example` to `.env` and fill in:

```
FRED_API_KEY=your_fred_api_key_here   # Optional — get free at fred.stlouisfed.org
```

---

## 📄 License

MIT License — feel free to use and adapt.
