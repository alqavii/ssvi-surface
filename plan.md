# Professional Quant Trading Dashboard

## AI Assistance Guide

🟢 **MINIMAL AI** - Core quant logic YOU write (math, models, analytics, pricing)
🟡 **MEDIUM AI** - Data processing/cleaning, validate AI suggestions critically
🔴 **FULL AI** - Infrastructure, boilerplate, plumbing (let AI handle)

## Architecture Overview

**Backend**: FastAPI with PostgreSQL, commodity futures, broker integration, screener logic
**Frontend**: Next.js 14 with TypeScript, TailwindCSS, Plotly, Leaflet
**Deployment**: Docker on Hetzner with nginx reverse proxy

---

## Phase 1: Infrastructure & Backend Foundation

### 1.1 Database Setup 🔴 FULL AI

- PostgreSQL with SQLAlchemy/asyncpg connection pooling
- Schema: `strategies`, `positions`, `screener_filters`, `commodity_data`, `geospatial_cache`
- Alembic migrations
- *Pure plumbing - let AI handle*

### 1.2 API Structure 🔴 FULL AI

- Refactor `apps/api/app.py` as proper FastAPI app (CORS, error handling)
- Endpoints: `/api/v1/{commodities,screener,strategies,broker,geospatial,surface}/*`
- *Routing boilerplate - full AI*

### 1.3 Data Adapters 🟡 MEDIUM AI

- `commodity_adapter.py` - fetch CL=F, NG=F, GC=F, SI=F, HG=F from yfinance
- `geospatial_adapter.py` - scrape MarineTraffic with rate limiting
- `broker_adapter.py` - IB/Alpaca paper trading wrapper
- Enhance `options_adapter.py` for futures options
- *Data cleaning - review carefully, understand data structures*

### 1.4 Core Engines

**screener_engine.py** 🟡 MEDIUM AI

- Filter options by IV percentile, volume, OI, bid-ask spread, DTE, moneyness
- Pre-built screens (high IV, cheap spreads)
- *Filtering logic - understand quant reasoning*

**strategy_executor.py** 🟡 MEDIUM AI

- Order validation (leg checks, spread validation)
- Broker API submission wrapper
- *Validation logic important, API calls can be AI-generated*

**risk_calculator.py** 🟢 MINIMAL AI ⚠️ KEY QUANT COMPONENT

- **Black-Scholes Greeks**: delta, gamma, vega, theta, rho for each position
- **Finite difference methods** for Greeks when closed-form unavailable
- **Portfolio-level Greek aggregation** (weighted by position size)
- **VaR calculation**: parametric (delta-normal) and historical simulation
- **Position sizing**: Kelly criterion, volatility-based sizing
- **Greeks by strike/expiry** for visualization
- *Core quant math YOU should master*

**implied_vol_engine.py** 🟢 MINIMAL AI (enhance existing)

- Improve `IV_surface.py` Newton-Raphson convergence
- Add SVI (Stochastic Volatility Inspired) parameterization
- Arbitrage-free interpolation checks
- *Pricing math - critical quant skill*

---

## Phase 2: Frontend Dashboard (Next.js)

### 2.1 Project Setup 🔴 FULL AI

- Next.js 14 with TypeScript in `apps/web`
- Install: TailwindCSS, shadcn/ui, Plotly.js, Leaflet, TanStack Query, Zustand
- API client with axios wrapper
- *Boilerplate setup*

### 2.2 Pages & Layout 🔴 FULL AI

- `/` - Landing with market overview
- `/surface` - Volatility surface (migrate from Streamlit)
- `/screener` - Multi-tab screener
- `/map` - Geospatial dashboard
- `/strategies` - Strategy builder
- `/portfolio` - Paper trading dashboard
- `/term-structure` - Futures curves, yield curve
- *UI scaffolding - full AI*

### 2.3 UI Components

**Chart Components** 🔴 FULL AI

- `VolatilitySurface3D.tsx`, `VolatilitySmile.tsx` (Plotly wrappers)
- `GreeksChart.tsx`, `TermStructureChart.tsx`
- *Visualization plumbing*

**Interactive Components** 🟡 MEDIUM AI

- `ScreenerTable.tsx` - sortable/filterable with export
- `StrategyBuilder.tsx` - drag-drop leg builder
- *Review UX logic, ensure data flows correctly*

**GeoMap.tsx** 🔴 FULL AI

- Leaflet map with tanker clusters, port markers, pipeline polylines
- *Mapping library integration*

### 2.4 State Management 🔴 FULL AI

- Zustand stores for ticker, date range, positions
- TanStack Query for API caching
- *State plumbing*

---

## Phase 3: Geospatial Features

### 3.1 Data Collection 🟡 MEDIUM AI

- Scrape MarineTraffic (VLCC, Suezmax, Aframax tankers)
- Static datasets: EIA ports, pipelines, refineries
- PostgreSQL caching with APScheduler refresh
- *Data parsing - understand energy market geography*

### 3.2 Map Visualization 🔴 FULL AI

- Dark mode Mapbox tiles, cluster markers
- Colored icons (green=ports, orange=refineries, blue=LNG)
- Popup tooltips with stats
- *Mapping UI*

### 3.3 Analytics Layer 🟡 MEDIUM AI

- Heatmap for tanker congestion
- Route analysis (MENA→Asia, US→EU)
- Correlation with Brent-WTI, Dubai-Brent spreads
- *Domain analysis - understand crude market flows*

---

## Phase 4: Screener & Strategy System

### 4.1 Options Screener Logic 🟡 MEDIUM AI

- IV rank/percentile calculations (historical comparison)
- Filters: volume, OI, spread width, DTE ranges
- *Understand why each filter matters for trading*

### 4.2 Commodity Screener Logic 🟢 MINIMAL AI

- **Backwardation/contango detection** (front month vs back months)
- **Roll yield calculation** (theoretical return from rolling futures)
- **Spread analysis**: crack spreads (crude vs gasoline), calendar spreads
- **Correlation matrices**: gold/silver ratio, oil/gas ratio
- *Key commodity trading concepts - write this yourself*

### 4.3 Strategy Templates 🟡 MEDIUM AI

- Vertical spreads, iron condors, butterflies, straddles/strangles
- Multi-leg validation (strike ordering, expiry matching)
- Simple backtest P&L (historical option prices)
- *Understand strategy payoffs, AI can help with plumbing*

### 4.4 Paper Trading Integration 🔴 FULL AI

- IB ib_insync or Alpaca API connection
- Order submission (limit/market), position tracking
- *Broker API boilerplate*

---

## Phase 5: Risk Management & Analytics

### 5.1 Portfolio Dashboard 🟡 MEDIUM AI

- Position table with live P&L (call backend risk calculator)
- Greek charts (AI can handle charting, YOU write calc logic)
- *UI layer over your risk math*

### 5.2 Term Structure Analysis 🟢 MINIMAL AI

- **Futures curve construction** (CL, NG, GC, SI, HG term structures)
- **Zero-coupon yield curve** from treasury bootstrapping (enhance `zero_rates.py`)
- **Implied forward rates** from zero curve
- **Calendar spread valuation** (fair value based on carry/convenience yield)
- *Fixed income math - core quant skill*

### 5.3 Alerts & Monitoring 🟡 MEDIUM AI

- Price alerts (threshold logic)
- Position limit checks
- IV percentile signals
- *Business logic - understand trigger conditions*

---

## Phase 6: Deployment & Polish

### 6.1 Docker Setup 🔴 FULL AI

- Dockerfiles (backend/frontend), docker-compose
- PostgreSQL container, nginx reverse proxy
- Environment variables for API keys
- *DevOps plumbing*

### 6.2 Hetzner Deployment 🔴 FULL AI

- VPS provisioning (CX32/CCX23 for 8GB RAM)
- Docker deployment, nginx config (frontend `/`, API `/api`)
- Let's Encrypt SSL, optional HTTP basic auth
- *Infrastructure*

### 6.3 CV-Worthy Polish 🔴 FULL AI

- Professional dark theme, responsive design
- Export reports (PDF), code docs
- GitHub README with diagrams, screenshots
- *Presentation layer*

---

## Summary: Where to Focus Your Quant Learning

### 🟢 MUST MASTER (Minimal AI) - Core Quant Skills

1. **Greeks Calculation** (`risk_calculator.py`) - delta, gamma, vega, theta, rho
2. **VaR Models** - parametric and historical simulation
3. **Implied Volatility** - Newton-Raphson, SVI fitting, arbitrage checks
4. **Term Structure Math** - bootstrapping, forward rates, calendar spreads
5. **Commodity Analytics** - backwardation/contango, roll yield, spread modeling

### 🟡 VALIDATE CAREFULLY (Medium AI) - Data & Domain Logic

1. **Data Adapters** - understand market data structures
2. **Screener Filters** - know why each metric matters for trading
3. **Strategy Validation** - understand option payoffs deeply
4. **Geospatial Analysis** - crude market geography and flows

### 🔴 LET AI HANDLE (Full AI) - Plumbing

1. **Database/API setup** - FastAPI routes, SQLAlchemy models
2. **Frontend scaffolding** - Next.js pages, component structure
3. **Docker/deployment** - infrastructure as code
4. **UI libraries** - Plotly/Leaflet integration

---

## Estimated Effort

- **Backend**: ~40 files, ~3000 LOC (60% full AI, 30% medium, 10% minimal)
- **Frontend**: ~60 files, ~5000 LOC (90% full AI, 10% medium)
- **Quant Core**: ~800 LOC YOU write/deeply review (Greeks, VaR, term structure, spreads)
- **Timeline**: 3-4 weeks (1 week on core quant logic, 2-3 weeks on plumbing/UI with AI)

This approach lets you **master the quant fundamentals** while building a **production-grade system** efficiently.