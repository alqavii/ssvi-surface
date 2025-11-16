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


### 1.2 API Structure 🔴 FULL AI

- Refactor `apps/api/app.py` as proper FastAPI app (CORS, error handling)
- **Comprehensive endpoints**: 
  - `/api/v1/rates/*` - Risk-free rates, treasury yields, SOFR
  - `/api/v1/options/*` - Options chains, pricing, Greeks, IV surfaces
  - `/api/v1/volatility/*` - Volatility models, calibration, surfaces
  - `/api/v1/strategies/*` - Strategy analysis, backtesting, execution
  - `/api/v1/portfolio/*` - Portfolio analytics, risk metrics, performance
  - `/api/v1/screener/*` - Options screening, filters, unusual activity
  - **Energy Trading Endpoints**:
    - `/api/v1/energy/futures/*` - WTI, Brent, NG, refined product futures
    - `/api/v1/energy/spreads/*` - Crack spreads, calendar spreads, basis analysis
    - `/api/v1/energy/fundamentals/*` - EIA data, OPEC+ production, storage
    - `/api/v1/energy/weather/*` - Hurricane tracking, HDD/CDD analysis
    - `/api/v1/energy/options/*` - Energy futures options chains
  - **Geospatial Energy Endpoints**:
    - `/api/v1/geospatial/tankers/*` - Tanker tracking, routes, congestion
    - `/api/v1/geospatial/pipelines/*` - Pipeline flows, capacity, bottlenecks
    - `/api/v1/geospatial/ports/*` - Port capacity, loading/unloading delays
    - `/api/v1/geospatial/storage/*` - Storage levels, capacity utilization
  - `/api/v1/broker/*` - Order management, position tracking
- *Routing boilerplate - full AI*

### 1.3 Data Adapters 🟡 MEDIUM AI

- **`energy_commodity_adapter.py`** - Advanced energy futures data collection
  - **Crude Oil**: WTI (CL), Brent (BZ), Dubai, Oman futures
  - **Natural Gas**: Henry Hub (NG), TTF, JKM LNG futures
  - **Refined Products**: RBOB (RB), Heating Oil (HO), Gasoil futures
  - **Energy Options**: Crude, NG, refined product options chains
  - **Basis Spreads**: WTI-Brent, NG-Calendar spreads
  - **Crack Spreads**: 3-2-1, 2-1-1 crack spreads
- **`geospatial_adapter.py`** - Marine traffic and energy infrastructure
  - **Tanker Tracking**: VLCC, Suezmax, Aframax positions and routes
  - **Pipeline Flows**: Major crude/gas pipeline monitoring
  - **Storage Data**: Cushing, SPR, LNG storage levels
  - **Port Congestion**: Loading/unloading delays and capacity
- **`fundamental_adapter.py`** - Energy market fundamentals
  - **EIA Data**: Weekly petroleum status, natural gas storage
  - **OPEC+ Production**: Monthly production quotas and compliance
  - **Weather Data**: Hurricane tracking, heating/cooling degree days
  - **Economic Indicators**: GDP, industrial production, energy intensity
- **`broker_adapter.py`** - IB/Alpaca paper trading wrapper
- **Enhanced `options_adapter.py`** - comprehensive options data collection
  - Real-time options chains (strikes, expiries, Greeks, IV)
  - Historical options data for backtesting
  - Options volume/OI tracking
  - Dividend-adjusted pricing
- *Data cleaning - review carefully, understand data structures*

### 1.4 Core Options Engines 🟢 MINIMAL AI ⚠️ KEY QUANT COMPONENTS

**heston_model.py** 🟢 MINIMAL AI ⚠️ CRITICAL QUANT COMPONENT

- **Heston Stochastic Volatility Model** implementation
- **Characteristic function** for European options pricing
- **Fast Fourier Transform (FFT)** for efficient pricing
- **Calibration** to market data using least squares optimization
- **Volatility surface fitting** with mean reversion parameters
- **Greeks calculation** under stochastic volatility
- **Monte Carlo simulation** for American/exotic options
- *Advanced derivatives pricing - master this deeply*

**black_scholes_engine.py** 🟢 MINIMAL AI

- **Black-Scholes-Merton** with dividend yield
- **Closed-form Greeks**: delta, gamma, vega, theta, rho
- **Finite difference Greeks** for complex payoffs
- **Implied volatility** calculation with Newton-Raphson
- **American options** approximation (Barone-Adesi-Whaley)
- **Binary options** pricing
- *Foundation pricing model - understand thoroughly*

**volatility_models.py** 🟢 MINIMAL AI

- **SVI (Stochastic Volatility Inspired)** parameterization
- **SABR model** for interest rate derivatives
- **Local volatility** surface construction
- **Volatility smile/skew** analysis
- **Term structure** of volatility
- **Arbitrage-free interpolation** methods
- *Volatility modeling - core quant skill*

**options_pricing_engine.py** 🟢 MINIMAL AI

- **Multi-model pricing** (BS, Heston, SABR)
- **Model selection** based on market conditions
- **Exotic options**: barriers, Asians, lookbacks
- **Path-dependent options** with Monte Carlo
- **Dividend modeling** and early exercise
- **Risk-neutral pricing** framework
- *Advanced options pricing - critical for trading*

**greeks_calculator.py** 🟢 MINIMAL AI

- **Portfolio Greeks** aggregation
- **Cross-Greeks**: vanna, volga, charm
- **Higher-order Greeks** for risk management
- **Scenario analysis** (parallel shifts, twists)
- **Greeks hedging** strategies
- **Real-time Greeks** updates
- *Risk management foundation*

### 1.5 Options Analytics & Screening 🟡 MEDIUM AI

**screener_engine.py** 🟡 MEDIUM AI

- **Advanced filters**: IV percentile, volume, OI, bid-ask spread, DTE, moneyness
- **Pre-built screens**: high IV, cheap spreads, earnings plays, volatility crush
- **Custom filter builder** with logical operators
- **Options flow** analysis and unusual activity detection
- **IV rank/percentile** calculations
- **Skew analysis** and relative value identification
- *Filtering logic - understand quant reasoning*

**strategy_analyzer.py** 🟡 MEDIUM AI

- **Multi-leg strategy** validation and analysis
- **P&L diagrams** and breakeven calculations
- **Risk metrics**: max loss, max profit, probability of profit
- **Strategy Greeks** and risk profiles
- **Backtesting framework** for strategies
- **Strategy ranking** and optimization
- *Strategy analysis - understand payoffs deeply*

**strategy_executor.py** 🟡 MEDIUM AI

- **Order validation** (leg checks, spread validation, margin requirements)
- **Broker API submission** wrapper
- **Position management** and monitoring
- **Risk limits** enforcement
- **Slippage modeling** and execution analysis
- *Validation logic important, API calls can be AI-generated*

### 1.6 Energy Commodities Analytics 🟢 MINIMAL AI ⚠️ KEY ENERGY TRADING COMPONENTS

**energy_spread_analyzer.py** 🟢 MINIMAL AI ⚠️ CRITICAL ENERGY COMPONENT

- **Crack Spread Analysis**: 3-2-1, 2-1-1, gasoline-heating oil spreads
- **Calendar Spread Analysis**: Front month vs back month spreads
- **Basis Spread Analysis**: WTI-Brent, NG-Calendar, location spreads
- **Storage Economics**: Contango/backwardation analysis, storage arbitrage
- **Seasonal Analysis**: Heating/cooling season patterns, hurricane impacts
- **Spread Regression Models**: Mean reversion, cointegration analysis
- *Energy spread trading - core skill for energy traders*

**energy_fundamental_analyzer.py** 🟢 MINIMAL AI

- **Supply-Demand Models**: EIA data integration, OPEC+ analysis
- **Storage Analysis**: Cushing levels, SPR releases, LNG storage
- **Weather Impact Models**: Hurricane tracking, HDD/CDD analysis
- **Economic Indicators**: Energy intensity, industrial production correlation
- **Geopolitical Risk**: Supply disruptions, sanctions impact modeling
- **Fundamental Valuation**: Fair value models based on fundamentals
- *Energy fundamentals - essential for fundamental trading*

**energy_volatility_analyzer.py** 🟢 MINIMAL AI

- **Energy Volatility Surfaces**: Crude, NG, refined product IV surfaces
- **Volatility Term Structure**: Energy-specific volatility patterns
- **Volatility Regimes**: High/low volatility period identification
- **Volatility Forecasting**: GARCH models for energy markets
- **Volatility Trading**: Volatility arbitrage opportunities
- *Energy volatility - critical for options trading*

**energy_geospatial_analyzer.py** 🟡 MEDIUM AI

- **Tanker Flow Analysis**: Crude flows, route optimization, congestion
- **Pipeline Analysis**: Flow rates, capacity utilization, bottlenecks
- **Port Congestion**: Loading/unloading delays, capacity constraints
- **Geographic Arbitrage**: Location-based trading opportunities
- **Supply Chain Risk**: Infrastructure disruption impact modeling
- *Geospatial analysis - unique edge for energy trading*

### 1.7 Risk Management & Portfolio Analytics 🟢 MINIMAL AI

**risk_calculator.py** 🟢 MINIMAL AI ⚠️ KEY QUANT COMPONENT

- **Portfolio VaR**: parametric (delta-normal) and historical simulation
- **Monte Carlo VaR** with full revaluation
- **Stress testing** scenarios (energy-specific: hurricanes, OPEC cuts)
- **Correlation analysis** and diversification metrics
- **Position sizing**: Kelly criterion, volatility-based sizing
- **Risk budgeting** and allocation
- **Real-time risk monitoring**
- *Core risk math YOU should master*

**portfolio_engine.py** 🟢 MINIMAL AI

- **Portfolio construction** and optimization
- **Factor models** and attribution analysis
- **Performance analytics** and benchmarking
- **Drawdown analysis** and risk-adjusted returns
- **Rebalancing** strategies
- *Portfolio theory - understand deeply*

---

## Phase 2: Frontend Dashboard (Next.js)

### 2.1 Project Setup 🔴 FULL AI

- Next.js 14 with TypeScript in `apps/web`
- Install: TailwindCSS, shadcn/ui, Plotly.js, Leaflet, TanStack Query, Zustand
- API client with axios wrapper
- *Boilerplate setup*

### 2.2 Pages & Layout 🔴 FULL AI

- `/` - Landing with market overview and energy flow dashboard
- `/options` - Comprehensive options dashboard
  - `/options/chains` - Real-time options chains with Greeks
  - `/options/volatility` - IV surfaces, smiles, term structure
  - `/options/pricing` - Model comparison, calibration tools
  - `/options/analytics` - Greeks analysis, scenario modeling
- **Energy Trading Dashboards**:
  - `/energy/overview` - Energy market overview with key spreads
  - `/energy/crude` - WTI/Brent analysis, crack spreads, storage
  - `/energy/natural-gas` - NG analysis, weather impact, storage
  - `/energy/refined` - Gasoline/heating oil, crack spread analysis
  - `/energy/spreads` - Comprehensive spread analysis and trading
  - `/energy/fundamentals` - EIA data, OPEC+ production, supply/demand
  - `/energy/weather` - Hurricane tracking, HDD/CDD analysis
- **Geospatial Energy**:
  - `/geospatial/tankers` - Tanker tracking and crude flow analysis
  - `/geospatial/pipelines` - Pipeline flows and capacity analysis
  - `/geospatial/ports` - Port congestion and loading analysis
  - `/geospatial/storage` - Storage levels and capacity utilization
- `/surface` - Advanced volatility surface visualization
- `/screener` - Multi-tab options screener with custom filters
- `/strategies` - Strategy builder with P&L diagrams
- `/portfolio` - Portfolio analytics with risk metrics
- `/backtesting` - Strategy backtesting and performance analysis
- `/risk` - Real-time risk monitoring and VaR analysis
- `/term-structure` - Futures curves, yield curves, commodity spreads
- *UI scaffolding - full AI*

### 2.3 UI Components

**Chart Components** 🔴 FULL AI

- **Options Visualization**:
  - `VolatilitySurface3D.tsx` - Interactive 3D IV surface with Plotly
  - `VolatilitySmile.tsx` - IV smile/skew analysis
  - `VolatilityTermStructure.tsx` - IV term structure curves
  - `OptionsChain.tsx` - Real-time options chain with Greeks
  - `GreeksChart.tsx` - Greeks visualization by strike/expiry
  - `P&LDiagram.tsx` - Strategy P&L diagrams
  - `RiskProfile.tsx` - Portfolio risk visualization
- **Energy Trading Charts**:
  - `CrackSpreadChart.tsx` - 3-2-1, 2-1-1 crack spread analysis
  - `CalendarSpreadChart.tsx` - Front/back month spread analysis
  - `BasisSpreadChart.tsx` - WTI-Brent, location spread analysis
  - `StorageChart.tsx` - Cushing, SPR, LNG storage levels
  - `WeatherImpactChart.tsx` - Hurricane tracking, HDD/CDD analysis
  - `EnergyFlowChart.tsx` - Tanker routes, pipeline flows
  - `FundamentalChart.tsx` - EIA data, OPEC+ production
- **Market Data Charts**:
  - `TermStructureChart.tsx` - Yield curves, futures curves
  - `PriceChart.tsx` - OHLCV charts with technical indicators
  - `SpreadChart.tsx` - Commodity spreads, calendar spreads
- **Analytics Charts**:
  - `VaRChart.tsx` - Value at Risk visualization
  - `PerformanceChart.tsx` - Portfolio performance attribution
  - `CorrelationMatrix.tsx` - Asset correlation heatmap
- *Visualization plumbing*

**Interactive Components** 🟡 MEDIUM AI

- **Options Components**:
  - `OptionsScreener.tsx` - Advanced filtering with custom criteria
  - `StrategyBuilder.tsx` - Drag-drop multi-leg strategy builder
  - `OptionsCalculator.tsx` - Interactive pricing calculator
  - `VolatilityCalibrator.tsx` - Model calibration interface
  - `GreeksAnalyzer.tsx` - Greeks analysis and scenario modeling
- **Energy Trading Components**:
  - `SpreadAnalyzer.tsx` - Crack spread, calendar spread analysis
  - `FundamentalAnalyzer.tsx` - EIA data, OPEC+ production analysis
  - `WeatherAnalyzer.tsx` - Hurricane tracking, HDD/CDD analysis
  - `StorageAnalyzer.tsx` - Storage levels and capacity analysis
  - `EnergyFlowAnalyzer.tsx` - Tanker routes, pipeline flow analysis
- **Portfolio Components**:
  - `PortfolioDashboard.tsx` - Real-time portfolio monitoring
  - `RiskMonitor.tsx` - Risk metrics and alerts
  - `PositionManager.tsx` - Position tracking and management
- **Market Components**:
  - `MarketScanner.tsx` - Unusual options activity scanner
  - `FlowAnalyzer.tsx` - Options flow analysis
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

1. **Heston Model Implementation** (`heston_model.py`) - Stochastic volatility pricing, FFT methods, calibration
2. **Advanced Options Pricing** (`options_pricing_engine.py`) - Multi-model pricing, exotic options, Monte Carlo
3. **Volatility Modeling** (`volatility_models.py`) - SVI, SABR, local volatility surfaces
4. **Greeks Calculation** (`greeks_calculator.py`) - Portfolio Greeks, cross-Greeks, scenario analysis
5. **Risk Management** (`risk_calculator.py`) - VaR models, stress testing, portfolio analytics
6. **Black-Scholes Engine** (`black_scholes_engine.py`) - Foundation pricing with dividends, American options
7. **Portfolio Analytics** (`portfolio_engine.py`) - Optimization, attribution, performance analysis
8. **Energy Spread Analysis** (`energy_spread_analyzer.py`) - Crack spreads, calendar spreads, storage economics
9. **Energy Fundamentals** (`energy_fundamental_analyzer.py`) - EIA data, OPEC+ analysis, weather impact
10. **Energy Volatility** (`energy_volatility_analyzer.py`) - Energy-specific volatility modeling

### 🟡 VALIDATE CAREFULLY (Medium AI) - Data & Domain Logic

1. **Options Data Adapters** - Understand options chains, Greeks, IV calculations
2. **Energy Data Adapters** - Understand energy futures, spreads, fundamentals
3. **Screener Logic** - Know why each filter matters for options/energy trading
4. **Strategy Analysis** - Understand multi-leg payoffs and risk profiles
5. **Backtesting Framework** - Validate strategy performance and assumptions
6. **Geospatial Analysis** - Crude market geography, tanker flows, pipeline analysis
7. **Weather Impact Models** - Hurricane tracking, HDD/CDD analysis
8. **Fundamental Analysis** - EIA data integration, OPEC+ production analysis

### 🔴 LET AI HANDLE (Full AI) - Plumbing

1. **API setup** - FastAPI routes
2. **Frontend scaffolding** - Next.js pages, component structure, routing
3. **Docker/deployment** - Infrastructure as code
4. **UI libraries** - Plotly/Leaflet integration, chart components
5. **Data collection** - API integrations, real-time feeds

---

## Estimated Effort

- **Backend**: ~80 files, ~7000 LOC (45% full AI, 35% medium, 20% minimal)
- **Frontend**: ~100 files, ~9000 LOC (90% full AI, 10% medium)
- **Quant Core**: ~2000 LOC YOU write/deeply review (Heston, pricing models, Greeks, VaR, energy analytics)
- **Timeline**: 6-8 weeks (3-4 weeks on core quant logic, 3-4 weeks on plumbing/UI with AI)

This approach lets you **master advanced derivatives pricing AND energy trading** while building a **production-grade energy trading system** efficiently.