# Database Migration Guide

This guide will help you migrate from CSV files to PostgreSQL database for the Petral trading dashboard.

## Prerequisites

1. **Docker** (recommended) or **PostgreSQL** installed locally
2. **Python dependencies** installed: `pip install -r backend/requirements.txt`
3. **FRED API Key** (optional, for updating rates data)

## Quick Start

### Option 1: Using Docker (Recommended)

1. **Start PostgreSQL**:
   ```bash
   docker-compose up -d postgres
   ```

2. **Run the migration**:
   ```bash
   python setup_database.py --start-db
   ```

### Option 2: Manual PostgreSQL Setup

1. **Install PostgreSQL** and create database:
   ```sql
   CREATE DATABASE petral_db;
   CREATE USER postgres WITH PASSWORD 'password';
   GRANT ALL PRIVILEGES ON DATABASE petral_db TO postgres;
   ```

2. **Set environment variables**:
   ```bash
   export DATABASE_URL="postgresql://postgres:password@localhost:5432/petral_db"
   export ASYNC_DATABASE_URL="postgresql+asyncpg://postgres:password@localhost:5432/petral_db"
   ```

3. **Run migration**:
   ```bash
   python setup_database.py
   ```

## What Gets Migrated

The following CSV files will be migrated to database tables:

- `backend/data/risk_free.csv` → `risk_free_rates` table
- `backend/data/treasury_par_yields.csv` → `treasury_yields` table  
- `backend/data/ticker_universe.csv` → `tickers` table
- `backend/data/sofr_data.csv` → `sofr_rates` table

## Database Schema

### Tables Created

1. **risk_free_rates**: Risk-free rate data (SOFR)
   - `id`, `date`, `value`, `created_at`, `updated_at`

2. **treasury_yields**: Treasury par yields for different maturities
   - `id`, `date`, `maturity_0_5`, `maturity_1_0`, `maturity_2_0`, `maturity_3_0`, `maturity_5_0`, `maturity_7_0`, `maturity_10_0`, `created_at`, `updated_at`

3. **tickers**: Ticker universe data
   - `id`, `ticker`, `name`, `mcap`, `created_at`, `updated_at`

4. **sofr_rates**: SOFR (Secured Overnight Financing Rate) data
   - `id`, `date`, `sofr`, `created_at`, `updated_at`

## Using the New Database Adapters

### Rates Adapter

Replace the old CSV-based adapter:

```python
# Old way (CSV-based)
from adapters.rates_adapter import RatesAdapter

# New way (Database-based)
from adapters.rates_adapter_db import RatesAdapter
```

The interface remains the same:

```python
# Get risk-free rate for a specific date
rate = RatesAdapter.getRiskFreeRate(target_date)

# Update rates data
RatesAdapter.updateAllRates()

# Get last update date
last_update = RatesAdapter.getLastUpdateDate()
```

### Database Services

You can also use the database services directly:

```python
from database.config import SessionLocal
from database.services import DatabaseRatesService, DatabaseTickerService

db = SessionLocal()
rates_service = DatabaseRatesService(db)
ticker_service = DatabaseTickerService(db)

# Get SOFR rate
sofr_rate = rates_service.get_sofr_rate(date(2024, 1, 1))

# Search tickers
tickers = ticker_service.search_tickers("AAPL")
```

## Updating Data

### Automatic Updates

The database adapters support incremental updates:

```python
from adapters.rates_adapter_db import RatesAdapter

# Update all rates data
RatesAdapter.updateAllRates()

# Update only SOFR
RatesAdapter.updateSOFR()

# Update only treasury yields
RatesAdapter.updateTreasuryYields()
```

### Manual Data Loading

To reload all data from FRED:

```python
from adapters.rates_adapter_db import RatesAdapter

# Generate fresh data
RatesAdapter.generateSOFR()
RatesAdapter.generateTreasuryYields()
```

## Troubleshooting

### Connection Issues

1. **Check PostgreSQL is running**:
   ```bash
   docker-compose ps
   ```

2. **Check database connection**:
   ```python
   from database.config import engine
   print(engine.url)
   ```

### Migration Issues

1. **Clear existing data**:
   ```python
   from database.config import SessionLocal
   from database.models import Base
   
   db = SessionLocal()
   Base.metadata.drop_all(bind=db.bind)
   Base.metadata.create_all(bind=db.bind)
   db.close()
   ```

2. **Re-run migration**:
   ```bash
   python setup_database.py
   ```

### Performance

For better performance with large datasets:

1. **Add indexes** (already included in models)
2. **Use connection pooling** (configured in `database/config.py`)
3. **Consider async operations** for high-frequency updates

## Next Steps

After successful migration:

1. **Update your application** to use `rates_adapter_db.py`
2. **Set up scheduled updates** using APScheduler
3. **Add monitoring** for database health
4. **Consider backup strategies** for production

## Files Created

- `backend/database/` - Database package
- `backend/database/config.py` - Database configuration
- `backend/database/models.py` - SQLAlchemy models
- `backend/database/services.py` - Database service classes
- `backend/database/migrate_csv.py` - Migration script
- `backend/adapters/rates_adapter_db.py` - Database-based rates adapter
- `docker-compose.yml` - PostgreSQL Docker setup
- `setup_database.py` - Setup and migration script
