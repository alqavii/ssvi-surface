# Petral Trading Dashboard - Quick Reference Guide

## 🚀 Quick Start Commands

### 1. Start the System

```bash
# Start PostgreSQL database
docker-compose up -d postgres

# Run database migration (first time only)
python setup_database.py --start-db

# Start API server
cd backend && python app.py
```

### 2. Test the System

```bash
# Run comprehensive tests
python test_api.py

# Quick health check
curl http://localhost:8000/
```

---

## 📊 API Quick Reference

### Base URL
```
http://localhost:8000
```

### Essential Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/api/v1/rates/summary` | GET | Database summary |
| `/api/v1/rates/risk-free` | GET | Risk-free rates |
| `/api/v1/rates/treasury` | GET | Treasury yields |
| `/api/v1/rates/sofr` | GET | SOFR rates |
| `/api/v1/tickers` | GET | Ticker search |
| `/api/v1/analytics/risk-free-rate` | GET | Calculate risk-free rate |

### Common API Calls

```bash
# Get database summary
curl http://localhost:8000/api/v1/rates/summary

# Get recent risk-free rates
curl "http://localhost:8000/api/v1/rates/risk-free?start_date=2024-01-01&limit=10"

# Search for Apple stock
curl "http://localhost:8000/api/v1/tickers?search=AAPL"

# Calculate risk-free rate for future date
curl "http://localhost:8000/api/v1/analytics/risk-free-rate?target_date=2024-06-15"
```

---

## 🗄️ Database Operations

### Direct Database Access

```python
from database.config import SessionLocal
from database.services import DatabaseRatesService, DatabaseTickerService

# Get database session
db = SessionLocal()

# Use services
rates_service = DatabaseRatesService(db)
ticker_service = DatabaseTickerService(db)

# Get latest SOFR rate
sofr_rate = rates_service.get_sofr_rate(date.today())

# Search tickers
tickers = ticker_service.search_tickers("Apple")

# Close session
db.close()
```

### Raw SQL Queries

```sql
-- Get latest SOFR rate
SELECT sofr, date FROM sofr_rates ORDER BY date DESC LIMIT 1;

-- Get latest treasury yields
SELECT * FROM treasury_yields ORDER BY date DESC LIMIT 1;

-- Search tickers
SELECT ticker, name FROM tickers WHERE ticker ILIKE '%AAPL%' OR name ILIKE '%Apple%';

-- Get risk-free rates for date range
SELECT date, value FROM risk_free_rates 
WHERE date BETWEEN '2024-01-01' AND '2024-01-31' 
ORDER BY date DESC;
```

---

## 🔄 Data Updates

### Update Rates Data

```python
from adapters.rates_adapter_db import RatesAdapter

# Update all rates
RatesAdapter.updateAllRates()

# Update only SOFR
RatesAdapter.updateSOFR()

# Update only treasury yields
RatesAdapter.updateTreasuryYields()

# Check last update dates
last_update = RatesAdapter.getLastUpdateDate()
print(f"Last SOFR update: {last_update['sofr']}")
print(f"Last treasury update: {last_update['treasury']}")
```

### Scheduled Updates

```python
from apscheduler.schedulers.blocking import BlockingScheduler
from adapters.rates_adapter_db import RatesAdapter

scheduler = BlockingScheduler()

# Update rates daily at 6 AM
@scheduler.scheduled_job('cron', hour=6, minute=0)
def update_rates():
    print("Updating rates data...")
    RatesAdapter.updateAllRates()
    print("Rates update complete")

scheduler.start()
```

---

## 🧮 Risk-Free Rate Calculations

### Using the Adapter

```python
from adapters.rates_adapter_db import RatesAdapter
from datetime import date, timedelta

# Get rate for today
today_rate = RatesAdapter.getRiskFreeRate(date.today())

# Get rate for future date (uses bootstrapped zero curve)
future_date = date.today() + timedelta(days=90)
future_rate = RatesAdapter.getRiskFreeRate(future_date)

# Get rate for past date (uses SOFR)
past_date = date.today() - timedelta(days=30)
past_rate = RatesAdapter.getRiskFreeRate(past_date)
```

### Using the API

```python
import requests

# Calculate risk-free rate via API
response = requests.get("http://localhost:8000/api/v1/analytics/risk-free-rate",
                       params={"target_date": "2024-06-15"})
data = response.json()
print(f"Rate: {data['risk_free_rate_percent']:.2f}%")
print(f"Method: {data['calculation_method']}")
```

---

## 🔍 Data Exploration

### Database Schema Overview

```sql
-- Check table sizes
SELECT 
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation
FROM pg_stats 
WHERE schemaname = 'public';

-- Check indexes
SELECT 
    tablename,
    indexname,
    indexdef
FROM pg_indexes 
WHERE schemaname = 'public';
```

### Data Quality Checks

```python
from database.config import SessionLocal
from database.models import RiskFreeRate, TreasuryYield, Ticker, SOFRRate

db = SessionLocal()

# Check for missing data
missing_sofr = db.query(SOFRRate).filter(SOFRRate.sofr.is_(None)).count()
missing_treasury = db.query(TreasuryYield).filter(TreasuryYield.maturity_10_0.is_(None)).count()

print(f"Missing SOFR records: {missing_sofr}")
print(f"Missing 10Y treasury records: {missing_treasury}")

# Check date ranges
sofr_range = db.query(SOFRRate.date.min(), SOFRRate.date.max()).first()
treasury_range = db.query(TreasuryYield.date.min(), TreasuryYield.date.max()).first()

print(f"SOFR date range: {sofr_range[0]} to {sofr_range[1]}")
print(f"Treasury date range: {treasury_range[0]} to {treasury_range[1]}")

db.close()
```

---

## 🛠️ Troubleshooting

### Common Issues

#### 1. Database Connection Error
```bash
# Check if PostgreSQL is running
docker-compose ps

# Check logs
docker-compose logs postgres

# Restart if needed
docker-compose restart postgres
```

#### 2. API Server Not Responding
```bash
# Check if server is running
ps aux | grep "python app.py"

# Check port 8000
netstat -tlnp | grep 8000

# Restart server
pkill -f "python app.py"
cd backend && python app.py
```

#### 3. No Data in Database
```bash
# Re-run migration
python setup_database.py

# Check if CSV files exist
ls -la backend/data/

# Verify data was loaded
python -c "
from database.config import SessionLocal
from database.models import SOFRRate
db = SessionLocal()
count = db.query(SOFRRate).count()
print(f'SOFR records: {count}')
db.close()
"
```

#### 4. FRED API Rate Limits
```python
# Check FRED API key
import os
print(f"FRED API Key: {'Set' if os.getenv('FRED_API_KEY') else 'Not set'}")

# Test FRED connection
from fredapi import Fred
fred = Fred(api_key=os.getenv('FRED_API_KEY'))
try:
    data = fred.get_series('SOFR', observation_start='2024-01-01', observation_end='2024-01-02')
    print(f"FRED connection: OK ({len(data)} records)")
except Exception as e:
    print(f"FRED connection error: {e}")
```

---

## 📈 Performance Tips

### Database Optimization

```sql
-- Analyze table statistics
ANALYZE risk_free_rates;
ANALYZE treasury_yields;
ANALYZE tickers;
ANALYZE sofr_rates;

-- Check query performance
EXPLAIN ANALYZE SELECT * FROM sofr_rates WHERE date >= '2024-01-01' ORDER BY date DESC LIMIT 10;
```

### API Optimization

```python
# Use pagination for large datasets
response = requests.get("http://localhost:8000/api/v1/rates/risk-free", 
                       params={"limit": 100, "offset": 0})

# Use date ranges to limit data
response = requests.get("http://localhost:8000/api/v1/rates/risk-free",
                       params={"start_date": "2024-01-01", "end_date": "2024-01-31"})
```

---

## 🔐 Security Checklist

### Development
- [ ] No sensitive data in logs
- [ ] Environment variables for secrets
- [ ] Input validation on all endpoints

### Production
- [ ] HTTPS enabled
- [ ] Authentication implemented
- [ ] Rate limiting configured
- [ ] CORS properly configured
- [ ] Database credentials secured
- [ ] API keys rotated regularly

---

## 📚 Additional Resources

### Documentation Files
- `INTERNAL_DOCUMENTATION.md` - Complete technical documentation
- `DATABASE_MIGRATION.md` - Migration guide
- `backend/app.py` - API implementation with Swagger docs

### Useful Commands
```bash
# View API documentation
open http://localhost:8000/docs

# View ReDoc documentation
open http://localhost:8000/redoc

# Run tests
python test_api.py

# Check database status
docker-compose ps postgres
```

### Key Files
- `backend/database/models.py` - Database schema
- `backend/database/services.py` - Database service classes
- `backend/adapters/rates_adapter_db.py` - Database-based rates adapter
- `docker-compose.yml` - PostgreSQL setup
- `setup_database.py` - Database migration script

This quick reference guide provides the essential commands and operations for working with the Petral trading dashboard system.
