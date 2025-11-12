# Petral Trading Dashboard - Internal Documentation

## Database Architecture & API Documentation

### Overview

The Petral trading dashboard has been migrated from CSV-based storage to a PostgreSQL database with a comprehensive FastAPI backend. This document provides internal documentation for developers working with the system.

---

## Database Schema

### Tables Structure

#### 1. `risk_free_rates`
Stores risk-free rate data (primarily SOFR data).

```sql
CREATE TABLE risk_free_rates (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    value FLOAT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_risk_free_date ON risk_free_rates(date);
```

**Fields:**
- `id`: Primary key
- `date`: Date of the rate observation
- `value`: Risk-free rate value (as decimal, e.g., 0.05 for 5%)
- `created_at`: Record creation timestamp
- `updated_at`: Record update timestamp

#### 2. `treasury_yields`
Stores Treasury par yields for different maturities.

```sql
CREATE TABLE treasury_yields (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    maturity_0_5 FLOAT,  -- 0.5 year yield
    maturity_1_0 FLOAT,  -- 1.0 year yield
    maturity_2_0 FLOAT,  -- 2.0 year yield
    maturity_3_0 FLOAT,  -- 3.0 year yield
    maturity_5_0 FLOAT,  -- 5.0 year yield
    maturity_7_0 FLOAT,  -- 7.0 year yield
    maturity_10_0 FLOAT, -- 10.0 year yield
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_treasury_date ON treasury_yields(date);
```

**Fields:**
- `id`: Primary key
- `date`: Date of the yield observation
- `maturity_X_Y`: Yield for X.Y year maturity (as decimal)
- `created_at`: Record creation timestamp
- `updated_at`: Record update timestamp

#### 3. `tickers`
Stores ticker universe data.

```sql
CREATE TABLE tickers (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL UNIQUE,
    name TEXT,
    mcap INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_ticker_symbol ON tickers(ticker);
```

**Fields:**
- `id`: Primary key
- `ticker`: Ticker symbol (e.g., 'AAPL', 'MSFT')
- `name`: Company name
- `mcap`: Market capitalization
- `created_at`: Record creation timestamp
- `updated_at`: Record update timestamp

#### 4. `sofr_rates`
Stores SOFR (Secured Overnight Financing Rate) data.

```sql
CREATE TABLE sofr_rates (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    sofr FLOAT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_sofr_date ON sofr_rates(date);
```

**Fields:**
- `id`: Primary key
- `date`: Date of the SOFR observation
- `sofr`: SOFR rate value (as percentage, e.g., 5.25 for 5.25%)
- `created_at`: Record creation timestamp
- `updated_at`: Record update timestamp

---

## API Endpoints

### Base URL
```
http://localhost:8000
```

### Authentication
Currently no authentication is implemented. For production, implement JWT or API key authentication.

### Rate Limiting
No rate limiting is currently implemented. Consider implementing rate limiting for production use.

---

### Health Check

#### GET /
**Description:** Health check endpoint

**Response:**
```json
{
  "message": "Petral Trading Dashboard API",
  "status": "healthy"
}
```

---

### Rates Endpoints

#### GET /api/v1/rates/summary
**Description:** Get a summary of all rates data in the database

**Response:**
```json
{
  "latest_sofr": 5.25,
  "latest_sofr_date": "2024-01-15",
  "latest_treasury_date": "2024-01-15",
  "total_risk_free_records": 6531,
  "total_treasury_records": 6701,
  "total_ticker_records": 3062,
  "total_sofr_records": 1930
}
```

#### GET /api/v1/rates/risk-free
**Description:** Get risk-free rates data

**Query Parameters:**
- `start_date` (optional): Start date for filtering (YYYY-MM-DD)
- `end_date` (optional): End date for filtering (YYYY-MM-DD)
- `limit` (optional): Maximum records to return (default: 100)

**Example:**
```
GET /api/v1/rates/risk-free?start_date=2024-01-01&end_date=2024-01-31&limit=50
```

**Response:**
```json
[
  {
    "id": 1,
    "date": "2024-01-15",
    "value": 0.0525,
    "created_at": "2024-01-15T10:00:00Z",
    "updated_at": null
  }
]
```

#### GET /api/v1/rates/risk-free/{target_date}
**Description:** Get risk-free rate for a specific date

**Path Parameters:**
- `target_date`: Target date (YYYY-MM-DD)

**Example:**
```
GET /api/v1/rates/risk-free/2024-01-15
```

**Response:**
```json
{
  "date": "2024-01-15",
  "rate": 0.0525,
  "rate_percent": 5.25
}
```

#### GET /api/v1/rates/treasury
**Description:** Get treasury yield data

**Query Parameters:**
- `start_date` (optional): Start date for filtering (YYYY-MM-DD)
- `end_date` (optional): End date for filtering (YYYY-MM-DD)
- `limit` (optional): Maximum records to return (default: 100)

**Example:**
```
GET /api/v1/rates/treasury?start_date=2024-01-01&limit=10
```

**Response:**
```json
[
  {
    "id": 1,
    "date": "2024-01-15",
    "maturity_0_5": 0.0481,
    "maturity_1_0": 0.0509,
    "maturity_2_0": 0.0538,
    "maturity_3_0": 0.0542,
    "maturity_5_0": 0.0550,
    "maturity_7_0": 0.0565,
    "maturity_10_0": 0.0558,
    "created_at": "2024-01-15T10:00:00Z",
    "updated_at": null
  }
]
```

#### GET /api/v1/rates/treasury/{target_date}
**Description:** Get treasury yields for a specific date

**Path Parameters:**
- `target_date`: Target date (YYYY-MM-DD)

**Example:**
```
GET /api/v1/rates/treasury/2024-01-15
```

**Response:**
```json
{
  "date": "2024-01-15",
  "yields": {
    "0.5": 0.0481,
    "1.0": 0.0509,
    "2.0": 0.0538,
    "3.0": 0.0542,
    "5.0": 0.0550,
    "7.0": 0.0565,
    "10.0": 0.0558
  }
}
```

#### GET /api/v1/rates/sofr
**Description:** Get SOFR rates data

**Query Parameters:**
- `start_date` (optional): Start date for filtering (YYYY-MM-DD)
- `end_date` (optional): End date for filtering (YYYY-MM-DD)
- `limit` (optional): Maximum records to return (default: 100)

**Example:**
```
GET /api/v1/rates/sofr?start_date=2024-01-01&limit=30
```

**Response:**
```json
[
  {
    "id": 1,
    "date": "2024-01-15",
    "sofr": 5.25,
    "created_at": "2024-01-15T10:00:00Z",
    "updated_at": null
  }
]
```

---

### Ticker Endpoints

#### GET /api/v1/tickers
**Description:** Get ticker information

**Query Parameters:**
- `search` (optional): Search by ticker symbol or company name
- `limit` (optional): Maximum records to return (default: 100)

**Example:**
```
GET /api/v1/tickers?search=AAPL&limit=10
```

**Response:**
```json
[
  {
    "id": 1,
    "ticker": "AAPL",
    "name": "Apple Inc.",
    "mcap": 3154142035968,
    "created_at": "2024-01-15T10:00:00Z",
    "updated_at": null
  }
]
```

#### GET /api/v1/tickers/{symbol}
**Description:** Get ticker information by symbol

**Path Parameters:**
- `symbol`: Ticker symbol (case-insensitive)

**Example:**
```
GET /api/v1/tickers/AAPL
```

**Response:**
```json
{
  "ticker": "AAPL",
  "name": "Apple Inc.",
  "mcap": 3154142035968
}
```

---

### Analytics Endpoints

#### GET /api/v1/analytics/risk-free-rate
**Description:** Calculate risk-free rate for a target date

**Query Parameters:**
- `target_date`: Date for which to calculate the risk-free rate (YYYY-MM-DD)
- `request_date` (optional): Reference date for calculation (defaults to today)

**Example:**
```
GET /api/v1/analytics/risk-free-rate?target_date=2024-06-15&request_date=2024-01-15
```

**Response:**
```json
{
  "target_date": "2024-06-15",
  "request_date": "2024-01-15",
  "risk_free_rate": 0.0525,
  "risk_free_rate_percent": 5.25,
  "calculation_method": "Bootstrapped Zero Curve"
}
```

---

## Database Service Classes

### DatabaseRatesService

Located in `backend/database/services.py`

**Methods:**

#### `get_risk_free_rate(target_date: date) -> Optional[float]`
Get risk-free rate for a specific date.

#### `get_risk_free_rates_range(start_date: date, end_date: date) -> List[Dict[str, Any]]`
Get risk-free rates for a date range.

#### `get_treasury_yields(target_date: date) -> Optional[Dict[str, float]]`
Get treasury yields for a specific date.

#### `get_treasury_yields_range(start_date: date, end_date: date) -> List[Dict[str, Any]]`
Get treasury yields for a date range.

#### `get_sofr_rate(target_date: date) -> Optional[float]`
Get SOFR rate for a specific date.

#### `get_sofr_rates_range(start_date: date, end_date: date) -> List[Dict[str, Any]]`
Get SOFR rates for a date range.

### DatabaseTickerService

Located in `backend/database/services.py`

**Methods:**

#### `get_ticker(symbol: str) -> Optional[Dict[str, Any]]`
Get ticker information by symbol.

#### `get_all_tickers() -> List[Dict[str, Any]]`
Get all tickers.

#### `search_tickers(query: str, limit: int = 100) -> List[Dict[str, Any]]`
Search tickers by symbol or name.

---

## Adapters

### RatesAdapter (Database Version)

Located in `backend/adapters/rates_adapter_db.py`

**Key Methods:**

#### `getRiskFreeRate(target_date, request_date=None)`
Calculate risk-free rate using the same logic as the original CSV-based adapter:
- For past dates: Uses SOFR rate
- For future dates: Uses bootstrapped zero rates from treasury yields

#### `updateSOFR()`
Incrementally update SOFR data from FRED API.

#### `updateTreasuryYields()`
Incrementally update treasury yield data from FRED API.

#### `updateAllRates()`
Update both SOFR and treasury yield data.

#### `getLastUpdateDate()`
Get the last update date for both SOFR and treasury data.

---

## Setup and Usage Commands

### 1. Database Setup

#### Start PostgreSQL with Docker:
```bash
docker-compose up -d postgres
```

#### Run Migration:
```bash
python setup_database.py --start-db
```

#### Manual Migration (if PostgreSQL is already running):
```bash
python setup_database.py
```

### 2. API Server

#### Start the FastAPI server:
```bash
cd backend
python app.py
```

#### Or with uvicorn directly:
```bash
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Data Updates

#### Update all rates data:
```python
from adapters.rates_adapter_db import RatesAdapter
RatesAdapter.updateAllRates()
```

#### Update only SOFR:
```python
from adapters.rates_adapter_db import RatesAdapter
RatesAdapter.updateSOFR()
```

#### Update only treasury yields:
```python
from adapters.rates_adapter_db import RatesAdapter
RatesAdapter.updateTreasuryYields()
```

### 4. Testing API Endpoints

#### Using curl:

```bash
# Health check
curl http://localhost:8000/

# Get rates summary
curl http://localhost:8000/api/v1/rates/summary

# Get risk-free rates for January 2024
curl "http://localhost:8000/api/v1/rates/risk-free?start_date=2024-01-01&end_date=2024-01-31&limit=10"

# Get treasury yields for a specific date
curl http://localhost:8000/api/v1/rates/treasury/2024-01-15

# Search tickers
curl "http://localhost:8000/api/v1/tickers?search=AAPL&limit=5"

# Calculate risk-free rate
curl "http://localhost:8000/api/v1/analytics/risk-free-rate?target_date=2024-06-15"
```

#### Using Python requests:

```python
import requests

# Get rates summary
response = requests.get("http://localhost:8000/api/v1/rates/summary")
print(response.json())

# Get risk-free rates
response = requests.get("http://localhost:8000/api/v1/rates/risk-free", 
                       params={"start_date": "2024-01-01", "limit": 10})
print(response.json())

# Calculate risk-free rate
response = requests.get("http://localhost:8000/api/v1/analytics/risk-free-rate",
                       params={"target_date": "2024-06-15"})
print(response.json())
```

---

## Error Handling

### HTTP Status Codes

- `200`: Success
- `404`: Resource not found
- `500`: Internal server error

### Error Response Format

```json
{
  "detail": "Error message description"
}
```

### Common Error Scenarios

1. **No data found for date**: Returns 404 with message "No risk-free rate found for {date}"
2. **Database connection error**: Returns 500 with database error details
3. **Invalid date format**: Returns 422 (validation error)
4. **Missing required parameters**: Returns 422 (validation error)

---

## Performance Considerations

### Database Indexes
- All date columns are indexed for fast lookups
- Ticker symbol is indexed for fast searches
- Primary keys are automatically indexed

### Query Optimization
- Use date ranges to limit result sets
- Use `limit` parameter to control response size
- Consider pagination for large datasets

### Caching
- No caching is currently implemented
- Consider Redis for frequently accessed data
- Implement response caching for static data

---

## Security Considerations

### Current State
- No authentication implemented
- CORS allows all origins
- No rate limiting

### Production Recommendations
1. Implement JWT authentication
2. Restrict CORS origins
3. Add rate limiting
4. Use HTTPS
5. Validate all input parameters
6. Implement API key authentication for external access

---

## Monitoring and Logging

### Current State
- Basic error logging in FastAPI
- No structured logging
- No metrics collection

### Production Recommendations
1. Implement structured logging (JSON format)
2. Add request/response logging
3. Implement health checks
4. Add metrics collection (Prometheus)
5. Set up alerting for errors
6. Monitor database performance

---

## Development Workflow

### Adding New Endpoints

1. **Define Pydantic models** in `app.py`
2. **Add route handler** with proper error handling
3. **Update service classes** if needed
4. **Add tests** for new functionality
5. **Update documentation**

### Database Schema Changes

1. **Create migration script** using Alembic
2. **Update SQLAlchemy models**
3. **Test migration** on development database
4. **Update API documentation**
5. **Deploy migration** to production

### Data Updates

1. **Use incremental updates** via `updateSOFR()` and `updateTreasuryYields()`
2. **Schedule regular updates** using APScheduler
3. **Monitor update success/failure**
4. **Handle API rate limits** from FRED

---

## Troubleshooting

### Common Issues

#### Database Connection Errors
```bash
# Check if PostgreSQL is running
docker-compose ps

# Check database logs
docker-compose logs postgres

# Test connection
psql -h localhost -p 5432 -U postgres -d petral_db
```

#### API Server Issues
```bash
# Check if server is running
curl http://localhost:8000/

# Check server logs
tail -f backend/logs/app.log

# Restart server
pkill -f "python app.py"
python app.py
```

#### Data Migration Issues
```bash
# Clear existing data and re-migrate
python -c "
from database.config import SessionLocal
from database.models import Base
db = SessionLocal()
Base.metadata.drop_all(bind=db.bind)
Base.metadata.create_all(bind=db.bind)
db.close()
"
python setup_database.py
```

### Performance Issues

1. **Check database indexes**: Ensure all date columns are indexed
2. **Monitor query performance**: Use PostgreSQL's `EXPLAIN ANALYZE`
3. **Limit result sets**: Use `limit` parameter in API calls
4. **Consider caching**: Implement Redis for frequently accessed data

---

## Future Enhancements

### Planned Features

1. **Authentication & Authorization**
2. **Rate Limiting**
3. **Caching Layer**
4. **Real-time Updates** (WebSockets)
5. **Advanced Analytics**
6. **Data Export** (CSV, Excel)
7. **Monitoring Dashboard**
8. **Automated Testing**

### Database Optimizations

1. **Partitioning** by date for large tables
2. **Materialized Views** for complex queries
3. **Read Replicas** for scaling
4. **Connection Pooling** optimization

This documentation provides a comprehensive guide for developers working with the Petral trading dashboard database and API system.
