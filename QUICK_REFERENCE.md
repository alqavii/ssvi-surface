# Petral Trading Dashboard - Quick Reference Guide

## 🚀 Quick Start Commands

### 1. Start the System

```bash
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

### Common API Calls

```bash
# Health check
curl http://localhost:8000/
```

---

## 🛠️ Troubleshooting

### Common Issues

#### 1. API Server Not Responding
```bash
# Check if server is running
ps aux | grep "python app.py"

# Check port 8000
netstat -tlnp | grep 8000

# Restart server
pkill -f "python app.py"
cd backend && python app.py
```

#### 2. FRED API Rate Limits
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

### API Optimization

```python
# Health check
response = requests.get("http://localhost:8000/")
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
- [ ] API keys rotated regularly

---

## 📚 Additional Resources

### Useful Commands
```bash
# View API documentation
open http://localhost:8000/docs

# View ReDoc documentation
open http://localhost:8000/redoc

# Run tests
python test_api.py
```

This quick reference guide provides the essential commands and operations for working with the Petral trading dashboard system.
