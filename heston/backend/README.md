# Heston Model Backend

FastAPI backend for the Heston Model Web Application, providing comprehensive volatility modeling capabilities with real-time market data integration.

## 🏗️ Architecture

### Core Components

- **Models**: Heston, Black-Scholes, SABR implementations
- **Data Layer**: Market data aggregation with validation
- **Calibration Engine**: Robust optimization framework
- **API Layer**: RESTful endpoints with automatic documentation

### Technology Stack

- **Framework**: FastAPI 0.103+
- **Python**: 3.11+
- **Numerical Computing**: NumPy, SciPy, Pandas
- **Quantitative Finance**: QuantLib-Python
- **Data Sources**: Yahoo Finance, Alpha Vantage
- **Caching**: Redis
- **Server**: Uvicorn

## 📦 Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Setup

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd heston-calibration/heston/backend
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables** (optional):
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## 🚀 Running the Application

### Development Mode

```bash
cd app
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
cd app
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at `http://localhost:8000`

## 📚 API Documentation

Once the server is running, visit:
- **Interactive API Docs**: `http://localhost:8000/docs`
- **ReDoc Documentation**: `http://localhost:8000/redoc`

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the backend directory:

```env
# API Keys
API_KEY_ALPHA_VANTAGE=your_alpha_vantage_key

# Redis Configuration
REDIS_URL=redis://localhost:6379

# Logging
LOG_LEVEL=INFO

# Server Configuration
HOST=0.0.0.0
PORT=8000
WORKERS=4
```

### Model Parameters

Each model has specific parameter bounds and validation:

#### Heston Model
- `v0`: Initial variance [0.001, 1.0]
- `kappa`: Mean reversion speed [0.1, 10.0]
- `theta`: Long-term variance [0.001, 1.0]
- `sigma`: Volatility of volatility [0.1, 2.0]
- `rho`: Correlation [-0.99, 0.99]

#### Black-Scholes Model
- `volatility`: Constant volatility [0.001, 2.0]

#### SABR Model
- `alpha`: Initial volatility [0.001, 1.0]
- `beta`: CEV parameter [0.01, 1.0]
- `rho`: Correlation [-0.99, 0.99]
- `nu`: Volatility of volatility [0.001, 2.0]

## 🧪 Testing

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test file
pytest tests/test_models.py

# Run with verbose output
pytest -v
```

### Test Structure

```
tests/
├── test_models/          # Model implementation tests
├── test_calibration/     # Calibration engine tests
├── test_data/           # Data integration tests
├── test_api/            # API endpoint tests
└── conftest.py          # Test configuration
```

## 📊 Model Implementations

### Heston Model

Based on Heston (1993) with moment matching approximation:

```python
# Key features
- Moment matching for computational efficiency
- Feller condition validation
- Robust optimization with constraints
- Finite difference Greeks calculation
```

### Black-Scholes Model

Classical implementation with extensions:

```python
# Key features
- Analytical option pricing
- Dividend yield support
- Analytical Greeks calculation
- Implied volatility calculation
```

### SABR Model

Based on Hagan et al. (2002):

```python
# Key features
- Asymptotic expansion for implied volatility
- Industry-standard parameter bounds
- Numerical stability enhancements
- Cross-asset class applicability
```

## 🔄 Data Flow

### Market Data Processing

1. **Data Acquisition**: Fetch from Yahoo Finance/Alpha Vantage
2. **Validation**: Filter for liquidity and quality
3. **Cleaning**: Remove outliers and stale data
4. **Caching**: Store in Redis for performance
5. **Calibration**: Prepare data for model fitting

### Calibration Process

1. **Data Preparation**: Flatten options chain data
2. **Objective Function**: Define optimization target
3. **Global Search**: Differential Evolution
4. **Local Refinement**: L-BFGS-B optimization
5. **Validation**: Parameter reasonableness checks
6. **Uncertainty**: Bootstrap analysis

## 📈 Performance Optimization

### Caching Strategy

- **Market Data**: 5-minute TTL
- **Calibration Results**: 1-hour TTL
- **Model Parameters**: Session-based

### Optimization Techniques

- **Parallel Processing**: Multi-core calibration
- **Memory Management**: Explicit cleanup
- **Rate Limiting**: API quota management
- **Error Handling**: Graceful degradation

## 🔍 Monitoring

### Health Checks

```bash
# Check API health
curl http://localhost:8000/health

# Check cache status
curl http://localhost:8000/cache/stats
```

### Logging

Configure logging levels in `.env`:

```env
LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR
```

## 🚀 Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Production Considerations

- **Load Balancing**: Use multiple workers
- **Caching**: Redis cluster for high availability
- **Monitoring**: Application performance monitoring
- **Security**: API key management, rate limiting
- **Backup**: Database and configuration backups

## 🤝 Contributing

### Development Setup

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/new-feature`
3. **Install dev dependencies**: `pip install -r requirements-dev.txt`
4. **Run tests**: `pytest`
5. **Submit pull request**

### Code Style

- **Formatting**: Black code formatter
- **Linting**: Flake8
- **Type Checking**: mypy
- **Documentation**: Google-style docstrings

## 📞 Support

For backend-specific issues:

1. Check the API documentation at `/docs`
2. Review the logs for error messages
3. Open an issue on GitHub
4. Contact the development team

---

**Note**: This backend is designed for educational and demonstration purposes. For production use in financial applications, additional security, compliance, and validation measures should be implemented. 