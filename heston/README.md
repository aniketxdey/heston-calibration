# Heston Model Web Application

A comprehensive quantitative finance platform for volatility modeling, featuring interactive calibration of the Heston stochastic volatility model, Black-Scholes, and SABR models with real-time market data integration.

## 🚀 Features

- **Multi-Model Calibration**: Calibrate Heston, Black-Scholes, and SABR models to real market data
- **Real-Time Market Data**: Live options data from Yahoo Finance with intelligent caching
- **Interactive 3D Visualization**: Rotatable and zoomable volatility surfaces
- **Parameter Controls**: Real-time parameter adjustment with immediate surface updates
- **Model Comparison**: Side-by-side comparison of different volatility models
- **Educational Content**: Interactive explanations and model demonstrations
- **Professional Interface**: Clean, modern UI designed for portfolio demonstrations

## 🏗️ Architecture

### Backend (Python/FastAPI)
- **Models**: Heston, Black-Scholes, SABR implementations with moment matching
- **Data Integration**: Yahoo Finance API with validation pipeline
- **Calibration Engine**: Robust optimization with multiple algorithms
- **API**: RESTful endpoints with automatic documentation

### Frontend (React/TypeScript)
- **UI Framework**: Material-UI with responsive design
- **Visualization**: D3.js for 3D surfaces, Plotly for charts
- **State Management**: React Query for efficient data fetching
- **Real-time Updates**: WebSocket integration for live data

## 📋 Prerequisites

- Python 3.11+
- Node.js 18+
- npm or yarn

## 🛠️ Installation

### Backend Setup

1. **Navigate to backend directory**:
   ```bash
   cd heston/backend
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

4. **Start the backend server**:
   ```bash
   cd app
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

The backend will be available at `http://localhost:8000`

### Frontend Setup

1. **Navigate to frontend directory**:
   ```bash
   cd heston/frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start the development server**:
   ```bash
   npm start
   ```

The frontend will be available at `http://localhost:3000`

## 🎯 Usage

### Basic Workflow

1. **Load Market Data**: Enter a stock symbol (e.g., AAPL, MSFT) to fetch live options data
2. **Select Model**: Choose from Heston, Black-Scholes, or SABR models
3. **Calibrate**: Run calibration to fit model parameters to market data
4. **Visualize**: Explore the 3D volatility surface and model comparison
5. **Adjust Parameters**: Use interactive sliders to see real-time changes

### API Endpoints

- `GET /models` - Get available volatility models
- `GET /market-data/{symbol}` - Fetch options chain data
- `POST /calibrate` - Calibrate model to market data
- `POST /price` - Price options with specified parameters
- `POST /greeks` - Calculate option Greeks
- `POST /volatility-surface` - Generate volatility surface
- `POST /compare-models` - Compare multiple models

## 📊 Model Implementations

### Heston Model
- **Paper**: Heston (1993) - "A Closed-Form Solution for Options with Stochastic Volatility"
- **Features**: Moment matching approximation for computational efficiency
- **Parameters**: v₀ (initial variance), κ (mean reversion), θ (long-term variance), σ (vol of vol), ρ (correlation)

### Black-Scholes Model
- **Paper**: Black & Scholes (1973) - "The Pricing of Options and Corporate Liabilities"
- **Features**: Classical option pricing with constant volatility
- **Parameters**: σ (volatility)

### SABR Model
- **Paper**: Hagan et al. (2002) - "Managing Smile Risk"
- **Features**: Stochastic volatility model for interest rates and FX
- **Parameters**: α (initial vol), β (CEV parameter), ρ (correlation), ν (vol of vol)

## 🧪 Testing

### Backend Tests
```bash
cd heston/backend
pytest tests/
```

### Frontend Tests
```bash
cd heston/frontend
npm test
```

## 📈 Performance

- **Calibration Speed**: < 5 seconds for 100+ options
- **Interactive Response**: < 200ms for parameter adjustments
- **Visualization Rendering**: < 1 second for surface generation
- **Concurrent Users**: 10+ simultaneous users supported

## 🔧 Configuration

### Environment Variables

Backend (`.env`):
```
API_KEY_ALPHA_VANTAGE=your_alpha_vantage_key
REDIS_URL=redis://localhost:6379
LOG_LEVEL=INFO
```

Frontend (`.env`):
```
REACT_APP_API_URL=http://localhost:8000
```

## 📚 Educational Content

The application includes interactive educational modules covering:

- **Volatility Smile/Skew**: Market pattern demonstrations
- **Model Assumptions**: Limitations and appropriate use cases
- **Parameter Interpretation**: Economic meaning of mathematical parameters
- **Practical Applications**: Real-world usage scenarios

## 🚀 Deployment

### Production Deployment

1. **Backend**: Deploy to Railway, Render, or AWS
2. **Frontend**: Deploy to Vercel or Netlify
3. **Database**: Set up Redis for caching
4. **Environment**: Configure production environment variables

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Academic Papers**: Heston (1993), Black & Scholes (1973), Hagan et al. (2002)
- **Libraries**: NumPy, SciPy, FastAPI, React, Material-UI, D3.js
- **Data Sources**: Yahoo Finance, Alpha Vantage

## 📞 Support

For questions or support, please open an issue on GitHub or contact the development team.

---

**Note**: This application is designed for educational and demonstration purposes. For production use in financial applications, additional validation, testing, and compliance measures should be implemented. 