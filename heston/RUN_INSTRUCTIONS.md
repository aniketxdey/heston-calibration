# 🚀 Heston Model Web Application - Run Instructions

## Quick Start (Automated)

The setup has been completed! Here are three ways to run the application:

### Option 1: Use the Setup Script (Recommended)
```bash
cd heston
./setup.sh
```

### Option 2: Use Individual Start Scripts
```bash
# Terminal 1 - Start Backend
cd heston
./start-backend.sh

# Terminal 2 - Start Frontend (in a new terminal)
cd heston
./start-frontend.sh
```

### Option 3: Manual Start (Step by Step)

#### Start the Backend (Terminal 1)
```bash
cd heston/backend
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Start the Frontend (Terminal 2)
```bash
cd heston/frontend
npm start
```

## Access the Application

- **Frontend (React App)**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs (FastAPI auto-generated docs)

## What You'll See

1. **Frontend**: Professional React interface with Material-UI components
   - Market Data Panel for real-time options data
   - Model Calibration Panel for Heston, Black-Scholes, and SABR models
   - Interactive parameter controls and 3D visualizations (planned)

2. **Backend**: FastAPI server with comprehensive REST API
   - `/models/` - Model implementations and pricing
   - `/data/` - Market data endpoints
   - `/calibration/` - Model calibration services
   - `/docs` - Interactive API documentation

## Troubleshooting

### If Backend Fails to Start:
```bash
cd heston/backend
source venv/bin/activate
pip install -r requirements.txt
```

### If Frontend Fails to Start:
```bash
cd heston/frontend
npm install
npm start
```

### Dependencies Issues:
- **Python**: Requires Python 3.11+ (tested with 3.12)
- **Node.js**: Requires Node.js 18+ (tested with 20.x)
- **Virtual Environment**: All Python dependencies are isolated in `backend/venv/`

## Features Available

✅ **Working Features:**
- FastAPI backend with comprehensive model implementations
- React frontend with Material-UI components
- Market data integration (Yahoo Finance, Alpha Vantage)
- Heston, Black-Scholes, and SABR model implementations
- Real-time API with automatic documentation
- Professional interface ready for portfolio demonstrations

🚧 **In Development:**
- 3D volatility surface visualization
- Real-time parameter adjustment
- Model comparison interfaces
- Educational content integration

## Project Structure
```
heston/
├── backend/           # FastAPI server
│   ├── app/          # Application code
│   ├── venv/         # Python virtual environment
│   └── requirements.txt
├── frontend/         # React application
│   ├── src/          # Source code
│   ├── public/       # Static files
│   └── package.json
├── start-backend.sh  # Backend start script
└── start-frontend.sh # Frontend start script
```

## Next Steps

1. **Portfolio Demo**: The application is ready for professional demonstrations
2. **Customization**: Modify models, add new features, or enhance UI
3. **Deployment**: Ready for deployment to cloud platforms
4. **Educational Use**: Perfect for explaining quantitative finance concepts

Happy coding! 🎯 