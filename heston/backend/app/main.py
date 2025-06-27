"""
Main FastAPI Application

Heston Model Web Application backend with comprehensive API endpoints
for volatility model calibration, pricing, and analysis.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import asyncio
import logging
import uvicorn

from .models import HestonModel, BlackScholesModel, SABRModel
from .data import MarketDataAggregator, DataValidationPipeline
from .calibration import RobustOptimizer, CalibrationValidator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Heston Model Web Application",
    description="A comprehensive quantitative finance platform for volatility modeling",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
market_data_aggregator = MarketDataAggregator()
data_validator = DataValidationPipeline()
optimizer = RobustOptimizer()
validator = CalibrationValidator()

# Initialize models
heston_model = HestonModel()
black_scholes_model = BlackScholesModel()
sabr_model = SABRModel()

# Pydantic models for API requests/responses
class CalibrationRequest(BaseModel):
    symbol: str
    model_type: str = "heston"  # heston, black_scholes, sabr
    use_puts: bool = False
    min_volume: int = 10
    max_bid_ask_spread: float = 0.5

class PricingRequest(BaseModel):
    spot: float
    strike: float
    expiry: float
    risk_free_rate: float = 0.05
    dividend_yield: float = 0.0
    model_type: str = "heston"
    parameters: Dict[str, float]

class GreeksRequest(BaseModel):
    spot: float
    strike: float
    expiry: float
    risk_free_rate: float = 0.05
    dividend_yield: float = 0.0
    model_type: str = "heston"
    parameters: Dict[str, float]

class VolatilitySurfaceRequest(BaseModel):
    spot: float
    strikes: List[float]
    expiries: List[float]
    risk_free_rate: float = 0.05
    dividend_yield: float = 0.0
    model_type: str = "heston"
    parameters: Dict[str, float]

@app.get("/")
async def root():
    """Root endpoint with application information."""
    return {
        "message": "Heston Model Web Application API",
        "version": "1.0.0",
        "description": "A comprehensive quantitative finance platform for volatility modeling",
        "endpoints": {
            "health": "/health",
            "models": "/models",
            "market_data": "/market-data/{symbol}",
            "calibrate": "/calibrate",
            "price": "/price",
            "greeks": "/greeks",
            "volatility_surface": "/volatility-surface",
            "compare_models": "/compare-models"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": "2024-01-01T00:00:00Z",
        "services": {
            "market_data": "operational",
            "models": "operational",
            "calibration": "operational"
        }
    }

@app.get("/models")
async def get_available_models():
    """Get available volatility models and their parameters."""
    return {
        "models": [
            {
                "name": "Heston",
                "description": "Stochastic volatility model with mean-reverting variance",
                "parameters": {
                    "v0": {"description": "Initial variance", "bounds": [0.001, 1.0]},
                    "kappa": {"description": "Mean reversion speed", "bounds": [0.1, 10.0]},
                    "theta": {"description": "Long-term variance", "bounds": [0.001, 1.0]},
                    "sigma": {"description": "Volatility of volatility", "bounds": [0.1, 2.0]},
                    "rho": {"description": "Correlation", "bounds": [-0.99, 0.99]}
                }
            },
            {
                "name": "Black-Scholes",
                "description": "Classical option pricing model with constant volatility",
                "parameters": {
                    "volatility": {"description": "Constant volatility", "bounds": [0.001, 2.0]}
                }
            },
            {
                "name": "SABR",
                "description": "Stochastic Alpha Beta Rho model for interest rates and FX",
                "parameters": {
                    "alpha": {"description": "Initial volatility", "bounds": [0.001, 1.0]},
                    "beta": {"description": "CEV parameter", "bounds": [0.01, 1.0]},
                    "rho": {"description": "Correlation", "bounds": [-0.99, 0.99]},
                    "nu": {"description": "Volatility of volatility", "bounds": [0.001, 2.0]}
                }
            }
        ]
    }

@app.get("/market-data/{symbol}")
async def get_market_data(symbol: str):
    """
    Fetch market data for a given symbol.
    
    Args:
        symbol: Stock symbol (e.g., AAPL, MSFT)
        
    Returns:
        Market data including options chain and validation statistics
    """
    try:
        # Fetch options chain
        options_chain = await market_data_aggregator.get_options_chain(symbol)
        
        if options_chain is None:
            raise HTTPException(status_code=404, detail=f"No data available for symbol {symbol}")
            
        # Get risk-free rate
        risk_free_rate = await market_data_aggregator.get_risk_free_rate()
        options_chain.risk_free_rate = risk_free_rate
        
        # Validate data
        raw_data = options_chain.to_dict()
        validated_data = data_validator.validate_options_chain(raw_data)
        
        return {
            "symbol": symbol,
            "data": validated_data,
            "cache_stats": market_data_aggregator.get_cache_stats()
        }
        
    except Exception as e:
        logger.error(f"Error fetching market data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching market data: {str(e)}")

@app.post("/calibrate")
async def calibrate_model(request: CalibrationRequest):
    """
    Calibrate a volatility model to market data.
    
    Args:
        request: Calibration request with symbol and model type
        
    Returns:
        Calibration results with parameters and validation
    """
    try:
        # Fetch and validate market data
        options_chain = await market_data_aggregator.get_options_chain(request.symbol)
        
        if options_chain is None:
            raise HTTPException(status_code=404, detail=f"No data available for symbol {request.symbol}")
            
        raw_data = options_chain.to_dict()
        validated_data = data_validator.validate_options_chain(raw_data)
        
        if validated_data.get('error', False):
            raise HTTPException(status_code=400, detail=validated_data['error_message'])
            
        # Prepare calibration data
        calibration_data = data_validator.prepare_calibration_data(validated_data)
        
        if calibration_data is None:
            raise HTTPException(status_code=400, detail="Insufficient data for calibration")
            
        # Select model
        if request.model_type.lower() == "heston":
            model = heston_model
        elif request.model_type.lower() == "black_scholes":
            model = black_scholes_model
        elif request.model_type.lower() == "sabr":
            model = sabr_model
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model type: {request.model_type}")
            
        # Perform calibration
        calibration_result = model.calibrate(calibration_data)
        
        # Validate calibration
        validation_result = validator.validate_calibration(
            calibration_result.__dict__, calibration_data, model.name)
        
        # Bootstrap uncertainty analysis
        bootstrap_results = validator.bootstrap_uncertainty(
            model, calibration_data)
        
        return {
            "symbol": request.symbol,
            "model_type": request.model_type,
            "calibration_result": calibration_result.__dict__,
            "validation_result": {
                "is_valid": validation_result.is_valid,
                "warnings": validation_result.warnings,
                "quality_score": validation_result.quality_score,
                "stability_score": validation_result.stability_score
            },
            "bootstrap_uncertainty": bootstrap_results,
            "market_data_stats": validated_data.get('validation_stats', {})
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in model calibration: {e}")
        raise HTTPException(status_code=500, detail=f"Calibration error: {str(e)}")

@app.post("/price")
async def price_option(request: PricingRequest):
    """
    Price an option using specified model and parameters.
    
    Args:
        request: Pricing request with option details and model parameters
        
    Returns:
        Option price and implied volatility
    """
    try:
        # Select model
        if request.model_type.lower() == "heston":
            model = heston_model
            # Set Heston parameters
            from .models.heston import HestonParameters
            params = HestonParameters(**request.parameters)
            model.set_parameters(params)
        elif request.model_type.lower() == "black_scholes":
            model = black_scholes_model
            model.set_volatility(request.parameters.get('volatility', 0.2))
        elif request.model_type.lower() == "sabr":
            model = sabr_model
            # Set SABR parameters
            from .models.sabr import SABRParameters
            params = SABRParameters(**request.parameters)
            model.set_parameters(params)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model type: {request.model_type}")
            
        # Price option
        option_price = model.price_option(
            request.spot, request.strike, request.expiry,
            request.risk_free_rate, request.dividend_yield
        )
        
        # Calculate implied volatility
        implied_vol = model.implied_volatility(
            request.spot, request.strike, request.expiry,
            option_price, request.risk_free_rate, request.dividend_yield
        )
        
        return {
            "option_price": option_price,
            "implied_volatility": implied_vol,
            "model_type": request.model_type,
            "parameters": request.parameters
        }
        
    except Exception as e:
        logger.error(f"Error in option pricing: {e}")
        raise HTTPException(status_code=500, detail=f"Pricing error: {str(e)}")

@app.post("/greeks")
async def calculate_greeks(request: GreeksRequest):
    """
    Calculate option Greeks using specified model and parameters.
    
    Args:
        request: Greeks request with option details and model parameters
        
    Returns:
        Option Greeks (delta, gamma, theta, vega, rho)
    """
    try:
        # Select model
        if request.model_type.lower() == "heston":
            model = heston_model
            # Set Heston parameters
            from .models.heston import HestonParameters
            params = HestonParameters(**request.parameters)
            model.set_parameters(params)
        elif request.model_type.lower() == "black_scholes":
            model = black_scholes_model
            model.set_volatility(request.parameters.get('volatility', 0.2))
        elif request.model_type.lower() == "sabr":
            model = sabr_model
            # Set SABR parameters
            from .models.sabr import SABRParameters
            params = SABRParameters(**request.parameters)
            model.set_parameters(params)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model type: {request.model_type}")
            
        # Calculate Greeks
        greeks = model.calculate_greeks(
            request.spot, request.strike, request.expiry,
            request.risk_free_rate, request.dividend_yield
        )
        
        return {
            "greeks": {
                "delta": greeks.delta,
                "gamma": greeks.gamma,
                "theta": greeks.theta,
                "vega": greeks.vega,
                "rho": greeks.rho
            },
            "model_type": request.model_type,
            "parameters": request.parameters
        }
        
    except Exception as e:
        logger.error(f"Error in Greeks calculation: {e}")
        raise HTTPException(status_code=500, detail=f"Greeks calculation error: {str(e)}")

@app.post("/volatility-surface")
async def generate_volatility_surface(request: VolatilitySurfaceRequest):
    """
    Generate implied volatility surface using specified model and parameters.
    
    Args:
        request: Volatility surface request with strikes, expiries, and model parameters
        
    Returns:
        Implied volatility surface as 2D array
    """
    try:
        # Select model
        if request.model_type.lower() == "heston":
            model = heston_model
            # Set Heston parameters
            from .models.heston import HestonParameters
            params = HestonParameters(**request.parameters)
            model.set_parameters(params)
        elif request.model_type.lower() == "black_scholes":
            model = black_scholes_model
            model.set_volatility(request.parameters.get('volatility', 0.2))
        elif request.model_type.lower() == "sabr":
            model = sabr_model
            # Set SABR parameters
            from .models.sabr import SABRParameters
            params = SABRParameters(**request.parameters)
            model.set_parameters(params)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model type: {request.model_type}")
            
        # Generate volatility surface
        surface = model.generate_volatility_surface(
            request.spot, request.strikes, request.expiries,
            request.risk_free_rate, request.dividend_yield
        )
        
        return {
            "volatility_surface": surface.tolist(),
            "strikes": request.strikes,
            "expiries": request.expiries,
            "model_type": request.model_type,
            "parameters": request.parameters
        }
        
    except Exception as e:
        logger.error(f"Error in volatility surface generation: {e}")
        raise HTTPException(status_code=500, detail=f"Surface generation error: {str(e)}")

@app.post("/compare-models")
async def compare_models(symbol: str):
    """
    Compare multiple models calibrated to the same market data.
    
    Args:
        symbol: Stock symbol for market data
        
    Returns:
        Comparison results for all models
    """
    try:
        # Fetch and validate market data
        options_chain = await market_data_aggregator.get_options_chain(symbol)
        
        if options_chain is None:
            raise HTTPException(status_code=404, detail=f"No data available for symbol {symbol}")
            
        raw_data = options_chain.to_dict()
        validated_data = data_validator.validate_options_chain(raw_data)
        
        if validated_data.get('error', False):
            raise HTTPException(status_code=400, detail=validated_data['error_message'])
            
        # Prepare calibration data
        calibration_data = data_validator.prepare_calibration_data(validated_data)
        
        if calibration_data is None:
            raise HTTPException(status_code=400, detail="Insufficient data for calibration")
            
        # Calibrate all models
        models = {
            "heston": heston_model,
            "black_scholes": black_scholes_model,
            "sabr": sabr_model
        }
        
        comparison_results = {}
        
        for model_name, model in models.items():
            try:
                # Calibrate model
                calibration_result = model.calibrate(calibration_data)
                
                # Validate calibration
                validation_result = validator.validate_calibration(
                    calibration_result.__dict__, calibration_data, model.name)
                
                comparison_results[model_name] = {
                    "calibration_result": calibration_result.__dict__,
                    "validation_result": {
                        "is_valid": validation_result.is_valid,
                        "warnings": validation_result.warnings,
                        "quality_score": validation_result.quality_score,
                        "stability_score": validation_result.stability_score
                    }
                }
                
            except Exception as e:
                logger.warning(f"Failed to calibrate {model_name}: {e}")
                comparison_results[model_name] = {
                    "error": str(e),
                    "calibration_result": None,
                    "validation_result": None
                }
                
        return {
            "symbol": symbol,
            "comparison_results": comparison_results,
            "market_data_stats": validated_data.get('validation_stats', {})
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in model comparison: {e}")
        raise HTTPException(status_code=500, detail=f"Comparison error: {str(e)}")

@app.get("/cache/clear")
async def clear_cache():
    """Clear all cached data."""
    try:
        market_data_aggregator.clear_cache()
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=f"Cache clearing error: {str(e)}")

@app.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics."""
    try:
        return market_data_aggregator.get_cache_stats()
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail=f"Cache stats error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 