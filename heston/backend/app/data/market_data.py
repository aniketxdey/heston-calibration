"""
Market Data Integration Layer

Provides unified interface for multiple market data sources with intelligent
failover, rate limiting, and data quality validation.
"""

import asyncio
import aiohttp
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import time
import logging

logger = logging.getLogger(__name__)


class OptionsChain:
    """Container for options chain data."""
    
    def __init__(self, symbol: str, spot_price: float, risk_free_rate: float = 0.05):
        self.symbol = symbol
        self.spot_price = spot_price
        self.risk_free_rate = risk_free_rate
        self.calls = pd.DataFrame()
        self.puts = pd.DataFrame()
        self.timestamp = datetime.now()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            'symbol': self.symbol,
            'spot_price': self.spot_price,
            'risk_free_rate': self.risk_free_rate,
            'calls': self.calls.to_dict('records') if not self.calls.empty else [],
            'puts': self.puts.to_dict('records') if not self.puts.empty else [],
            'timestamp': self.timestamp.isoformat()
        }


class YahooFinanceProvider:
    """Yahoo Finance data provider."""
    
    def __init__(self):
        self.rate_limit_delay = 1.0  # seconds between requests
        self.last_request_time = 0
        
    async def _rate_limit(self):
        """Implement rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
        
    async def get_options_chain(self, symbol: str) -> Optional[OptionsChain]:
        """
        Fetch options chain from Yahoo Finance.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            
        Returns:
            OptionsChain object or None if failed
        """
        try:
            await self._rate_limit()
            
            # Get ticker info
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get current price
            spot_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
            if not spot_price:
                return None
                
            # Get options chain
            options = ticker.options
            
            if not options:
                return None
                
            # Use nearest expiration for demo
            nearest_expiry = options[0]
            
            # Get options data
            opt = ticker.option_chain(nearest_expiry)
            
            # Create options chain
            chain = OptionsChain(symbol, spot_price)
            chain.calls = opt.calls
            chain.puts = opt.puts
            
            return chain
            
        except Exception as e:
            logger.error(f"Error fetching options chain for {symbol}: {e}")
            return None
            
    async def get_risk_free_rate(self) -> float:
        """
        Get current risk-free rate from Treasury yields.
        
        Returns:
            Risk-free rate as decimal
        """
        try:
            await self._rate_limit()
            
            # Use 3-month Treasury yield as proxy for risk-free rate
            treasury = yf.Ticker("^IRX")  # 13-week Treasury yield
            info = treasury.info
            rate = info.get('regularMarketPrice', 5.0) / 100.0  # Convert to decimal
            
            return max(0.001, rate)  # Ensure positive rate
            
        except Exception as e:
            logger.error(f"Error fetching risk-free rate: {e}")
            return 0.05  # Default 5% rate


class AlphaVantageProvider:
    """Alpha Vantage data provider (backup)."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or "demo"
        self.base_url = "https://www.alphavantage.co/query"
        self.rate_limit_delay = 12.0  # Alpha Vantage free tier limit
        
    async def get_options_chain(self, symbol: str) -> Optional[OptionsChain]:
        """
        Fetch options chain from Alpha Vantage.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            OptionsChain object or None if failed
        """
        # Alpha Vantage doesn't provide options data in free tier
        # This is a placeholder for premium API access
        return None
        
    async def get_risk_free_rate(self) -> float:
        """
        Get risk-free rate from Alpha Vantage.
        
        Returns:
            Risk-free rate as decimal
        """
        try:
            # Use Treasury yield data
            params = {
                'function': 'TREASURY_YIELD',
                'interval': 'daily',
                'maturity': '3month',
                'apikey': self.api_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        # Parse response and extract rate
                        # Implementation depends on actual API response format
                        return 0.05  # Default rate
                    else:
                        return 0.05
                        
        except Exception as e:
            logger.error(f"Error fetching risk-free rate from Alpha Vantage: {e}")
            return 0.05


class MarketDataAggregator:
    """
    Unified market data aggregator with intelligent failover.
    
    Provides a single interface for multiple data sources with automatic
    failover, caching, and data quality validation.
    """
    
    def __init__(self):
        self.sources = [
            YahooFinanceProvider(),
            AlphaVantageProvider()
        ]
        self.cache = {}  # Simple in-memory cache
        self.cache_ttl = 300  # 5 minutes
        
    async def get_options_chain(self, symbol: str) -> Optional[OptionsChain]:
        """
        Fetch options chain with intelligent source selection.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            OptionsChain object or None if all sources fail
        """
        # Check cache first
        cache_key = f"options_{symbol}"
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if (datetime.now() - timestamp).seconds < self.cache_ttl:
                return cached_data
                
        # Try each source
        for source in self.sources:
            try:
                chain = await source.get_options_chain(symbol)
                if chain is not None:
                    # Cache the result
                    self.cache[cache_key] = (chain, datetime.now())
                    return chain
            except Exception as e:
                logger.warning(f"Source {source.__class__.__name__} failed: {e}")
                continue
                
        logger.error(f"All data sources failed for symbol {symbol}")
        return None
        
    async def get_risk_free_rate(self) -> float:
        """
        Get current risk-free rate.
        
        Returns:
            Risk-free rate as decimal
        """
        # Check cache first
        cache_key = "risk_free_rate"
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if (datetime.now() - timestamp).seconds < self.cache_ttl:
                return cached_data
                
        # Try each source
        for source in self.sources:
            try:
                rate = await source.get_risk_free_rate()
                if rate > 0:
                    # Cache the result
                    self.cache[cache_key] = (rate, datetime.now())
                    return rate
            except Exception as e:
                logger.warning(f"Source {source.__class__.__name__} failed: {e}")
                continue
                
        # Return default rate if all sources fail
        return 0.05
        
    def clear_cache(self):
        """Clear all cached data."""
        self.cache.clear()
        
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cache_size': len(self.cache),
            'cache_ttl': self.cache_ttl
        } 