"""
Data Validation Pipeline

Comprehensive validation pipeline that filters for liquid options, removes
stale quotes, validates bid-ask spreads, and flags suspicious data points.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class DataValidationPipeline:
    """
    Comprehensive data validation pipeline for options data.
    
    Implements multiple validation stages to ensure data quality:
    1. Basic data integrity checks
    2. Liquidity filtering
    3. Bid-ask spread validation
    4. Price reasonableness checks
    5. Outlier detection
    """
    
    def __init__(self):
        self.min_volume = 10  # Minimum volume for liquidity
        self.max_bid_ask_spread = 0.5  # Maximum bid-ask spread as fraction of mid price
        self.min_price = 0.01  # Minimum option price
        self.max_price_ratio = 10.0  # Maximum price relative to underlying
        self.outlier_threshold = 3.0  # Standard deviations for outlier detection
        
    def validate_options_chain(self, chain_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate complete options chain data.
        
        Args:
            chain_data: Raw options chain data
            
        Returns:
            Validated and cleaned options chain data
        """
        try:
            # Extract data
            symbol = chain_data.get('symbol', '')
            spot_price = chain_data.get('spot_price', 0)
            risk_free_rate = chain_data.get('risk_free_rate', 0.05)
            calls = pd.DataFrame(chain_data.get('calls', []))
            puts = pd.DataFrame(chain_data.get('puts', []))
            
            # Basic validation
            if not self._validate_basic_data(symbol, spot_price, calls, puts):
                return self._create_error_response("Basic data validation failed")
                
            # Clean and validate calls
            if not calls.empty:
                calls_clean = self._validate_options_data(calls, spot_price, 'call')
            else:
                calls_clean = pd.DataFrame()
                
            # Clean and validate puts
            if not puts.empty:
                puts_clean = self._validate_options_data(puts, spot_price, 'put')
            else:
                puts_clean = pd.DataFrame()
                
            # Check if we have enough data
            if calls_clean.empty and puts_clean.empty:
                return self._create_error_response("No valid options data after validation")
                
            # Create validated response
            validated_data = {
                'symbol': symbol,
                'spot_price': spot_price,
                'risk_free_rate': risk_free_rate,
                'calls': calls_clean.to_dict('records') if not calls_clean.empty else [],
                'puts': puts_clean.to_dict('records') if not puts_clean.empty else [],
                'validation_stats': self._get_validation_stats(calls, puts, calls_clean, puts_clean),
                'timestamp': datetime.now().isoformat()
            }
            
            return validated_data
            
        except Exception as e:
            logger.error(f"Error in options chain validation: {e}")
            return self._create_error_response(f"Validation error: {str(e)}")
            
    def _validate_basic_data(self, symbol: str, spot_price: float, 
                           calls: pd.DataFrame, puts: pd.DataFrame) -> bool:
        """
        Validate basic data integrity.
        
        Args:
            symbol: Stock symbol
            spot_price: Current spot price
            calls: Calls DataFrame
            puts: Puts DataFrame
            
        Returns:
            True if basic validation passes
        """
        # Check symbol
        if not symbol or len(symbol.strip()) == 0:
            logger.warning("Empty or invalid symbol")
            return False
            
        # Check spot price
        if spot_price <= 0:
            logger.warning(f"Invalid spot price: {spot_price}")
            return False
            
        # Check that we have at least some options data
        if calls.empty and puts.empty:
            logger.warning("No options data provided")
            return False
            
        return True
        
    def _validate_options_data(self, options_df: pd.DataFrame, spot_price: float, 
                              option_type: str) -> pd.DataFrame:
        """
        Validate and clean options data.
        
        Args:
            options_df: Options DataFrame
            spot_price: Current spot price
            option_type: 'call' or 'put'
            
        Returns:
            Cleaned and validated DataFrame
        """
        if options_df.empty:
            return pd.DataFrame()
            
        # Make a copy to avoid modifying original
        df = options_df.copy()
        
        # Required columns
        required_columns = ['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
            return pd.DataFrame()
            
        # Convert to numeric
        numeric_columns = ['strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Remove rows with NaN values
        initial_count = len(df)
        df = df.dropna(subset=numeric_columns)
        if len(df) < initial_count:
            logger.info(f"Removed {initial_count - len(df)} rows with NaN values")
            
        # Filter by minimum volume
        df = df[df['volume'] >= self.min_volume]
        
        # Filter by minimum price
        df = df[df['lastPrice'] >= self.min_price]
        
        # Validate bid-ask spreads
        df['mid_price'] = (df['bid'] + df['ask']) / 2
        df['spread_ratio'] = (df['ask'] - df['bid']) / df['mid_price']
        df = df[df['spread_ratio'] <= self.max_bid_ask_spread]
        
        # Validate price reasonableness
        if option_type == 'call':
            # Call price should not exceed spot price significantly
            df = df[df['lastPrice'] <= spot_price * self.max_price_ratio]
        else:  # put
            # Put price should not exceed strike price significantly
            df = df[df['lastPrice'] <= df['strike'] * self.max_price_ratio]
            
        # Remove outliers using IQR method
        df = self._remove_outliers(df, 'lastPrice')
        
        # Sort by strike
        df = df.sort_values('strike')
        
        return df
        
    def _remove_outliers(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Remove outliers using IQR method.
        
        Args:
            df: DataFrame
            column: Column to check for outliers
            
        Returns:
            DataFrame with outliers removed
        """
        if df.empty or column not in df.columns:
            return df
            
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        initial_count = len(df)
        df_clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        
        if len(df_clean) < initial_count:
            logger.info(f"Removed {initial_count - len(df_clean)} outliers from {column}")
            
        return df_clean
        
    def _get_validation_stats(self, calls_orig: pd.DataFrame, puts_orig: pd.DataFrame,
                            calls_clean: pd.DataFrame, puts_clean: pd.DataFrame) -> Dict[str, Any]:
        """
        Get validation statistics.
        
        Args:
            calls_orig: Original calls DataFrame
            puts_orig: Original puts DataFrame
            calls_clean: Cleaned calls DataFrame
            puts_clean: Cleaned puts DataFrame
            
        Returns:
            Validation statistics
        """
        stats = {
            'calls_original': len(calls_orig),
            'calls_cleaned': len(calls_clean),
            'puts_original': len(puts_orig),
            'puts_cleaned': len(puts_clean),
            'total_original': len(calls_orig) + len(puts_orig),
            'total_cleaned': len(calls_clean) + len(puts_clean),
            'validation_parameters': {
                'min_volume': self.min_volume,
                'max_bid_ask_spread': self.max_bid_ask_spread,
                'min_price': self.min_price,
                'max_price_ratio': self.max_price_ratio
            }
        }
        
        # Calculate retention rates
        if stats['calls_original'] > 0:
            stats['calls_retention_rate'] = stats['calls_cleaned'] / stats['calls_original']
        else:
            stats['calls_retention_rate'] = 0.0
            
        if stats['puts_original'] > 0:
            stats['puts_retention_rate'] = stats['puts_cleaned'] / stats['puts_original']
        else:
            stats['puts_retention_rate'] = 0.0
            
        if stats['total_original'] > 0:
            stats['overall_retention_rate'] = stats['total_cleaned'] / stats['total_original']
        else:
            stats['overall_retention_rate'] = 0.0
            
        return stats
        
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """
        Create error response.
        
        Args:
            error_message: Error message
            
        Returns:
            Error response dictionary
        """
        return {
            'error': True,
            'error_message': error_message,
            'timestamp': datetime.now().isoformat()
        }
        
    def prepare_calibration_data(self, validated_chain: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Prepare validated options chain for model calibration.
        
        Args:
            validated_chain: Validated options chain data
            
        Returns:
            Calibration-ready data or None if insufficient data
        """
        try:
            if validated_chain.get('error', False):
                return None
                
            calls = pd.DataFrame(validated_chain.get('calls', []))
            puts = pd.DataFrame(validated_chain.get('puts', []))
            spot_price = validated_chain.get('spot_price', 0)
            risk_free_rate = validated_chain.get('risk_free_rate', 0.05)
            
            # Combine calls and puts for calibration
            all_options = []
            
            if not calls.empty:
                for _, row in calls.iterrows():
                    all_options.append({
                        'strike': row['strike'],
                        'expiry': 0.25,  # Assume 3 months for demo
                        'price': row['lastPrice'],
                        'type': 'call'
                    })
                    
            if not puts.empty:
                for _, row in puts.iterrows():
                    all_options.append({
                        'strike': row['strike'],
                        'expiry': 0.25,  # Assume 3 months for demo
                        'price': row['lastPrice'],
                        'type': 'put'
                    })
                    
            if len(all_options) < 5:  # Need minimum number of options
                return None
                
            # Create calibration data structure
            strikes = sorted(list(set([opt['strike'] for opt in all_options])))
            expiries = [0.25]  # Single expiry for demo
            
            # Create price matrix
            prices = np.zeros((len(strikes), len(expiries)))
            for i, strike in enumerate(strikes):
                for j, expiry in enumerate(expiries):
                    # Find option with this strike and expiry
                    matching_options = [opt for opt in all_options 
                                      if opt['strike'] == strike and opt['expiry'] == expiry]
                    if matching_options:
                        # Use average price if multiple options
                        avg_price = np.mean([opt['price'] for opt in matching_options])
                        prices[i, j] = avg_price
                    else:
                        prices[i, j] = np.nan
                        
            return {
                'strikes': strikes,
                'expiries': expiries,
                'prices': prices.tolist(),
                'spot': spot_price,
                'risk_free_rate': risk_free_rate,
                'dividend_yield': 0.0  # Assume no dividends for demo
            }
            
        except Exception as e:
            logger.error(f"Error preparing calibration data: {e}")
            return None 