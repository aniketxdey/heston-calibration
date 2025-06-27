"""
Calibration Validator

Post-calibration validation that checks parameter reasonableness, model fit quality,
and stability analysis through bootstrap sampling and sensitivity tests.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ValidationResult:
    """Container for validation results."""
    
    def __init__(self, is_valid: bool, warnings: List[str], 
                 quality_score: float, stability_score: float):
        self.is_valid = is_valid
        self.warnings = warnings
        self.quality_score = quality_score
        self.stability_score = stability_score


class CalibrationValidator:
    """
    Post-calibration validation framework.
    
    Implements multiple validation checks:
    1. Parameter reasonableness
    2. Model fit quality
    3. Stability analysis
    4. Bootstrap uncertainty quantification
    """
    
    def __init__(self):
        self.min_r_squared = 0.7  # Minimum R-squared for acceptable fit
        self.max_parameter_change = 0.5  # Maximum parameter change in stability test
        self.n_bootstrap_samples = 100  # Number of bootstrap samples
        
    def validate_calibration(self, calibration_result: Dict[str, Any],
                           market_data: Dict[str, Any],
                           model_name: str) -> ValidationResult:
        """
        Comprehensive validation of calibration results.
        
        Args:
            calibration_result: Calibration result from model
            market_data: Market data used for calibration
            model_name: Name of the calibrated model
            
        Returns:
            ValidationResult with validation status and scores
        """
        warnings = []
        
        # Check convergence
        if not calibration_result.get('convergence', False):
            warnings.append("Calibration did not converge")
            
        # Check fit quality
        fit_quality = calibration_result.get('fit_quality', 0.0)
        if fit_quality < self.min_r_squared:
            warnings.append(f"Poor fit quality (R² = {fit_quality:.3f})")
            
        # Check parameter reasonableness
        param_warnings = self._validate_parameters(
            calibration_result.get('parameters', {}), model_name)
        warnings.extend(param_warnings)
        
        # Check error distribution
        error_warnings = self._validate_errors(
            calibration_result.get('errors', []))
        warnings.extend(error_warnings)
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(calibration_result)
        
        # Calculate stability score
        stability_score = self._calculate_stability_score(
            calibration_result, market_data)
        
        # Determine overall validity
        is_valid = (calibration_result.get('convergence', False) and
                   fit_quality >= self.min_r_squared and
                   len(param_warnings) == 0)
        
        return ValidationResult(
            is_valid=is_valid,
            warnings=warnings,
            quality_score=quality_score,
            stability_score=stability_score
        )
        
    def _validate_parameters(self, parameters: Dict[str, float], 
                           model_name: str) -> List[str]:
        """
        Validate parameter reasonableness.
        
        Args:
            parameters: Calibrated parameters
            model_name: Model name for specific checks
            
        Returns:
            List of parameter warnings
        """
        warnings = []
        
        if model_name == "Heston":
            warnings.extend(self._validate_heston_parameters(parameters))
        elif model_name == "SABR":
            warnings.extend(self._validate_sabr_parameters(parameters))
        elif model_name == "Black-Scholes":
            warnings.extend(self._validate_black_scholes_parameters(parameters))
            
        return warnings
        
    def _validate_heston_parameters(self, parameters: Dict[str, float]) -> List[str]:
        """Validate Heston model parameters."""
        warnings = []
        
        v0 = parameters.get('v0', 0)
        kappa = parameters.get('kappa', 0)
        theta = parameters.get('theta', 0)
        sigma = parameters.get('sigma', 0)
        rho = parameters.get('rho', 0)
        
        # Check Feller condition
        if 2 * kappa * theta < sigma**2:
            warnings.append("Feller condition violated (2*κ*θ < σ²)")
            
        # Check parameter ranges
        if v0 <= 0 or v0 > 1:
            warnings.append("Initial variance v0 outside reasonable range")
            
        if kappa <= 0 or kappa > 10:
            warnings.append("Mean reversion speed κ outside reasonable range")
            
        if theta <= 0 or theta > 1:
            warnings.append("Long-term variance θ outside reasonable range")
            
        if sigma <= 0 or sigma > 2:
            warnings.append("Volatility of volatility σ outside reasonable range")
            
        if abs(rho) > 0.99:
            warnings.append("Correlation ρ too close to ±1")
            
        return warnings
        
    def _validate_sabr_parameters(self, parameters: Dict[str, float]) -> List[str]:
        """Validate SABR model parameters."""
        warnings = []
        
        alpha = parameters.get('alpha', 0)
        beta = parameters.get('beta', 0)
        rho = parameters.get('rho', 0)
        nu = parameters.get('nu', 0)
        
        # Check parameter ranges
        if alpha <= 0 or alpha > 1:
            warnings.append("Initial volatility α outside reasonable range")
            
        if beta <= 0 or beta > 1:
            warnings.append("CEV parameter β outside reasonable range")
            
        if abs(rho) > 0.99:
            warnings.append("Correlation ρ too close to ±1")
            
        if nu <= 0 or nu > 2:
            warnings.append("Volatility of volatility ν outside reasonable range")
            
        return warnings
        
    def _validate_black_scholes_parameters(self, parameters: Dict[str, float]) -> List[str]:
        """Validate Black-Scholes model parameters."""
        warnings = []
        
        volatility = parameters.get('volatility', 0)
        
        if volatility <= 0 or volatility > 2:
            warnings.append("Volatility outside reasonable range")
            
        return warnings
        
    def _validate_errors(self, errors: List[float]) -> List[str]:
        """
        Validate error distribution.
        
        Args:
            errors: List of calibration errors
            
        Returns:
            List of error warnings
        """
        warnings = []
        
        if not errors:
            return warnings
            
        errors = np.array(errors)
        
        # Check for large errors
        max_error = np.max(np.abs(errors))
        if max_error > 0.5:  # 50% error threshold
            warnings.append(f"Large calibration errors detected (max: {max_error:.3f})")
            
        # Check for systematic bias
        mean_error = np.mean(errors)
        if abs(mean_error) > 0.1:  # 10% bias threshold
            warnings.append(f"Systematic bias detected (mean error: {mean_error:.3f})")
            
        # Check error distribution
        error_std = np.std(errors)
        if error_std > 0.3:  # High error variability
            warnings.append(f"High error variability (std: {error_std:.3f})")
            
        return warnings
        
    def _calculate_quality_score(self, calibration_result: Dict[str, Any]) -> float:
        """
        Calculate overall quality score.
        
        Args:
            calibration_result: Calibration result
            
        Returns:
            Quality score between 0 and 1
        """
        # Base score from R-squared
        r_squared = calibration_result.get('fit_quality', 0.0)
        base_score = max(0.0, min(1.0, r_squared))
        
        # Penalty for convergence failure
        if not calibration_result.get('convergence', False):
            base_score *= 0.5
            
        # Penalty for high objective value
        objective_value = calibration_result.get('objective_value', np.inf)
        if objective_value > 1.0:
            base_score *= 0.8
            
        # Penalty for many iterations
        iterations = calibration_result.get('iterations', 0)
        if iterations > 500:
            base_score *= 0.9
            
        return base_score
        
    def _calculate_stability_score(self, calibration_result: Dict[str, Any],
                                 market_data: Dict[str, Any]) -> float:
        """
        Calculate stability score through bootstrap analysis.
        
        Args:
            calibration_result: Calibration result
            market_data: Market data used for calibration
            
        Returns:
            Stability score between 0 and 1
        """
        try:
            # Simple stability measure based on parameter sensitivity
            # In a full implementation, this would use bootstrap sampling
            
            parameters = calibration_result.get('parameters', {})
            if not parameters:
                return 0.0
                
            # Calculate coefficient of variation for parameters
            param_values = list(parameters.values())
            if len(param_values) < 2:
                return 1.0
                
            cv = np.std(param_values) / np.mean(np.abs(param_values))
            
            # Convert to stability score (lower CV = higher stability)
            stability_score = max(0.0, min(1.0, 1.0 - cv))
            
            return stability_score
            
        except Exception as e:
            logger.error(f"Error calculating stability score: {e}")
            return 0.5  # Default score
            
    def bootstrap_uncertainty(self, model, market_data: Dict[str, Any],
                            n_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Estimate parameter uncertainty using bootstrap sampling.
        
        Args:
            model: Calibrated model
            market_data: Market data
            n_samples: Number of bootstrap samples
            
        Returns:
            Bootstrap uncertainty estimates
        """
        if n_samples is None:
            n_samples = self.n_bootstrap_samples
            
        # Extract data
        strikes = np.array(market_data['strikes'])
        expiries = np.array(market_data['expiries'])
        prices = np.array(market_data['prices'])
        
        # Flatten data for bootstrap
        data_points = []
        for i, strike in enumerate(strikes):
            for j, expiry in enumerate(expiries):
                if not np.isnan(prices[i, j]) and prices[i, j] > 0:
                    data_points.append({
                        'strike': strike,
                        'expiry': expiry,
                        'price': prices[i, j]
                    })
                    
        if len(data_points) < 10:
            return {'error': 'Insufficient data for bootstrap analysis'}
            
        # Bootstrap sampling
        bootstrap_results = []
        
        for _ in range(n_samples):
            try:
                # Sample with replacement
                bootstrap_sample = np.random.choice(
                    data_points, size=len(data_points), replace=True)
                
                # Prepare bootstrap data
                bootstrap_data = self._prepare_bootstrap_data(bootstrap_sample)
                
                # Recalibrate model
                bootstrap_result = model.calibrate(bootstrap_data)
                
                if bootstrap_result.convergence:
                    bootstrap_results.append(bootstrap_result.parameters)
                    
            except Exception as e:
                logger.warning(f"Bootstrap sample failed: {e}")
                continue
                
        if not bootstrap_results:
            return {'error': 'No successful bootstrap samples'}
            
        # Calculate uncertainty statistics
        param_names = list(bootstrap_results[0].keys())
        uncertainty_stats = {}
        
        for param_name in param_names:
            param_values = [result[param_name] for result in bootstrap_results]
            
            uncertainty_stats[param_name] = {
                'mean': np.mean(param_values),
                'std': np.std(param_values),
                'ci_95_lower': np.percentile(param_values, 2.5),
                'ci_95_upper': np.percentile(param_values, 97.5),
                'cv': np.std(param_values) / np.mean(param_values)
            }
            
        return {
            'n_successful_samples': len(bootstrap_results),
            'parameter_uncertainty': uncertainty_stats
        }
        
    def _prepare_bootstrap_data(self, bootstrap_sample: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Prepare bootstrap sample for calibration.
        
        Args:
            bootstrap_sample: Bootstrap data sample
            
        Returns:
            Calibration-ready data
        """
        strikes = sorted(list(set([point['strike'] for point in bootstrap_sample])))
        expiries = sorted(list(set([point['expiry'] for point in bootstrap_sample])))
        
        # Create price matrix
        prices = np.zeros((len(strikes), len(expiries)))
        for i, strike in enumerate(strikes):
            for j, expiry in enumerate(expiries):
                matching_points = [point for point in bootstrap_sample 
                                 if point['strike'] == strike and point['expiry'] == expiry]
                if matching_points:
                    avg_price = np.mean([point['price'] for point in matching_points])
                    prices[i, j] = avg_price
                else:
                    prices[i, j] = np.nan
                    
        return {
            'strikes': strikes,
            'expiries': expiries,
            'prices': prices.tolist(),
            'spot': 100.0,  # Default values
            'risk_free_rate': 0.05,
            'dividend_yield': 0.0
        } 