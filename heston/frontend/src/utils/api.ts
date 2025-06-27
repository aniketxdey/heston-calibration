/**
 * API utilities for backend communication
 */

export const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export interface CalibrationRequest {
  symbol: string;
  model_type: string;
  use_puts?: boolean;
  min_volume?: number;
  max_bid_ask_spread?: number;
}

export interface PricingRequest {
  spot: number;
  strike: number;
  expiry: number;
  risk_free_rate?: number;
  dividend_yield?: number;
  model_type: string;
  parameters: Record<string, number>;
}

export interface GreeksRequest {
  spot: number;
  strike: number;
  expiry: number;
  risk_free_rate?: number;
  dividend_yield?: number;
  model_type: string;
  parameters: Record<string, number>;
}

export interface VolatilitySurfaceRequest {
  spot: number;
  strikes: number[];
  expiries: number[];
  risk_free_rate?: number;
  dividend_yield?: number;
  model_type: string;
  parameters: Record<string, number>;
}

export interface MarketData {
  symbol: string;
  data: {
    symbol: string;
    spot_price: number;
    risk_free_rate: number;
    calls: any[];
    puts: any[];
    validation_stats: any;
    timestamp: string;
  };
  cache_stats: any;
}

export interface CalibrationResult {
  symbol: string;
  model_type: string;
  calibration_result: {
    parameters: Record<string, number>;
    objective_value: number;
    convergence: boolean;
    iterations: number;
    fit_quality: number;
    errors: number[];
  };
  validation_result: {
    is_valid: boolean;
    warnings: string[];
    quality_score: number;
    stability_score: number;
  };
  bootstrap_uncertainty: any;
  market_data_stats: any;
}

export interface ModelInfo {
  name: string;
  description: string;
  parameters: Record<string, {
    description: string;
    bounds: [number, number];
  }>;
}

export interface ModelsResponse {
  models: ModelInfo[];
}

/**
 * Generic API request function
 */
async function apiRequest<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  
  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`API request failed: ${response.status} ${errorText}`);
  }

  return response.json();
}

/**
 * API functions
 */
export const api = {
  // Get available models
  getModels: (): Promise<ModelsResponse> => 
    apiRequest<ModelsResponse>('/models'),

  // Get market data
  getMarketData: (symbol: string): Promise<MarketData> => 
    apiRequest<MarketData>(`/market-data/${symbol}`),

  // Calibrate model
  calibrateModel: (request: CalibrationRequest): Promise<CalibrationResult> => 
    apiRequest<CalibrationResult>('/calibrate', {
      method: 'POST',
      body: JSON.stringify(request),
    }),

  // Price option
  priceOption: (request: PricingRequest): Promise<any> => 
    apiRequest<any>('/price', {
      method: 'POST',
      body: JSON.stringify(request),
    }),

  // Calculate Greeks
  calculateGreeks: (request: GreeksRequest): Promise<any> => 
    apiRequest<any>('/greeks', {
      method: 'POST',
      body: JSON.stringify(request),
    }),

  // Generate volatility surface
  generateVolatilitySurface: (request: VolatilitySurfaceRequest): Promise<any> => 
    apiRequest<any>('/volatility-surface', {
      method: 'POST',
      body: JSON.stringify(request),
    }),

  // Compare models
  compareModels: (symbol: string): Promise<any> => 
    apiRequest<any>('/compare-models', {
      method: 'POST',
      body: JSON.stringify({ symbol }),
    }),

  // Health check
  healthCheck: (): Promise<any> => 
    apiRequest<any>('/health'),

  // Clear cache
  clearCache: (): Promise<any> => 
    apiRequest<any>('/cache/clear'),

  // Get cache stats
  getCacheStats: (): Promise<any> => 
    apiRequest<any>('/cache/stats'),
}; 