import React, { useState } from 'react';
import {
  Box,
  Typography,
  TextField,
  Button,
  Alert,
  CircularProgress,
  Card,
  CardContent,
  Grid,
  Chip,
} from '@mui/material';
import { useQuery } from '@tanstack/react-query';
import { api, MarketData } from '../utils/api';

interface MarketDataPanelProps {
  selectedSymbol: string;
  onSymbolChange: (symbol: string) => void;
}

const MarketDataPanel: React.FC<MarketDataPanelProps> = ({
  selectedSymbol,
  onSymbolChange,
}) => {
  const [inputSymbol, setInputSymbol] = useState(selectedSymbol);

  // Fetch market data
  const {
    data: marketData,
    isLoading,
    error,
    refetch,
  } = useQuery<MarketData>({
    queryKey: ['marketData', selectedSymbol],
    queryFn: () => api.getMarketData(selectedSymbol),
    enabled: !!selectedSymbol,
    refetchInterval: 300000, // Refetch every 5 minutes
  });

  const handleSymbolSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputSymbol.trim()) {
      onSymbolChange(inputSymbol.trim().toUpperCase());
    }
  };

  const handleRefresh = () => {
    refetch();
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Market Data
      </Typography>

      {/* Symbol Input */}
      <Box component="form" onSubmit={handleSymbolSubmit} sx={{ mb: 2 }}>
        <Grid container spacing={1}>
          <Grid item xs={8}>
            <TextField
              fullWidth
              size="small"
              label="Symbol"
              value={inputSymbol}
              onChange={(e) => setInputSymbol(e.target.value)}
              placeholder="e.g., AAPL, MSFT"
            />
          </Grid>
          <Grid item xs={4}>
            <Button
              fullWidth
              variant="contained"
              type="submit"
              size="small"
              disabled={!inputSymbol.trim()}
            >
              Load
            </Button>
          </Grid>
        </Grid>
      </Box>

      {/* Loading State */}
      {isLoading && (
        <Box display="flex" justifyContent="center" p={2}>
          <CircularProgress size={24} />
        </Box>
      )}

      {/* Error State */}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          Failed to load market data for {selectedSymbol}
        </Alert>
      )}

      {/* Market Data Display */}
      {marketData && (
        <Box>
          {/* Current Price */}
          <Card sx={{ mb: 2 }}>
            <CardContent>
              <Typography variant="h5" component="div">
                ${marketData.data.spot_price?.toFixed(2) || 'N/A'}
              </Typography>
              <Typography color="text.secondary">
                {marketData.symbol} Current Price
              </Typography>
            </CardContent>
          </Card>

          {/* Risk-Free Rate */}
          <Card sx={{ mb: 2 }}>
            <CardContent>
              <Typography variant="h6" component="div">
                {(marketData.data.risk_free_rate * 100).toFixed(2)}%
              </Typography>
              <Typography color="text.secondary">
                Risk-Free Rate
              </Typography>
            </CardContent>
          </Card>

          {/* Options Summary */}
          <Card sx={{ mb: 2 }}>
            <CardContent>
              <Typography variant="subtitle1" gutterBottom>
                Options Summary
              </Typography>
              <Grid container spacing={1}>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    Calls: {marketData.data.calls?.length || 0}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    Puts: {marketData.data.puts?.length || 0}
                  </Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>

          {/* Validation Stats */}
          {marketData.data.validation_stats && (
            <Card sx={{ mb: 2 }}>
              <CardContent>
                <Typography variant="subtitle1" gutterBottom>
                  Data Quality
                </Typography>
                <Grid container spacing={1}>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">
                      Total: {marketData.data.validation_stats.total_original || 0}
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="body2" color="text.secondary">
                      Valid: {marketData.data.validation_stats.total_cleaned || 0}
                    </Typography>
                  </Grid>
                </Grid>
                <Box sx={{ mt: 1 }}>
                  <Chip
                    label={`${((marketData.data.validation_stats.overall_retention_rate || 0) * 100).toFixed(1)}% Retention`}
                    color={
                      (marketData.data.validation_stats.overall_retention_rate || 0) > 0.8
                        ? 'success'
                        : (marketData.data.validation_stats.overall_retention_rate || 0) > 0.5
                        ? 'warning'
                        : 'error'
                    }
                    size="small"
                  />
                </Box>
              </CardContent>
            </Card>
          )}

          {/* Cache Info */}
          {marketData.cache_stats && (
            <Card>
              <CardContent>
                <Typography variant="subtitle1" gutterBottom>
                  Cache Status
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Cache Size: {marketData.cache_stats.cache_size || 0}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  TTL: {marketData.cache_stats.cache_ttl || 0}s
                </Typography>
              </CardContent>
            </Card>
          )}

          {/* Refresh Button */}
          <Box sx={{ mt: 2 }}>
            <Button
              fullWidth
              variant="outlined"
              onClick={handleRefresh}
              disabled={isLoading}
            >
              Refresh Data
            </Button>
          </Box>
        </Box>
      )}
    </Box>
  );
};

export default MarketDataPanel; 