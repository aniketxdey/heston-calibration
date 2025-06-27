import React from 'react';
import { Box, Typography, Paper } from '@mui/material';

interface ModelComparisonPanelProps {
  symbol: string;
  models: any[];
}

const ModelComparisonPanel: React.FC<ModelComparisonPanelProps> = ({
  symbol,
  models,
}) => {
  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Model Comparison
      </Typography>
      <Paper sx={{ p: 3, textAlign: 'center' }}>
        <Typography color="text.secondary">
          Side-by-side model comparison will be implemented here.
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
          This will compare Heston, Black-Scholes, and SABR models for {symbol}.
        </Typography>
      </Paper>
    </Box>
  );
};

export default ModelComparisonPanel; 