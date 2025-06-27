import React from 'react';
import { Box, Typography, Paper } from '@mui/material';

interface VolatilitySurfacePanelProps {
  symbol: string;
  calibrationResults: any;
}

const VolatilitySurfacePanel: React.FC<VolatilitySurfacePanelProps> = ({
  symbol,
  calibrationResults,
}) => {
  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        3D Volatility Surface
      </Typography>
      <Paper sx={{ p: 3, textAlign: 'center' }}>
        <Typography color="text.secondary">
          3D volatility surface visualization will be implemented here using D3.js.
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
          This will show the implied volatility surface for {symbol} with interactive rotation and zoom.
        </Typography>
      </Paper>
    </Box>
  );
};

export default VolatilitySurfacePanel; 