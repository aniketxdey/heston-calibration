import React from 'react';
import {
  Box,
  Container,
  AppBar,
  Toolbar,
  Typography,
  Paper,
  Button,
} from '@mui/material';

function App() {
  const handleTestBackend = async () => {
    try {
      const response = await fetch('http://localhost:8000');
      const data = await response.json();
      alert(`Backend connected! ${data.message}`);
    } catch (error) {
      alert('Backend connection failed. Make sure backend is running on localhost:8000');
    }
  };

  return (
    <Box sx={{ flexGrow: 1 }}>
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Heston Model Web Application
          </Typography>
          <Typography variant="body2" sx={{ opacity: 0.8 }}>
            Quantitative Finance Platform
          </Typography>
        </Toolbar>
      </AppBar>

      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        <Paper sx={{ p: 4, textAlign: 'center' }}>
          <Typography variant="h4" gutterBottom>
            🚀 Application Running Successfully!
          </Typography>
          
          <Typography variant="body1" sx={{ mb: 4 }}>
            Your Heston Model Web Application is working. 
            The backend provides comprehensive quantitative finance models and APIs.
          </Typography>

          <Button 
            variant="contained" 
            size="large" 
            onClick={handleTestBackend}
            sx={{ mb: 2 }}
          >
            Test Backend Connection
          </Button>

          <Box sx={{ mt: 4 }}>
            <Typography variant="h6" gutterBottom>
              Available Features:
            </Typography>
            <Typography variant="body2" color="text.secondary">
              • Heston Model Implementation<br/>
              • Black-Scholes Model<br/>
              • SABR Model<br/>
              • Real-time Market Data Integration<br/>
              • Model Calibration Engine<br/>
              • Interactive API Documentation at localhost:8000/docs
            </Typography>
          </Box>
        </Paper>
      </Container>
    </Box>
  );
}

export default App; 