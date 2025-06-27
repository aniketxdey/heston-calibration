import React, { useState } from 'react';
import {
  Box,
  Container,
  AppBar,
  Toolbar,
  Typography,
  Grid,
  Paper,
  Tabs,
  Tab,
  Alert,
  CircularProgress,
} from '@mui/material';
import { useQuery } from '@tanstack/react-query';

import MarketDataPanel from './components/MarketDataPanel';
import ModelCalibrationPanel from './components/ModelCalibrationPanel';
import VolatilitySurfacePanel from './components/VolatilitySurfacePanel';
import ModelComparisonPanel from './components/ModelComparisonPanel';
import ParameterControlPanel from './components/ParameterControlPanel';
import { API_BASE_URL } from './utils/api';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

function App() {
  const [selectedTab, setSelectedTab] = useState(0);
  const [selectedSymbol, setSelectedSymbol] = useState('AAPL');
  const [calibrationResults, setCalibrationResults] = useState<any>(null);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setSelectedTab(newValue);
  };

  // Fetch available models
  const { data: models, isLoading: modelsLoading, error: modelsError } = useQuery({
    queryKey: ['models'],
    queryFn: async () => {
      const response = await fetch(`${API_BASE_URL}/models`);
      if (!response.ok) {
        throw new Error('Failed to fetch models');
      }
      return response.json();
    }
  });

  if (modelsLoading) {
    return (
      <Box
        display="flex"
        justifyContent="center"
        alignItems="center"
        minHeight="100vh"
      >
        <CircularProgress />
      </Box>
    );
  }

  if (modelsError) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4 }}>
        <Alert severity="error">
          Failed to load application data. Please check your connection and try again.
        </Alert>
      </Container>
    );
  }

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

      <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
        <Grid container spacing={3}>
          {/* Left Panel - Market Data and Controls */}
          <Grid item xs={12} md={4}>
            <Paper sx={{ p: 2, mb: 2 }}>
              <MarketDataPanel
                selectedSymbol={selectedSymbol}
                onSymbolChange={setSelectedSymbol}
              />
            </Paper>

            <Paper sx={{ p: 2, mb: 2 }}>
              <ModelCalibrationPanel
                symbol={selectedSymbol}
                models={(models as any)?.models || []}
                onCalibrationComplete={setCalibrationResults}
              />
            </Paper>

            {calibrationResults && (
              <Paper sx={{ p: 2 }}>
                <ParameterControlPanel
                  calibrationResults={calibrationResults}
                  models={(models as any)?.models || []}
                />
              </Paper>
            )}
          </Grid>

          {/* Right Panel - Visualizations and Analysis */}
          <Grid item xs={12} md={8}>
            <Paper sx={{ width: '100%' }}>
              <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
                <Tabs value={selectedTab} onChange={handleTabChange}>
                  <Tab label="Volatility Surface" />
                  <Tab label="Model Comparison" />
                  <Tab label="Greeks Analysis" />
                  <Tab label="Educational Content" />
                </Tabs>
              </Box>

              <TabPanel value={selectedTab} index={0}>
                <VolatilitySurfacePanel
                  symbol={selectedSymbol}
                  calibrationResults={calibrationResults}
                />
              </TabPanel>

              <TabPanel value={selectedTab} index={1}>
                <ModelComparisonPanel
                  symbol={selectedSymbol}
                  models={(models as any)?.models || []}
                />
              </TabPanel>

              <TabPanel value={selectedTab} index={2}>
                <Typography variant="h6" gutterBottom>
                  Greeks Analysis
                </Typography>
                <Typography color="text.secondary">
                  Advanced Greeks analysis and risk metrics will be implemented here.
                </Typography>
              </TabPanel>

              <TabPanel value={selectedTab} index={3}>
                <Typography variant="h6" gutterBottom>
                  Educational Content
                </Typography>
                <Typography color="text.secondary">
                  Interactive explanations and model demonstrations will be implemented here.
                </Typography>
              </TabPanel>
            </Paper>
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
}

export default App; 