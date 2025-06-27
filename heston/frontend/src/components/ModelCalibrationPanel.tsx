import React, { useState } from 'react';
import {
  Box,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button,
  Alert,
  CircularProgress,
  Card,
  CardContent,
  Grid,
  Chip,
} from '@mui/material';
import { useMutation } from '@tanstack/react-query';
import { api, CalibrationRequest, CalibrationResult, ModelInfo } from '../utils/api';

interface ModelCalibrationPanelProps {
  symbol: string;
  models: ModelInfo[];
  onCalibrationComplete: (results: CalibrationResult) => void;
}

const ModelCalibrationPanel: React.FC<ModelCalibrationPanelProps> = ({
  symbol,
  models,
  onCalibrationComplete,
}) => {
  const [selectedModel, setSelectedModel] = useState<string>('heston');

  // Calibration mutation
  const calibrationMutation = useMutation<CalibrationResult, Error, CalibrationRequest>({
    mutationFn: (request) => api.calibrateModel(request),
    onSuccess: (data) => {
      onCalibrationComplete(data);
    },
  });

  const handleCalibrate = () => {
    if (!symbol || !selectedModel) return;

    const request: CalibrationRequest = {
      symbol,
      model_type: selectedModel,
    };

    calibrationMutation.mutate(request);
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Model Calibration
      </Typography>

      {/* Model Selection */}
      <FormControl fullWidth size="small" sx={{ mb: 2 }}>
        <InputLabel>Model</InputLabel>
        <Select
          value={selectedModel}
          label="Model"
          onChange={(e) => setSelectedModel(e.target.value)}
        >
          {models.map((model) => (
            <MenuItem key={model.name.toLowerCase()} value={model.name.toLowerCase()}>
              {model.name}
            </MenuItem>
          ))}
        </Select>
      </FormControl>

      {/* Calibrate Button */}
      <Button
        fullWidth
        variant="contained"
        onClick={handleCalibrate}
        disabled={!symbol || calibrationMutation.isLoading}
        sx={{ mb: 2 }}
      >
        {calibrationMutation.isLoading ? (
          <CircularProgress size={20} color="inherit" />
        ) : (
          'Calibrate Model'
        )}
      </Button>

      {/* Error Display */}
      {calibrationMutation.error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {calibrationMutation.error.message}
        </Alert>
      )}

      {/* Calibration Results */}
      {calibrationMutation.data && (
        <Card>
          <CardContent>
            <Typography variant="subtitle1" gutterBottom>
              Calibration Results
            </Typography>

            {/* Model Info */}
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" color="text.secondary">
                Model: {calibrationMutation.data.model_type}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Symbol: {calibrationMutation.data.symbol}
              </Typography>
            </Box>

            {/* Parameters */}
            <Typography variant="subtitle2" gutterBottom>
              Parameters:
            </Typography>
            <Grid container spacing={1} sx={{ mb: 2 }}>
              {Object.entries(calibrationMutation.data.calibration_result.parameters).map(
                ([key, value]) => (
                  <Grid item xs={6} key={key}>
                    <Typography variant="body2">
                      {key}: {value.toFixed(4)}
                    </Typography>
                  </Grid>
                )
              )}
            </Grid>

            {/* Quality Metrics */}
            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Quality Metrics:
              </Typography>
              <Grid container spacing={1}>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    R²: {(calibrationMutation.data.calibration_result.fit_quality * 100).toFixed(1)}%
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">
                    Iterations: {calibrationMutation.data.calibration_result.iterations}
                  </Typography>
                </Grid>
              </Grid>
            </Box>

            {/* Validation Status */}
            <Box>
              <Chip
                label={calibrationMutation.data.validation_result.is_valid ? 'Valid' : 'Invalid'}
                color={calibrationMutation.data.validation_result.is_valid ? 'success' : 'error'}
                size="small"
                sx={{ mr: 1 }}
              />
              <Chip
                label={`Quality: ${(calibrationMutation.data.validation_result.quality_score * 100).toFixed(0)}%`}
                color={
                  calibrationMutation.data.validation_result.quality_score > 0.8
                    ? 'success'
                    : calibrationMutation.data.validation_result.quality_score > 0.5
                    ? 'warning'
                    : 'error'
                }
                size="small"
              />
            </Box>

            {/* Warnings */}
            {calibrationMutation.data.validation_result.warnings.length > 0 && (
              <Alert severity="warning" sx={{ mt: 2 }}>
                <Typography variant="body2">
                  {calibrationMutation.data.validation_result.warnings.join(', ')}
                </Typography>
              </Alert>
            )}
          </CardContent>
        </Card>
      )}
    </Box>
  );
};

export default ModelCalibrationPanel; 