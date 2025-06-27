# Heston Model Frontend

React TypeScript frontend for the Heston Model Web Application, providing an interactive interface for volatility modeling and analysis.

## 🏗️ Architecture

### Core Components

- **Market Data Panel**: Real-time data fetching and display
- **Model Calibration Panel**: Interactive model fitting
- **Volatility Surface Panel**: 3D visualization with D3.js
- **Parameter Control Panel**: Real-time parameter adjustment
- **Model Comparison Panel**: Side-by-side model analysis

### Technology Stack

- **Framework**: React 18+ with TypeScript
- **UI Library**: Material-UI v5
- **State Management**: React Query
- **Visualization**: D3.js, Plotly.js
- **Build Tool**: Create React App
- **Package Manager**: npm/yarn

## 📦 Installation

### Prerequisites

- Node.js 18 or higher
- npm or yarn package manager

### Setup

1. **Navigate to frontend directory**:
   ```bash
   cd heston/frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   # or
   yarn install
   ```

3. **Set up environment variables** (optional):
   ```bash
   cp .env.example .env
   # Edit .env with your API configuration
   ```

## 🚀 Running the Application

### Development Mode

```bash
npm start
# or
yarn start
```

The application will be available at `http://localhost:3000`

### Production Build

```bash
npm run build
# or
yarn build
```

### Testing

```bash
npm test
# or
yarn test
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the frontend directory:

```env
# API Configuration
REACT_APP_API_URL=http://localhost:8000

# Development Settings
REACT_APP_DEBUG=true
REACT_APP_LOG_LEVEL=info
```

### API Proxy

The application is configured to proxy API requests to the backend during development:

```json
{
  "proxy": "http://localhost:8000"
}
```

## 📱 User Interface

### Layout Structure

```
┌─────────────────────────────────────────┐
│              App Bar                    │
├─────────────┬───────────────────────────┤
│             │                           │
│   Left      │        Right Panel        │
│   Panel     │                           │
│             │                           │
│ • Market    │ • Volatility Surface      │
│   Data      │ • Model Comparison        │
│ • Model     │ • Greeks Analysis         │
│   Calib.    │ • Educational Content     │
│ • Parameter │                           │
│   Controls  │                           │
│             │                           │
└─────────────┴───────────────────────────┘
```

### Component Hierarchy

```
App
├── MarketDataPanel
├── ModelCalibrationPanel
├── ParameterControlPanel
└── Main Content Area
    ├── VolatilitySurfacePanel
    ├── ModelComparisonPanel
    ├── GreeksAnalysisPanel
    └── EducationalContentPanel
```

## 🎨 UI Components

### Material-UI Theme

Custom theme configuration with:

- **Primary Color**: Blue (#1976d2)
- **Secondary Color**: Pink (#dc004e)
- **Typography**: Roboto font family
- **Components**: Custom card shadows and spacing

### Responsive Design

- **Mobile**: Single column layout
- **Tablet**: Two column layout
- **Desktop**: Full layout with side panels

## 📊 Data Visualization

### 3D Volatility Surface

Planned implementation using D3.js:

```typescript
// Features
- Interactive rotation and zoom
- Real-time parameter updates
- Multiple model overlays
- Color-coded volatility levels
```

### Charts and Graphs

Using Plotly.js for:

- **Parameter Evolution**: Time series of calibrated parameters
- **Model Comparison**: Side-by-side performance metrics
- **Greeks Analysis**: Risk metric visualizations
- **Error Analysis**: Calibration quality plots

## 🔄 State Management

### React Query

Used for server state management:

```typescript
// Market data fetching
const { data, isLoading, error } = useQuery(
  ['marketData', symbol],
  () => api.getMarketData(symbol)
);

// Model calibration
const mutation = useMutation(
  (request) => api.calibrateModel(request),
  {
    onSuccess: (data) => {
      // Handle successful calibration
    }
  }
);
```

### Local State

Component-level state for:

- **UI State**: Tab selection, form inputs
- **User Preferences**: Theme, layout settings
- **Temporary Data**: Form validation, loading states

## 🎯 User Workflow

### Basic Usage

1. **Load Market Data**
   - Enter stock symbol (e.g., AAPL)
   - View current price and options summary
   - Check data quality metrics

2. **Calibrate Model**
   - Select volatility model (Heston, Black-Scholes, SABR)
   - Run calibration process
   - Review parameter estimates and quality metrics

3. **Visualize Results**
   - Explore 3D volatility surface
   - Compare multiple models
   - Analyze Greeks and risk metrics

4. **Interactive Analysis**
   - Adjust parameters with real-time updates
   - Explore different scenarios
   - Educational content and explanations

## 🧪 Testing

### Test Structure

```
src/
├── __tests__/
│   ├── components/
│   │   ├── MarketDataPanel.test.tsx
│   │   ├── ModelCalibrationPanel.test.tsx
│   │   └── ...
│   ├── utils/
│   │   └── api.test.ts
│   └── App.test.tsx
```

### Running Tests

```bash
# Run all tests
npm test

# Run with coverage
npm test -- --coverage

# Run specific test file
npm test -- MarketDataPanel.test.tsx

# Watch mode
npm test -- --watch
```

## 🚀 Performance Optimization

### Code Splitting

```typescript
// Lazy load components
const VolatilitySurfacePanel = lazy(() => import('./components/VolatilitySurfacePanel'));
const ModelComparisonPanel = lazy(() => import('./components/ModelComparisonPanel'));
```

### Memoization

```typescript
// Optimize expensive calculations
const memoizedSurface = useMemo(() => {
  return generateVolatilitySurface(parameters);
}, [parameters]);
```

### Bundle Optimization

- **Tree Shaking**: Remove unused code
- **Dynamic Imports**: Lazy load heavy components
- **Image Optimization**: Compress and lazy load images

## 🔧 Development Tools

### Development Scripts

```json
{
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject",
    "lint": "eslint src --ext .ts,.tsx",
    "lint:fix": "eslint src --ext .ts,.tsx --fix"
  }
}
```

### Code Quality

- **ESLint**: Code linting and formatting
- **Prettier**: Code formatting
- **TypeScript**: Type checking
- **Husky**: Git hooks for pre-commit checks

## 🚀 Deployment

### Build for Production

```bash
npm run build
```

### Deployment Options

1. **Vercel**: Automatic deployment from Git
2. **Netlify**: Drag and drop deployment
3. **AWS S3**: Static website hosting
4. **GitHub Pages**: Free hosting for public repos

### Environment Configuration

```bash
# Production environment
REACT_APP_API_URL=https://your-api-domain.com
REACT_APP_DEBUG=false
```

## 🤝 Contributing

### Development Setup

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/new-feature`
3. **Install dependencies**: `npm install`
4. **Run tests**: `npm test`
5. **Submit pull request**

### Code Style

- **TypeScript**: Strict mode enabled
- **ESLint**: Airbnb configuration
- **Prettier**: Consistent formatting
- **Components**: Functional components with hooks

## 📞 Support

For frontend-specific issues:

1. Check the browser console for errors
2. Review the React DevTools
3. Open an issue on GitHub
4. Contact the development team

---

**Note**: This frontend is designed for educational and demonstration purposes. For production use in financial applications, additional security, accessibility, and compliance measures should be implemented. 