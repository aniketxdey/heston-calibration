#!/bin/bash

# Heston Model Web Application Setup Script
# This script automates the setup process for both backend and frontend

set -e  # Exit on any error

echo "🚀 Setting up Heston Model Web Application..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "design/DESIGN.md" ]; then
    print_error "Please run this script from the heston-calibration directory"
    exit 1
fi

# Check prerequisites
print_status "Checking prerequisites..."

# Check Python version
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
    if [ "$(echo "$PYTHON_VERSION >= 3.11" | bc -l)" -eq 1 ]; then
        print_success "Python $PYTHON_VERSION found"
    else
        print_error "Python 3.11+ required, found $PYTHON_VERSION"
        exit 1
    fi
else
    print_error "Python 3.11+ not found. Please install Python first."
    exit 1
fi

# Check Node.js version
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version | cut -d'v' -f2 | cut -d. -f1)
    if [ "$NODE_VERSION" -ge 18 ]; then
        print_success "Node.js $(node --version) found"
    else
        print_error "Node.js 18+ required, found $(node --version)"
        exit 1
    fi
else
    print_error "Node.js 18+ not found. Please install Node.js first."
    exit 1
fi

# Check npm
if command -v npm &> /dev/null; then
    print_success "npm $(npm --version) found"
else
    print_error "npm not found. Please install npm first."
    exit 1
fi

print_success "All prerequisites satisfied!"

# Setup Backend
print_status "Setting up backend..."

cd backend

# Create virtual environment
if [ ! -d "venv" ]; then
    print_status "Creating Python virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Install Python dependencies
print_status "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

print_success "Backend setup complete!"

# Setup Frontend
print_status "Setting up frontend..."

cd ../frontend

# Install Node.js dependencies
print_status "Installing Node.js dependencies..."
npm install

print_success "Frontend setup complete!"

# Create environment files
print_status "Creating environment files..."

# Backend .env
cd ../backend
if [ ! -f ".env" ]; then
    cat > .env << EOF
# API Keys
API_KEY_ALPHA_VANTAGE=your_alpha_vantage_key

# Redis Configuration
REDIS_URL=redis://localhost:6379

# Logging
LOG_LEVEL=INFO

# Server Configuration
HOST=0.0.0.0
PORT=8000
WORKERS=4
EOF
    print_success "Backend .env file created"
else
    print_warning "Backend .env file already exists"
fi

# Frontend .env
cd ../frontend
if [ ! -f ".env" ]; then
    cat > .env << EOF
# API Configuration
REACT_APP_API_URL=http://localhost:8000

# Development Settings
REACT_APP_DEBUG=true
REACT_APP_LOG_LEVEL=info
EOF
    print_success "Frontend .env file created"
else
    print_warning "Frontend .env file already exists"
fi

# Create start scripts
print_status "Creating start scripts..."

cd ..

# Start backend script
cat > start_backend.sh << 'EOF'
#!/bin/bash
cd heston/backend
source venv/bin/activate
cd app
uvicorn main:app --reload --host 0.0.0.0 --port 8000
EOF
chmod +x start_backend.sh

# Start frontend script
cat > start_frontend.sh << 'EOF'
#!/bin/bash
cd heston/frontend
npm start
EOF
chmod +x start_frontend.sh

# Start both script
cat > start_all.sh << 'EOF'
#!/bin/bash
# Start both backend and frontend in separate terminals
# This script requires tmux or similar terminal multiplexer

if command -v tmux &> /dev/null; then
    tmux new-session -d -s heston-app
    tmux split-window -h
    tmux send-keys -t 0 "cd heston/backend && source venv/bin/activate && cd app && uvicorn main:app --reload --host 0.0.0.0 --port 8000" C-m
    tmux send-keys -t 1 "cd heston/frontend && npm start" C-m
    tmux attach-session -t heston-app
else
    echo "tmux not found. Please install tmux or run start_backend.sh and start_frontend.sh in separate terminals."
fi
EOF
chmod +x start_all.sh

print_success "Start scripts created!"

# Print final instructions
echo ""
echo "🎉 Setup complete! Here's how to get started:"
echo ""
echo "1. Start the backend server:"
echo "   ./start_backend.sh"
echo ""
echo "2. In a new terminal, start the frontend:"
echo "   ./start_frontend.sh"
echo ""
echo "3. Or start both at once (requires tmux):"
echo "   ./start_all.sh"
echo ""
echo "4. Open your browser and navigate to:"
echo "   Frontend: http://localhost:3000"
echo "   Backend API: http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "📚 For more information, see the README files:"
echo "   - heston/README.md (main documentation)"
echo "   - heston/backend/README.md (backend details)"
echo "   - heston/frontend/README.md (frontend details)"
echo ""
echo "🔧 Optional: Configure API keys in heston/backend/.env"
echo ""

print_success "Heston Model Web Application is ready to use!" 