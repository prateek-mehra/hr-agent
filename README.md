# AI Agent HR Chat App

This project provides an agentic backend for HR tasks and a minimal chat UI frontend.

## Backend (Python, FastAPI)

### Setup
1. Install Python 3.8+ and pip.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the backend server:
   ```bash
   python main.py --serve
   ```
   The server will start at http://localhost:8000

## Frontend (React)

The frontend is in the `ui/` directory.

### Setup
1. Install Node.js (v16+ recommended).
2. In a new terminal:
   ```bash
   cd ui
   npm install
   npm start
   ```
   The app will run at http://localhost:3000 and connect to the backend at http://localhost:8000.

## Usage
- Enter your prompt in the chat bar and press the up-arrow button or Enter.
- User messages appear on the right, agent responses on the left.

## Notes
- Backend requirements are in `requirements.txt`.
- Frontend dependencies are managed in `ui/package.json`.
- Make sure the backend is running before using the frontend. 