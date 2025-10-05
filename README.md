# ExoVision
AstroNova subsection for NASA Space Apps Challenge

## 🚀 Live Demo
**Try ExoVision now:** [https://huggingface.co/spaces/E5haan/ExoVision](https://huggingface.co/spaces/E5haan/ExoVision)

---
ExoVision uses an OpenAI Agents framework with ensemble machine learning models to automatically analyze exoplanet data from NASA's Kepler Objects of Interest (KOI) dataset. The system combines multiple ensemble ML algorithms for robust exoplanet detection and provides a Gradio web interface for interactive analysis and visualization.

## Local Setup

This project uses [uv](https://docs.astral.sh/uv/) for Python package management. To get started:

Open your terminal and follow these steps:

1. Install uv if you haven't already:
   
   **On Windows (PowerShell):**
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
   
   **On macOS/Linux:**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Clone this repository and navigate to the project directory:
   ```bash
   git clone https://github.com/AstroNova67/ExoVision.git
   cd ExoVision
   ```

3. Install dependencies:
   ```bash
   uv sync
   ```

4. Set up your OpenAI API key:
   ```bash
   # Create a .env file 
   echo "OPENAI_API_KEY=sk-proj-your-key-here" > .env
   ```
   
   **Note:** Replace `sk-proj-your-key-here` with your actual OpenAI API key. You can get one from [OpenAI's website](https://platform.openai.com/api-keys).

## API Key Setup

**Important:** ExoVision uses OpenAI's API for the AI chat functionality. You'll need your own API key:

1. **Get an API key** from [OpenAI Platform](https://platform.openai.com/api-keys)
2. **Create a `.env` file** in the `backend/` directory
3. **Add your key** to the file:
   ```
   OPENAI_API_KEY=sk-proj-your-actual-key-here
   ```

**Cost:** The AI chat uses GPT-4o-mini (~$0.00015 per 1K tokens). Typical usage costs $0.01-0.05 per conversation.

## Running the Models

Open your terminal and navigate to the backend directory, then run the desired model:

### Running the Main Agent
To run the main agent with all models:
```bash
cd backend
uv run agent.py
```

### Running Individual Models
To run individual models (KOI)
```bash
cd backend
uv run KOI.py
```

