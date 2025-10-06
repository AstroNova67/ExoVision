import asyncio
from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    Runner,
    input_guardrail,
    trace,
    function_tool,
)
from agents.model_settings import ModelSettings
from dotenv import load_dotenv
from openai import AsyncOpenAI
import gradio as gr
import os
import pandas as pd
from typing import Dict, List, Tuple
from IPython.display import display, Markdown
from KOI import KOI
import numpy as np
from pypdf import PdfReader

load_dotenv(override=True)

openai_client = AsyncOpenAI(
    base_url="https://api.openai.com/v1", api_key=os.getenv("OPENAI_API_KEY")
)
openai_model = OpenAIChatCompletionsModel(
    model="gpt-4o-mini", openai_client=openai_client
)

# Comprehensive context about the machine learning models used for exoplanet prediction
ml_context = """
MACHINE LEARNING MODELS FOR EXOPLANET PREDICTION:

Our system uses 5 ensemble machine learning models to predict whether Kepler Objects of Interest (KOIs) are CONFIRMED exoplanets or CANDIDATE exoplanets:

1. ADABOOST CLASSIFIER:
   - Uses adaptive boosting with decision tree stumps (max_depth=1)
   - 100 estimators with learning rate 0.1
   - Works by iteratively focusing on misclassified examples
   - Each subsequent model learns from the mistakes of previous models
   - Good for handling complex decision boundaries

2. RANDOM FOREST CLASSIFIER:
   - Ensemble of 50 decision trees using entropy criterion
   - Uses bootstrap sampling and random feature selection
   - Reduces overfitting through averaging multiple trees
   - Provides feature importance rankings
   - Robust to outliers and noise

3. EXTRA TREES CLASSIFIER:
   - 200 estimators with sqrt(max_features) and entropy criterion
   - Uses random splits instead of optimal splits
   - Faster training than Random Forest
   - More randomization reduces overfitting
   - Good generalization performance

4. STACKING CLASSIFIER:
   - Meta-learning approach using Random Forest and AdaBoost as base estimators
   - Uses Logistic Regression as final estimator
   - 5-fold cross-validation for meta-feature generation
   - Combines predictions from multiple models intelligently
   - Often achieves highest accuracy

5. SUBSPACE CLASSIFIER (Bagging):
   - Uses 1000 decision trees with 50% feature subspace sampling
   - 80% bootstrap sampling with replacement
   - Reduces variance through ensemble averaging
   - Each tree sees different random subset of features
   - Good for high-dimensional data

CONSENSUS APPROACH:
- All 5 models make independent predictions
- Final decision uses majority voting
- Confidence is averaged across all models
- This ensemble approach provides robust, reliable predictions

INPUT FEATURES:
The models analyze 13 key astronomical parameters including orbital period, transit duration/depth, planet radius, temperature, stellar properties, and signal-to-noise ratios.

Be ready to explain what the parameters are and what they mean.
"""

# General LLM instructions for space/astronomy questions
general_instructions = f"""You are an AI assistant from AstroNova specializing in space, astronomy, and exoplanet science. You can answer general questions about:

- Exoplanets and their discovery methods
- Astronomy concepts and phenomena
- Space missions and telescopes
- Stellar and planetary science
- The search for life beyond Earth

HANDOFF TO PREDICTION AGENT:
When users provide numerical exoplanet parameters (even in human-readable terms), you MUST hand off to the prediction agent. Look for these parameter types:
- Orbital period (days)
- Transit duration (hours) 
- Transit depth (ppm)
- Planet radius (Earth radii)
- Temperature (Kelvin/K)
- Insolation (Earth units)
- Signal-to-noise ratio (SNR)
- Star temperature (Kelvin/K)
- Star radius (Solar radii)

The prediction agent can interpret human-readable parameter names and convert them to the correct function parameters.

Be friendly, informative, and use clear language to explain complex astronomical concepts.

{ml_context}

When discussing exoplanet prediction methods, you can explain how our machine learning models work and their advantages in analyzing astronomical data."""

# Prediction agent instructions
prediction_instructions = """You are a specialized AI assistant focused on exoplanet predictions using machine learning models.

CRITICAL RULES:
1. ALWAYS call the predict_koi_exoplanet function when numerical parameters are provided
2. ONLY use information returned by the function - do not add explanations or interpretations
3. Follow the exact response template below

PARAMETER MAPPING:
You can interpret human-readable parameter names and map them to function parameters:
- "Orbital Period" or "period" → koi_period
- "Transit Duration" or "duration" → koi_duration  
- "Transit Depth" or "depth" → koi_depth
- "Planet Radius" or "radius" → koi_prad
- "Temperature" or "temp" → koi_teq
- "Insolation" → koi_insol
- "Signal-to-Noise Ratio" or "SNR" → koi_model_snr
- "Star Temperature" or "stellar temperature" → koi_steff
- "Star Radius" or "stellar radius" → koi_srad

RESPONSE TEMPLATE (use exactly this format):
```
**PREDICTION RESULT**

Consensus: [CONFIRMED/CANDIDATE from function]
Confidence: [actual confidence number]% (average across all models)

**YOUR INPUT PARAMETERS**
- Orbital Period: [koi_period] days
- Transit Duration: [koi_duration] hours  
- Transit Depth: [koi_depth] ppm
- Planet Radius: [koi_prad] Earth radii
- Temperature: [koi_teq] K ([converted to]°C)
- Insolation: [koi_insol] Earth units
- SNR: [koi_model_snr]
- Star Temperature: [koi_steff] K
- Star Radius: [koi_srad] Solar radii

**WHAT THIS MEANS**
- CONFIRMED = Verified exoplanet through additional observations
- CANDIDATE = Requires further verification

[Include individual model predictions if available]
```

TEMPLATE RULES:
- Replace [bracketed items] with actual values from function output
- Do NOT add scientific explanations beyond what's in the template
- Do NOT interpret or explain mechanisms - just report results
- Keep it concise and factual only
"""

# Initialize models
koi_model = None


def initialize_models():
    global koi_model
    try:
        print("Loading KOI model...")
        # Use absolute path to the dataset
        import os

        dataset_path = os.path.join(
            os.path.dirname(__file__), "datasets", "filtered_file.csv"
        )
        koi_model = KOI(dataset_path)
        koi_model.load_model()
        print("KOI model loaded successfully!")
        print(
            f"KOI model attributes: {[attr for attr in dir(koi_model) if not attr.startswith('_')]}"
        )

    except Exception as e:
        print(f"Error loading models: {e}")
        import traceback

        traceback.print_exc()


# Initialize models
initialize_models()

# Create general agent for space/astronomy questions
general_agent = Agent(
    name="GeneralAstronomyAssistant",
    instructions=general_instructions,
    model=openai_model,
)


@function_tool
def predict_koi_exoplanet(
    koi_period: float,
    koi_duration: float,
    koi_depth: float,
    koi_prad: float,
    koi_teq: float,
    koi_insol: float,
    koi_model_snr: float,
    koi_steff: float,
    koi_srad: float,
) -> Dict[str, any]:
    """
    Predicts whether a Kepler Object is CONFIRMED or CANDIDATE exoplanet using 5 ML models.

    INPUT PARAMETERS:
    - koi_period: orbital period (days)
    - koi_duration: transit duration (hours)
    - koi_depth: transit depth (ppm)
    - koi_prad: planet radius (Earth radii)
    - koi_teq: temperature (Kelvin)
    - koi_insol: insolation (Earth units)
    - koi_model_snr: signal-to-noise ratio
    - koi_steff: star temperature (Kelvin)
    - koi_srad: star radius (Solar radii)

    RETURNS:
    - consensus_prediction: "CONFIRMED" or "CANDIDATE"
    - average_confidence: confidence score (decimal 0-1)
    - individual_predictions: all 5 model results
    - input_parameters: user inputs echo

    TEMPERATURE: Convert K to °C by subtracting 273.15 (288K = +15°C)
    """
    print(
        f"Prediction called with: period={koi_period}, duration={koi_duration}, depth={koi_depth}"
    )

    if koi_model is None:
        print("ERROR: KOI model is None")
        return {"error": "KOI model not loaded"}

    # Check if models are properly loaded
    if not hasattr(koi_model, "adaboost") or koi_model.adaboost is None:
        print("ERROR: AdaBoost model not loaded")
        return {"error": "AdaBoost model not loaded"}

    print("Models appear to be loaded correctly")

    try:
        # Set flags to 0 (no flags raised)
        koi_fpflag_nt = 0
        koi_fpflag_ss = 0
        koi_fpflag_co = 0
        koi_fpflag_ec = 0

        # Convert input to numpy array in the correct order
        input_data = np.array(
            [
                koi_fpflag_nt,
                koi_fpflag_ss,
                koi_fpflag_co,
                koi_fpflag_ec,
                koi_period,
                koi_duration,
                koi_depth,
                koi_prad,
                koi_teq,
                koi_insol,
                koi_model_snr,
                koi_steff,
                koi_srad,
            ]
        ).reshape(1, -1)

        print(f"Input data shape: {input_data.shape}")
        print(f"Input data: {input_data}")

        # Apply scaling if scaler is available
        if hasattr(koi_model, "sc_x") and koi_model.sc_x is not None:
            input_data = koi_model.sc_x.transform(input_data)
            print(f"Scaled input data: {input_data}")
        else:
            print("WARNING: No scaler found, using raw data")

        # Make predictions with all models
        predictions = {}

        # AdaBoost
        print("Making AdaBoost prediction...")
        adaboost_pred = koi_model.adaboost.predict(input_data)[0]
        adaboost_prob = koi_model.adaboost.predict_proba(input_data)[0]
        print(f"AdaBoost prediction: {adaboost_pred}, probabilities: {adaboost_prob}")
        predictions["adaboost"] = {
            "prediction": int(adaboost_pred),
            "confidence": float(max(adaboost_prob)),
            "probabilities": {
                "CANDIDATE": float(adaboost_prob[0]),
                "CONFIRMED": float(adaboost_prob[1]),
            },
        }

        # Random Forest
        forest_pred = koi_model.forest_classifier.predict(input_data)[0]
        forest_prob = koi_model.forest_classifier.predict_proba(input_data)[0]
        predictions["forest_classifier"] = {
            "prediction": int(forest_pred),
            "confidence": float(max(forest_prob)),
            "probabilities": {
                "CANDIDATE": float(forest_prob[0]),
                "CONFIRMED": float(forest_prob[1]),
            },
        }

        # Subspace
        subspace_pred = koi_model.subspace.predict(input_data)[0]
        subspace_prob = koi_model.subspace.predict_proba(input_data)[0]
        predictions["subspace"] = {
            "prediction": int(subspace_pred),
            "confidence": float(max(subspace_prob)),
            "probabilities": {
                "CANDIDATE": float(subspace_prob[0]),
                "CONFIRMED": float(subspace_prob[1]),
            },
        }

        # Stacking
        stacking_pred = koi_model.stacking.predict(input_data)[0]
        stacking_prob = koi_model.stacking.predict_proba(input_data)[0]
        predictions["stacking"] = {
            "prediction": int(stacking_pred),
            "confidence": float(max(stacking_prob)),
            "probabilities": {
                "CANDIDATE": float(stacking_prob[0]),
                "CONFIRMED": float(stacking_prob[1]),
            },
        }

        # Extra Trees
        extra_trees_pred = koi_model.extra_trees.predict(input_data)[0]
        extra_trees_prob = koi_model.extra_trees.predict_proba(input_data)[0]
        predictions["extra_trees"] = {
            "prediction": int(extra_trees_pred),
            "confidence": float(max(extra_trees_prob)),
            "probabilities": {
                "CANDIDATE": float(extra_trees_prob[0]),
                "CONFIRMED": float(extra_trees_prob[1]),
            },
        }

        # Calculate average confidence
        avg_confidence = np.mean([p["confidence"] for p in predictions.values()])
        print(f"Average confidence: {avg_confidence}")

        # Determine consensus (majority vote)
        # Note: LabelEncoder assigns CANDIDATE=0, CONFIRMED=1 alphabetically
        candidate_votes = sum(1 for p in predictions.values() if p["prediction"] == 0)
        confirmed_votes = sum(1 for p in predictions.values() if p["prediction"] == 1)
        print(f"Confirmed votes: {confirmed_votes}, Candidate votes: {candidate_votes}")

        if confirmed_votes > candidate_votes:
            consensus = "CONFIRMED"
        else:
            consensus = "CANDIDATE"

        print(f"Final consensus: {consensus}")

        result = {
            "consensus_prediction": consensus,
            "average_confidence": float(avg_confidence),
            "individual_predictions": predictions,
            "input_parameters": {
                "orbital_period_days": koi_period,
                "transit_duration_hours": koi_duration,
                "transit_depth_ppm": koi_depth,
                "planet_radius_earth_radii": koi_prad,
                "equilibrium_temperature_k": koi_teq,
                "insolation_flux_earth_units": koi_insol,
                "signal_to_noise_ratio": koi_model_snr,
                "stellar_temperature_k": koi_steff,
                "stellar_radius_solar_radii": koi_srad,
            },
        }

        print(f"Returning result: {result}")
        return result

    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}


# Create prediction agent with tools
prediction_agent = Agent(
    name="ExoplanetPredictor",
    instructions=prediction_instructions,
    model=openai_model,
    tools=[predict_koi_exoplanet],
)

# Set up handoff from general agent to prediction agent
general_agent.handoffs = [prediction_agent]


class ExoVisionGradioInterface:
    """Enhanced Gradio interface for ExoVision with all features"""

    def __init__(self):
        self.general_agent = general_agent
        self.prediction_agent = prediction_agent

    async def chat_with_agent(self, message: str, history: list) -> tuple:
        """Chat with the general agent (which will hand off to prediction agent when needed)"""
        try:
            with trace("ExoVision Assistant"):
                result = await Runner.run(self.general_agent, message, max_turns=5)
                # Extract the final output from the result
                if hasattr(result, "final_output"):
                    response = result.final_output
                else:
                    response = str(result)

                # Add to history in messages format
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": response})
                return history, ""

        except Exception as e:
            error_msg = f"Error: {e}"
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_msg})
            return history, ""

    def get_model_statistics(self):
        """Get model performance statistics"""
        if koi_model is None:
            return pd.DataFrame()
        return koi_model.get_model_statistics()

    def get_feature_importance(self):
        """Get feature importance data"""
        if koi_model is None:
            return pd.DataFrame()
        return koi_model.get_feature_importance()

    def create_model_comparison_plot(self):
        """Create model comparison visualization"""
        if koi_model is None:
            return None
        return koi_model.create_model_comparison_plot()

    def create_feature_importance_plot(self):
        """Create feature importance visualization"""
        if koi_model is None:
            return None
        return koi_model.create_feature_importance_plot()

    def create_confusion_matrix_plot(self, model_name="Forest"):
        """Create confusion matrix plot"""
        if koi_model is None:
            return None
        return koi_model.create_confusion_matrix_plot(model_name)

    def make_direct_prediction(
        self,
        koi_period,
        koi_duration,
        koi_depth,
        koi_prad,
        koi_teq,
        koi_insol,
        koi_model_snr,
        koi_steff,
        koi_srad,
    ):
        """Make direct prediction without agent system"""
        if koi_model is None:
            return {"error": "KOI model not loaded"}

        try:
            # Set flags to 0 (no flags raised)
            koi_fpflag_nt = 0
            koi_fpflag_ss = 0
            koi_fpflag_co = 0
            koi_fpflag_ec = 0

            # Convert input to numpy array in the correct order
            input_data = np.array(
                [
                    koi_fpflag_nt,
                    koi_fpflag_ss,
                    koi_fpflag_co,
                    koi_fpflag_ec,
                    koi_period,
                    koi_duration,
                    koi_depth,
                    koi_prad,
                    koi_teq,
                    koi_insol,
                    koi_model_snr,
                    koi_steff,
                    koi_srad,
                ]
            ).reshape(1, -1)

            # Make predictions with all models
            predictions = {}

            # AdaBoost
            adaboost_pred = koi_model.adaboost.predict(input_data)[0]
            adaboost_prob = koi_model.adaboost.predict_proba(input_data)[0]
            predictions["adaboost"] = {
                "prediction": int(adaboost_pred),
                "confidence": float(max(adaboost_prob)),
                "probabilities": {
                    "CANDIDATE": float(adaboost_prob[0]),
                    "CONFIRMED": float(adaboost_prob[1]),
                },
            }

            # Random Forest
            forest_pred = koi_model.forest_classifier.predict(input_data)[0]
            forest_prob = koi_model.forest_classifier.predict_proba(input_data)[0]
            predictions["forest_classifier"] = {
                "prediction": int(forest_pred),
                "confidence": float(max(forest_prob)),
                "probabilities": {
                    "CANDIDATE": float(forest_prob[0]),
                    "CONFIRMED": float(forest_prob[1]),
                },
            }

            # Subspace
            subspace_pred = koi_model.subspace.predict(input_data)[0]
            subspace_prob = koi_model.subspace.predict_proba(input_data)[0]
            predictions["subspace"] = {
                "prediction": int(subspace_pred),
                "confidence": float(max(subspace_prob)),
                "probabilities": {
                    "CANDIDATE": float(subspace_prob[0]),
                    "CONFIRMED": float(subspace_prob[1]),
                },
            }

            # Stacking
            stacking_pred = koi_model.stacking.predict(input_data)[0]
            stacking_prob = koi_model.stacking.predict_proba(input_data)[0]
            predictions["stacking"] = {
                "prediction": int(stacking_pred),
                "confidence": float(max(stacking_prob)),
                "probabilities": {
                    "CANDIDATE": float(stacking_prob[0]),
                    "CONFIRMED": float(stacking_prob[1]),
                },
            }

            # Extra Trees
            extra_trees_pred = koi_model.extra_trees.predict(input_data)[0]
            extra_trees_prob = koi_model.extra_trees.predict_proba(input_data)[0]
            predictions["extra_trees"] = {
                "prediction": int(extra_trees_pred),
                "confidence": float(max(extra_trees_prob)),
                "probabilities": {
                    "CANDIDATE": float(extra_trees_prob[0]),
                    "CONFIRMED": float(extra_trees_prob[1]),
                },
            }

            # Calculate average confidence
            avg_confidence = np.mean([p["confidence"] for p in predictions.values()])

            # Determine consensus (majority vote)
            # Note: LabelEncoder assigns CANDIDATE=0, CONFIRMED=1 alphabetically
            candidate_votes = sum(
                1 for p in predictions.values() if p["prediction"] == 0
            )
            confirmed_votes = sum(
                1 for p in predictions.values() if p["prediction"] == 1
            )

            if confirmed_votes > candidate_votes:
                consensus = "CONFIRMED"
            else:
                consensus = "CANDIDATE"

            return {
                "consensus_prediction": consensus,
                "average_confidence": float(avg_confidence),
                "individual_predictions": predictions,
                "input_parameters": {
                    "orbital_period_days": koi_period,
                    "transit_duration_hours": koi_duration,
                    "transit_depth_ppm": koi_depth,
                    "planet_radius_earth_radii": koi_prad,
                    "equilibrium_temperature_k": koi_teq,
                    "insolation_flux_earth_units": koi_insol,
                    "signal_to_noise_ratio": koi_model_snr,
                    "stellar_temperature_k": koi_steff,
                    "stellar_radius_solar_radii": koi_srad,
                },
            }

        except Exception as e:
            return {"error": f"Prediction error: {str(e)}"}

    def process_batch_file(self, file):
        """Process uploaded CSV file for batch predictions"""
        if file is None:
            return None, None, "No file uploaded"

        try:
            # Use robust CSV parsing with error handling
            df = pd.read_csv(
                file.name, encoding="utf-8", on_bad_lines="skip", engine="python"
            )

            # Validate required columns
            required_cols = [
                "koi_period",
                "koi_duration",
                "koi_depth",
                "koi_prad",
                "koi_teq",
                "koi_insol",
                "koi_model_snr",
                "koi_steff",
                "koi_srad",
            ]

            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                return None, None, f"❌ Missing columns: {missing}"

            # Make predictions
            predictions = []
            confidences = []

            for _, row in df.iterrows():
                try:
                    result = self.make_direct_prediction(
                        row["koi_period"],
                        row["koi_duration"],
                        row["koi_depth"],
                        row["koi_prad"],
                        row["koi_teq"],
                        row["koi_insol"],
                        row["koi_model_snr"],
                        row["koi_steff"],
                        row["koi_srad"],
                    )
                    if "error" in result:
                        predictions.append("ERROR")
                        confidences.append(0)
                    else:
                        predictions.append(result["consensus_prediction"])
                        confidences.append(round(result["average_confidence"] * 100, 2))
                except Exception as e:
                    predictions.append("ERROR")
                    confidences.append(0)

            # Create results dataframe with only relevant columns
            results_df = df[required_cols].copy()
            results_df["prediction"] = predictions
            results_df["confidence"] = confidences

            preview = results_df.head(5)
            results = results_df

            return preview, results, f"✅ Processed {len(df)} rows successfully!"

        except Exception as e:
            return None, None, f"❌ Error processing file: {str(e)}"

    def create_gradio_interface(self):
        """Create the comprehensive Gradio interface"""

        # Enhanced CSS for expandable side panel
        custom_css = """
        .gradio-container {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        
        .plot-container {
            border: 1px solid #e5e7eb !important;
            border-radius: 8px !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
            background: #ffffff !important;
        }
        
        .dataframe {
            border: 1px solid #e5e7eb !important;
            border-radius: 8px !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
            background: #ffffff !important;
        }
        
        /* Side panel styles */
        .side-panel {
            position: fixed !important;
            top: 0 !important;
            right: -400px !important;
            width: 400px !important;
            height: 100vh !important;
            background: #ffffff !important;
            border-left: 1px solid #e5e7eb !important;
            box-shadow: -2px 0 10px rgba(0, 0, 0, 0.1) !important;
            transition: right 0.3s ease !important;
            z-index: 1000 !important;
            overflow-y: auto !important;
            padding: 20px !important;
        }
        
        .side-panel.open {
            right: 0 !important;
        }
        
        .panel-toggle {
            position: fixed !important;
            top: 20px !important;
            right: 20px !important;
            z-index: 1001 !important;
            background: #3b82f6 !important;
            color: white !important;
            border: none !important;
            border-radius: 50% !important;
            width: 50px !important;
            height: 50px !important;
            cursor: pointer !important;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2) !important;
            transition: all 0.3s ease !important;
        }
        
        .panel-toggle:hover {
            background: #2563eb !important;
            transform: scale(1.05) !important;
        }
        
        .panel-toggle.open {
            right: 420px !important;
        }
        
        .panel-header {
            display: flex !important;
            justify-content: space-between !important;
            align-items: center !important;
            margin-bottom: 20px !important;
            padding-bottom: 15px !important;
            border-bottom: 1px solid #e5e7eb !important;
        }
        
        .panel-title {
            font-size: 18px !important;
            font-weight: 600 !important;
            color: #1f2937 !important;
        }
        
        .close-btn {
            background: #ef4444 !important;
            color: white !important;
            border: none !important;
            border-radius: 50% !important;
            width: 30px !important;
            height: 30px !important;
            cursor: pointer !important;
            font-size: 16px !important;
        }
        
        .close-btn:hover {
            background: #dc2626 !important;
        }
        
        .chat-container {
            height: calc(100vh - 120px) !important;
            display: flex !important;
            flex-direction: column !important;
        }
        
        .chatbot-panel {
            flex: 1 !important;
            margin-bottom: 15px !important;
        }
        
        .chat-input-container {
            display: flex !important;
            gap: 10px !important;
        }
        
        .chat-input {
            flex: 1 !important;
        }
        
        .send-btn {
            background: #3b82f6 !important;
            color: white !important;
            border: none !important;
            border-radius: 6px !important;
            padding: 8px 16px !important;
            cursor: pointer !important;
        }
        
        .send-btn:hover {
            background: #2563eb !important;
        }
        
        .clear-btn {
            background: #6b7280 !important;
            color: white !important;
            border: none !important;
            border-radius: 6px !important;
            padding: 8px 16px !important;
            cursor: pointer !important;
            margin-top: 10px !important;
        }
        
        .clear-btn:hover {
            background: #4b5563 !important;
        }
        """

        with gr.Blocks(
            title="ExoVision - AI Exoplanet Assistant",
            theme=gr.themes.Soft(),
            css=custom_css,
        ) as demo:
            # Header with custom logo
            with gr.Row():
                with gr.Column(scale=1):
                    # Add your custom logo here
                    logo = gr.Image(
                        value=os.path.join(
                            os.path.dirname(__file__), "astronova_image.png"
                        ),  # Absolute path to PNG file
                        label="",
                        show_label=False,
                        container=False,
                        height=80,
                        width=80,
                    )
                with gr.Column(scale=4):
                    gr.Markdown(
                        """
                        # ExoVision - AI Exoplanet Assistant
                        
                        Welcome to ExoVision! Chat with our AI assistant to get information about space, astronomy, and make exoplanet predictions using our machine learning models.
                        
                        ## What you can do:
                        - Ask general questions about space, astronomy, and exoplanets
                        - Get predictions by providing numerical parameters (automatically switches to prediction mode)
                        - Analyze model performance and feature importance
                        - Upload CSV files for batch processing
                        
                        ## Example queries:
                        - "What is an exoplanet?"
                        - "How do we detect exoplanets?"
                        - "Tell me about the James Webb Space Telescope"
                        - "koi_period: 10, koi_duration: 3, koi_depth: 1000, koi_prad: 1.2, koi_teq: 300, koi_insol: 1.0, koi_model_snr: 15, koi_steff: 5800, koi_srad: 1.0"
                        - "How accurate are your models?"
                        """
                    )

            # Chat interface in a collapsible section
            with gr.Accordion("💬 AI Assistant", open=False) as chat_accordion:
                chatbot = gr.Chatbot(
                    label="ExoVision Assistant",
                    height=400,
                    show_label=False,
                    container=True,
                    show_copy_button=True,
                    type="messages",
                )

                # Input area
                with gr.Row():
                    msg = gr.Textbox(
                        label="",
                        placeholder="Ask me about exoplanets or provide parameters for prediction...",
                        lines=1,
                        max_lines=3,
                        scale=4,
                    )
                    send_btn = gr.Button("Send", variant="primary", size="sm", scale=1)

                clear_btn = gr.Button("Clear Chat", variant="secondary", size="sm")

            # Main content area with tabs (excluding chat tab)
            with gr.Tabs():

                # Parameter Input Tab
                with gr.Tab("🔧 Parameter Input"):
                    gr.Markdown("### Enter Exoplanet Parameters for Prediction")
                    gr.Markdown(
                        "Fill in the values below and click 'Generate Prediction Query' to create a formatted query for the AI assistant."
                    )

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### Orbital Parameters")
                            koi_period = gr.Number(
                                label="Orbital Period (days)",
                                value=10.0,
                                info="Time for one complete orbit around the star",
                            )
                            koi_duration = gr.Number(
                                label="Transit Duration (hours)",
                                value=3.0,
                                info="Duration of the transit event",
                            )
                            koi_depth = gr.Number(
                                label="Transit Depth (ppm)",
                                value=1000.0,
                                info="Depth of the transit in parts per million",
                            )

                        with gr.Column():
                            gr.Markdown("#### Planetary Parameters")
                            koi_prad = gr.Number(
                                label="Planet Radius (Earth radii)",
                                value=1.2,
                                info="Radius compared to Earth",
                            )
                            koi_teq = gr.Number(
                                label="Equilibrium Temperature (K)",
                                value=300.0,
                                info="Planet's equilibrium temperature in Kelvin",
                            )
                            koi_insol = gr.Number(
                                label="Insolation (Earth units)",
                                value=1.0,
                                info="Stellar flux compared to Earth",
                            )

                        with gr.Column():
                            gr.Markdown("#### Stellar & Signal Parameters")
                            koi_model_snr = gr.Number(
                                label="Signal-to-Noise Ratio",
                                value=15.0,
                                info="Quality of the transit signal",
                            )
                            koi_steff = gr.Number(
                                label="Star Temperature (K)",
                                value=5800.0,
                                info="Stellar effective temperature",
                            )
                            koi_srad = gr.Number(
                                label="Star Radius (Solar radii)",
                                value=1.0,
                                info="Stellar radius compared to Sun",
                            )

                    with gr.Row():
                        generate_query_btn = gr.Button(
                            "Generate Prediction Query", variant="primary", size="lg"
                        )
                        clear_params_btn = gr.Button(
                            "Clear Parameters", variant="secondary", size="lg"
                        )

                    # Output area for generated query
                    generated_query = gr.Textbox(
                        label="Generated Query",
                        placeholder="Click 'Generate Prediction Query' to create a formatted query...",
                        lines=3,
                        interactive=False,
                    )

                    with gr.Row():
                        copy_query_btn = gr.Button(
                            "Copy Query to Chat", variant="primary"
                        )

                # Model Analytics Tab
                with gr.Tab("📊 Model Analytics"):
                    gr.Markdown("### Model Performance Analysis")

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### Model Performance Statistics")
                            stats_df = gr.DataFrame(
                                label="Model Performance (%)",
                                value=self.get_model_statistics(),
                                interactive=False,
                            )

                            model_comparison_plot = gr.Plot(
                                label="Model Comparison Chart",
                                value=self.create_model_comparison_plot(),
                            )

                        with gr.Column():
                            gr.Markdown("#### Feature Importance Analysis")
                            feature_importance_plot = gr.Plot(
                                label="Feature Importance",
                                value=self.create_feature_importance_plot(),
                            )

                            gr.Markdown("#### Confusion Matrix Analysis")
                            model_selector = gr.Radio(
                                choices=[
                                    "Forest",
                                    "AdaBoost",
                                    "Stacking",
                                    "Subspace",
                                    "Extra Trees",
                                ],
                                label="Select Model",
                                value="Forest",
                            )

                            confusion_matrix_plot = gr.Plot(
                                label="Confusion Matrix",
                                value=self.create_confusion_matrix_plot("Forest"),
                            )

                # Batch Processing Tab
                with gr.Tab("📁 Batch Processing"):
                    gr.Markdown("### Upload CSV File for Batch Predictions")

                    gr.Markdown(
                        """
                    **Required CSV Columns:** `koi_period`, `koi_duration`, `koi_depth`, `koi_prad`, `koi_teq`, `koi_insol`, `koi_model_snr`, `koi_steff`, `koi_srad`
                    """
                    )

                    file_input = gr.File(
                        label="Upload CSV File",
                        file_types=[".csv"],
                        file_count="single",
                        show_label=True,
                        container=True,
                    )

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### Sample Data Preview")
                            sample_data = gr.DataFrame(
                                label="First 5 rows",
                                headers=[
                                    "koi_period",
                                    "koi_duration",
                                    "koi_depth",
                                    "koi_prad",
                                    "koi_teq",
                                    "koi_insol",
                                    "koi_model_snr",
                                    "koi_steff",
                                    "koi_srad",
                                ],
                            )

                            upload_btn = gr.Button("Process File", variant="primary")
                            status = gr.Markdown()

                        with gr.Column():
                            gr.Markdown("#### Batch Results")
                            results_df = gr.DataFrame(
                                label="Predictions",
                                headers=[
                                    "koi_period",
                                    "koi_duration",
                                    "koi_depth",
                                    "koi_prad",
                                    "koi_teq",
                                    "koi_insol",
                                    "koi_model_snr",
                                    "koi_steff",
                                    "koi_srad",
                                    "prediction",
                                    "confidence",
                                ],
                            )

                            download_btn = gr.DownloadButton(
                                label="Download Results", variant="secondary"
                            )

            # Event handlers
            def respond(message, history):
                """Handle user input and get agent response"""
                if not message.strip():
                    return history, ""

                # Run the async function
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    history, _ = loop.run_until_complete(
                        self.chat_with_agent(message, history)
                    )
                finally:
                    loop.close()

                return history, ""

            def clear_chat():
                """Clear the chat history"""
                return [], ""

            def generate_prediction_query(
                period, duration, depth, prad, teq, insol, snr, steff, srad
            ):
                """Generate a formatted prediction query from the parameter inputs"""
                query = f"koi_period: {period}, koi_duration: {duration}, koi_depth: {depth}, koi_prad: {prad}, koi_teq: {teq}, koi_insol: {insol}, koi_model_snr: {snr}, koi_steff: {steff}, koi_srad: {srad}"
                return query

            def clear_parameters():
                """Reset all parameter inputs to default values"""
                return 10.0, 3.0, 1000.0, 1.2, 300.0, 1.0, 15.0, 5800.0, 1.0, ""

            def copy_query_to_chat(query):
                """Copy the generated query to the chat input"""
                return query

            def update_confusion_matrix(model_name):
                """Update confusion matrix when model selection changes"""
                return self.create_confusion_matrix_plot(model_name)

            # Connect events with Enter key support
            msg.submit(respond, [msg, chatbot], [chatbot, msg])
            send_btn.click(respond, [msg, chatbot], [chatbot, msg])
            clear_btn.click(clear_chat, outputs=[chatbot, msg])

            # Parameter input event handlers
            generate_query_btn.click(
                generate_prediction_query,
                inputs=[
                    koi_period,
                    koi_duration,
                    koi_depth,
                    koi_prad,
                    koi_teq,
                    koi_insol,
                    koi_model_snr,
                    koi_steff,
                    koi_srad,
                ],
                outputs=[generated_query],
            )

            clear_params_btn.click(
                clear_parameters,
                outputs=[
                    koi_period,
                    koi_duration,
                    koi_depth,
                    koi_prad,
                    koi_teq,
                    koi_insol,
                    koi_model_snr,
                    koi_steff,
                    koi_srad,
                    generated_query,
                ],
            )

            copy_query_btn.click(
                copy_query_to_chat, inputs=[generated_query], outputs=[msg]
            )

            # Analytics event handlers
            model_selector.change(
                update_confusion_matrix,
                inputs=[model_selector],
                outputs=[confusion_matrix_plot],
            )

            # Batch processing event handlers
            upload_btn.click(
                self.process_batch_file,
                inputs=[file_input],
                outputs=[sample_data, results_df, status],
            )

        return demo


def create_gradio_interface():
    """Create and return the Gradio interface"""
    interface = ExoVisionGradioInterface()
    return interface.create_gradio_interface()


async def main():
    print("🚀 ExoVision Agent is ready!")
    print("Starting Gradio interface...")

    # Create and launch the Gradio interface
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,  # Hugging Face Spaces expects port 7860
        share=False,
        show_error=True,
        inbrowser=True,  # Don't try to open browser in container
    )


if __name__ == "__main__":
    asyncio.run(main())
