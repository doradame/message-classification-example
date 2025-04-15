# app.py
import os
import re
import json
import logging
from typing import Dict, List, Optional, Tuple, Any

from flask import Flask, request, jsonify
import openai 
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from sentence_transformers import SentenceTransformer, util
from langdetect import detect, LangDetectException
import torch
from dotenv import load_dotenv

# Load environment variables from .env file BEFORE accessing them
load_dotenv()

# --- Configuration ---
# Fetches from environment variables loaded by load_dotenv() or system env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    # Critical error if API key is missing, raise immediately
    raise ValueError("FATAL: OPENAI_API_KEY environment variable not set.")

# Model Names (configurable via environment variables)
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", 'distiluse-base-multilingual-cased-v2')
SENTIMENT_MODEL_NAME = os.getenv("SENTIMENT_MODEL_NAME", "cardiffnlp/twitter-xlm-roberta-base-sentiment")
GPT_MODEL_NAME = os.getenv("GPT_MODEL_NAME", "gpt-4") # 

# Thresholds (configurable via environment variables with defaults)
EMBEDDING_CONFIDENCE_THRESHOLD = float(os.getenv("EMBEDDING_CONFIDENCE_THRESHOLD", 0.60))
SENTIMENT_POSITIVE_THRESHOLD = float(os.getenv("SENTIMENT_POSITIVE_THRESHOLD", 0.85))
SENTIMENT_NEGATIVE_THRESHOLD = float(os.getenv("SENTIMENT_NEGATIVE_THRESHOLD", 0.85))
SENTIMENT_NEUTRAL_THRESHOLD = float(os.getenv("SENTIMENT_NEUTRAL_THRESHOLD", 0.8))

# Confidence Scores for Different Methods
KEYWORD_UNSUB_CONFIDENCE = float(os.getenv("KEYWORD_UNSUB_CONFIDENCE", 0.95))
KEYWORD_INTERESTED_CONFIDENCE = float(os.getenv("KEYWORD_INTERESTED_CONFIDENCE", 0.90))
KEYWORD_NOT_INTERESTED_CONFIDENCE = float(os.getenv("KEYWORD_NOT_INTERESTED_CONFIDENCE", 0.90))
SENTIMENT_POSITIVE_CONFIDENCE = float(os.getenv("SENTIMENT_POSITIVE_CONFIDENCE", 0.75))
SENTIMENT_NEGATIVE_CONFIDENCE = float(os.getenv("SENTIMENT_NEGATIVE_CONFIDENCE", 0.75))
SENTIMENT_NEUTRAL_CONFIDENCE = float(os.getenv("SENTIMENT_NEUTRAL_CONFIDENCE", 0.65))
GPT_FALLBACK_CONFIDENCE_FLOOR = float(os.getenv("GPT_FALLBACK_CONFIDENCE_FLOOR", 0.5))

# Supported Languages
SUPPORTED_LANGUAGES = ["en", "it"]
DEFAULT_LANG = "en" # Fallback language for replies if specific lang not found

# --- Logging Setup ---
# Basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Variables & Model Placeholders ---
# Initialize Flask app
app = Flask(__name__)

# Initialize OpenAI client (ensure API key was loaded)
try:
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    logger.critical(f"Failed to initialize OpenAI client: {e}", exc_info=True)
    raise RuntimeError(f"Failed to initialize OpenAI client: {e}") from e

# Model variables, initialized to None before loading
embedding_model = None
sentiment_tokenizer = None
sentiment_model = None
precomputed_embeddings = {} # Store precomputed embeddings

# --- Data Structures ---
# (Could also be loaded from JSON/YAML files)
EMBEDDING_EXAMPLES = {
    "interested": ["yes please", "sure thing", "call me", "i want info", "interested",
                   "sì", "certo", "chiamami", "voglio info"],
    "not_interested": ["no thanks", "not interested", "maybe later", "don't want it",
                       "no", "non mi interessa", "forse dopo"],
    "unsubscribe": ["stop", "remove me", "unsubscribe me", "don't text me again",
                    "stop", "cancellami", "non scrivermi più"]
}

CUSTOM_REPLIES = {
    "interested": { "en": "Thanks! We’ll call you shortly.", "it": "Grazie! Ti chiameremo a breve." },
    "not_interested": { "en": "No problem, we’ll try again another time.", "it": "Va bene, magari alla prossima." },
    "unsubscribe": { "en": "Sorry to bother you. You won’t hear from us again.", "it": "Ci dispiace disturbarti. Non ti contatteremo più." },
    "unclear": { "en": "Just to be sure—did you mean yes, no, or stop?", "it": "Sei interessato, non interessato, o vuoi annullare l'iscrizione?" },
    "unsupported_language": { "en": "Sorry, this service currently supports only English and Italian.", "it": "Al momento supportiamo solo Italiano e Inglese." },
     "error": { "en": "Sorry, we encountered an technical issue. Please try again later.", "it": "Siamo spiacenti, si è verificato un problema tecnico. Riprova più tardi." }
}

KEYWORD_CATEGORIES = {
    "unsubscribe": { "keywords": { "en": ["stop", "unsubscribe", "don't text", "remove me"], "it": ["stop", "cancellami", "non scrivermi più"] }, "confidence": KEYWORD_UNSUB_CONFIDENCE },
    "interested": { "keywords": { "en": ["yes", "sure", "interested", "call me", "i want", "okay", "yeah", "please call"], "it": ["sì", "certo", "interessato", "chiamami", "voglio info"] }, "confidence": KEYWORD_INTERESTED_CONFIDENCE },
    "not_interested": { "keywords": { "en": ["no", "not interested", "don't want", "maybe later", "no thanks", "nah"], "it": ["no", "non mi interessa", "forse dopo"] }, "confidence": KEYWORD_NOT_INTERESTED_CONFIDENCE }
}

# --- Model Loading Function ---
def load_models():
    """Loads all ML models into global variables. Called once at startup."""
    global embedding_model, sentiment_tokenizer, sentiment_model, precomputed_embeddings
    logger.info("--- Starting Model Loading ---")
    try:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device='cpu') # Use 'cpu' for compatibility if you have no GPU !!!!
        logger.info("Embedding model loaded successfully.")
        # Check if embedding model loaded successfully
        if not embedding_model:
            raise RuntimeError("Failed to load embedding model.")
        # Load sentiment model
        logger.info(f"Loading sentiment model: {SENTIMENT_MODEL_NAME}")
        sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)
        sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME)
        logger.info("Sentiment model loaded successfully.")

        logger.info("Precomputing embeddings for examples...")
        # Precompute embeddings for labeled examples
        for label, examples in EMBEDDING_EXAMPLES.items():
            if embedding_model: # Check if embedding model loaded successfully
                 precomputed_embeddings[label] = embedding_model.encode(examples, convert_to_tensor=True)
            else:
                 raise RuntimeError("Cannot precompute embeddings, embedding model failed to load.")
        logger.info("Example embeddings precomputed successfully.")
        logger.info("--- Model Loading Finished ---")

    except Exception as e:
        # Log the error and re-raise to signal failure
        logger.error(f"CRITICAL ERROR during model loading: {e}", exc_info=True)
        raise RuntimeError(f"Failed to load models: {e}") from e

# --- Helper Functions ---
def get_reply(intent: str, lang: str) -> str:
    """Gets the appropriate reply for a given intent and language, with fallback."""
    if intent not in CUSTOM_REPLIES:
        logger.warning(f"Intent '{intent}' not found in CUSTOM_REPLIES. Falling back to 'unclear'.")
        intent = "unclear"
    return CUSTOM_REPLIES[intent].get(lang, CUSTOM_REPLIES[intent].get(DEFAULT_LANG, next(iter(CUSTOM_REPLIES[intent].values()))))

def detect_language(text: str) -> str:
    """Detects language, falling back to DEFAULT_LANG if unsupported or detection fails."""
    try:
        lang = detect(text)
        if lang not in SUPPORTED_LANGUAGES:
            logger.warning(f"Detected language '{lang}' not in supported list {SUPPORTED_LANGUAGES}. Using {DEFAULT_LANG} for processing.")
            return DEFAULT_LANG
        return lang
    except LangDetectException:
        logger.warning(f"Language detection failed for text: '{text[:50]}...'. Falling back to {DEFAULT_LANG}.", exc_info=False) # Avoid logging full stack trace usually
        return DEFAULT_LANG

def create_response(intent: str, confidence: float, lang: str, error: Optional[str] = None) -> Dict[str, Any]:
    """Standardizes the creation of the response dictionary."""
    response = {
        "intent": intent,
        "reply": get_reply(intent, lang),
        "confidence": round(confidence, 2),
        "language": lang
    }
    if error:
        response["error"] = error
    return response

# --- Classification Logic Functions ---
def classify_by_keyword(text: str, lang: str) -> Optional[Dict[str, Any]]:
    """Classifies intent based on keywords."""
    text_lower = text.strip().lower()
    for intent, category_data in KEYWORD_CATEGORIES.items():
        keywords_for_lang = category_data["keywords"].get(lang, [])
        for keyword in keywords_for_lang:
            if re.search(rf"\b{re.escape(keyword)}\b", text_lower):
                logger.info(f"Keyword match: '{keyword}' -> {intent}")
                return create_response(intent, category_data["confidence"], lang)
    return None

def classify_by_sentiment(text: str, lang: str) -> Optional[Dict[str, Any]]:
    """Classifies intent based on sentiment analysis."""
    if not sentiment_model or not sentiment_tokenizer:
         logger.error("Sentiment model/tokenizer not loaded. Skipping sentiment classification.")
         return None
    try:
        encoded = sentiment_tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            output = sentiment_model(**encoded)
        scores = softmax(output.logits.detach().cpu().numpy()[0])
        sentiment_scores = {'negative': scores[0], 'neutral': scores[1], 'positive': scores[2]}
        logger.info(f"Sentiment scores: Neg={sentiment_scores['negative']:.2f}, Neu={sentiment_scores['neutral']:.2f}, Pos={sentiment_scores['positive']:.2f}")

        if sentiment_scores['positive'] > SENTIMENT_POSITIVE_THRESHOLD:
            logger.info("Sentiment -> interested")
            return create_response("interested", SENTIMENT_POSITIVE_CONFIDENCE, lang)
        elif sentiment_scores['negative'] > SENTIMENT_NEGATIVE_THRESHOLD:
            logger.info("Sentiment -> unsubscribe")
            return create_response("unsubscribe", SENTIMENT_NEGATIVE_CONFIDENCE, lang)
        elif sentiment_scores['neutral'] > SENTIMENT_NEUTRAL_THRESHOLD:
             logger.info("Sentiment -> not_interested")
             return create_response("not_interested", SENTIMENT_NEUTRAL_CONFIDENCE, lang)
        else:
            logger.info("Sentiment -> unclear")
            return None
    except Exception as e:
        logger.error(f"Error during sentiment analysis: {e}", exc_info=True)
        return None

def classify_by_embedding(text: str, lang: str) -> Optional[Dict[str, Any]]:
    """Classifies intent based on semantic similarity using embeddings."""
    logging.info("Classifying using embeddings...")
    if not embedding_model or not precomputed_embeddings:
        logger.error("Embedding model/precomputed embeddings not available. Skipping embedding classification.")
        return None
    try:
        text_emb = embedding_model.encode(text, convert_to_tensor=True)
        best_score = 0.0
        best_label = "unclear"

        for label, example_embs in precomputed_embeddings.items():
            scores = util.cos_sim(text_emb, example_embs)
            max_score_for_label = float(scores.max())
            # logger.debug(f"Embedding score for label '{label}': {max_score_for_label:.4f}")
            if max_score_for_label > best_score:
                best_score = max_score_for_label
                best_label = label

        logger.info(f"Best embedding match: label='{best_label}', score={best_score:.4f}")
        if best_score >= EMBEDDING_CONFIDENCE_THRESHOLD and best_label in CUSTOM_REPLIES:
            logger.info(f"Embedding -> {best_label}")
            return create_response(best_label, best_score, lang)
        else:
            return None
    except Exception as e:
        logger.error(f"Error during embedding classification: {e}", exc_info=True)
        return None

def classify_with_gpt(text: str, lang: str) -> Dict[str, Any]:
    """Uses OpenAI GPT as a fallback for classification."""
    logger.info("Falling back to GPT for classification.")
    prompt = (
        f"You are an intent classifier for SMS replies to marketing messages. The user replied with the following message (language: {lang}):\n"
        f"\"{text}\"\n\n"
        f"Classify the user's intent strictly as one of: 'interested', 'not_interested', 'unsubscribe', 'unclear'. Consider the context.\n"
        f"If you detect negative reactions, classify as 'unsubscribe'. If you detect clear disconfort about the marketing message or offensive word classify as 'unsubscribe'.\n"
        f"Respond ONLY with a JSON object containing three keys: 'intent', 'reasoning' (brief), and 'confidence' (float between {GPT_FALLBACK_CONFIDENCE_FLOOR} and 1.0)." )
    try:
        response = openai_client.chat.completions.create(
            model=GPT_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an accurate and concise intent classification assistant outputting JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        gpt_response_content = response.choices[0].message.content
        logger.info(f"Raw GPT response: {gpt_response_content}")
        try:
            gpt_data = json.loads(gpt_response_content)
            intent = gpt_data.get("intent")
            confidence = float(gpt_data.get("confidence", 0.0))
            if intent not in CUSTOM_REPLIES or confidence < GPT_FALLBACK_CONFIDENCE_FLOOR:
                 logger.warning(f"Invalid intent ('{intent}') or low confidence ({confidence}) from GPT. Treating as 'unclear'.")
                 return create_response("unclear", 0.4, lang, error="Invalid GPT response or low confidence.")
            logger.info(f"GPT -> {intent} (Conf: {confidence}). Reason: {gpt_data.get('reasoning', 'N/A')}")
            final_confidence = max(GPT_FALLBACK_CONFIDENCE_FLOOR, min(1.0, confidence))
            return create_response(intent, final_confidence, lang)
        except (json.JSONDecodeError, TypeError, KeyError, ValueError) as json_e:
            logger.error(f"Failed to parse GPT JSON response: {json_e}. Content: {gpt_response_content}", exc_info=True)
            return create_response("unclear", 0.4, lang, error=f"GPT response parsing error: {json_e}")
    except openai.APIError as api_e:
        logger.error(f"OpenAI API error: {api_e}", exc_info=True)
        return create_response("unclear", 0.4, lang, error=f"OpenAI API error: {api_e}")
    except Exception as e:
        logger.error(f"Unexpected error during GPT classification: {e}", exc_info=True)
        return create_response("unclear", 0.4, lang, error=f"Unexpected GPT error: {e}")

# --- Main Classification Pipeline Function ---
def classify_message(text: str) -> Dict[str, Any]:
    """Main classification pipeline: Keyword -> Sentiment -> Embedding -> GPT."""
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
         logger.warning("Received empty or invalid text message.")
         return create_response("unclear", 0.1, DEFAULT_LANG, error="Empty message received")

    # Ensure models are loaded before proceeding (belt-and-suspenders check)
    if not embedding_model or not sentiment_model:
        logger.error("Models are not loaded. Cannot perform classification.")
        return create_response("error", 0.0, DEFAULT_LANG, error="Internal server error: Classification models unavailable.")

    logger.info(f"Processing message: '{text}'")
    lang = detect_language(text)
    logger.info(f"Detected language: {lang}")

    # Execute classification steps in order
    result = classify_by_keyword(text, lang)
    if result: return result

    result = classify_by_sentiment(text, lang)
    if result: return result

    result = classify_by_embedding(text, lang)
    if result: return result

    result = classify_with_gpt(text, lang)
    return result

# --- Load Models Globally at Startup ---
# This is crucial for Gunicorn with --preload. It runs when the module is imported.
try:
    load_models()
except Exception as startup_error:
    # Log critical error and prevent app from starting if models fail
    logger.critical(f"FATAL: Failed to load models during app initialization: {startup_error}", exc_info=True)
    # Exit helps ensure Gunicorn doesn't try to run workers with broken state
    exit(1)

# --- Flask Routes ---
@app.route("/classify", methods=["POST"])
def classify_endpoint():
    """Flask endpoint to classify a message."""
    # Check if models are loaded before processing request
    if not embedding_model or not sentiment_model:
         logger.error("Models not loaded, cannot classify request.")
         # Return 503 Service Unavailable if models aren't ready
         return jsonify(create_response("error", 0.0, DEFAULT_LANG, error="Service temporarily unavailable: Models not loaded.")), 503

    if not request.is_json:
        logger.warning("Received non-JSON request")
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    message_text = data.get("message")

    if not message_text:
        logger.warning("Missing 'message' field in request JSON")
        return jsonify({"error": "Missing 'message' field in JSON payload"}), 400

    try:
        classification_result = classify_message(message_text)
        return jsonify(classification_result), 200
    except Exception as e:
        logger.error(f"Unexpected error during classification for message: '{message_text[:50]}...': {e}", exc_info=True)
        error_response = create_response("error", 0.0, DEFAULT_LANG, error="Internal server error during classification.")
        return jsonify(error_response), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Basic health check endpoint, verifies models are loaded."""
    models_loaded = embedding_model is not None and sentiment_model is not None and sentiment_tokenizer is not None and precomputed_embeddings is not None
    status = "ok" if models_loaded else "error"
    http_code = 200 if models_loaded else 503 # Use 503 Service Unavailable if not ready
    logger.info(f"Health check endpoint called. Models loaded: {models_loaded}")
    return jsonify({"status": status, "models_loaded": models_loaded}), http_code

# --- Main Execution Block (for direct 'python app.py' runs ONLY) ---
if __name__ == "__main__":
    # This block is ignored when running with Gunicorn
    logger.info("Attempting to start Flask development server directly (python app.py)...")
    # Models should already be loaded globally above.
    # We add a final check here before trying to run Flask's dev server.
    if not (embedding_model and sentiment_model):
         logger.critical("Models not loaded successfully earlier. Aborting Flask development server start.")
         exit(1)
    try:
        # Run Flask's development server (NOT for production)
        # Use host='0.0.0.0' to make it accessible on your network
        app.run(debug=False, host='0.0.0.0', port=5000)
    except Exception as run_error:
         logger.critical(f"Flask development server failed: {run_error}", exc_info=True)
         exit(1)
