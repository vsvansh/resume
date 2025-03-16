import nltk
import logging
from app import app

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Download NLTK data
try:
    # Download all necessary NLTK resources
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger')  # For POS tagging
    logger.info("NLTK data downloaded successfully")
except Exception as e:
    logger.error(f"Error downloading NLTK data: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
