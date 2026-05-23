import logging
from typing import Tuple, Dict, List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("IntentRouter")

class IntentRouter:
    """
    IntentRouter routes incoming queries to either 'quantitative' or 'textual_sentiment'
    based on TF-IDF representation and cosine similarity against predefined anchor profiles.
    """

    def __init__(self) -> None:
        # Predefined anchor queries representing the 'quantitative' intent
        self.quant_anchors: List[str] = [
            "what is the revenue growth rate of the company",
            "calculate the EBITDA or net income for the quarter",
            "get the financial metrics and EPS of the ticker",
            "retrieve the average P/E ratio across the tech sector",
            "what are the numerical values for cash flow and operating margin",
            "compute the financial ratios and sales figures for Q2 or Q3",
            "sum of all revenue and operating income statistics",
            "show the statistical financial table and quarterly metrics"
        ]

        # Predefined anchor queries representing the 'textual_sentiment' intent
        self.textual_anchors: List[str] = [
            "what are the major risks, uncertainties, or threats discussed in the report",
            "analyze the tone, sentiment, and outlook of the shareholder letter",
            "is there any negative sentiment, regulatory challenge, or potential litigation risk",
            "summarize the qualitative risk factors, competitive landscape, and challenges",
            "explain the regulatory issues and environmental compliance concerns",
            "what does the CEO letter say about market competition and product obsolescence",
            "evaluate the overall risk sentiment of the corporate outlook",
            "read the qualitative statements about supply chain disruptions and logistics delays"
        ]

        # Combine anchors for vectorizer fitting
        self.all_anchors: List[str] = self.quant_anchors + self.textual_anchors
        self.vectorizer: TfidfVectorizer = TfidfVectorizer(stop_words='english')

        try:
            # Fit vectorizer and transform the anchor profiles
            self.anchor_matrices: np.ndarray = self.vectorizer.fit_transform(self.all_anchors)
            # Separate transformed matrices for quick comparison
            self.quant_matrix = self.anchor_matrices[:len(self.quant_anchors)]
            self.textual_matrix = self.anchor_matrices[len(self.quant_anchors):]
            logger.info("IntentRouter successfully initialized and vectorizer fitted.")
        except Exception as e:
            logger.error(f"Error during TF-IDF Vectorizer initialization: {str(e)}")
            raise

    def route_query(self, query: str) -> Tuple[str, float]:
        """
        Compares the incoming query against predefined anchor profiles using cosine similarity.

        Args:
            query (str): The natural language query from the user.

        Returns:
            Tuple[str, float]: The routed agent name ('quantitative' or 'textual_sentiment') 
                               and the associated confidence score (cosine similarity).
        """
        if not query or not query.strip():
            logger.warning("Empty query received. Routing to 'textual_sentiment' by default.")
            return "textual_sentiment", 0.0

        try:
            # Transform query into TF-IDF space
            query_vector = self.vectorizer.transform([query])

            # Calculate cosine similarity with both sets of anchors
            quant_similarities = cosine_similarity(query_vector, self.quant_matrix)[0]
            textual_similarities = cosine_similarity(query_vector, self.textual_matrix)[0]

            # We use the maximum similarity to any single anchor in each category to classify,
            # as a query might match a specific anchor very closely.
            max_quant_sim: float = float(np.max(quant_similarities))
            max_textual_sim: float = float(np.max(textual_similarities))

            logger.info(f"Query similarities - Quantitative: {max_quant_sim:.4f}, Textual: {max_textual_sim:.4f}")

            # Decide route based on higher similarity score
            if max_quant_sim >= max_textual_sim:
                # Return quantitative route; default threshold is 0.0 but can be used for out-of-domain detection
                route = "quantitative"
                confidence = max_quant_sim
            else:
                route = "textual_sentiment"
                confidence = max_textual_sim

            logger.info(f"Routed query to '{route}' with confidence {confidence:.4f}")
            return route, confidence

        except Exception as e:
            logger.error(f"Error occurred during query routing: {str(e)}")
            # Fallback safe response
            return "textual_sentiment", 0.0
