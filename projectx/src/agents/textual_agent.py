import os
import re
import logging
from typing import Dict, Any, List, Set

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TextualSentimentAgent")

class TextualSentimentAgent:
    """
    TextualSentimentAgent processes unstructured markdown or plain text files,
    performs rule-based RAG segmentation, retrieves relevant text passages based
    on query overlap, and computes numerical sentiment and risk metrics.
    """

    def __init__(self, data_path: str = "data/annual_report.txt") -> None:
        self.data_path: str = data_path
        
        # Predefined sentiment vocabulary mappings
        self.positive_vocab: Set[str] = {
            "growth", "strong", "positive", "increase", "profit", "succeed",
            "expansion", "benefit", "opportunity", "record", "leadership",
            "innovative", "optimistic", "flexibility", "successful", "pleased"
        }
        
        self.negative_vocab: Set[str] = {
            "risk", "uncertainty", "litigation", "lawsuit", "decline", "loss",
            "weakness", "threat", "adversely", "competition", "regulation",
            "challenge", "failure", "bottleneck", "disruption", "vulnerability"
        }

    def _read_file_safely(self) -> str:
        """
        Opens and reads unstructured data files safely, using read-only access
        to prevent blocking issues or concurrency file-locking bottlenecks.
        """
        if not os.path.exists(self.data_path):
            logger.error(f"Textual corpus file not found at: {self.data_path}")
            return ""

        try:
            # Open file in read-only mode explicitly
            with open(self.data_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            return content
        except Exception as e:
            logger.error(f"Failed to read textual file safely: {str(e)}")
            return ""

    def segment_text(self, text: str) -> List[str]:
        """
        Segments raw document string into cohesive paragraphs.
        """
        if not text:
            return []
        
        # Split on double newlines to isolate paragraphs/sections
        raw_segments = re.split(r'\n\s*\n', text)
        cleaned_segments = []
        for segment in raw_segments:
            cleaned = segment.strip()
            if cleaned:
                cleaned_segments.append(cleaned)
        return cleaned_segments

    def retrieve_relevant_segments(self, query: str, segments: List[str], top_k: int = 2) -> List[str]:
        """
        Basic RAG step: Ranks and retrieves the top-K segments matching query keywords.
        """
        if not segments:
            return []

        # Tokenize query, filter out stop words and lowercase
        stop_words = {"what", "is", "the", "are", "about", "for", "in", "of", "and", "a", "an", "to"}
        query_words = {word.lower() for word in re.findall(r'\b\w+\b', query) if word.lower() not in stop_words}
        
        if not query_words:
            # If no query terms remain, return first top_k segments
            return segments[:top_k]

        scored_segments = []
        for segment in segments:
            segment_words = {w.lower() for w in re.findall(r'\b\w+\b', segment)}
            # Score based on vocabulary intersection
            intersection = query_words.intersection(segment_words)
            score = len(intersection)
            scored_segments.append((score, segment))

        # Sort by score in descending order
        scored_segments.sort(key=lambda x: x[0], reverse=True)
        
        # Return top_k text segments
        return [seg for score, seg in scored_segments[:top_k]]

    def calculate_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Calculates sentiment and risk score of a given piece of text based on
        directional keyword densities.
        """
        # Tokenize text into words
        words = re.findall(r'\b\w+\b', text.lower())
        total_words = len(words)
        
        if total_words == 0:
            return {
                "sentiment_score": 0.0,
                "risk_score": 0.0,
                "positive_matches": [],
                "negative_matches": [],
                "word_count": 0
            }

        pos_matches = [w for w in words if w in self.positive_vocab]
        neg_matches = [w for w in words if w in self.negative_vocab]

        pos_count = len(pos_matches)
        neg_count = len(neg_matches)

        # Net sentiment index: (Pos - Neg) / (Pos + Neg)
        # Ranges from -1.0 (purely negative) to +1.0 (purely positive)
        divisor = pos_count + neg_count
        sentiment_score = (pos_count - neg_count) / divisor if divisor > 0 else 0.0

        # Risk index: density of risk vocabulary in the text
        risk_score = neg_count / total_words

        return {
            "sentiment_score": round(sentiment_score, 4),
            "risk_score": round(risk_score, 4),
            "positive_matches": list(set(pos_matches)),
            "negative_matches": list(set(neg_matches)),
            "word_count": total_words
        }

    def execute_query(self, query: str) -> Dict[str, Any]:
        """
        Executes textual analyst logic: reads file, retrieves context matching
        the query, and calculates explicit sentiment and risk values.

        Args:
            query (str): The natural language query.

        Returns:
            Dict[str, Any]: Payload containing matched text passages, scores,
                            and detailed logs.
        """
        raw_text = self._read_file_safely()
        if not raw_text:
            return {
                "status": "error",
                "message": "Textual database is empty or could not be loaded safely.",
                "value": None
            }

        try:
            # Segment text corpus
            segments = self.segment_text(raw_text)
            
            # Retrieve relevant context
            retrieved = self.retrieve_relevant_segments(query, segments, top_k=2)
            retrieved_context = "\n\n".join(retrieved)

            # Perform sentiment and risk assessment
            scores = self.calculate_sentiment(retrieved_context)
            
            # Synthesize text result summary
            sentiment_label = "Positive" if scores["sentiment_score"] > 0.1 else ("Negative" if scores["sentiment_score"] < -0.1 else "Neutral")
            formatted_result = (
                f"Textual sentiment analysis of the retrieved context:\n"
                f"- Overall Sentiment: {sentiment_label} (Score: {scores['sentiment_score']})\n"
                f"- Risk/Uncertainty Density: {scores['risk_score'] * 100:.2f}%\n"
                f"- Key Positive Words: {', '.join(scores['positive_matches']) or 'None'}\n"
                f"- Key Risk Words: {', '.join(scores['negative_matches']) or 'None'}\n\n"
                f"Retrieved Document Context:\n{retrieved_context}"
            )

            return {
                "status": "success",
                "query": query,
                "sentiment_score": scores["sentiment_score"],
                "risk_score": scores["risk_score"],
                "positive_matches": scores["positive_matches"],
                "negative_matches": scores["negative_matches"],
                "retrieved_context": retrieved,
                "formatted_result": formatted_result
            }

        except Exception as e:
            logger.error(f"Error during textual sentiment execution: {str(e)}")
            return {
                "status": "error",
                "message": f"Sentiment analysis error: {str(e)}",
                "value": None
            }
