import os
import time
import logging
from typing import Dict, Any, List

# Add parent directory of src to path if needed (done automatically by standard running)
from router import IntentRouter
from agents.quant_agent import QuantAnalystAgent
from agents.textual_agent import TextualSentimentAgent

# Configure logging to console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("RAGOrchestrator")

class RAGOrchestrator:
    """
    RAGOrchestrator coordinates the IntentRouter and specialized agents
    (QuantAnalystAgent and TextualSentimentAgent) to answer natural language
    financial and corporate queries.
    """

    def __init__(self, csv_path: str = "data/financial_metrics.csv", txt_path: str = "data/annual_report.txt") -> None:
        logger.info("Initializing RAG Orchestrator system components...")
        self.router: IntentRouter = IntentRouter()
        self.quant_agent: QuantAnalystAgent = QuantAnalystAgent(data_path=csv_path)
        self.textual_agent: TextualSentimentAgent = TextualSentimentAgent(data_path=txt_path)
        logger.info("All orchestrator sub-agents and routers initialized successfully.")

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Routes the user query and delegates execution to the appropriate specialized agent.

        Args:
            query (str): The natural language query.

        Returns:
            Dict[str, Any]: Consolidated response containing route, confidence, and output payload.
        """
        start_time = time.perf_counter()
        logger.info(f"Processing query: '{query}'")

        try:
            # 1. Intent Routing Step
            route, confidence = self.router.route_query(query)
            logger.info(f"Route selected: {route} (Confidence: {confidence:.4f})")

            # 2. Agent Execution Step
            if route == "quantitative":
                agent_response = self.quant_agent.execute_query(query)
            elif route == "textual_sentiment":
                agent_response = self.textual_agent.execute_query(query)
            else:
                raise ValueError(f"Unknown route '{route}' suggested by IntentRouter.")

            execution_time = time.perf_counter() - start_time
            logger.info(f"Query processed successfully in {execution_time * 1000:.2f} ms")

            # Combine metadata into final report
            return {
                "query": query,
                "route": route,
                "confidence": confidence,
                "status": agent_response.get("status", "success"),
                "agent_response": agent_response,
                "execution_time_ms": round(execution_time * 1000, 2)
            }

        except Exception as e:
            execution_time = time.perf_counter() - start_time
            logger.error(f"Failed to process query '{query}': {str(e)}")
            return {
                "query": query,
                "route": "error",
                "confidence": 0.0,
                "status": "error",
                "message": f"Orchestrator execution error: {str(e)}",
                "execution_time_ms": round(execution_time * 1000, 2)
            }


def main() -> None:
    # Verify the local directories and write paths
    csv_file = "data/financial_metrics.csv"
    txt_file = "data/annual_report.txt"

    print("=" * 80)
    print("        MULTI-AGENT RAG ANALYST ROUTING ENGINE - PRODUCTION TEST RUN        ")
    print("=" * 80)

    # Initialize the master coordinator
    try:
        orchestrator = RAGOrchestrator(csv_path=csv_file, txt_path=txt_file)
    except Exception as e:
        print(f"CRITICAL: Failed to initialize orchestration engine: {e}")
        return

    # Define validation queries crossing quantitative and sentiment domains
    test_queries: List[str] = [
        # Quantitative Intent
        "What was Apple's net income for Q2?",
        "What is the average revenue across all tickers?",
        "What is the EBITDA of MSFT in Q3?",
        "Calculate the highest PE ratio in the dataset",
        
        # Textual/Sentiment Intent
        "Explain the major regulatory risks and compliance challenges described in the report.",
        "What does the letter to shareholders say about our market growth and innovative product engineering?",
        "Summarize key competition and supply chain threats.",
        "How is artificial intelligence mentioned in the outlook?"
    ]

    print("\nStarting execution of query workload...\n")

    for i, q in enumerate(test_queries, 1):
        print("-" * 80)
        print(f"QUERY {i}: \"{q}\"")
        print("-" * 80)
        
        result = orchestrator.process_query(q)
        
        print(f"\n[ROUTING]: Agent Selected -> '{result['route']}' (Confidence: {result['confidence']:.4f})")
        print(f"[LATENCY]: {result['execution_time_ms']} ms")
        
        agent_res = result["agent_response"]
        if result["status"] == "success":
            print("\n[AGENT OUTPUT]:")
            print(agent_res.get("formatted_result", "No output formatted."))
        else:
            print(f"\n[ERROR]: {result.get('message', 'Unknown agent processing error.')}")
        
        print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
