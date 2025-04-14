from typing import Dict, List, Any
# TODO!!!!!!
# TODO!!!!!!
# TODO!!!!!!
class Reranker:
    """
    Base class for rerankers that improve retrieval ranking.
    """
    def __init__(self):
        """
        Initialize the base reranker.
        """
        pass
    
    def rerank(self, query: str, contexts: List[Dict[str, Any]], query_intent: Dict[str, Any], 
               answer_components: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Rerank the contexts based on additional criteria.
        
        Args:
            query: Query string
            contexts: List of context dictionaries
            query_intent: Extracted intent from the query
            answer_components: Expected answer components
            
        Returns:
            Reranked list of contexts
        """
        raise NotImplementedError("Subclasses must implement rerank method")