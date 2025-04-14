from typing import Dict, List, Any, Optional
from reranking.reranker import Reranker
from reranking.code_structure_reranker import CodeStructureReranker
from reranking.signature_reranker import SignatureReranker

class CombinedReranker(Reranker):
    """
    Reranker that combines multiple reranking strategies.
    """
    def __init__(self, rerankers: Optional[List[Reranker]] = None, weights: Optional[Dict[str, float]] = None):
        """
        Initialize the combined reranker.
        
        Args:
            rerankers: List of rerankers to combine (defaults to CodeStructureReranker and SignatureReranker)
            weights: Dictionary mapping reranker class names to weights (defaults to equal weights)
        """
        super().__init__()
        self.rerankers = rerankers or [
            CodeStructureReranker(weight=1.0),
            SignatureReranker(weight=1.5)
        ]
        
        # Apply custom weights if provided
        if weights:
            for reranker in self.rerankers:
                class_name = reranker.__class__.__name__
                if class_name in weights:
                    reranker.weight = weights[class_name]
    
    def rerank(self, query: str, contexts: List[Dict[str, Any]], query_intent: Dict[str, Any],
               answer_components: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Rerank the contexts by applying multiple rerankers and combining their scores.
        
        Args:
            query: Query string
            contexts: List of context dictionaries
            query_intent: Extracted intent from the query
            answer_components: Expected answer components
            
        Returns:
            Reranked list of contexts
        """
        try:
            # Start with the original scores
            for ctx in contexts:
                ctx['final_score'] = ctx['score']
            
            # Apply each reranker in sequence
            current_contexts = contexts
            for reranker in self.rerankers:
                current_contexts = reranker.rerank(query, current_contexts, query_intent, answer_components)
            
            # Apply a normalizing factor to adjust for combined weights
            normalizing_factor = sum(reranker.weight for reranker in self.rerankers)
            
            if normalizing_factor > 0:
                for ctx in current_contexts:
                    # Original score contributes 40%, rerankers' scores contribute 60%
                    base_weight = 0.4
                    reranker_weight = 0.6
                    
                    # Normalize the contribution from rerankers
                    reranker_score = (ctx['final_score'] - ctx['score']) / normalizing_factor
                    
                    # Compute final score
                    ctx['final_score'] = base_weight * ctx['score'] + reranker_weight * reranker_score
            
            return sorted(current_contexts, key=lambda x: x['final_score'], reverse=True)
        except Exception as e:
            print(f"Error in combined reranker: {e}")
            return contexts