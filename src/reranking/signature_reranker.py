from typing import Dict, List, Any
from src.reranking.reranker import Reranker

class SignatureReranker(Reranker):
    """
    Reranker that prioritizes code with matching function signatures.
    """
    def __init__(self, weight: float = 1.5):
        """
        Initialize the signature reranker.
        
        Args:
            weight: Weight to apply to the signature score
        """
        super().__init__()
        self.weight = weight
    
    def compute_signature_similarity(self, ctx_components: Dict[str, Any], 
                                    query_intent: Dict[str, Any]) -> float:
        """
        Compute the signature similarity between context and query intent.
        
        Args:
            ctx_components: Components extracted from context
            query_intent: Extracted intent from the query
            
        Returns:
            Similarity score between 0 and 1
        """
        score = 0.0
        total_weight = 0.0
        
        # Compare function names if available
        function_weight = 2.0
        
        ctx_function = ctx_components.get('function_name', '')
        
        if 'function_name' in query_intent:
            total_weight += function_weight
            function_similarity = 1.0 if ctx_function == query_intent['function_name'] else 0.0
            score += function_weight * function_similarity
        elif 'potential_names' in query_intent and query_intent['potential_names']:
            total_weight += function_weight
            
            # Check if any potential name matches
            if any(name in ctx_function for name in query_intent['potential_names']):
                score += function_weight * 0.7  # Partial match
            elif any(name == ctx_function for name in query_intent['potential_names']):
                score += function_weight  # Exact match
        
        # Compare parameters
        if 'parameters' in query_intent and query_intent['parameters']:
            param_weight = 1.5
            total_weight += param_weight
            
            ctx_params = set(ctx_components.get('parameters', []))
            query_params = set(query_intent['parameters'])
            
            if query_params:
                param_similarity = len(ctx_params.intersection(query_params)) / len(query_params)
                score += param_weight * param_similarity
        
        # Check if return value handling matches
        if 'has_return' in query_intent and query_intent['has_return']:
            return_weight = 0.8
            total_weight += return_weight
            
            ctx_returns = ctx_components.get('return_type', [])
            has_return = len(ctx_returns) > 0
            
            score += return_weight if has_return else 0
        
        # Check if error handling matches
        if 'error_handling' in query_intent and query_intent['error_handling']:
            error_weight = 1.0
            total_weight += error_weight
            
            has_error_handling = ctx_components.get('try_count', 0) > 0 and ctx_components.get('except_count', 0) > 0
            
            score += error_weight if has_error_handling else 0
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def rerank(self, query: str, contexts: List[Dict[str, Any]], query_intent: Dict[str, Any],
               answer_components: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Rerank the contexts based on function signature similarity to query intent.
        
        Args:
            query: Query string
            contexts: List of context dictionaries
            query_intent: Extracted intent from the query
            answer_components: Expected answer components
            
        Returns:
            Reranked list of contexts
        """
        for ctx in contexts:
            signature_score = self.compute_signature_similarity(ctx['components'], query_intent)
            ctx['signature_score'] = signature_score
            
            # Update final score if it exists, otherwise create it
            if 'final_score' in ctx:
                ctx['final_score'] += self.weight * signature_score
            else:
                ctx['final_score'] = ctx['score'] + self.weight * signature_score
        
        return sorted(contexts, key=lambda x: x['final_score'], reverse=True)