from typing import Dict, List, Any
from reranking.reranker import Reranker

class CodeStructureReranker(Reranker):
    """
    Reranker that prioritizes code with similar structure to the expected answer.
    """
    def __init__(self, weight: float = 1.0):
        """
        Initialize the code structure reranker.
        
        Args:
            weight: Weight to apply to the structure score
        """
        super().__init__()
        self.weight = weight
    
    def compute_structure_similarity(self, ctx_components: Dict[str, Any], 
                                    answer_components: Dict[str, Any]) -> float:
        """
        Compute the structural similarity between context and answer.
        
        Args:
            ctx_components: Components extracted from context
            answer_components: Components extracted from answer
            
        Returns:
            Similarity score between 0 and 1
        """
        score = 0.0
        total_weight = 0.0
        
        # Compare control flow structures
        control_flow_features = [
            'if_count', 'else_count', 'for_count', 'while_count', 
            'try_count', 'except_count', 'function_calls'
        ]
        
        for feature in control_flow_features:
            weight = 0.5  # Weight for control flow features
            total_weight += weight
            
            ctx_value = ctx_components.get(feature, 0)
            answer_value = answer_components.get(feature, 0)
            
            # Normalize difference
            max_value = max(ctx_value, answer_value)
            if max_value > 0:
                similarity = 1.0 - abs(ctx_value - answer_value) / max_value
            else:
                similarity = 1.0  # Both zero is a perfect match
            
            score += weight * similarity
        
        # Compare imports (important for functionality)
        ctx_imports = set(ctx_components.get('imports', []) + ctx_components.get('from_imports', []))
        answer_imports = set(answer_components.get('imports', []) + answer_components.get('from_imports', []))
        
        if answer_imports:
            import_weight = 1.5
            total_weight += import_weight
            import_similarity = len(ctx_imports.intersection(answer_imports)) / len(answer_imports)
            score += import_weight * import_similarity
        
        # Code length can be indicative of complexity match
        if 'line_count' in ctx_components and 'line_count' in answer_components:
            length_weight = 0.3
            total_weight += length_weight
            
            ctx_lines = ctx_components['line_count']
            answer_lines = answer_components['line_count']
            
            max_lines = max(ctx_lines, answer_lines)
            length_similarity = 1.0 - abs(ctx_lines - answer_lines) / max_lines
            
            score += length_weight * length_similarity
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def rerank(self, query: str, contexts: List[Dict[str, Any]], query_intent: Dict[str, Any],
               answer_components: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Rerank the contexts based on code structure similarity.
        
        Args:
            query: Query string
            contexts: List of context dictionaries
            query_intent: Extracted intent from the query
            answer_components: Expected answer components
            
        Returns:
            Reranked list of contexts
        """
        try:
            for ctx in contexts:
                structure_score = self.compute_structure_similarity(ctx['components'], answer_components)
                ctx['structure_score'] = structure_score
                ctx['final_score'] = ctx['score'] + self.weight * structure_score
            
            return sorted(contexts, key=lambda x: x['final_score'], reverse=True)
        except Exception as e:
            print(f"Error in structure reranker: {e}")
            return contexts