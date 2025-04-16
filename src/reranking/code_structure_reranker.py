from typing import Dict, List, Any
from reranking.reranker import Reranker

class CodeStructureReranker(Reranker):
    """Reranker that prioritizes code with similar structure to the expected answer."""
    
    def __init__(self, weight: float = 1.0):
        """Initialize with given weight."""
        super().__init__()
        self.weight = weight
    
    def compute_structure_similarity(self, ctx_components: Dict[str, Any], 
                                    answer_components: Dict[str, Any]) -> float:
        """Compute structural similarity between context and answer components."""
        score = 0.0
        total_weight = 0.0
        
        # Compare control flow structures
        control_flow_features = [
            'if_count', 'else_count', 'for_count', 'while_count', 
            'try_count', 'except_count'
        ]
        
        for feature in control_flow_features:
            weight = 0.5
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
        
        # Handle function calls separately
        if 'function_calls' in ctx_components and 'function_calls' in answer_components:
            weight = 0.5
            total_weight += weight
            
            ctx_calls = set(ctx_components['function_calls']) if isinstance(ctx_components['function_calls'], list) else set()
            answer_calls = set(answer_components['function_calls']) if isinstance(answer_components['function_calls'], list) else set()
            
            if answer_calls:
                call_similarity = len(ctx_calls.intersection(answer_calls)) / len(answer_calls)
                score += weight * call_similarity
            elif not ctx_calls:
                # Both empty is a match
                score += weight
        
        # Compare imports
        ctx_imports = set(ctx_components.get('imports', []) + ctx_components.get('from_imports', []))
        answer_imports = set(answer_components.get('imports', []) + answer_components.get('from_imports', []))
        
        if answer_imports:
            import_weight = 1.5
            total_weight += import_weight
            import_similarity = len(ctx_imports.intersection(answer_imports)) / len(answer_imports)
            score += import_weight * import_similarity
        
        # Compare code length
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
        """Rerank contexts based on code structure similarity."""
        try:
            for ctx in contexts:
                structure_score = self.compute_structure_similarity(ctx['components'], answer_components)
                ctx['structure_score'] = structure_score
                ctx['final_score'] = ctx['score'] + self.weight * structure_score
            
            return sorted(contexts, key=lambda x: x['final_score'], reverse=True)
        except Exception as e:
            print(f"Error in structure reranker: {e}")
            return contexts