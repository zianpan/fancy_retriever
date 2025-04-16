from typing import Dict, List, Any
from reranking.reranker import Reranker
import re

class SignatureReranker(Reranker):
    """
    Reranker that prioritizes code with matching function signatures.
    """
    def __init__(self, weight: float = 1.5):
        """Initialize the signature reranker."""
        super().__init__()
        self.weight = weight
    
    def compute_signature_similarity(self, ctx_components: Dict[str, Any], 
                                    query_intent: Dict[str, Any]) -> float:
        """
        Compute similarity between context function signature and query intent.
        
        Returns a score between 0 and 1.
        """
        score = 0.0
        total_weight = 0.0
        
        # Compare function names
        function_weight = 2.0
        ctx_function = ctx_components.get('function_name', '')
        
        if 'function_name' in query_intent and query_intent['function_name']:
            total_weight += function_weight
            # Exact match
            if ctx_function.lower() == query_intent['function_name'].lower():
                score += function_weight
            # Partial match
            elif query_intent['function_name'].lower() in ctx_function.lower():
                score += function_weight * 0.8
            # Token similarity
            else:
                query_tokens = set(re.findall(r'[a-z]+', query_intent['function_name'].lower()))
                ctx_tokens = set(re.findall(r'[a-z]+', ctx_function.lower()))
                if query_tokens and ctx_tokens:
                    token_similarity = len(query_tokens.intersection(ctx_tokens)) / len(query_tokens)
                    score += function_weight * token_similarity * 0.5
        
        elif 'potential_names' in query_intent and query_intent['potential_names']:
            total_weight += function_weight
            
            # Exact match with potential names
            if any(name.lower() == ctx_function.lower() for name in query_intent['potential_names']):
                score += function_weight
            # Partial match with potential names
            elif any(name.lower() in ctx_function.lower() for name in query_intent['potential_names']):
                score += function_weight * 0.7
            # Token similarity
            else:
                max_token_sim = 0.0
                for name in query_intent['potential_names']:
                    query_tokens = set(re.findall(r'[a-z]+', name.lower()))
                    ctx_tokens = set(re.findall(r'[a-z]+', ctx_function.lower()))
                    if query_tokens and ctx_tokens:
                        token_similarity = len(query_tokens.intersection(ctx_tokens)) / len(query_tokens)
                        max_token_sim = max(max_token_sim, token_similarity)
                score += function_weight * max_token_sim * 0.5
        
        # Compare parameters
        param_weight = 2.0
        if 'parameters' in query_intent and query_intent['parameters']:
            total_weight += param_weight
            
            ctx_params = ctx_components.get('parameters', [])
            query_params = query_intent['parameters']
            
            if query_params and ctx_params:
                # Exact parameter match
                if set(ctx_params) == set(query_params):
                    score += param_weight
                else:
                    # Parameter name similarity
                    ctx_param_names = set(ctx_params)
                    query_param_names = set(query_params)
                    
                    if query_param_names:
                        param_similarity = len(ctx_param_names.intersection(query_param_names)) / len(query_param_names)
                        position_matches = sum(1 for i, param in enumerate(query_params) 
                                             if i < len(ctx_params) and param == ctx_params[i])
                        position_similarity = position_matches / len(query_params) if query_params else 0
                        
                        combined_param_similarity = 0.7 * param_similarity + 0.3 * position_similarity
                        score += param_weight * combined_param_similarity
        
        # Check for docstring
        if 'has_docstring' in query_intent and query_intent['has_docstring']:
            docstring_weight = 1.0
            total_weight += docstring_weight
            
            ctx_text = ctx_components.get('text', '')
            has_docstring = ('"""' in ctx_text and '"""' in ctx_text[ctx_text.find('"""')+3:]) or \
                           ("'''" in ctx_text and "'''" in ctx_text[ctx_text.find("'''")+3:])
            
            score += docstring_weight if has_docstring else 0
        
        # Check for return value
        if 'has_return' in query_intent and query_intent['has_return']:
            return_weight = 1.0
            total_weight += return_weight
            score += return_weight if re.search(r'\breturn\b', ctx_components.get('text', '')) else 0
        
        # Check for error handling
        if 'error_handling' in query_intent and query_intent['error_handling']:
            error_weight = 1.2
            total_weight += error_weight
            has_error_handling = ctx_components.get('try_count', 0) > 0 and ctx_components.get('except_count', 0) > 0
            score += error_weight if has_error_handling else 0
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def rerank(self, query: str, contexts: List[Dict[str, Any]], query_intent: Dict[str, Any],
               answer_components: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rerank contexts based on function signature similarity."""
        try:
            for ctx in contexts:
                if 'components' not in ctx:
                    ctx['components'] = self._extract_components(ctx.get('text', ''))
                
                signature_score = self.compute_signature_similarity(ctx['components'], query_intent)
                ctx['signature_score'] = signature_score
                
                if 'final_score' in ctx:
                    ctx['final_score'] += self.weight * signature_score
                else:
                    ctx['final_score'] = ctx['score'] + self.weight * signature_score
            
            return sorted(contexts, key=lambda x: x['final_score'], reverse=True)
        except Exception as e:
            print(f"Error in signature reranker: {e}")
            return contexts
    
    def _extract_components(self, code_text: str) -> Dict[str, Any]:
        """Extract basic components from code text."""
        components = {
            'function_name': '',
            'parameters': [],
            'return_type': [],
            'try_count': 0,
            'except_count': 0,
            'text': code_text
        }
        
        # Extract function name and parameters
        func_match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)', code_text)
        
        if func_match:
            components['function_name'] = func_match.group(1)
            
            # Extract parameters
            params_str = func_match.group(2).strip()
            if params_str:
                param_list = []
                for param in params_str.split(','):
                    param = param.strip()
                    param_name_match = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)', param)
                    if param_name_match:
                        param_list.append(param_name_match.group(1))
                
                components['parameters'] = param_list
        
        # Count try/except blocks
        components['try_count'] = len(re.findall(r'\btry\s*:', code_text))
        components['except_count'] = len(re.findall(r'\bexcept\b', code_text))
        
        # Check for return statements
        return_matches = re.findall(r'\breturn\b\s+([^:;]+)', code_text)
        if return_matches:
            components['return_type'] = ['unknown_type'] * len(return_matches)
        
        return components