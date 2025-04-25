import re
import ast
import difflib
import Levenshtein
from typing import Dict, List, Any, Tuple, Optional

class CodeMetrics:
    """
    Metrics for evaluating code generation.
    """
    @staticmethod
    def exact_match(generated: str, reference: str) -> bool:
        """
        Check if generated code exactly matches reference.
        
        Args:
            generated: Generated code
            reference: Reference code
            
        Returns:
            True if exact match, False otherwise
        """
        return generated.strip() == reference.strip()
    
    @staticmethod
    def normalized_edit_distance(generated: str, reference: str) -> float:
        """
        Compute normalized edit distance between generated and reference code.
        
        Args:
            generated: Generated code
            reference: Reference code
            
        Returns:
            Normalized edit distance (0-1, lower is better)
        """
        distance = Levenshtein.distance(generated.strip(), reference.strip())
        max_len = max(len(generated.strip()), len(reference.strip()))
        return 1 - (distance / max_len if max_len > 0 else 0)
    
    @staticmethod
    def _normalize_code(code: str) -> str:
        """
        Normalize code by removing comments, docstrings, and whitespace.
        
        Args:
            code: Code to normalize
            
        Returns:
            Normalized code
        """
        # Remove docstrings
        code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
        code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
        
        # Remove comments
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        
        # Normalize whitespace
        code = re.sub(r'\s+', ' ', code).strip()
        
        return code
    
    @staticmethod
    def bleu_score(generated: str, reference: str, n: int = 4) -> float:
        """
        Compute a simplified BLEU score for code.
        
        Args:
            generated: Generated code
            reference: Reference code
            n: Maximum n-gram size
            
        Returns:
            BLEU score (0-1, higher is better)
        """
        # Tokenize by splitting on whitespace and punctuation
        def tokenize(code):
            code = CodeMetrics._normalize_code(code)
            tokens = re.findall(r'\b\w+\b|[^\w\s]', code)
            return tokens
        
        gen_tokens = tokenize(generated)
        ref_tokens = tokenize(reference)
        
        if not gen_tokens or not ref_tokens:
            return 0.0
        
        # Calculate n-gram precision
        precisions = []
        
        for i in range(1, min(n+1, len(gen_tokens)+1)):
            gen_ngrams = [tuple(gen_tokens[j:j+i]) for j in range(len(gen_tokens)-i+1)]
            ref_ngrams = [tuple(ref_tokens[j:j+i]) for j in range(len(ref_tokens)-i+1)]
            
            # Count matching n-grams
            matches = sum(1 for ngram in gen_ngrams if ngram in ref_ngrams)
            
            # Calculate precision
            precision = matches / len(gen_ngrams) if gen_ngrams else 0
            precisions.append(precision)
        
        # Calculate geometric mean of precisions
        if not precisions or 0 in precisions:
            return 0.0
        
        import math
        score = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
        
        # Apply brevity penalty
        bp = 1.0
        if len(gen_tokens) < len(ref_tokens):
            bp = math.exp(1 - len(ref_tokens) / len(gen_tokens))
        
        return bp * score
    
    @staticmethod
    def ast_match(generated: str, reference: str) -> float:
        """
        Compare AST structures of generated and reference code.
        
        Args:
            generated: Generated code
            reference: Reference code
            
        Returns:
            AST similarity score (0-1, higher is better)
        """
        try:
            gen_ast = ast.parse(generated)
            ref_ast = ast.parse(reference)
        except SyntaxError:
            return 0.0
        
        # Count node types in each AST
        def count_node_types(tree):
            counter = {}
            for node in ast.walk(tree):
                node_type = type(node).__name__
                counter[node_type] = counter.get(node_type, 0) + 1
            return counter
        
        gen_counts = count_node_types(gen_ast)
        ref_counts = count_node_types(ref_ast)
        
        # Calculate similarity based on node type frequencies
        all_types = set(gen_counts.keys()).union(ref_counts.keys())
        similarity = 0.0
        
        for node_type in all_types:
            gen_count = gen_counts.get(node_type, 0)
            ref_count = ref_counts.get(node_type, 0)
            
            # Similarity for this node type
            type_similarity = min(gen_count, ref_count) / max(gen_count, ref_count) if max(gen_count, ref_count) > 0 else 1.0
            similarity += type_similarity
        
        return similarity / len(all_types) if all_types else 0.0
    
    @staticmethod
    def functional_similarity(generated: str, reference: str) -> float:
        """
        Evaluate functional similarity by checking function signatures and structure.
        
        Args:
            generated: Generated code
            reference: Reference code
            
        Returns:
            Functional similarity score (0-1, higher is better)
        """
        try:
            gen_ast = ast.parse(generated)
            ref_ast = ast.parse(reference)
            
            # Extract function definitions
            gen_funcs = [node for node in ast.walk(gen_ast) if isinstance(node, ast.FunctionDef)]
            ref_funcs = [node for node in ast.walk(ref_ast) if isinstance(node, ast.FunctionDef)]
            
            if not gen_funcs or not ref_funcs:
                return 0.0
            
            # Compare function names
            gen_func_name = gen_funcs[0].name
            ref_func_name = ref_funcs[0].name
            name_match = gen_func_name == ref_func_name
            
            # Compare parameters
            gen_params = [arg.arg for arg in gen_funcs[0].args.args]
            ref_params = [arg.arg for arg in ref_funcs[0].args.args]
            
            param_overlap = len(set(gen_params).intersection(set(ref_params)))
            param_max = max(len(gen_params), len(ref_params))
            param_score = param_overlap / param_max if param_max > 0 else 1.0
            
            # Calculate weighted score
            return 0.3 * name_match + 0.7 * param_score
        
        except SyntaxError:
            return 0.0