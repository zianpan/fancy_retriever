from typing import Dict, List, Any, Optional
from reranking.reranker import Reranker
from reranking.code_structure_reranker import CodeStructureReranker
from reranking.signature_reranker import SignatureReranker
from reranking.semantic_reranker import SemanticReranker
from function_relationship_analyzer import FunctionRelationshipAnalyzer
from ast_component_extractor import ASTComponentExtractor
from query_intent_extractor import QueryIntentExtractor

class EnhancedCombinedReranker(Reranker):
    """
    Enhanced reranker that combines multiple reranking strategies with dynamic weighting.
    """
    def __init__(self, rerankers: Optional[List[Reranker]] = None, weights: Optional[Dict[str, float]] = None):
        """Initialize the enhanced combined reranker."""
        super().__init__()
        
        # Default weights
        self.base_weights = {
            'CodeStructureReranker': 1.2,
            'SignatureReranker': 1.8,
            'SemanticReranker': 1.6
        }
        
        # Initialize rerankers
        self.rerankers = rerankers or [
            CodeStructureReranker(weight=self.base_weights['CodeStructureReranker']),
            SignatureReranker(weight=self.base_weights['SignatureReranker']),
            SemanticReranker(weight=self.base_weights['SemanticReranker'])
        ]
        
        # Apply custom weights if provided
        if weights:
            for reranker in self.rerankers:
                class_name = reranker.__class__.__name__
                if class_name in weights:
                    reranker.weight = weights[class_name]
        
        # Initialize extractors
        self.component_extractor = ASTComponentExtractor()
        self.intent_extractor = QueryIntentExtractor()
        
        # Track feature impacts
        self.feature_impacts = {
            'structure_score': [],
            'signature_score': [],
            'semantic_score': []
        }
    
    def rerank(self, query: str, contexts: List[Dict[str, Any]], query_intent: Dict[str, Any] = None,
               answer_components: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Rerank contexts using multiple strategies."""
        try:
            # Extract query intent if not provided
            if query_intent is None:
                query_intent = self.intent_extractor.extract_intent(query)
            
            # Preprocess contexts
            contexts = self._preprocess_contexts(contexts, query, query_intent)
            
            # Get answer components if not provided
            if not answer_components:
                if contexts and 'components' in contexts[0]:
                    answer_components = contexts[0]['components']
                else:
                    answer_components = {}
            
            # Adjust weights based on query
            self._adjust_weights_for_query(query, query_intent)
            
            # Apply each reranker
            for reranker in self.rerankers:
                contexts = reranker.rerank(query, contexts, query_intent, answer_components)
                
                # Store scores for analysis
                reranker_name = reranker.__class__.__name__
                score_key = None
                
                if reranker_name == 'CodeStructureReranker':
                    score_key = 'structure_score'
                elif reranker_name == 'SignatureReranker':
                    score_key = 'signature_score'
                elif reranker_name == 'SemanticReranker':
                    score_key = 'semantic_score'
                
                if score_key:
                    self.feature_impacts[score_key] = [ctx.get(score_key, 0) for ctx in contexts[:5]]
                    
                    # Store scores in context
                    for ctx in contexts:
                        if 'previous_scores' not in ctx:
                            ctx['previous_scores'] = {}
                        ctx['previous_scores'][score_key] = ctx.get(score_key, 0)
            
            # Boost main functions
            relationship_analyzer = FunctionRelationshipAnalyzer(contexts)
            contexts = relationship_analyzer.boost_main_functions(contexts)
            
            # Normalize scores
            contexts = self._normalize_scores(contexts)
            
            # Apply heuristics
            self._apply_code_quality_heuristics(contexts, query_intent)
            self._apply_query_specific_adjustments(contexts, query, query_intent)
            
            # Sort by final score
            return sorted(contexts, key=lambda x: x.get('final_score', 0.0), reverse=True)
            
        except Exception as e:
            print(f"Error in enhanced combined reranker: {e}")
            return contexts
    
    def _preprocess_contexts(self, contexts: List[Dict[str, Any]], query: str, query_intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Preprocess contexts to extract components and set initial scores."""
        for ctx in contexts:
            # Extract components if needed
            if 'components' not in ctx:
                ctx['components'] = self.component_extractor.extract_components(ctx.get('text', ''))
            
            # Set initial scores
            if 'score' not in ctx:
                ctx['score'] = 0.0
            ctx['final_score'] = ctx['score']
            
            # Check for function name match
            if query_intent.get('function_name'):
                ctx_func_name = ctx['components'].get('function_name', '').lower()
                query_func_name = query_intent['function_name'].lower()
                
                # Apply name match boost
                if ctx_func_name == query_func_name:
                    ctx['name_match_boost'] = 0.3
                    ctx['final_score'] += 0.3
                elif query_func_name in ctx_func_name or ctx_func_name in query_func_name:
                    ctx['name_match_boost'] = 0.15
                    ctx['final_score'] += 0.15
                elif query_intent.get('potential_names') and any(
                    name.lower() in ctx_func_name or ctx_func_name in name.lower()
                    for name in query_intent['potential_names']):
                    ctx['name_match_boost'] = 0.1
                    ctx['final_score'] += 0.1
                else:
                    ctx['name_match_boost'] = 0.0
        
        return contexts
    
    def _adjust_weights_for_query(self, query: str, query_intent: Dict[str, Any]) -> None:
        """Adjust reranker weights based on query type."""
        # Default adjustments
        adjustments = {
            'CodeStructureReranker': 0.0,
            'SignatureReranker': 0.0,
            'SemanticReranker': 0.0
        }
        
        # Adjust based on domain
        domain = query_intent.get('domain', 'general')
        
        if domain == 'file_io':
            adjustments['SignatureReranker'] += 0.2
            adjustments['SemanticReranker'] += 0.1
        elif domain in ['string_processing', 'data_structures']:
            adjustments['CodeStructureReranker'] += 0.2
        elif domain == 'error_handling':
            adjustments['CodeStructureReranker'] += 0.1
            adjustments['SignatureReranker'] += 0.1
        elif domain == 'algorithms':
            adjustments['CodeStructureReranker'] += 0.3
            adjustments['SemanticReranker'] += 0.1
        
        # Adjust based on complexity
        complexity = query_intent.get('complexity', 'medium')
        
        if complexity == 'high':
            adjustments['CodeStructureReranker'] += 0.2
            adjustments['SemanticReranker'] += 0.1
        elif complexity == 'low':
            adjustments['SignatureReranker'] += 0.2
        
        # Adjust based on query length
        query_words = len(query.split())
        if query_words > 50:
            adjustments['SemanticReranker'] += 0.2
        elif query_words < 20:
            adjustments['SignatureReranker'] += 0.1
        
        # Apply adjustments
        for reranker in self.rerankers:
            class_name = reranker.__class__.__name__
            if class_name in adjustments:
                reranker.weight = self.base_weights[class_name] + adjustments[class_name]
    
    def _normalize_scores(self, contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize scores to 0-1 range."""
        all_scores = [ctx['final_score'] for ctx in contexts]
        if not all_scores:
            return contexts
            
        max_score = max(all_scores)
        min_score = min(all_scores)
        
        if max_score > min_score:
            score_range = max_score - min_score
            for ctx in contexts:
                normalized_score = (ctx['final_score'] - min_score) / score_range
                ctx['normalized_score'] = normalized_score
                ctx['final_score'] = normalized_score
        
        return contexts
    
    def _apply_code_quality_heuristics(self, contexts: List[Dict[str, Any]], query_intent: Dict[str, Any]) -> None:
        """Apply heuristics to boost high-quality code."""
        for ctx in contexts:
            boost = 0.0
            components = ctx.get('components', {})
            
            # Boost for docstring
            if query_intent.get('has_docstring', False) and components.get('has_docstring', False):
                boost += 0.08
            
            # Boost for error handling
            if query_intent.get('error_handling', False) and components.get('error_handling', False):
                boost += 0.1
            
            # Boost for appropriate code length
            complexity = query_intent.get('complexity', 'medium')
            line_count = components.get('line_count', 0)
            
            if (complexity == 'low' and 3 <= line_count <= 15) or \
               (complexity == 'medium' and 10 <= line_count <= 30) or \
               (complexity == 'high' and line_count >= 20):
                boost += 0.05
            
            # Boost for matching parameter count
            if query_intent.get('parameters'):
                expected_param_count = len(query_intent['parameters'])
                actual_param_count = len(components.get('parameters', []))
                if expected_param_count == actual_param_count:
                    boost += 0.08
                elif abs(expected_param_count - actual_param_count) <= 1:
                    boost += 0.04
            
            # Boost for return value
            if query_intent.get('return_value', {}).get('has_return', False) and components.get('return_type') is not None:
                boost += 0.05
            
            # Apply boost
            ctx['quality_boost'] = boost
            ctx['final_score'] += boost
    
    def _apply_query_specific_adjustments(self, contexts: List[Dict[str, Any]], query: str, query_intent: Dict[str, Any]) -> None:
        """Apply adjustments based on query type."""
        query_lower = query.lower()
        
        # File handling queries
        if any(term in query_lower for term in ['file', 'directory', 'open', 'read', 'write']):
            for ctx in contexts:
                components = ctx.get('components', {})
                function_calls = components.get('function_calls', [])
                
                file_ops = ['open', 'read', 'write', 'close', 'makedirs', 'exists', 'isdir', 'isfile']
                file_op_count = sum(1 for op in file_ops if op in function_calls)
                
                if file_op_count > 0:
                    boost = 0.05 * min(file_op_count, 3)
                    ctx['file_op_boost'] = boost
                    ctx['final_score'] += boost
        
        # String processing queries
        if any(term in query_lower for term in ['string', 'text', 'parse', 'format', 'split', 'join']):
            for ctx in contexts:
                components = ctx.get('components', {})
                function_calls = components.get('function_calls', [])
                
                string_ops = ['split', 'join', 'strip', 'replace', 'format', 'lower', 'upper', 'find', 'index']
                string_op_count = sum(1 for op in string_ops if op in function_calls)
                
                if string_op_count > 0:
                    boost = 0.04 * min(string_op_count, 3)
                    ctx['string_op_boost'] = boost
                    ctx['final_score'] += boost
        
        # Helper vs. main function detection
        function_name = query_intent.get('function_name', '')
        if function_name:
            for ctx in contexts:
                ctx_function_name = ctx.get('components', {}).get('function_name', '')
                
                helper_terms = ['_helper', 'helper_', 'util', 'internal', 'impl']
                main_terms = ['main', 'public', 'api', 'interface']
                
                is_helper = any(term in ctx_function_name.lower() for term in helper_terms)
                is_main = any(term in ctx_function_name.lower() for term in main_terms)
                
                if is_helper and 'helper' in query_lower:
                    ctx['helper_boost'] = 0.08
                    ctx['final_score'] += 0.08
                elif is_main and not 'helper' in query_lower:
                    ctx['main_boost'] = 0.08
                    ctx['final_score'] += 0.08