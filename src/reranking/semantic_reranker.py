from typing import Dict, List, Any
import re
from reranking.reranker import Reranker

class SemanticReranker(Reranker):
    """Reranker that uses semantic understanding to match queries with code snippets."""
    
    def __init__(self, weight: float = 1.6):
        """Initialize the semantic reranker with given weight."""
        super().__init__()
        self.weight = weight
        
    def compute_semantic_similarity(self, query: str, ctx_text: str, 
                                   query_intent: Dict[str, Any]) -> float:
        """Compute semantic similarity between query and code context."""
        score = 0.0
        total_weight = 0.0
        
        # Extract key phrases from the query
        key_phrases = self._extract_key_phrases(query)
        if key_phrases:
            phrase_weight = 1.5
            total_weight += phrase_weight
            matches = sum(1 for phrase in key_phrases if phrase.lower() in ctx_text.lower())
            score += phrase_weight * (matches / len(key_phrases))
        
        # Check for domain-specific terminology matches
        domain_terms = self._extract_domain_terms(query)
        if domain_terms:
            domain_weight = 1.2
            total_weight += domain_weight
            term_matches = sum(1 for term in domain_terms if term.lower() in ctx_text.lower())
            score += domain_weight * (term_matches / len(domain_terms))
        
        # Check for algorithm description matches
        if 'algorithm_description' in query_intent and query_intent['algorithm_description']:
            algo_weight = 1.8
            total_weight += algo_weight
            algo_steps = query_intent['algorithm_description']
            step_matches = 0
            for step in algo_steps:
                step_terms = re.findall(r'\b[a-z][a-z_]+\b', step.lower())
                if any(term in ctx_text.lower() for term in step_terms):
                    step_matches += 1
            score += algo_weight * (step_matches / len(algo_steps))
                
        # Check for key function calls
        if 'key_functions' in query_intent and query_intent['key_functions']:
            func_weight = 1.4
            total_weight += func_weight
            func_calls = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', ctx_text)
            func_calls = [f.lower() for f in func_calls]
            key_funcs = [f.lower() for f in query_intent['key_functions']]
            func_matches = sum(1 for func in key_funcs if func in func_calls)
            score += func_weight * (func_matches / len(key_funcs))
        
        # Docstring quality assessment
        docstring_match = self._assess_docstring_match(ctx_text, query)
        if docstring_match > 0:
            doc_weight = 1.0
            total_weight += doc_weight
            score += doc_weight * docstring_match
            
        return score / total_weight if total_weight > 0 else 0.0
    
    def _extract_key_phrases(self, query: str) -> List[str]:
        """Extract important phrases from the query."""
        phrases = []
        
        # Look for verb phrases (often describe actions)
        verb_phrases = re.findall(r'([a-z]+(?:\s+[a-z]+){1,3}(?:\s+the\s+[a-z]+)?)', query.lower())
        phrases.extend([p for p in verb_phrases if len(p.split()) >= 2])
        
        # Look for phrases in RST-style parameter descriptions
        param_desc = re.findall(r':param\s+\w+:\s+([^:\n]+)', query)
        phrases.extend(param_desc)
        
        # Look for phrases in RST-style return descriptions
        return_desc = re.findall(r':return:\s+([^:\n]+)', query)
        phrases.extend(return_desc)
        
        return phrases
    
    def _extract_domain_terms(self, query: str) -> List[str]:
        """Extract domain-specific terminology from the query."""
        # Common programming and computing terms
        common_domains = {
            'file': ['file', 'directory', 'path', 'open', 'read', 'write', 'close'],
            'string': ['string', 'parse', 'format', 'concatenate', 'split', 'join'],
            'math': ['calculate', 'compute', 'sum', 'average', 'median', 'normalize'],
            'web': ['http', 'request', 'response', 'url', 'api', 'json', 'endpoint'],
            'database': ['query', 'database', 'sql', 'table', 'row', 'column', 'record'],
            'error': ['exception', 'error', 'handle', 'try', 'except', 'finally', 'raise'],
        }
        
        # Identify which domains appear in the query
        domains_found = []
        query_lower = query.lower()
        
        for domain, terms in common_domains.items():
            if any(term in query_lower for term in terms):
                domains_found.append(domain)
        
        # Extract all domain-specific terms that appear in the query
        domain_terms = []
        for domain in domains_found:
            for term in common_domains[domain]:
                if term in query_lower:
                    domain_terms.append(term)
        
        return domain_terms
    
    def _assess_docstring_match(self, ctx_text: str, query: str) -> float:
        """Assess how well a docstring matches the query."""
        # Extract docstring if present
        docstring_match = re.search(r'"""(.*?)"""', ctx_text, re.DOTALL)
        if not docstring_match:
            docstring_match = re.search(r"'''(.*?)'''", ctx_text, re.DOTALL)
            
        if not docstring_match:
            return 0.0
            
        docstring = docstring_match.group(1)
        
        # Extract key terms from query and docstring
        query_terms = set(re.findall(r'\b[a-z][a-z_]{2,}\b', query.lower()))
        docstring_terms = set(re.findall(r'\b[a-z][a-z_]{2,}\b', docstring.lower()))
        
        # Calculate term overlap
        if not query_terms:
            return 0.0
            
        overlap = len(query_terms.intersection(docstring_terms)) / len(query_terms)
        
        # Check for parameter documentation
        query_params = re.findall(r':param\s+(\w+):', query)
        doc_params = re.findall(r':param\s+(\w+):', docstring)
        
        param_coverage = 0.0
        if query_params:
            param_coverage = len(set(query_params).intersection(set(doc_params))) / len(query_params)
            
        # Combined score with higher weight on parameter coverage
        return 0.4 * overlap + 0.6 * param_coverage
    
    def rerank(self, query: str, contexts: List[Dict[str, Any]], query_intent: Dict[str, Any],
               answer_components: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rerank contexts based on semantic similarity to the query."""
        try:
            for ctx in contexts:
                semantic_score = self.compute_semantic_similarity(
                    query, ctx.get('text', ''), query_intent)
                ctx['semantic_score'] = semantic_score
                
                # Update final score
                if 'final_score' in ctx:
                    ctx['final_score'] += self.weight * semantic_score
                else:
                    ctx['final_score'] = ctx.get('score', 0.0) + self.weight * semantic_score
            
            return sorted(contexts, key=lambda x: x.get('final_score', 0.0), reverse=True)
        except Exception as e:
            print(f"Error in semantic reranker: {e}")
            return contexts