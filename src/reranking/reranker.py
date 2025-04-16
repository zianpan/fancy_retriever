from typing import Dict, List, Any
import re

class Reranker:
    """
    Base class for rerankers that improve retrieval ranking.
    """
    def __init__(self, weight: float = 1.0):
        """Initialize the base reranker."""
        self.weight = weight
    
    def rerank(self, query: str, contexts: List[Dict[str, Any]], query_intent: Dict[str, Any] = None, 
               answer_components: Dict[str, Any] = None) -> List[Dict[str, Any]]:
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
        try:
            # Extract important keywords from the query
            keywords = self._extract_keywords(query)
            
            for ctx in contexts:
                # Ensure score is a float
                if 'score' in ctx and isinstance(ctx['score'], str):
                    try:
                        ctx['score'] = float(ctx['score'])
                    except (ValueError, TypeError):
                        ctx['score'] = 0.0
                
                # Calculate basic scores
                ctx['final_score'] = float(ctx.get('score', 0.0))
                lexical_score = self._calculate_lexical_score(keywords, ctx.get('text', ''))
                quality_score = self._calculate_quality_score(ctx)
                length_score = self._calculate_length_score(ctx)
                
                # Combine scores with simple weighting
                rerank_score = (lexical_score * 0.5 + quality_score * 0.3 + length_score * 0.2)
                ctx['rerank_score'] = rerank_score
                ctx['final_score'] += self.weight * rerank_score
            
            # Sort by final score
            return sorted(contexts, key=lambda x: float(x.get('final_score', 0.0)), reverse=True)
        
        except Exception as e:
            print(f"Error in base reranker: {e}")
            # Fallback to original ranking
            for ctx in contexts:
                if 'score' in ctx and isinstance(ctx['score'], str):
                    ctx['score'] = float(ctx['score']) if ctx['score'].replace('.', '', 1).isdigit() else 0.0
            return sorted(contexts, key=lambda x: float(x.get('score', 0.0)), reverse=True)
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from the query."""
        query = query.lower()
        stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'for', 'to', 'in', 'on', 'by', 'is', 'are'}
        
        # Extract parameters and function names
        params = re.findall(r':param\s+(\w+):', query)
        func_names = re.findall(r'def\s+(\w+)', query)
        
        # Extract other terms
        words = re.findall(r'\b([a-z][a-z_]+)\b', query)
        keywords = [w for w in words if w not in stopwords]
        
        # Return unique keywords
        return list(set(keywords + params + func_names))
    
    def _calculate_lexical_score(self, keywords: List[str], text: str) -> float:
        """Calculate lexical match score between keywords and text."""
        if not keywords or not text:
            return 0.0
        
        text = text.lower()
        matched = sum(1 for keyword in keywords if keyword.lower() in text)
        return matched / len(keywords) if keywords else 0.0
    
    def _calculate_quality_score(self, ctx: Dict[str, Any]) -> float:
        """Calculate code quality score based on simple heuristics."""
        text = ctx.get('text', '')
        components = ctx.get('components', {})
        
        # Check for basic quality indicators
        indicators = {
            'has_docstring': '"""' in text or "'''" in text,
            'has_comments': '#' in text,
            'has_function': 'def ' in text,
            'has_error_handling': 'try' in text and 'except' in text,
            'has_imports': 'import ' in text or 'from ' in text,
        }
        
        return sum(1 for present in indicators.values() if present) / len(indicators)
    
    def _calculate_length_score(self, ctx: Dict[str, Any]) -> float:
        """Calculate score based on code length appropriateness."""
        text = ctx.get('text', '')
        line_count = len(text.split('\n'))
        
        # Simple scoring based on line count
        if line_count < 3:
            return 0.3  # Too short
        elif line_count > 100:
            return 0.5  # Too long
        elif 5 <= line_count <= 50:
            return 1.0  # Ideal range
        else:
            # Scale linearly between ranges
            return 0.7  # Acceptable length