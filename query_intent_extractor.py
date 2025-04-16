import re
from typing import Dict, Any, List, Tuple
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter

class QueryIntentExtractor:
    """Extract structured query intent from natural language queries."""
    
    def __init__(self):
        """Initialize the query intent extractor."""
        self.lemmatizer = WordNetLemmatizer()
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
            self.advanced_nlp = True
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            self.advanced_nlp = False
    
    def extract_intent(self, query: str) -> Dict[str, Any]:
        """
        Extract intent from query string.
        
        Args:
            query: Raw query string
            
        Returns:
            Dictionary of extracted intent information
        """
        intent = {
            'function_name': self._extract_function_name(query),
            'parameters': self._extract_parameters(query),
            'return_value': self._extract_return_value(query),
            'has_docstring': self._has_docstring(query),
            'error_handling': self._needs_error_handling(query),
            'complexity': self._determine_complexity(query),
            'domain': self._determine_domain(query),
            'key_functions': self._extract_key_functions(query),
            'algorithm_description': self._extract_algorithm_steps(query),
            'potential_names': self._extract_potential_names(query),
        }
        
        return intent
    
    def extract_answer_components(self, answer: str) -> Dict[str, Any]:
        """
        Extract components from an answer string.
        
        Args:
            answer: Answer string containing code
            
        Returns:
            Dictionary of extracted components
        """
        components = {
            'function_name': self._extract_function_name(answer),
            'parameters': self._extract_parameters(answer),
            'return_value': self._extract_return_value(answer),
            'has_docstring': self._has_docstring(answer),
            'error_handling': self._needs_error_handling(answer),
            'complexity': self._determine_complexity(answer),
            'domain': self._determine_domain(answer),
            'key_functions': self._extract_key_functions(answer),
            'algorithm_description': self._extract_algorithm_steps(answer)
        }
        
        return components
    
    def _extract_function_name(self, query: str) -> str:
        """Extract the likely function name from the query."""
        # Check for explicit function definition
        func_def_match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)', query)
        if func_def_match:
            return func_def_match.group(1)
        
        # Check for RST format
        rst_match = re.search(r'`([a-zA-Z_][a-zA-Z0-9_]*)`', query)
        if rst_match:
            return rst_match.group(1)
            
        # Try verb + noun pairs
        verb_noun_pairs = self._extract_verb_noun_pairs(query)
        if verb_noun_pairs:
            top_pair = verb_noun_pairs[0]
            return f"{top_pair[0]}_{top_pair[1]}"
            
        return ""
    
    def _extract_verb_noun_pairs(self, query: str) -> List[Tuple[str, str]]:
        """Extract verb+noun pairs that could be function names."""
        if not self.advanced_nlp:
            return []
            
        try:
            tokens = word_tokenize(query.lower())
            pos_tags = nltk.pos_tag(tokens)
            
            pairs = []
            for i in range(len(pos_tags) - 1):
                if pos_tags[i][1].startswith('VB') and pos_tags[i+1][1].startswith('NN'):
                    verb = self.lemmatizer.lemmatize(pos_tags[i][0], 'v')
                    noun = self.lemmatizer.lemmatize(pos_tags[i+1][0], 'n')
                    pairs.append((verb, noun))
                    
            return pairs
        except:
            return []
    
    def _extract_parameters(self, query: str) -> List[str]:
        """Extract parameter names from the query."""
        # Check for RST-style parameters
        param_matches = re.findall(r':param\s+(\w+):', query)
        if param_matches:
            return param_matches
            
        # Check function definition
        def_match = re.search(r'def\s+\w+\s*\(([^)]*)\)', query)
        if def_match:
            params = []
            params_str = def_match.group(1)
            for param in params_str.split(','):
                param = param.strip()
                param_match = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)', param)
                if param_match:
                    params.append(param_match.group(1))
            return params
            
        # Check for mentions in text
        param_candidates = re.findall(r'(?:parameter|argument|param)\s+(?:named|called)?\s*[\'"]?([a-zA-Z_][a-zA-Z0-9_]*)[\'"]?', query, re.IGNORECASE)
        if param_candidates:
            return param_candidates
            
        return []
    
    def _extract_return_value(self, query: str) -> Dict[str, Any]:
        """Extract information about the return value."""
        return_info = {
            'has_return': False,
            'return_type': None,
            'description': ''
        }
        
        if re.search(r':return:|:rtype:|returns|returning', query, re.IGNORECASE):
            return_info['has_return'] = True
            
            rtype_match = re.search(r':rtype:\s*([a-zA-Z_][a-zA-Z0-9_]*)', query)
            if rtype_match:
                return_info['return_type'] = rtype_match.group(1)
                
            rdesc_match = re.search(r':return:\s*([^\n]+)', query)
            if rdesc_match:
                return_info['description'] = rdesc_match.group(1).strip()
                
        return return_info
    
    def _has_docstring(self, query: str) -> bool:
        """Determine if the function should have a docstring."""
        has_rst = any(pattern in query for pattern in [':param', ':type', ':return', ':rtype'])
        is_detailed = len(query.split()) > 30
        return has_rst or is_detailed
    
    def _needs_error_handling(self, query: str) -> bool:
        """Determine if the function needs error handling."""
        error_terms = ['error', 'exception', 'raise', 'try', 'except', 'handle', 'catch']
        return any(term in query.lower() for term in error_terms)
    
    def _determine_complexity(self, query: str) -> str:
        """Estimate the complexity of the required function."""
        factors = 0
        
        if len(self._extract_parameters(query)) >= 3:
            factors += 1
            
        if any(term in query.lower() for term in ['loop', 'iteration', 'recursion', 'recursive', 'nested']):
            factors += 1
            
        if any(term in query.lower() for term in ['algorithm', 'optimize', 'efficient', 'complexity']):
            factors += 1
            
        if any(term in query.lower() for term in ['dictionary', 'list', 'array', 'tree', 'graph', 'hash']):
            factors += 1
            
        if factors >= 3:
            return 'high'
        elif factors >= 1:
            return 'medium'
        else:
            return 'low'
    
    def _determine_domain(self, query: str) -> str:
        """Identify the domain or category of the function."""
        domains = {
            'file_io': ['file', 'directory', 'path', 'open', 'read', 'write', 'close'],
            'string_processing': ['string', 'text', 'parse', 'format', 'concatenate', 'split', 'join'],
            'math': ['calculate', 'compute', 'sum', 'average', 'median', 'normalize'],
            'web': ['http', 'request', 'response', 'url', 'api', 'json', 'endpoint'],
            'database': ['query', 'database', 'sql', 'table', 'row', 'column', 'record'],
            'error_handling': ['exception', 'error', 'handle', 'try', 'except', 'finally', 'raise'],
            'data_structures': ['list', 'dict', 'tuple', 'set', 'array', 'collection'],
            'algorithms': ['algorithm', 'sort', 'search', 'find', 'filter', 'map', 'reduce'],
        }
        
        domain_counts = {domain: 0 for domain in domains}
        query_lower = query.lower()
        
        for domain, terms in domains.items():
            for term in terms:
                if term in query_lower:
                    domain_counts[domain] += 1
        
        max_count = max(domain_counts.values()) if domain_counts else 0
        if max_count > 0:
            for domain, count in domain_counts.items():
                if count == max_count:
                    return domain
        
        return 'general'
    
    def _extract_key_functions(self, query: str) -> List[str]:
        """Extract potential key function calls from the query."""
        common_funcs = [
            'open', 'read', 'write', 'close', 'append', 'split', 'join', 'strip',
            'replace', 'format', 'parse', 'sort', 'filter', 'map', 'reduce',
            'sum', 'min', 'max', 'len', 'range', 'enumerate', 'zip', 'list', 'dict',
            'set', 'tuple', 'int', 'float', 'str', 'bool', 'print', 'input'
        ]
        
        query_words = set(re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', query.lower()))
        mentioned_funcs = [func for func in common_funcs if func in query_words]
        
        code_funcs = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', query)
        
        all_funcs = set(mentioned_funcs + code_funcs)
        return list(all_funcs)
    
    def _extract_algorithm_steps(self, query: str) -> List[str]:
        """Extract algorithm steps if the query describes an algorithm."""
        steps = re.findall(r'(?:\d+\.\s+|\*\s+)([^\n]+)', query)
        
        if not steps:
            sentences = re.split(r'(?<=[.!?])\s+', query)
            action_verbs = ['check', 'find', 'return', 'compute', 'calculate', 
                           'get', 'set', 'create', 'update', 'delete',
                           'convert', 'transform', 'parse', 'iterate']
            
            action_sentences = []
            for sentence in sentences:
                words = sentence.strip().split()
                if words and words[0].lower() in action_verbs:
                    action_sentences.append(sentence)
            
            steps = action_sentences
        
        return steps
    
    def _extract_potential_names(self, query: str) -> List[str]:
        """Extract potential function names beyond the primary guess."""
        candidates = []
        
        # Find camelCase or snake_case words
        potential_names = re.findall(r'\b([a-z][a-zA-Z0-9_]*(?:_[a-z][a-zA-Z0-9_]*)+)\b|\b([a-z][a-z]*[A-Z][a-zA-Z0-9]*)\b', query)
        for match in potential_names:
            name = match[0] if match[0] else match[1]
            if name:
                candidates.append(name)
        
        # Add verb-noun combinations
        verb_noun_pairs = self._extract_verb_noun_pairs(query)
        for verb, noun in verb_noun_pairs:
            candidates.append(f"{verb}_{noun}")
        
        # Add frequent words
        if self.advanced_nlp:
            words = [word.lower() for word in word_tokenize(query) 
                    if word.isalpha() and word.lower() not in self.stop_words]
            
            word_counts = Counter(words)
            frequent_words = [word for word, count in word_counts.most_common(3) if count > 1]
            
            if len(frequent_words) >= 2:
                candidates.append('_'.join(frequent_words[:2]))
        
        # Return unique candidates (max 5)
        unique_candidates = list(set(candidates))
        return unique_candidates[:5]