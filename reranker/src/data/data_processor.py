import re
import ast
from typing import Dict, List, Any, Tuple, Optional

class DataProcessor:
    """
    Processes code data, extracts features, and prepares it for retrieval and reranking.
    """
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data
    
    def extract_function_signature(self, code: str) -> Tuple[str, List[str], List[str]]:
        """
        Extract function name, parameters, and return type from code.
        """
        try:
            # Try parsing with AST first
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    function_name = node.name
                    parameters = [arg.arg for arg in node.args.args]
                    
                    # Extract return type from docstring or annotations
                    return_type = []
                    if node.returns:
                        if hasattr(ast, 'unparse'):  # Python 3.9+
                            return_type.append(ast.unparse(node.returns))
                        else:
                            return_type.append(str(node.returns))
                    
                    # Look for return type in docstring
                    if hasattr(node, 'body') and len(node.body) > 0:
                        first_node = node.body[0]
                        if isinstance(first_node, ast.Expr) and isinstance(first_node.value, ast.Constant):
                            docstring = first_node.value.value
                            if docstring and ':return:' in docstring:
                                try:
                                    return_parts = docstring.split(':return:')[1].strip().split('\n')[0]
                                    return_type.append(return_parts)
                                except Exception:
                                    pass
                    
                    return function_name, parameters, return_type
        except Exception as e:
            pass
        
        # If AST parsing fails, use regex as fallback
        try:
            function_match = re.search(r'def\s+(\w+)\s*\((.*?)\)', code)
            if function_match:
                function_name = function_match.group(1)
                params_str = function_match.group(2)
                parameters = [p.strip().split(':')[0].strip() for p in params_str.split(',') if p.strip()]
                
                # Try to find return type in docstring
                return_match = re.search(r':return:(.*?)(?:$|\n)', code)
                return_type = [return_match.group(1).strip()] if return_match else []
                
                return function_name, parameters, return_type
        except Exception:
            # Regex fallback also failed
            pass
        
        return "", [], []
    
    def extract_code_components(self, code: str) -> Dict[str, Any]:
        """
        Extract various components and metrics from code.
        """
        components = {}
        
        # Extract function signature
        function_name, parameters, return_type = self.extract_function_signature(code)
        components['function_name'] = function_name
        components['parameters'] = parameters
        components['return_type'] = return_type
        
        # Handle bad/empty code gracefully
        if not code or not isinstance(code, str):
            return components
        
        # Count control flow structures - using raw strings for regex
        components['if_count'] = len(re.findall(r'\bif\b', code))
        components['else_count'] = len(re.findall(r'\belse\b', code))
        components['for_count'] = len(re.findall(r'\bfor\b', code))
        components['while_count'] = len(re.findall(r'\bwhile\b', code))
        
        # Count error handling
        components['try_count'] = len(re.findall(r'\btry\b', code))
        components['except_count'] = len(re.findall(r'\bexcept\b', code))
        
        # Identify imports - using raw strings for regex
        components['imports'] = re.findall(r'import\s+(\w+)', code)
        components['from_imports'] = re.findall(r'from\s+(\w+)\s+import', code)
        
        # Calculate code complexity metrics
        components['line_count'] = len(code.split('\n'))
        components['function_calls'] = len(re.findall(r'\w+\(', code))
        
        return components
    
    def extract_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Extract intent and requirements from the query.
        """
        intent = {}
        
        # Handle bad/empty queries gracefully
        if not query or not isinstance(query, str):
            return intent
        
        # Extract function name if present - using raw strings for regex
        function_match = re.search(r'def\s+(\w+)', query)
        if function_match:
            intent['function_name'] = function_match.group(1)
        else:
            # Look for potential function names in the query
            words = re.findall(r'\b([a-z][a-z_]+)\b', query.lower())
            candidates = [w for w in words if w not in ['def', 'return', 'if', 'else', 'for', 'while', 'try', 'except']]
            if candidates:
                intent['potential_names'] = candidates
        
        # Extract parameter information
        param_matches = re.findall(r':param\s+(\w+):', query)
        if param_matches:
            intent['parameters'] = param_matches
        
        # Detect if return value is expected
        if ':return:' in query or ':rtype:' in query:
            intent['has_return'] = True
        
        # Detect if error handling is mentioned
        if 'exception' in query.lower() or 'error' in query.lower() or 'except' in query.lower():
            intent['error_handling'] = True
        
        return intent
    
    def process_data(self) -> List[Dict[str, Any]]:
        """
        Process the data to extract additional features.
        
        Returns:
            List of dictionaries with processed data
        """
        processed_data = []
        
        for i, item in enumerate(self.data):
            try:
                # Skip items if they don't have the expected structure
                if not isinstance(item, dict) or 'question' not in item:
                    print(f"Skipping item {i} - missing required fields")
                    continue
                
                # Handle different answer field names
                answer = item.get('answers', item.get('answer', ''))
                
                processed_item = {
                    'question': item['question'],
                    'answer': answer,
                    'answer_components': self.extract_code_components(answer),
                    'query_intent': self.extract_query_intent(item['question']),
                    'contexts': []
                }
                
                # Process contexts if they exist
                if 'ctxs' in item and isinstance(item['ctxs'], list):
                    for ctx in item['ctxs']:
                        if not isinstance(ctx, dict) or 'text' not in ctx:
                            continue
                            
                        try:
                            # Prepare default values for missing fields
                            ctx_id = ctx.get('id', 'unknown')
                            ctx_score = float(ctx.get('score', 0.0))
                            ctx_has_answer = ctx.get('has_answer', False)
                            
                            # Handle score that might be a string
                            if isinstance(ctx_score, str):
                                try:
                                    ctx_score = float(ctx_score)
                                except:
                                    ctx_score = 0.0
                            
                            processed_ctx = {
                                'id': ctx_id,
                                'score': ctx_score,
                                'text': ctx['text'],
                                'has_answer': ctx_has_answer,
                                'components': self.extract_code_components(ctx['text'])
                            }
                            processed_item['contexts'].append(processed_ctx)
                        except Exception as e:
                            # Skip problematic contexts
                            print(f"Error processing context in item {i}: {str(e)}")
                            continue
                
                processed_data.append(processed_item)
                
            except Exception as e:
                # Skip problematic items
                print(f"Error processing item {i}: {str(e)}")
                continue
        
        print(f"Successfully processed {len(processed_data)} items")
        return processed_data
    
    def get_processed_item(self, index: int) -> Optional[Dict[str, Any]]:
        """
        Get a processed item by index.
        """
        processed_data = self.process_data()
        if 0 <= index < len(processed_data):
            return processed_data[index]
        return None