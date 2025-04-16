import ast
import re
from typing import Dict, List, Set, Any
from collections import defaultdict

class FunctionRelationshipAnalyzer:
    """
    Analyze relationships between functions to identify main vs helper functions.
    """
    def __init__(self, contexts: List[Dict[str, Any]]):
        """
        Initialize the function relationship analyzer.
        
        Args:
            contexts: List of context dictionaries containing code snippets
        """
        self.contexts = contexts
        self.call_graph = self._build_call_graph()
        self.function_signatures = self._collect_function_signatures()
        self.function_roles = self._determine_function_roles()
    
    def _build_call_graph(self) -> Dict[str, Set[str]]:
        """
        Build a directed graph of function calls.
        
        Returns:
            Dictionary mapping function names to sets of called function names
        """
        call_graph = defaultdict(set)
        
        for ctx in self.contexts:
            code_text = ctx.get('text', '')
            if not code_text:
                continue
                
            try:
                tree = ast.parse(code_text)
                function_defs = {}
                
                # Collect function definitions
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        function_defs[node.name] = node
                
                # Find function calls
                for func_name, func_node in function_defs.items():
                    for node in ast.walk(func_node):
                        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                            called_func = node.func.id
                            if called_func in function_defs:
                                call_graph[func_name].add(called_func)
            except SyntaxError:
                # Simple regex fallback
                func_defs = re.findall(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code_text)
                for func_name in func_defs:
                    for other_func in func_defs:
                        if other_func != func_name and re.search(fr'\b{re.escape(other_func)}\s*\(', code_text):
                            call_graph[func_name].add(other_func)
                
        return call_graph
    
    def _collect_function_signatures(self) -> Dict[str, Dict[str, Any]]:
        """
        Collect function signatures from all contexts.
        
        Returns:
            Dictionary mapping function names to signature information
        """
        signatures = {}
        
        for ctx in self.contexts:
            code_text = ctx.get('text', '')
            if not code_text:
                continue
                
            try:
                tree = ast.parse(code_text)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Extract basic signature info
                        params = [arg.arg for arg in node.args.args if hasattr(arg, 'arg')]
                        docstring = ast.get_docstring(node)
                        
                        # Check for return statement
                        has_return = any(
                            isinstance(n, ast.Return) and n.value is not None 
                            for n in ast.walk(node)
                        )
                        
                        signatures[node.name] = {
                            'parameters': params,
                            'has_docstring': docstring is not None,
                            'has_return': has_return,
                            'line_count': len(ast.unparse(node).split('\n')) if hasattr(ast, 'unparse') else 0
                        }
            except SyntaxError:
                # Simple regex fallback
                for match in re.finditer(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)\s*:', code_text):
                    func_name = match.group(1)
                    params = [p.strip().split(':')[0].split('=')[0].strip() 
                             for p in match.group(2).split(',') if p.strip()]
                    
                    # Simple checks
                    func_body = code_text[match.end():]
                    has_docstring = bool(re.match(r'\s*[\'"]', func_body))
                    has_return = 'return ' in func_body
                    
                    signatures[func_name] = {
                        'parameters': params,
                        'has_docstring': has_docstring,
                        'has_return': has_return,
                        'line_count': len(func_body.split('\n', 20)[:20])
                    }
                
        return signatures
    
    def _determine_function_roles(self) -> Dict[str, str]:
        """
        Determine the role of each function (main, helper, utility).
        
        Returns:
            Dictionary mapping function names to roles
        """
        roles = {}
        
        # Count incoming calls
        incoming_calls = defaultdict(int)
        for caller, callees in self.call_graph.items():
            for callee in callees:
                incoming_calls[callee] += 1
        
        # Count outgoing calls
        outgoing_calls = {func: len(callees) for func, callees in self.call_graph.items()}
        
        # Classify functions
        for func_name, sig_info in self.function_signatures.items():
            is_called = incoming_calls.get(func_name, 0) > 0
            calls_others = outgoing_calls.get(func_name, 0) > 0
            has_docstring = sig_info.get('has_docstring', False)
            param_count = len(sig_info.get('parameters', []))
            line_count = sig_info.get('line_count', 0)
            
            # Simple classification rules
            if is_called and not calls_others and line_count <= 10:
                roles[func_name] = 'helper'
            elif not is_called and calls_others and has_docstring:
                roles[func_name] = 'main'
            elif not is_called and not calls_others and line_count <= 5:
                roles[func_name] = 'utility'
            else:
                roles[func_name] = 'general'
        
        return roles
    
    def is_helper_function(self, function_name: str) -> bool:
        """Check if a function is a helper function."""
        return self.function_roles.get(function_name) == 'helper'
    
    def is_main_function(self, function_name: str) -> bool:
        """Check if a function is a main function."""
        return self.function_roles.get(function_name) == 'main'
    
    def boost_main_functions(self, contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Boost scores for main functions and reduce scores for helper functions.
        
        Args:
            contexts: List of context dictionaries to adjust
            
        Returns:
            Adjusted list of context dictionaries
        """
        for ctx in contexts:
            function_name = ctx.get('components', {}).get('function_name', '')
            
            if function_name:
                role = self.function_roles.get(function_name, 'general')
                
                if role == 'main':
                    ctx['final_score'] = ctx.get('final_score', 0.0) + 0.1
                elif role == 'helper':
                    ctx['final_score'] = ctx.get('final_score', 0.0) - 0.05
        
        return contexts