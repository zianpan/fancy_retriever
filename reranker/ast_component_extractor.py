import ast
import re
from typing import Dict, Any, List, Optional

class ASTComponentExtractor:
    """Extract code components using Python's Abstract Syntax Tree module."""
    
    def extract_components(self, code_text: str) -> Dict[str, Any]:
        """
        Extract components from code text using AST parsing.
        
        Args:
            code_text: The code text to analyze
            
        Returns:
            Dictionary of extracted components
        """
        # Default structure for components
        components = {
            'function_name': '',
            'parameters': [],
            'parameter_types': {},
            'return_type': None,
            'has_docstring': False,
            'docstring': '',
            'imports': [],
            'from_imports': [],
            'if_count': 0,
            'else_count': 0,
            'for_count': 0,
            'while_count': 0,
            'try_count': 0,
            'except_count': 0,
            'function_calls': [],
            'functions_defined': [],
            'classes_defined': [],
            'line_count': len(code_text.split('\n')),
            'complexity': 0,
            'error_handling': False,
            'text': code_text,
        }
        
        try:
            # Parse code into AST
            tree = ast.parse(code_text)
            
            # Extract information from AST
            self._extract_from_ast(tree, components)
            
            # Compute a rough complexity metric
            self._compute_complexity(components)
            
            return components
        
        except SyntaxError:
            # If AST parsing fails, fall back to regex-based extraction
            return self._fallback_extraction(code_text)
    
    def _extract_from_ast(self, tree: ast.AST, components: Dict[str, Any]) -> None:
        """Extract components by walking the AST."""
        function_defs = []
        class_defs = []
        
        # Collect top-level function and class definitions
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef):
                function_defs.append(node)
            elif isinstance(node, ast.ClassDef):
                class_defs.append(node)
        
        # Extract main function details if present
        if function_defs:
            main_func = function_defs[0]  # Assume first function is the main one
            components['function_name'] = main_func.name
            components['functions_defined'].append(main_func.name)
            
            # Extract parameters
            for arg in main_func.args.args:
                if hasattr(arg, 'arg'):
                    components['parameters'].append(arg.arg)
                    
                    # Check for type annotations
                    if hasattr(arg, 'annotation') and arg.annotation:
                        type_name = self._get_type_name(arg.annotation)
                        if type_name:
                            components['parameter_types'][arg.arg] = type_name
            
            # Check for return type annotation
            if hasattr(main_func, 'returns') and main_func.returns:
                components['return_type'] = self._get_type_name(main_func.returns)
            
            # Extract docstring
            docstring = ast.get_docstring(main_func)
            if docstring:
                components['has_docstring'] = True
                components['docstring'] = docstring
            
            # Analyze function body
            self._analyze_function_body(main_func, components)
        
        # Record all defined classes
        for class_def in class_defs:
            components['classes_defined'].append(class_def.name)
        
        # Extract imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    components['imports'].append(name.name)
            
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    components['from_imports'].append(node.module)
    
    def _get_type_name(self, annotation: ast.AST) -> Optional[str]:
        """Extract type name from annotation node."""
        if isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Attribute):
            return f"{self._get_type_name(annotation.value)}.{annotation.attr}"
        elif isinstance(annotation, ast.Subscript):
            value_name = self._get_type_name(annotation.value)
            return f"{value_name}[...]"
        return None
    
    def _analyze_function_body(self, func_node: ast.FunctionDef, components: Dict[str, Any]) -> None:
        """Count control structures and extract function calls."""
        for node in ast.walk(func_node):
            if isinstance(node, ast.If):
                components['if_count'] += 1
            elif isinstance(node, ast.For):
                components['for_count'] += 1
            elif isinstance(node, ast.While):
                components['while_count'] += 1
            elif isinstance(node, ast.Try):
                components['try_count'] += 1
                components['error_handling'] = True
            elif isinstance(node, ast.ExceptHandler):
                components['except_count'] += 1
                components['error_handling'] = True
            
            # Extract function calls
            elif isinstance(node, ast.Call):
                func_name = None
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                
                if func_name and func_name not in components['function_calls']:
                    if func_name not in components['functions_defined']:
                        components['function_calls'].append(func_name)
    
    def _compute_complexity(self, components: Dict[str, Any]) -> None:
        """Compute a simple complexity metric for the code."""
        complexity = 0
        
        # Control structures
        complexity += components['if_count'] + components['for_count'] * 2 + components['while_count'] * 2
        complexity += components['try_count'] + components['except_count'] * 0.5
        
        # Function calls and parameters
        complexity += len(components['function_calls']) * 0.5 + len(components['parameters']) * 0.5
        
        # Line count
        complexity += components['line_count'] * 0.1
        
        components['complexity'] = round(complexity, 1)
    
    def _fallback_extraction(self, code_text: str) -> Dict[str, Any]:
        """Extract components using regex when AST parsing fails."""
        components = {
            'function_name': '',
            'parameters': [],
            'parameter_types': {},
            'return_type': None,
            'has_docstring': False,
            'docstring': '',
            'imports': [],
            'from_imports': [],
            'if_count': 0,
            'else_count': 0,
            'for_count': 0,
            'while_count': 0,
            'try_count': 0,
            'except_count': 0,
            'function_calls': [],
            'functions_defined': [],
            'classes_defined': [],
            'line_count': len(code_text.split('\n')),
            'complexity': 0,
            'error_handling': False,
            'text': code_text,
        }
        
        # Extract function name and parameters
        func_match = re.search(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)', code_text)
        if func_match:
            components['function_name'] = func_match.group(1)
            components['functions_defined'].append(func_match.group(1))
            
            # Extract parameters
            params_str = func_match.group(2).strip()
            if params_str:
                for param in params_str.split(','):
                    param = param.strip()
                    param_match = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)', param)
                    if param_match:
                        components['parameters'].append(param_match.group(1))
        
        # Check for docstring
        docstring_match = re.search(r'"""(.*?)"""', code_text, re.DOTALL)
        if docstring_match:
            components['has_docstring'] = True
            components['docstring'] = docstring_match.group(1).strip()
        else:
            docstring_match = re.search(r"'''(.*?)'''", code_text, re.DOTALL)
            if docstring_match:
                components['has_docstring'] = True
                components['docstring'] = docstring_match.group(1).strip()
        
        # Extract imports and count control structures
        components['imports'] = re.findall(r'import\s+([a-zA-Z0-9_.]+)', code_text)
        components['from_imports'] = re.findall(r'from\s+([a-zA-Z0-9_.]+)\s+import', code_text)
        components['if_count'] = len(re.findall(r'\bif\b\s+', code_text))
        components['else_count'] = len(re.findall(r'\belse\b\s*:', code_text))
        components['for_count'] = len(re.findall(r'\bfor\b\s+', code_text))
        components['while_count'] = len(re.findall(r'\bwhile\b\s+', code_text))
        components['try_count'] = len(re.findall(r'\btry\s*:', code_text))
        components['except_count'] = len(re.findall(r'\bexcept\b', code_text))
        
        # Check for error handling
        components['error_handling'] = components['try_count'] > 0 and components['except_count'] > 0
        
        # Extract function calls
        all_calls = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code_text)
        components['function_calls'] = [
            call for call in all_calls 
            if call != components['function_name'] and 
            call not in ['if', 'for', 'while', 'with']
        ]
        
        # Extract class definitions
        components['classes_defined'] = re.findall(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)', code_text)
        
        # Compute complexity
        self._compute_complexity(components)
        
        return components