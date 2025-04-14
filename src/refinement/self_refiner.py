# TODO!!!!!!


from typing import Dict, List, Any, Optional
import re
import ast
from refinement.code_refiner import CodeRefiner

class SelfRefiner(CodeRefiner):
    """
    A self-refining code generator that combines and improves code from multiple sources.
    """
    def __init__(self, max_contexts: int = 3):
        """
        Initialize the self refiner.
        
        Args:
            max_contexts: Maximum number of contexts to consider
        """
        super().__init__()
        self.max_contexts = max_contexts
    
    def extract_docstring_from_query(self, query: str) -> str:
        """
        Extract and format a docstring from the query.
        
        Args:
            query: Query string
            
        Returns:
            Formatted docstring
        """
        docstring_parts = []
        
        # Extract function description (first paragraph)
        description_match = re.search(r'^(.*?)(?=:type|:param|:return|:rtype|$)', query, re.DOTALL)
        if description_match:
            description = description_match.group(1).strip()
            if description:
                docstring_parts.append(description)
        
        # Extract parameter descriptions
        param_matches = re.findall(r':param\s+(\w+):\s*(.*?)(?=:param|:type|:return|:rtype|$)', query, re.DOTALL)
        for param, desc in param_matches:
            docstring_parts.append(f":param {param}: {desc.strip()}")
        
        # Extract parameter types
        type_matches = re.findall(r':type\s+(\w+):\s*(.*?)(?=:param|:type|:return|:rtype|$)', query, re.DOTALL)
        for param, type_desc in type_matches:
            docstring_parts.append(f":type {param}: {type_desc.strip()}")
        
        # Extract return description
        return_match = re.search(r':return:\s*(.*?)(?=:param|:type|:rtype|$)', query, re.DOTALL)
        if return_match:
            docstring_parts.append(f":return: {return_match.group(1).strip()}")
        
        # Extract return type
        rtype_match = re.search(r':rtype:\s*(.*?)(?=:param|:type|:return|$)', query, re.DOTALL)
        if rtype_match:
            docstring_parts.append(f":rtype: {rtype_match.group(1).strip()}")
        
        if not docstring_parts:
            return ""
        
        # Create the docstring
        docstring = '"""' + '\n'.join(docstring_parts) + '\n"""'
        return docstring
    
    def determine_function_signature(self, query: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Determine the function signature from query and contexts.
        
        Args:
            query: Query string
            contexts: List of context dictionaries (ranked)
            
        Returns:
            Dictionary with function signature components
        """
        signature = {}
        
        # Extract function name from query or top contexts
        function_name = None
        
        # Look for function name in query
        fn_match = re.search(r'def\s+(\w+)', query)
        if fn_match:
            function_name = fn_match.group(1)
        
        # If not found, look for it in :param descriptions
        if not function_name:
            for ctx in contexts[:min(3, len(contexts))]:
                if ctx['components'].get('function_name'):
                    if function_name is None or ctx['final_score'] > signature.get('score', 0):
                        function_name = ctx['components']['function_name']
                        signature['score'] = ctx['final_score']
        
        # If still not found, look for potential names in query
        if not function_name:
            words = re.findall(r'\b([a-z][a-z_]+)\b', query.lower())
            potential_names = [w for w in words if w not in ['def', 'return', 'if', 'else', 'for', 'while', 'try', 'except']]
            if potential_names:
                function_name = potential_names[0]
        
        signature['function_name'] = function_name
        
        # Extract parameters from query
        parameters = []
        param_matches = re.findall(r':param\s+(\w+):', query)
        if param_matches:
            parameters = param_matches
        else:
            # If no parameters in query, use the top context's parameters
            for ctx in contexts[:min(3, len(contexts))]:
                if ctx['components'].get('parameters'):
                    if not parameters or ctx['final_score'] > signature.get('param_score', 0):
                        parameters = ctx['components']['parameters']
                        signature['param_score'] = ctx['final_score']
        
        signature['parameters'] = parameters
        
        return signature
    
    def combine_code_from_contexts(self, contexts: List[Dict[str, Any]], 
                                  signature: Dict[str, Any]) -> str:
        """
        Combine code snippets from top contexts.
        
        Args:
            contexts: List of context dictionaries (ranked)
            signature: Function signature information
            
        Returns:
            Combined code
        """
        if not contexts:
            return ""
        
        # Start with the top context as the base
        base_context = contexts[0]
        base_code = base_context['text']
        
        # Clean up code (remove escapes, normalize whitespace)
        base_code = base_code.replace('\\n', '\n').replace('\\"', '"').replace("\\'", "'")
        
        # If the base context's function name doesn't match our target,
        # replace it with the correct one
        function_name = signature.get('function_name')
        if function_name and function_name != base_context['components'].get('function_name', ''):
            base_lines = base_code.split('\n')
            for i, line in enumerate(base_lines):
                if line.strip().startswith('def ') and '(' in line:
                    current_name = line.strip().split('def ')[1].split('(')[0].strip()
                    base_lines[i] = line.replace(f"def {current_name}", f"def {function_name}")
                    break
            base_code = '\n'.join(base_lines)
        
        # Now we'll examine the other top contexts for error handling patterns
        if base_context['components'].get('try_count', 0) == 0:
            for ctx in contexts[1:min(3, len(contexts))]:
                if ctx['components'].get('try_count', 0) > 0 and ctx['components'].get('except_count', 0) > 0:
                    pass
        
        return base_code
    
    def fix_syntax_issues(self, code: str) -> str:
        """
        Fix common syntax issues in the code.
        
        Args:
            code: Code to fix
            
        Returns:
            Fixed code
        """
        try:
            # Try parsing with AST to check for syntax errors
            ast.parse(code)
            return code
        except SyntaxError as e:
            # Basic syntax error fixing
            # In a real system, this would be more sophisticated
            lines = code.split('\n')
            
            # Fix indentation issues
            if "indentation" in str(e) or "unindent" in str(e):
                fixed_lines = []
                indent_level = 0
                
                for line in lines:
                    stripped = line.strip()
                    
                    if not stripped:
                        fixed_lines.append('')
                        continue
                    
                    # Adjust indent level based on the line content
                    if stripped.startswith('def ') or stripped.startswith('class ') or stripped.endswith(':'):
                        fixed_lines.append('    ' * indent_level + stripped)
                        indent_level += 1
                    elif stripped in ['else:', 'elif:', 'except:', 'finally:']:
                        indent_level = max(0, indent_level - 1)
                        fixed_lines.append('    ' * indent_level + stripped)
                        indent_level += 1
                    else:
                        fixed_lines.append('    ' * indent_level + stripped)
                
                fixed_code = '\n'.join(fixed_lines)
                
                # Try parsing again
                try:
                    ast.parse(fixed_code)
                    return fixed_code
                except SyntaxError:
                    pass  # If still has errors, continue with other fixes
            
            # Fix missing parentheses
            if "unexpected EOF" in str(e) or "expected" in str(e):
                for char in ['(', '[', '{']:
                    open_count = code.count(char)
                    close_char = {'(': ')', '[': ']', '{': '}'}[char]
                    close_count = code.count(close_char)
                    
                    if open_count > close_count:
                        code += close_char * (open_count - close_count)
            
            # Try fixing again
            try:
                ast.parse(code)
                return code
            except SyntaxError:
                # If all fixes fail, return the original code
                return code
    
    def add_docstring(self, code: str, docstring: str) -> str:
        """
        Add a docstring to the code.
        
        Args:
            code: Code to add docstring to
            docstring: Docstring to add
            
        Returns:
            Code with docstring
        """
        if not docstring:
            return code
        
        # Check if the code already has a docstring
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if (node.body and isinstance(node.body[0], ast.Expr) and 
                        isinstance(node.body[0].value, ast.Constant) and
                        isinstance(node.body[0].value.value, str)):
                        # Already has a docstring
                        return code
        except:
            # If parsing fails, continue with docstring insertion attempt
            pass
        
        # Insert docstring after function definition
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith('def ') and line.strip().endswith(':'):
                # Find the indentation of the next line
                next_line_indent = ''
                if i+1 < len(lines):
                    next_line_indent = re.match(r'^(\s*)', lines[i+1]).group(1)
                else:
                    next_line_indent = '    '  # Default indentation
                
                # Indent docstring lines
                docstring_lines = [next_line_indent + line for line in docstring.split('\n')]
                
                # Insert docstring
                lines.insert(i+1, '\n'.join(docstring_lines))
                break
        
        return '\n'.join(lines)
    
    def refine(self, query: str, contexts: List[Dict[str, Any]], 
              query_intent: Dict[str, Any]) -> str:
        """
        Refine code based on the query, contexts, and query intent.
        
        Args:
            query: Query string
            contexts: List of context dictionaries (ranked)
            query_intent: Extracted intent from the query
            
        Returns:
            Refined code
        """
        try:
            # Limit to top contexts
            contexts = contexts[:self.max_contexts]
            
            # Extract docstring from query
            docstring = self.extract_docstring_from_query(query)
            
            # Determine function signature
            signature = self.determine_function_signature(query, contexts)
            
            # Combine code from contexts
            combined_code = self.combine_code_from_contexts(contexts, signature)
            
            # Fix any syntax issues
            fixed_code = self.fix_syntax_issues(combined_code)
            
            # Add docstring
            final_code = self.add_docstring(fixed_code, docstring)
            
            return final_code
        except Exception as e:
            print(f"Error during code refinement: {e}")
            
            # Fallback to the top context
            if contexts and len(contexts) > 0:
                return contexts[0]['text']
            
            # If all else fails, return an empty string
            return ""