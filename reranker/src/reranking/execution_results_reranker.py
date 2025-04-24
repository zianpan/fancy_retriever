# from typing import Dict, List, Any
# # Removed import from reranking.reranker
# import tempfile
# import subprocess
# import os
# import json
# import ast

# class ExecutionResultsReranker:
#     """
#     Reranker that evaluates code snippets based on execution results.
#     """
#     def __init__(self, weight: float = 2.0, timeout: int = 5):
#         """
#         Initialize the execution results reranker.
        
#         Args:
#             weight: Weight to apply to the execution score
#             timeout: Maximum execution time in seconds
#         """
#         # No longer calls super().__init__()
#         self.weight = weight
#         self.timeout = timeout
        
#     def generate_test_inputs(self, function_name: str, parameters: List[str], 
#                            query_intent: Dict[str, Any]) -> List[Dict[str, Any]]:
#         """
#         Generate test inputs based on function signature and query intent.
        
#         Args:
#             function_name: Name of the function to test
#             parameters: List of parameter names
#             query_intent: Extracted intent from the query
            
#         Returns:
#             List of test input dictionaries
#         """
#         test_inputs = []
        
#         # Generate simple test cases based on parameter types
#         param_types = query_intent.get('parameter_types', {})
        
#         # Default test case with basic values
#         default_case = {}
#         for param in parameters:
#             param_type = param_types.get(param, 'str')
#             if 'str' in param_type or 'path' in param or 'file' in param:
#                 default_case[param] = "test_string"
#             elif 'int' in param_type or 'num' in param:
#                 default_case[param] = 0
#             elif 'list' in param_type or 'array' in param:
#                 default_case[param] = [1, 2, 3]
#             elif 'dict' in param_type or 'map' in param:
#                 default_case[param] = {"key": "value"}
#             elif 'bool' in param_type:
#                 default_case[param] = False
#             else:
#                 default_case[param] = None
                
#         test_inputs.append(default_case)
        
#         # Domain-specific test cases
#         domain = query_intent.get('domain', 'general')
        
#         if domain == 'file_io':
#             # Test with file paths
#             file_case = default_case.copy()
#             for param in parameters:
#                 if 'path' in param or 'file' in param:
#                     file_case[param] = "test_file.txt"
#             test_inputs.append(file_case)
            
#         elif domain == 'string_processing':
#             # Test with special string cases
#             string_cases = [
#                 {param: "" for param in parameters},  # Empty string
#                 {param: "Hello, World!" for param in parameters}  # Standard test string
#             ]
#             test_inputs.extend(string_cases)
            
#         # Add more domain-specific test cases as needed
        
#         return test_inputs
    
#     def execute_code_safely(self, code: str, function_name: str, test_inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
#         """
#         Execute code in a sandboxed environment and evaluate results.
        
#         Args:
#             code: Code snippet to execute
#             function_name: Name of the function to test
#             test_inputs: List of test input dictionaries
            
#         Returns:
#             Dictionary with execution results
#         """
#         results = {
#             'success': False,
#             'error': None,
#             'test_results': [],
#             'execution_score': 0.0
#         }
        
#         try:
#             # Create a temporary Python file
#             with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as temp_file:
#                 temp_file_path = temp_file.name
                
#                 # Write the code to the file
#                 temp_file.write(code)
                
#             # Create a test script
#             with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as test_file:
#                 test_file_path = test_file.name
                
#                 # Write test script that imports the function and runs tests
#                 test_script = f"""
#                 import sys
#                 import json
#                 import importlib.util
#                 import traceback

#                 # Load the module
#                 spec = importlib.util.spec_from_file_location('test_module', '{temp_file_path}')
#                 module = importlib.util.module_from_spec(spec)
#                 spec.loader.exec_module(module)

#                 # Test results
#                 results = []

#                 try:
#                     # Check if function exists
#                     if not hasattr(module, '{function_name}'):
#                         print(json.dumps({{'success': False, 'error': 'Function not found'}}))
#                         sys.exit(1)
                        
#                     # Run tests
#                     test_inputs = {json.dumps(test_inputs)}
                    
#                     for i, test_input in enumerate(test_inputs):
#                         try:
#                             # Call the function
#                             result = module.{function_name}(**test_input)
#                             results.append({{'test_id': i, 'success': True, 'result': result}})
#                         except Exception as e:
#                             # Function execution failed
#                             results.append({{'test_id': i, 'success': False, 'error': str(e), 'traceback': traceback.format_exc()}})
                    
#                     # Return results
#                     print(json.dumps({{'success': True, 'results': results}}))
                    
#                 except Exception as e:
#                     # Module loading failed
#                     print(json.dumps({{'success': False, 'error': str(e), 'traceback': traceback.format_exc()}}))
#                 """
#                 test_file.write(test_script)
                
#             # Execute the test script with a timeout
#             try:
#                 process = subprocess.run(
#                     ['python', test_file_path],
#                     capture_output=True,
#                     text=True,
#                     timeout=self.timeout
#                 )
                
#                 # Parse output
#                 if process.returncode == 0 and process.stdout:
#                     try:
#                         execution_data = json.loads(process.stdout.strip())
                        
#                         results['success'] = execution_data.get('success', False)
                        
#                         if 'results' in execution_data:
#                             results['test_results'] = execution_data['results']
                            
#                             # Calculate execution score based on test successes
#                             successful_tests = sum(1 for test in execution_data['results'] if test.get('success', False))
#                             total_tests = len(execution_data['results'])
                            
#                             if total_tests > 0:
#                                 results['execution_score'] = successful_tests / total_tests
                        
#                         if 'error' in execution_data:
#                             results['error'] = execution_data['error']
                            
#                     except json.JSONDecodeError:
#                         results['error'] = 'Failed to parse execution results'
#                 else:
#                     results['error'] = process.stderr.strip() if process.stderr else 'Unknown execution error'
                    
#             except subprocess.TimeoutExpired:
#                 results['error'] = f'Execution timed out after {self.timeout} seconds'
                
#         except Exception as e:
#             results['error'] = f'Error setting up execution environment: {str(e)}'
            
#         finally:
#             # Clean up temporary files
#             for file_path in [temp_file_path, test_file_path]:
#                 try:
#                     os.unlink(file_path)
#                 except:
#                     pass
                    
#         return results
    
#     def rerank(self, query: str, contexts: List[Dict[str, Any]], query_intent: Dict[str, Any],
#               answer_components: Dict[str, Any]) -> List[Dict[str, Any]]:
#         """
#         Rerank contexts based on code execution results.
        
#         Args:
#             query: Query string
#             contexts: List of context dictionaries
#             query_intent: Extracted intent from the query
#             answer_components: Expected answer components
            
#         Returns:
#             Reranked list of contexts
#         """
#         try:
#             # Only execute the top N contexts to save time
#             top_n = min(5, len(contexts))
            
#             for i, ctx in enumerate(contexts[:top_n]):
#                 # Skip if there's no code text
#                 if 'text' not in ctx:
#                     ctx['execution_score'] = 0.0
#                     continue
                    
#                 # Extract function name from context
#                 function_name = ctx.get('components', {}).get('function_name', '')
                
#                 if not function_name:
#                     # Try to extract function name using AST
#                     try:
#                         tree = ast.parse(ctx['text'])
#                         for node in ast.walk(tree):
#                             if isinstance(node, ast.FunctionDef):
#                                 function_name = node.name
#                                 break
#                     except:
#                         pass
                        
#                 if not function_name:
#                     # No function found, can't execute
#                     ctx['execution_score'] = 0.0
#                     continue
                    
#                 # Get parameters
#                 parameters = ctx.get('components', {}).get('parameters', [])
                
#                 # Generate test inputs
#                 test_inputs = self.generate_test_inputs(function_name, parameters, query_intent)
                
#                 # Execute code and get results
#                 execution_results = self.execute_code_safely(ctx['text'], function_name, test_inputs)
                
#                 # Store execution results and score
#                 ctx['execution_results'] = execution_results
#                 ctx['execution_score'] = execution_results.get('execution_score', 0.0)
                
#                 # Update final score
#                 ctx['final_score'] = ctx.get('final_score', 0.0) + self.weight * ctx['execution_score']
            
#             # For contexts we didn't execute, assign a default execution score
#             for ctx in contexts[top_n:]:
#                 ctx['execution_score'] = 0.0
                
#             # Sort by final score
#             return sorted(contexts, key=lambda x: x.get('final_score', 0.0), reverse=True)
            
#         except Exception as e:
#             print(f"Error in execution results reranker: {e}")
#             return contexts

from typing import Dict, List, Any
import tempfile
import subprocess
import os
import json
import ast
import re
import sys
import traceback

class ExecutionResultsReranker:
    """
    Reranker that evaluates code snippets based on execution results.
    """
    def __init__(self, weight: float = 2.0, timeout: int = 5):
        """
        Initialize the execution results reranker.
        
        Args:
            weight: Weight to apply to the execution score
            timeout: Maximum execution time in seconds
        """
        self.weight = weight
        self.timeout = timeout
    
    def normalize_code(self, code: str) -> str:
        """
        Normalize code formatting - fix whitespace, indentation, etc.
        """
        # Remove extra spaces around parentheses, commas, etc.
        code = re.sub(r'\s*\(\s*', '(', code)
        code = re.sub(r'\s*\)\s*', ')', code)
        code = re.sub(r'\s*,\s*', ', ', code)
        code = re.sub(r'\s*:\s*', ': ', code)
        code = re.sub(r'\s*=\s*', '=', code)
        
        # Handle single-line function definitions
        if code.startswith('def') and ':' in code and code.count('\n') == 0:
            # Split at the colon and properly indent the body
            parts = code.split(':', 1)
            if len(parts) == 2:
                header = parts[0] + ':'
                body = parts[1].strip()
                code = f"{header}\n    {body}"
        
        return code
    
    def extract_function_info(self, code: str) -> tuple:
        """
        Extract function name and parameters from code snippet.
        Returns: (function_name, parameters)
        """
        try:
            # Normalize the code first
            code = self.normalize_code(code)
            
            # Try parsing as AST
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    parameters = []
                    for arg in node.args.args:
                        param_name = arg.arg
                        # Check if there's a default value
                        default_value = None
                        if node.args.defaults:
                            idx = node.args.args.index(arg) - (len(node.args.args) - len(node.args.defaults))
                            if idx >= 0:
                                try:
                                    default_value = ast.literal_eval(node.args.defaults[idx])
                                except:
                                    default_value = None
                        parameters.append((param_name, default_value))
                    return node.name, parameters
        except:
            pass
        
        # Fallback to regex if AST parsing fails
        function_pattern = r'def\s+(\w+)\s*\((.*?)\):'
        match = re.search(function_pattern, code)
        
        if match:
            function_name = match.group(1)
            params_str = match.group(2)
            parameters = []
            for p in params_str.split(','):
                p = p.strip()
                if '=' in p:
                    name, default = p.split('=', 1)
                    try:
                        default_value = ast.literal_eval(default.strip())
                    except:
                        default_value = default.strip()
                    parameters.append((name.strip(), default_value))
                elif p:
                    parameters.append((p, None))
            return function_name, parameters
        
        return None, []
    
    def generate_test_inputs(self, function_name: str, parameters: List[tuple], 
                           question: str) -> List[Dict[str, Any]]:
        """
        Generate test inputs based on function signature and question context.
        """
        test_inputs = []
        
        # Create default test case
        default_case = {}
        for param_name, default_value in parameters:
            if default_value is not None:
                # Use the default value if available
                default_case[param_name] = default_value
            else:
                # Infer type from parameter name or question context
                if 'taxonomy' in question.lower() or 'phylogeny' in question.lower():
                    if param_name == 'p':
                        default_case[param_name] = 'k__Bacteria; p__Firmicutes; c__Bacilli; o__Lactobacillales; f__Lactobacillaceae; g__Lactobacillus; s__Lactobacillus_delbrueckii'
                    elif param_name == 'level':
                        default_case[param_name] = 's'
                elif 'str' in param_name.lower() or 'text' in param_name.lower():
                    default_case[param_name] = "test_string"
                elif 'int' in param_name.lower() or 'num' in param_name.lower():
                    default_case[param_name] = 0
                elif 'list' in param_name.lower() or 'array' in param_name.lower():
                    default_case[param_name] = [1, 2, 3]
                elif 'dict' in param_name.lower() or 'map' in param_name.lower():
                    default_case[param_name] = {"key": "value"}
                elif 'bool' in param_name.lower():
                    default_case[param_name] = False
                else:
                    default_case[param_name] = "test_value"
        
        test_inputs.append(default_case)
        
        # Add specific test cases based on question content
        if 'taxonomy' in question.lower() or 'phylogeny' in question.lower():
            specific_cases = [
                {'p': 'k__Bacteria; p__Proteobacteria; c__Gammaproteobacteria', 'level': 'p'},
                {'p': 'k__Archaea; p__Euryarchaeota', 'level': 'k'}
            ]
            for case in specific_cases:
                test_case = default_case.copy()
                test_case.update(case)
                test_inputs.append(test_case)
        
        return test_inputs
    
    def execute_code_safely(self, code: str, function_name: str, test_inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute code in a sandboxed environment and evaluate results.
        """
        results = {
            'success': False,
            'error': None,
            'test_results': [],
            'execution_score': 0.0
        }
        
        try:
            # Normalize code
            code = self.normalize_code(code)
            
            # Create a temporary Python file
            with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as temp_file:
                temp_file_path = temp_file.name
                temp_file.write(code)
                
            # Create a test script
            with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as test_file:
                test_file_path = test_file.name
                
                test_script = f"""
import sys
import json
import importlib.util
import traceback

try:
    # Load the module
    spec = importlib.util.spec_from_file_location('test_module', '{temp_file_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Test results
    results = []

    # Check if function exists
    if not hasattr(module, '{function_name}'):
        print(json.dumps({{'success': False, 'error': 'Function {function_name} not found'}}))
        sys.exit(1)
        
    # Run tests
    test_inputs = {json.dumps(test_inputs)}
    
    for i, test_input in enumerate(test_inputs):
        try:
            # Call the function
            result = getattr(module, '{function_name}')(**test_input)
            results.append({{'test_id': i, 'success': True, 'result': str(result)}})
        except Exception as e:
            results.append({{'test_id': i, 'success': False, 'error': str(e), 'traceback': traceback.format_exc()[-200:]}})
    
    print(json.dumps({{'success': True, 'results': results}}))
    
except Exception as e:
    print(json.dumps({{'success': False, 'error': str(e), 'traceback': traceback.format_exc()[-200:]}}))
"""
                test_file.write(test_script)
                
            # Execute the test script
            try:
                process = subprocess.run(
                    [sys.executable, test_file_path],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )
                
                if process.returncode == 0 and process.stdout:
                    try:
                        execution_data = json.loads(process.stdout.strip())
                        results['success'] = execution_data.get('success', False)
                        
                        if 'results' in execution_data:
                            results['test_results'] = execution_data['results']
                            successful_tests = sum(1 for test in execution_data['results'] if test.get('success', False))
                            total_tests = len(execution_data['results'])
                            
                            if total_tests > 0:
                                results['execution_score'] = successful_tests / total_tests
                        
                        if 'error' in execution_data:
                            results['error'] = execution_data['error']
                            
                    except json.JSONDecodeError:
                        results['error'] = 'Failed to parse execution results'
                else:
                    results['error'] = process.stderr.strip() if process.stderr else 'Unknown execution error'
                    
            except subprocess.TimeoutExpired:
                results['error'] = f'Execution timed out after {self.timeout} seconds'
                
        except Exception as e:
            results['error'] = f'Error setting up execution environment: {str(e)}'
            
        finally:
            # Clean up temporary files
            for file_path in [temp_file_path, test_file_path]:
                try:
                    if os.path.exists(file_path):
                        os.unlink(file_path)
                except:
                    pass
                    
        return results
    
    def rerank(self, query: str, contexts: List[Dict[str, Any]], query_intent: Dict[str, Any] = None,
              answer_components: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Rerank contexts based on code execution results.
        """
        for ctx in contexts:
            if 'text' not in ctx:
                ctx['execution_score'] = 0.0
                ctx['final_score'] = float(ctx.get('score', 0.0))
                continue
                
            code = ctx['text']
            
            # Extract function info
            function_name, parameters = self.extract_function_info(code)
            
            if not function_name:
                # Can't find a function, skip execution
                ctx['execution_score'] = 0.0
                ctx['final_score'] = float(ctx.get('score', 0.0))
                ctx['execution_error'] = 'No function found'
                continue
            
            # Generate test inputs
            test_inputs = self.generate_test_inputs(function_name, parameters, query)
            
            # Execute code and get results
            execution_results = self.execute_code_safely(code, function_name, test_inputs)
            
            # Store execution results and score
            ctx['execution_results'] = execution_results
            ctx['execution_score'] = execution_results.get('execution_score', 0.0)
            
            # Update final score
            original_score = float(ctx.get('score', 0.0))
            ctx['final_score'] = original_score + self.weight * ctx['execution_score']
            
        # Sort by final score
        return sorted(contexts, key=lambda x: x.get('final_score', 0.0), reverse=True)