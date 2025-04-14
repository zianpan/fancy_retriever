from typing import Dict, List, Any
import subprocess
import tempfile
import os
import ast
from evaluation.metrics import CodeMetrics

class Evaluator:
    """
    Evaluates code generation results.
    """
    def __init__(self):
        """
        Initialize the evaluator.
        """
        pass
    
    def evaluate(self, generated: str, reference: str) -> Dict[str, Any]:
        """
        Evaluate generated code against a reference solution.
        """
        try:
            metrics = {}
            
            # Exact match (perfect match)
            metrics['exact_match'] = int(CodeMetrics.exact_match(generated, reference))
            
            # Edit distance (character-level similarity)
            metrics['edit_distance'] = CodeMetrics.normalized_edit_distance(generated, reference)
            
            # BLEU score (token-level similarity)
            metrics['bleu'] = CodeMetrics.bleu_score(generated, reference)
            
            # AST similarity (structure similarity)
            metrics['ast_similarity'] = CodeMetrics.ast_match(generated, reference)
            
            # Functional similarity (signature and parameter match)
            metrics['functional_match'] = CodeMetrics.functional_similarity(generated, reference)
            
            # Execution-based evaluation - skip this in error-prone environments
            try:
                execution_result = self.execution_evaluation(generated, reference)
                metrics.update(execution_result)
            except Exception as e:
                print(f"Skipping execution evaluation due to error: {e}")
                metrics['executes'] = False
                metrics['execution_match'] = False
            
            return metrics
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return {
                'error': str(e),
                'exact_match': 0,
                'functional_match': 0,
                'bleu': 0.0,
                'edit_distance': 0.0,
                'executes': False,
                'execution_match': False
            }
    
    def execution_evaluation(self, generated: str, reference: str) -> Dict[str, Any]:
        """
        Evaluate code by executing it and comparing outputs.
        
        Args:
            generated: Generated code
            reference: Reference code
            
        Returns:
            Dictionary with execution-based metrics
        """
        metrics = {
            'executes': False,
            'execution_match': False
        }
        
        # Skip execution if code has syntax errors
        try:
            ast.parse(generated)
        except SyntaxError:
            return metrics
        
        # Extract function name
        try:
            tree = ast.parse(generated)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    function_name = node.name
                    break
            else:
                # No function definition found
                return metrics
        except:
            return metrics
        
        # Create test cases (in a real system, these would be more sophisticated)
        test_cases = self._generate_test_cases(reference, function_name)
        
        if not test_cases:
            return metrics
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as gen_file, \
             tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as ref_file:
            
            gen_file.write(generated)
            ref_file.write(reference)
            
            gen_file_path = gen_file.name
            ref_file_path = ref_file.name
        
        try:
            # Create test script
            with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as test_file:
                test_file_path = test_file.name
                
                test_script = (
                    f"import sys\n"
                    f"sys.path.append('.')\n"
                    f"import importlib.util\n"
                    f"import traceback\n\n"
                    
                    f"# Import generated function\n"
                    f"gen_spec = importlib.util.spec_from_file_location('gen_module', '{gen_file_path}')\n"
                    f"gen_module = importlib.util.module_from_spec(gen_spec)\n"
                    f"gen_spec.loader.exec_module(gen_module)\n\n"
                    
                    f"# Import reference function\n"
                    f"ref_spec = importlib.util.spec_from_file_location('ref_module', '{ref_file_path}')\n"
                    f"ref_module = importlib.util.module_from_spec(ref_spec)\n"
                    f"ref_spec.loader.exec_module(ref_module)\n\n"
                    
                    f"results = {{}}\n"
                    f"success = True\n\n"
                    
                    f"# Run test cases\n"
                    f"for i, test_case in enumerate({test_cases}):\n"
                    f"    args, kwargs = test_case\n"
                    f"    results[i] = {{}}\n"
                    f"    try:\n"
                    f"        gen_result = gen_module.{function_name}(*args, **kwargs)\n"
                    f"        results[i]['gen_executes'] = True\n"
                    f"        results[i]['gen_result'] = gen_result\n"
                    f"    except Exception as e:\n"
                    f"        results[i]['gen_executes'] = False\n"
                    f"        results[i]['gen_error'] = str(e)\n"
                    f"        success = False\n"
                    f"    \n"
                    f"    try:\n"
                    f"        ref_result = ref_module.{function_name}(*args, **kwargs)\n"
                    f"        results[i]['ref_executes'] = True\n"
                    f"        results[i]['ref_result'] = ref_result\n"
                    f"    except Exception as e:\n"
                    f"        results[i]['ref_executes'] = False\n"
                    f"        results[i]['ref_error'] = str(e)\n"
                    f"    \n"
                    f"    if results[i].get('gen_executes', False) and results[i].get('ref_executes', False):\n"
                    f"        results[i]['match'] = gen_result == ref_result\n"
                    f"        if not results[i]['match']:\n"
                    f"            success = False\n"
                    f"    else:\n"
                    f"        results[i]['match'] = False\n"
                    f"        success = False\n\n"
                    
                    f"print('EXECUTION_RESULTS:')\n"
                    f"import json\n"
                    f"print(json.dumps({{'success': success, 'results': results}}))\n"
                )
                
                test_file.write(test_script)
            
            # Execute test script
            try:
                result = subprocess.run(
                    ['python', test_file_path], 
                    capture_output=True, 
                    text=True,
                    timeout=5  # Timeout after 5 seconds
                )
                
                # Parse results
                if result.stdout and 'EXECUTION_RESULTS:' in result.stdout:
                    import json
                    results_text = result.stdout.split('EXECUTION_RESULTS:')[1].strip()
                    execution_results = json.loads(results_text)
                    
                    metrics['executes'] = all(case['gen_executes'] for case in execution_results['results'].values())
                    metrics['execution_match'] = execution_results['success']
                    metrics['execution_details'] = execution_results['results']
                
            except subprocess.TimeoutExpired:
                metrics['execution_timeout'] = True
            except Exception as e:
                print(f"Error during execution evaluation: {e}")
            
        finally:
            # Clean up temporary files
            for file_path in [gen_file_path, ref_file_path, test_file_path]:
                try:
                    os.unlink(file_path)
                except:
                    pass
        
        return metrics
    
    def _generate_test_cases(self, reference: str, function_name: str) -> List[Any]:
        """
        Generate test cases for a function.
        
        Args:
            reference: Reference code
            function_name: Name of the function to test
            
        Returns:
            List of test cases (args, kwargs)
        """
        try:
            # Parse the reference code to extract parameters
            tree = ast.parse(reference)
            params = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    params = [arg.arg for arg in node.args.args]
                    break
            
            if not params:
                return []
            
            # Generate simple test cases based on parameter names and types
            test_cases = []
            
            # Basic test case: all default parameters (strings, empty, etc.)
            basic_args = ['' if 'path' in p or 'file' in p or 'dir' in p or 'str' in p else 0 for p in params]
            test_cases.append((basic_args, {}))
            
            # Test case for common parameter names
            special_cases = []
            for i, param in enumerate(params):
                arg_copy = basic_args.copy()
                
                # String parameters
                if 'path' in param or 'file' in param or 'dir' in param or 'str' in param:
                    arg_copy[i] = 'test_string'
                    special_cases.append((arg_copy.copy(), {}))
                
                # Level parameters (often used in taxonomy/tree functions)
                elif 'level' in param:
                    for level_val in ['k', 'p', 'c', 'o', 'f', 'g', 's']:
                        arg_copy[i] = level_val
                        special_cases.append((arg_copy.copy(), {}))
            
            test_cases.extend(special_cases)
            
            return test_cases
        
        except Exception as e:
            print(f"Error generating test cases: {e}")
            # Fallback to minimal test case if parsing fails
            return [([], {})]