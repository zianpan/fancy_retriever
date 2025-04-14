import os
import sys
import json
import argparse
from typing import Dict, List, Any, Optional

# Add the src directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Now import from src package
from data.data_processor import DataProcessor
from reranking.combined_reranker import CombinedReranker
from refinement.self_refiner import SelfRefiner
from evaluation.evaluator import Evaluator

class CodeEnhancementSystem:
    """
    Main class for enhancing previously retrieved code examples through reranking and refinement.
    """
    def __init__(self, retrieved_data_path: str, use_reranking: bool = True, use_refinement: bool = True):
        """
        Initialize the code enhancement system.
        
        Args:
            retrieved_data_path: Path to the JSON file containing retrieval results from scode-g/scode-r
            use_reranking: Whether to use the reranking component
            use_refinement: Whether to use the refinement component
        """
        # Load pre-retrieved data
        with open(retrieved_data_path, 'r') as f:
            self.retrieved_data = json.load(f)
        
        # Process the retrieved data to extract features
        self.processor = DataProcessor(self.retrieved_data)
        self.processed_data = self.processor.process_data()
        
        # Initialize enhancement components (no retriever needed)
        self.reranker = CombinedReranker() if use_reranking else None
        self.refiner = SelfRefiner() if use_refinement else None
        self.evaluator = Evaluator()
        
        # Track component usage
        self.use_reranking = use_reranking
        self.use_refinement = use_refinement
    
    def enhance_code(self, query_idx: int, k: int = 5) -> Dict[str, Any]:
        """
        Enhance code for a specific query index from the pre-retrieved data.
        
        Args:
            query_idx: Index of the query in the retrieved data
            k: Number of results to keep in output
            
        Returns:
            Dictionary with enhanced results
        """
        if query_idx >= len(self.processed_data):
            return {
                "error": f"Query index {query_idx} out of range"
            }
        
        # Get the item from processed data
        item = self.processed_data[query_idx]
        query = item['question']
        
        # Get the pre-retrieved contexts
        contexts = item['contexts']
        
        if not contexts:
            return {
                "query": query,
                "error": "No contexts found in the pre-retrieved data"
            }
        
        # Extract query intent and answer components
        query_intent = item['query_intent']
        answer_components = item['answer_components']
        
        # Apply reranking if enabled
        if self.use_reranking:
            reranked_contexts = self.reranker.rerank(query, contexts, query_intent, answer_components)
        else:
            reranked_contexts = contexts  # Use original contexts if reranking is disabled
        
        # Apply refinement if enabled
        if self.use_refinement:
            generated_code = self.refiner.refine(query, reranked_contexts, query_intent)
        else:
            # If refinement is disabled, use the top context's code
            generated_code = reranked_contexts[0]['text'] if reranked_contexts else ""
        
        # Evaluate results
        evaluation = self.evaluator.evaluate(generated_code, item['answer'])
        
        return {
            "query": query,
            "retrieved_contexts": reranked_contexts[:k],  # Ensure only top k contexts are returned
            "generated_code": generated_code,
            "ground_truth": item['answer'],
            "evaluation": evaluation
        }
    
    def batch_enhance(self, query_indices: List[int] = None, k: int = 5) -> Dict[str, Any]:
        """
        Enhance code for multiple queries.
        
        Args:
            query_indices: List of query indices. If None, process all queries.
            k: Number of results to keep per query
            
        Returns:
            Dictionary with evaluation results
        """
        # If no query indices provided, process all
        if query_indices is None:
            query_indices = list(range(len(self.processed_data)))
        
        results = []
        metrics = {
            "exact_match": 0,
            "functional_match": 0,
            "bleu": 0.0,
            "edit_distance": 0.0
        }
        
        for idx in query_indices:
            result = self.enhance_code(idx, k)
            results.append(result)
            
            # Update metrics
            if 'evaluation' in result:
                for metric in metrics:
                    if metric in result['evaluation']:
                        metrics[metric] += result['evaluation'][metric]
        
        # Average metrics
        num_queries = len(query_indices)
        if num_queries > 0:
            for metric in metrics:
                metrics[metric] /= num_queries
        
        return {
            "individual_results": results,
            "aggregated_metrics": metrics
        }

def main():
    parser = argparse.ArgumentParser(description='Code Enhancement System for scode-g/scode-r results')
    parser.add_argument('--data', type=str, required=True, help='Path to the JSON file with retrieved results')
    parser.add_argument('--query_idx', type=int, help='Index of specific query to enhance')
    parser.add_argument('--process_all', action='store_true', help='Process all queries in the data')
    parser.add_argument('--mode', type=str, choices=['reranking', 'refinement', 'both'], default='both',
                        help='Mode of operation: reranking, refinement, or both')
    parser.add_argument('--k', type=int, default=5, help='Number of retrievals to keep')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save output files')
    
    args = parser.parse_args()
    
    # Determine which components to use based on mode
    use_reranking = args.mode in ['reranking', 'both']
    use_refinement = args.mode in ['refinement', 'both']
    
    # Initialize the system with the specified components
    system = CodeEnhancementSystem(args.data, use_reranking=use_reranking, use_refinement=use_refinement)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    mode_name = args.mode
    k_value = args.k
    
    if args.query_idx is not None:
        # Process a single query by index
        result = system.enhance_code(args.query_idx, k=k_value)
        
        # Construct output filename
        query_slug = f"query_{args.query_idx}"
        if 'query' in result:
            query_text = result['query'].replace(' ', '_')[:30]
            query_slug = f"{args.query_idx}_{query_text}"
        
        output_file = os.path.join(args.output_dir, f"{mode_name}_{query_slug}.json")
        
        # Save result to JSON file
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
            
        print(f"Result saved to {output_file}")
    
    elif args.process_all:
        # Process all queries
        batch_results = system.batch_enhance(k=k_value)
        
        # Construct output filename for batch results
        output_file = os.path.join(args.output_dir, f"{mode_name}_all_results.json")
        
        # Save batch results to JSON file
        with open(output_file, 'w') as f:
            json.dump(batch_results, f, indent=2)
        
        # Save individual results to separate files
        for i, result in enumerate(batch_results['individual_results']):
            if 'query' in result:
                query_text = result['query'].replace(' ', '_')[:30]
                query_slug = f"{i}_{query_text}"
            else:
                query_slug = f"query_{i}"
                
            individual_output_file = os.path.join(args.output_dir, f"{mode_name}_{query_slug}.json")
            with open(individual_output_file, 'w') as f:
                json.dump(result, f, indent=2)
        
        print(f"Batch results saved to {output_file}")
        print(f"Individual results saved to {args.output_dir}")
    
    else:
        print("Please provide either a query index or use --process_all")

if __name__ == "__main__":
    main()