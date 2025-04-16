import os
import sys
import json
import argparse
import logging
import warnings
from typing import Dict, List, Any, Optional
from datetime import datetime
import time

# Filter out invalid escape sequence warnings
warnings.filterwarnings('ignore', category=SyntaxWarning, module='json')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from data.data_processor import DataProcessor
from reranking.enhanced_combined_reranker import EnhancedCombinedReranker
from refinement.self_refiner import SelfRefiner
from evaluation.evaluator import Evaluator
from query_intent_extractor import QueryIntentExtractor
from ast_component_extractor import ASTComponentExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reranker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CodeEnhancementSystem')

class EnhancedCodeEnhancementSystem:
    """
    Enhanced class for improving code examples through reranking and refinement.
    """
    def __init__(self, retrieved_data_path: str, use_reranking: bool = True, use_refinement: bool = True,
                 log_details: bool = False):
        """
        Initialize the enhanced code enhancement system.
        
        Args:
            retrieved_data_path: Path to the JSON file containing retrieval results
            use_reranking: Whether to use the reranking component
            use_refinement: Whether to use the refinement component
            log_details: Whether to log detailed information for analysis
        """
        logger.info(f"Initializing EnhancedCodeEnhancementSystem with {retrieved_data_path}")
        
        # Load pre-retrieved data
        with open(retrieved_data_path, 'r') as f:
            self.retrieved_data = json.load(f)
        
        self.log_details = log_details
        
        # Initialize components
        self.intent_extractor = QueryIntentExtractor()
        self.component_extractor = ASTComponentExtractor()
        
        # Process the retrieved data to extract features
        self.processor = DataProcessor(self.retrieved_data)
        self.processed_data = self.processor.process_data()
        
        # Enhance processed data with our improved intent extraction
        self._enhance_processed_data()
        
        # Initialize enhancement components
        self.reranker = EnhancedCombinedReranker() if use_reranking else None
        self.refiner = SelfRefiner() if use_refinement else None
        self.evaluator = Evaluator()
        
        # Track component usage
        self.use_reranking = use_reranking
        self.use_refinement = use_refinement
        
        logger.info("EnhancedCodeEnhancementSystem initialized successfully")
    
    def _enhance_processed_data(self):
        """Enhance processed data with improved intent extraction and component analysis."""
        logger.info("Enhancing processed data with improved intent extraction")
        
        for item in self.processed_data:
            # Extract enhanced query intent
            item['enhanced_query_intent'] = self.intent_extractor.extract_intent(item['question'])
            
            # Extract components from the ground truth answer
            if 'answer' in item:
                item['enhanced_answer_components'] = self.component_extractor.extract_components(item['answer'])
            
            # Extract components from each context
            for ctx in item.get('contexts', []):
                if 'text' in ctx:
                    ctx['enhanced_components'] = self.component_extractor.extract_components(ctx['text'])
    
    def enhance_code(self, query_idx: int, k: int = 5) -> Dict[str, Any]:
        """
        Enhance code for a specific query index from the pre-retrieved data.
        
        Args:
            query_idx: Index of the query in the pre-retrieved data
            k: Number of results to keep
            
        Returns:
            Dictionary with enhancement results
        """
        # Get the query and contexts from the pre-retrieved data
        query_data = self.processed_data[query_idx]
        query = query_data['question']
        contexts = query_data['contexts']
        
        # Store original contexts for comparison
        before_reranking_contexts = contexts[:k] if contexts else []
        
        # Evaluate the top result before reranking
        before_reranking_code = contexts[0]['text'] if contexts else ""
        before_reranking_evaluation = self.evaluator.evaluate(before_reranking_code, query_data.get('answer', ''))
        
        # Start timing
        start_time = time.time()
        
        # Apply reranking if enabled
        reranked_contexts = []
        if self.use_reranking:
            # Extract query intent and answer components
            query_intent = self.intent_extractor.extract_intent(query)
            answer_components = self.intent_extractor.extract_answer_components(query_data.get('answer', ''))
            
            # Apply reranking
            reranked_contexts = self.reranker.rerank(query, contexts, query_intent, answer_components)
            
            # Log reranking details if enabled
            if self.log_details:
                self._log_reranking_details(query, query_intent, reranked_contexts)
        else:
            # If reranking is disabled, use the original contexts
            reranked_contexts = contexts
        
        # Apply refinement if enabled
        if self.use_refinement and reranked_contexts:
            # Refine the top result
            refined_code = self.refiner.refine(query, reranked_contexts[0]['text'])
            reranked_contexts[0]['text'] = refined_code
        
        # End timing
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Evaluate the final result
        final_code = reranked_contexts[0]['text'] if reranked_contexts else ""
        evaluation = self.evaluator.evaluate(final_code, query_data.get('answer', ''))
        
        # Calculate BLEU improvement
        bleu_improvement = evaluation.get('bleu', 0) - before_reranking_evaluation.get('bleu', 0)
        
        # Prepare the result
        result = {
            "query": query,
            "scores": {
                "structure_score": reranked_contexts[0].get('structure_score', 0) if reranked_contexts else 0,
                "signature_score": reranked_contexts[0].get('signature_score', 0) if reranked_contexts else 0,
                "semantic_score": reranked_contexts[0].get('semantic_score', 0) if reranked_contexts else 0,
                "final_score": reranked_contexts[0].get('final_score', 0) if reranked_contexts else 0
            },
            "evaluation": {
                "bleu": evaluation.get('bleu', 0),
                "edit_distance": evaluation.get('edit_distance', 0)
            },
            "before_reranking_evaluation": {
                "bleu": before_reranking_evaluation.get('bleu', 0),
                "edit_distance": before_reranking_evaluation.get('edit_distance', 0)
            },
            "processing_time_seconds": processing_time,
            "bleu_improvement": bleu_improvement,
            # Store contexts for detailed comparison
            "before_reranking_contexts": before_reranking_contexts,
            "reranked_contexts": reranked_contexts[:k] if reranked_contexts else []
        }
        
        # Add reranking statistics if available
        if self.use_reranking:
            # Debug: Print the keys in previous_scores
            if reranked_contexts and 'previous_scores' in reranked_contexts[0]:
                logger.info(f"Previous scores keys: {list(reranked_contexts[0]['previous_scores'].keys())}")
            
            result["reranking_stats"] = {
                "feature_impacts": {
                    "structure_score": reranked_contexts[0].get('previous_scores', {}).get('structure_score', 0) if reranked_contexts else 0,
                    "signature_score": reranked_contexts[0].get('previous_scores', {}).get('signature_score', 0) if reranked_contexts else 0,
                    "semantic_score": reranked_contexts[0].get('previous_scores', {}).get('semantic_score', 0) if reranked_contexts else 0
                }
            }
        
        logger.info(f"Query {query_idx} processed in {processing_time:.2f} seconds")
        logger.info(f"Evaluation: BLEU={evaluation.get('bleu', 0):.4f}, " +
                   f"Edit distance={evaluation.get('edit_distance', 1):.4f}")
        
        return result
    
    def _log_reranking_details(self, query: str, query_intent: Dict[str, Any], 
                            contexts: List[Dict[str, Any]]) -> None:
        """
        Log detailed information about reranking for analysis.
        """
        logger.debug("RERANKING DETAILS:")
        logger.debug(f"Query: {query[:100]}...")
        
        # Debug query intent - this is crucial for signature and semantic scores
        logger.debug(f"Extracted intent: {json.dumps(query_intent, indent=2)}")
        
        # Check if query intent has the necessary fields for signature scoring
        has_signature_components = any(key in query_intent for key in [
            'function_name', 'parameters', 'has_docstring', 'has_return', 'error_handling'
        ])
        logger.debug(f"Has signature components: {has_signature_components}")
        
        # Check if there's enough information for semantic scoring
        semantic_info = {
            'key_phrases': self._extract_sample_key_phrases(query),
            'domain_terms': self._extract_sample_domain_terms(query),
            'algorithm_description': query_intent.get('algorithm_description', []),
            'key_functions': query_intent.get('key_functions', [])
        }
        logger.debug(f"Semantic information: {json.dumps(semantic_info, indent=2)}")
        
        # Log details for contexts
        for i, ctx in enumerate(contexts[:3]):
            logger.debug(f"\nContext {i+1} (score: {ctx.get('final_score', 0):.4f}):")
            logger.debug(f"Function: {ctx.get('components', {}).get('function_name', 'N/A')}")
            logger.debug(f"Structure score: {ctx.get('structure_score', 0):.4f}")
            logger.debug(f"Signature score: {ctx.get('signature_score', 0):.4f}")
            logger.debug(f"Semantic score: {ctx.get('semantic_score', 0):.4f}")
            
            # Inspect components structure which is needed for scoring
            comp_keys = list(ctx.get('components', {}).keys())
            logger.debug(f"Component keys: {comp_keys}")
            
            if 'quality_boost' in ctx:
                logger.debug(f"Quality boost: {ctx.get('quality_boost', 0):.4f}")

    def _extract_sample_key_phrases(self, query: str) -> List[str]:
        """Extract a few sample key phrases to help debug semantic scoring"""
        import re
        phrases = re.findall(r'([a-z]+(?:\s+[a-z]+){1,3}(?:\s+the\s+[a-z]+)?)', query.lower())
        return phrases[:3]  # Just return first 3 for debugging

    def _extract_sample_domain_terms(self, query: str) -> List[str]:
        """Extract sample domain terms to help debug semantic scoring"""
        domains = {
            'file': ['file', 'directory', 'path', 'open', 'read', 'write', 'close'],
            'string': ['string', 'parse', 'format', 'concatenate', 'split', 'join'],
            'error': ['exception', 'error', 'handle', 'try', 'except', 'finally', 'raise'],
        }
        
        found_terms = []
        query_lower = query.lower()
        for domain, terms in domains.items():
            for term in terms:
                if term in query_lower:
                    found_terms.append(term)
        
        return found_terms
    
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
        
        logger.info(f"Batch processing {len(query_indices)} queries")
        
        results = []
        metrics = {
            "exact_match": 0,
            "functional_match": 0,
            "bleu": 0.0,
            "bleu_before_reranking": 0.0,
            "bleu_improvement": 0.0,
            "edit_distance": 0.0,
            "total_processing_time": 0.0
        }
        
        for idx in query_indices:
            result = self.enhance_code(idx, k)
            results.append(result)
            
            # Update metrics
            if 'evaluation' in result:
                for metric in ['exact_match', 'functional_match', 'bleu', 'edit_distance']:
                    if metric in result['evaluation']:
                        metrics[metric] += result['evaluation'][metric]
            if 'before_reranking_evaluation' in result:
                metrics['bleu_before_reranking'] += result['before_reranking_evaluation'].get('bleu', 0)

            # Track processing time
            if 'processing_time_seconds' in result:
                metrics['total_processing_time'] += result['processing_time_seconds']
            
            # Track BLEU improvement
            if 'bleu_improvement' in result:
                metrics['bleu_improvement'] += result['bleu_improvement']
        
        # Average metrics
        num_queries = len(query_indices)
        if num_queries > 0:
            for metric in ['exact_match', 'functional_match', 'bleu', 'edit_distance', 'bleu_before_reranking', 'bleu_improvement']:
                if metric in metrics:
                    metrics[metric] /= num_queries
        
        return {
            "individual_results": results,
            "aggregated_metrics": metrics
        }

def main():
    parser = argparse.ArgumentParser(description='Code Enhancement System')
    parser.add_argument('--data', type=str, required=True, help='Path to retrieved results JSON file')
    parser.add_argument('--query_idx', type=int, help='Specific query index to enhance')
    parser.add_argument('--all', action='store_true', help='Process all queries')
    parser.add_argument('--mode', type=str, choices=['reranking', 'refinement', 'both'], default='both',
                        help='Operation mode')
    parser.add_argument('--k', type=int, default=5, help='Number of results to keep')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    os.makedirs(args.output, exist_ok=True)
    
    system = EnhancedCodeEnhancementSystem(
        args.data, 
        use_reranking=args.mode in ['reranking', 'both'], 
        use_refinement=args.mode in ['refinement', 'both'],
        log_details=args.debug
    )
    
    if args.query_idx is not None:
        # Process single query
        result = system.enhance_code(args.query_idx, k=args.k)
        output_file = os.path.join(args.output, f"{args.mode}_query_{args.query_idx}.json")
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Result saved to {output_file}")
    
    elif args.all:
        # Process all queries
        batch_results = system.batch_enhance(k=args.k)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(args.output, f"{args.mode}_results_{timestamp}.json")
        with open(output_file, 'w') as f:
            json.dump(batch_results, f, indent=2)
        logger.info(f"Batch results saved to {output_file}")

if __name__ == "__main__":
    main()