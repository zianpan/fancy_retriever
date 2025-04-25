# Code Reranker
A component for reranking and improving code search results based on semantic analysis and code understanding.

## Components
- `main.py`: Main entry point for the reranking system
- `function_relationship_analyzer.py`: Analyzes relationships between functions in code
- `query_intent_extractor.py`: Extracts and processes search query intents
- `ast_component_extractor.py`: Extracts code components using Abstract Syntax Tree analysis
- `src/`:
  - `data/`:
    - `data_loader.py`: Handles loading and preprocessing of code search results and queries
    - `data_processor.py`: Processes and transforms raw data into reranker-ready format
  - `reranking/`:
    - `enhanced_combined_reranker.py`: Main reranking orchestrator, combines scores from different reranking strategies
    - `semantic_reranker.py`: Semantic similarity-based reranking
    - `code_structure_reranker.py`: Structure-based reranking
    - `signature_reranker.py`: Function signature analysis
    - `execution_results_reranker.py`: Execution results analysis
- `evaluation/`: Evaluation scripts and metrics

## Usage

To use the reranker:

1. Ensure all dependencies are installed
2. Run the reranker using:
```bash
python main.py --data your_input_filename.json --all --mode reranking --output your_output_dir/
```

The reranker will process code search results and improve their ranking based on:
- Function relationships and dependencies
- Query intent analysis
- Code structure and components
- Semantic relevance
