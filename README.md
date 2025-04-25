# Biencoder Code Instances Retriever



# Retrieved Code Instances Reranker
A component for reranking and improving retrieved code instance results based on semantic analysis and code understanding.

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


# Dynamic Augmented Text2Code Generation
A text to code generator module augmented with dynamic number of retrieved relevant code instances. Our results show text to code generation overall performance improves and becomes saturated while augmented with increasing amount of retrieved code instances. However, different text command achieves its highest performance at different saturation speed: while some at 0 or 5 code instances, others at 20 or 40 code instances. To achieve an overall highest performance with the least amount of compute, we designed a LLM router, using the Qwen2.5-14B-1M long context model, to determine the difficulty of a given problem by classifying it into easy, medium, or hard levels. This provides instruction for how many retrieved code instances that are necessary as input for code generation, using the same Qwen2.5-14B-1M model.

## Components
- `codegen/`: code generation brance, run below sequentially
  - `Generation/`:
    - `pipeline_augmented.ipynb`: LLM router and augmented code generator
    - `evaluation_SCODE_G.ipynb`: EM, BLEU, CodeBLEU evaluation for code generation
    - `evaluation_SCODE_R.ipynb`: EM, BLEU, CodeBLEU evaluation for code retriever
