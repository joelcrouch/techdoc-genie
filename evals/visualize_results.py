import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any
import sys

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# --- CONFIGURATION ---
RESULTS_DIR = Path("evals/results")
VISUALIZATIONS_DIR = Path("evals/visualizations")

# Ensure visualization directory exists
VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)


# --- HELPER FUNCTIONS ---
def load_all_results(results_dir: Path) -> List[Dict]:
    """
    Loads metadata and summary statistics from all JSON result files
    in the specified directory.
    """
    all_summaries = []
    json_files = list(results_dir.glob("*.json"))

    if not json_files:
        logger.warning(f"No JSON result files found in {results_dir}.")
        return []

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Ensure the top-level object is a dictionary
            if not isinstance(data, dict):
                logger.warning(f"Skipping {json_file}: Top-level JSON object is not a dictionary.")
                continue

            metadata = data.get('metadata', {})
            
            # Extract relevant info from metadata
            summary_info = {
                "document": metadata.get("document", "Unknown Document"),
                "llm_provider": metadata.get("llm_provider", "Unknown LLM Provider"),
                "llm_model_id": metadata.get("llm_model_id", "Unknown LLM Model"),
                "chunking_strategy_name": metadata.get("chunking_strategy_name", "Unknown Strategy"),
                "chunk_size": metadata.get("chunk_size"),
                "chunk_overlap": metadata.get("chunk_overlap"),
                "chunking_method": metadata.get("chunking_method"),
                "embedder_provider": metadata.get("embedder_provider", "Unknown Embedder"),
                "embedder_model": metadata.get("embedder_model", "Unknown Embedder Model"),
                "average_semantic_similarity": metadata.get("average_semantic_similarity"),
                "answer_rate": metadata.get("answer_rate")
            }
            all_summaries.append(summary_info)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {json_file}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error processing {json_file}: {e}")
            
    return all_summaries


def create_comparison_plots(df: pd.DataFrame, output_dir: Path):
    """
    Generates and saves various comparison plots from the evaluation DataFrame.
    """
    if df.empty:
        logger.warning("No data to plot. Skipping plot generation.")
        return

    # Set up plot style
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))

    # --- Plot 1: Average Semantic Similarity by Chunking Strategy ---
    if 'average_semantic_similarity' in df.columns and 'chunking_strategy_name' in df.columns:
        plt.figure(figsize=(14, 7))
        sns.barplot(
            data=df,
            x='chunking_strategy_name',
            y='average_semantic_similarity',
            hue='llm_model_id',
            palette='viridis'
        )
        plt.title('Average Semantic Similarity Across Chunking Strategies')
        plt.xlabel('Chunking Strategy')
        plt.ylabel('Average Semantic Similarity')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plot_filename = output_dir / "avg_semantic_similarity_by_chunking.png"
        plt.savefig(plot_filename)
        logger.info(f"Saved plot: {plot_filename}")
        plt.close()

    # --- Plot 2: Answer Rate by Chunking Strategy ---
    if 'answer_rate' in df.columns and 'chunking_strategy_name' in df.columns:
        plt.figure(figsize=(14, 7))
        sns.barplot(
            data=df,
            x='chunking_strategy_name',
            y='answer_rate',
            hue='llm_model_id',
            palette='magma'
        )
        plt.title('Answer Rate Across Chunking Strategies')
        plt.xlabel('Chunking Strategy')
        plt.ylabel('Answer Rate (%)')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 100) # Answer rate is a percentage
        plt.tight_layout()
        plot_filename = output_dir / "answer_rate_by_chunking.png"
        plt.savefig(plot_filename)
        logger.info(f"Saved plot: {plot_filename}")
        plt.close()
        
    logger.info("Plot generation complete.")


# --- MAIN EXECUTION ---
def main():
    logger.info("--- Starting Visualization Script ---")
    
    summaries = load_all_results(RESULTS_DIR)
    if not summaries:
        logger.error("No summaries loaded. Exiting visualization script.")
        return
        
    df = pd.DataFrame(summaries)
    logger.info(f"Loaded {len(df)} experiment summaries.")

    # Convert numeric columns to appropriate types
    df['average_semantic_similarity'] = pd.to_numeric(df['average_semantic_similarity'], errors='coerce')
    df['answer_rate'] = pd.to_numeric(df['answer_rate'], errors='coerce')

    create_comparison_plots(df, VISUALIZATIONS_DIR)
    logger.info("--- Visualization Script Finished ---")

if __name__ == "__main__":
    main()
