import os
import argparse
import types
from accelerate import Accelerator
from train import train, get_yaml_file
from getemb import generate_embeddings
from test import evaluate_model

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = 'true'
os.environ["WANDB_IGNORE_GLOBS"] = '*.bin'

def main():
    parser = argparse.ArgumentParser(description="Run DPR Pipeline: Train -> Embed -> Evaluate")
    
    # DPR args
    parser.add_argument("--config_file", default='recall/dpr/config/train_dpr_nq.yaml')
    parser.add_argument("--share_encoder", type=lambda x: x.lower() == 'true', default=None)
    
    # GetEmb args
    parser.add_argument("--corpus_path", default="data/corpus.jsonl")
    parser.add_argument("--num_docs", type=int, default=18605582)
    parser.add_argument("--emb_batch_size", type=int, default=2048, help="Batch size for embedding generation")
    parser.add_argument("--doc_encoder_path", default=None, help="Override path for doc encoder")
    parser.add_argument("--output_dir", default="doc_embedding")
    
    # Test args
    parser.add_argument("--queries_file", default="recall/data/test.queries.tsv")
    parser.add_argument("--qrels_file", default="recall/data/test.qrels.tsv")
    parser.add_argument("--test_batch_size", type=int, default=512, help="Batch size for query encoding in test")
    parser.add_argument("--top_k", type=int, default=100)
    
    # Parse CLI args
    args = parser.parse_args()
    
    # Helper to check main process before Accelerator init
    def is_main_process():
        return int(os.environ.get("LOCAL_RANK", -1)) in [-1, 0]
    
    # --- Step 1: Training ---
    if is_main_process():
        print("\n" + "="*50)
        print("STEP 1: Training DPR Model")
        print("="*50)
    
    # Load DPR config from yaml
    yaml_config = {}
    if os.path.exists(args.config_file):
        yaml_config = get_yaml_file(args.config_file)
    
    # Merge logic: YAML < CLI (if not None)
    # 1. Start with YAML
    final_config = yaml_config.copy()
    
    # 2. Update with CLI args that are NOT None
    cli_args = {k: v for k, v in vars(args).items() if v is not None}
    final_config.update(cli_args)
    
    # 3. Update args namespace
    for k, v in final_config.items():
        setattr(args, k, v)

    # Execute training
    # Note: train() expects args to have attributes like args.seed, args.save_dir, etc.
    # train() will initialize Accelerator with specific kwargs.
    train(args)
    
    # Now get the accelerator instance (reusing the one created in train)
    accelerator = Accelerator()
    
    # Wait for all processes to finish training
    accelerator.wait_for_everyone()
    
    # Define paths based on training output
    save_dir = getattr(args, 'save_dir', 'checkpoint') # Fallback if save_dir not in args
    doc_encoder_path = os.path.join(save_dir, "best", "doc_encoder")
    query_encoder_path = os.path.join(save_dir, "best", "query_encoder")
    
    # --- Step 2: Generate Embeddings ---
    if accelerator.is_main_process:
        print("\n" + "="*50)
        print("STEP 2: Generating Document Embeddings")
        print("="*50)
    
    # Prepare args for getemb
    emb_args = types.SimpleNamespace()
    emb_args.corpus_path = args.corpus_path
    emb_args.num_docs = args.num_docs
    emb_args.encoding_batch_size = args.emb_batch_size
    # Use trained doc encoder unless overridden
    emb_args.pretrained_model_path = args.doc_encoder_path if args.doc_encoder_path else doc_encoder_path
    emb_args.output_dir = args.output_dir
    
    # Execute embedding generation
    generate_embeddings(emb_args)
    
    # Wait for all processes to finish embedding
    accelerator.wait_for_everyone()
    
    # --- Step 3: Evaluation ---
    # Only run evaluation on main process
    if accelerator.is_main_process:
        print("\n" + "="*50)
        print("STEP 3: Evaluating Performance")
        print("="*50)
        
        # Prepare args for test
        test_args = types.SimpleNamespace()
        test_args.corpus_file = args.corpus_path
        test_args.queries_file = args.queries_file
        test_args.qrels_file = args.qrels_file
        test_args.encoding_batch_size = args.test_batch_size
        test_args.num_shards = accelerator.num_processes
        test_args.embedding_dir = args.output_dir
        # Use trained query encoder
        test_args.pretrained_model_path = query_encoder_path
        test_args.top_k = args.top_k
        
        # Execute evaluation
        evaluate_model(test_args)
        
    accelerator.wait_for_everyone()

if __name__ == "__main__":
    main()
