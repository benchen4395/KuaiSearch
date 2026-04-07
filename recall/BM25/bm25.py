from BM25 import SparseRetriever
import json
from typing import Dict, List
from collections import defaultdict
import json
from typing import Iterator,Dict, List, Tuple, Optional
from collections import defaultdict

class Processor:
    @staticmethod
    def load_queries_from_file(file_path: str) -> Dict[str, str]:
        queries = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    qid = parts[0]
                    query_text = parts[1]
                    queries[qid] = query_text
        return queries
    
    @staticmethod
    def load_qrels_from_file(file_path: str) -> Dict[str, Dict[str, int]]:
        qrels = defaultdict(dict)
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    qid, _, docid, rel = parts[:4]
                    qrels[qid][docid] = int(rel)
        return dict(qrels)


class Evaluator:
    
    def __init__(self):
        self.processor = Processor()
        self.item_pseudo_queries = {}

    def _load_pseudo_queries(self, pseudo_query_file: str):
        """
        Load pseudo_query.all.jsonl.

        Each line has the format:
        {"item_id": 6381242858, "pseudo_query": "航海镜带罗盘"}

        One item_id typically appears in 10 lines (i.e., 10 pseudo-queries per item).
        """
        print(f"Loading pseudo-query file: {pseudo_query_file}")
        mp = defaultdict(list)

        with open(pseudo_query_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                item_id = str(obj["item_id"])
                q = obj.get("pseudo_query", "")
                if q:
                    mp[item_id].append(q)

        self.item_pseudo_queries = mp
        print(f"Pseudo-query loading completed. {len(self.item_pseudo_queries)} items have pseudo-queries.")

            
    def build_index(self, corpus_file: str, index_name: str,
                    pseudo_query_file: str = None):
    
        print("Initializing retriever...")
        self.retriever = SparseRetriever(
            index_name=index_name,
            model="bm25",
            min_df=1,
            tokenizer="jieba",
            stemmer=None,
            stopwords="chinese",
            do_lowercasing=True,
            do_ampersand_normalization=False,
            do_special_chars_normalization=True,
            do_acronyms_normalization=True,
            do_punctuation_removal=True,
        )
    
        # Load pseudo-query mapping first (if provided)
        if pseudo_query_file is not None:
            self._load_pseudo_queries(pseudo_query_file)
        else:
            self.item_pseudo_queries = {}
    
        def safe_get(doc, key):
            """
            Safely retrieve a field value from the document.
            If the field is missing or None, return an empty string.
            All returned values are converted to string.
            """
            val = doc.get(key, "")
            if val is None:
                return ""
            return str(val)
    
        def build_text(doc):
            """
            Construct the final indexed text by concatenating:
            [item_title] [pseudo_query_1] ... [pseudo_query_10]
            """
    
            item_id_str = str(doc.get("item_id", ""))
    
            # Original textual fields
            fields = [
                safe_get(doc, "item_title"),
            ]
    
            # Append pseudo-queries at the end
            pseudo_list = self.item_pseudo_queries.get(item_id_str, [])
            fields.extend(pseudo_list)
    
            # Remove empty strings and whitespace-only fields
            parts = [f.strip() for f in fields if f and f.strip()]
            return " ".join(parts)
    
        print("Building index...")
        self.retriever = self.retriever.index_file(
            path=corpus_file,
            show_progress=True,
            callback=lambda doc: {
                # Ensure item_id is always converted to string (even if originally int)
                "id": str(doc.get("item_id")),
                "text": build_text(doc),
            },
        )


        
    def load_evaluation_data(self, queries_file: str, qrels_file: str):
        self.queries = self.processor.load_queries_from_file(queries_file)
        self.qrels = self.processor.load_qrels_from_file(qrels_file)
        
    def run_retrieval(self, top_k: int = 1000):
    
        print("Starting retrieval...")
        
        results = {}
        for qid, query_text in self.queries.items():
            # Perform retrieval using retriv
            search_results = self.retriever.search(
                query=query_text,
                return_docs=True,
                cutoff=top_k
            )
            
            # Convert results into the required evaluation format
            results[qid] = []
            for result in search_results:
                results[qid].append({
                    "doc_id": result["id"],
                    "score": result["score"]
                })
        
        return results
    
    def calculate_metrics(self, results: Dict, cutoffs: List[int] = [10, 100, 1000]):
        from collections import defaultdict
        import numpy as np
        
        metrics = defaultdict(list)
        
        for qid in results:
            if qid not in self.qrels:
                continue
                
            relevant_docs = set(doc_id for doc_id, rel in self.qrels[qid].items() if rel > 0)
            retrieved_docs = [item["doc_id"] for item in results[qid]]
            
            for cutoff in cutoffs:
                retrieved_at_k = retrieved_docs[:cutoff]
                relevant_retrieved = len(set(retrieved_at_k) & relevant_docs)
                
                # Recall@k
                recall_k = relevant_retrieved / len(relevant_docs) if relevant_docs else 0
                metrics[f"recall@{cutoff}"].append(recall_k)
                
                # Precision@k  
                precision_k = relevant_retrieved / len(retrieved_at_k) if retrieved_at_k else 0
                metrics[f"precision@{cutoff}"].append(precision_k)
                
                # MRR@k
                mrr_k = 0
                for i, doc_id in enumerate(retrieved_at_k):
                    if doc_id in relevant_docs:
                        mrr_k = 1 / (i + 1)
                        break
                metrics[f"mrr@{cutoff}"].append(mrr_k)
                #HR@k
                hr_k = 1 if relevant_retrieved > 0 else 0
                metrics[f"hr@{cutoff}"].append(hr_k)
        avg_metrics = {}
        for metric_name, values in metrics.items():
            avg_metrics[metric_name] = np.mean(values)
        
        return avg_metrics


        
    def run_full_evaluation(
        self,
        index_name: str,
        corpus_file: str,
        queries_file: str,
        qrels_file: str,
        top_k: int = 10,
        cutoffs=[10],
        cutoff: int = 10,
        pseudo_query_file: str = None,  # Newly added parameter
    ):
        """Run the complete evaluation pipeline."""
        
        # Step 1: Build the retrieval index
        self.build_index(corpus_file, index_name, pseudo_query_file=pseudo_query_file)
    
        # Step 2: Load evaluation queries and relevance judgments
        self.load_evaluation_data(queries_file, qrels_file)
        
        # Step 3: Execute retrieval
        results = self.run_retrieval(top_k)
        
        # Step 4: Compute evaluation metrics
        metrics = self.calculate_metrics(results, cutoffs)
        
        # Step 5: Print evaluation results
        print("\n=== evaluation result ===")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
            
        return metrics, results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BM25 Retrieval Evaluation")
    parser.add_argument("--corpus_file",       type=str, default="data/corpus.jsonl")
    parser.add_argument("--queries_file",      type=str, default="recall/data/test.queries.tsv")
    parser.add_argument("--qrels_file",        type=str, default="recall/data/test.qrels.tsv")
    parser.add_argument("--pseudo_query_file", type=str, default=None)
    parser.add_argument("--index_name",        type=str, default="kuaisearch")
    parser.add_argument("--top_k",             type=int, default=100)
    parser.add_argument("--cutoffs",           type=int, nargs="+", default=[10, 20, 50, 100])
    args = parser.parse_args()

    evaluator = Evaluator()
    metrics, results = evaluator.run_full_evaluation(
        index_name=args.index_name,
        corpus_file=args.corpus_file,
        queries_file=args.queries_file,
        qrels_file=args.qrels_file,
        top_k=args.top_k,
        cutoffs=args.cutoffs,
        pseudo_query_file=args.pseudo_query_file,
    )