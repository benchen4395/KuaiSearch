import os
import json
import torch
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoTokenizer,
    MT5ForConditionalGeneration,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)

from parameters import load_parameters
from dataset import (
    Item2QueryTrainDataset,
    Item2QueryCollator,
    GenerateDataset,
    QueryEvalCollator,
)


def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def main():
    set_seed(42)
    model_args, data_args, training_args = load_parameters()

    # =========================
    # Part 1) Train
    # =========================
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
    )
    model = MT5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
    )

    full_dataset = Item2QueryTrainDataset(
        path_to_data=data_args.train_file,
        max_length=data_args.max_seq_length,
        tokenizer=tokenizer,
    )

    n_total = len(full_dataset)
    n_eval = max(1, int(n_total * 0.01))
    n_train = n_total - n_eval
    generator = torch.Generator().manual_seed(42)

    train_dataset, eval_dataset = random_split(
        full_dataset,
        [n_train, n_eval],
        generator=generator,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=Item2QueryCollator(tokenizer, padding="longest"),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    # Save trained model
    save_dir = "best_model"
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)

    # Use the trained model instance directly (no reload)
    model = trainer.model

    # =========================
    # Part 2) Generate
    # =========================
    dataloader_num_workers = 8

    generate_dataset = GenerateDataset(
        path_to_data=data_args.test_file,
        max_length=64,
        tokenizer=tokenizer,
    )

    rank, world_size = get_dist_info()
    is_main = (rank == 0)

    # Build dataloader with DistributedSampler (torchrun)
    if world_size > 1:
        sampler = DistributedSampler(generate_dataset, shuffle=False, drop_last=False)
    else:
        sampler = None

    dataloader = DataLoader(
        generate_dataset,
        batch_size= training_args.per_device_eval_batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=dataloader_num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=QueryEvalCollator(tokenizer, padding="longest"),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    if is_main:
        print(f"[Rank {rank}/{world_size}] Generating (streaming, no accumulation in GPU)...")

    TOP_K = 30
    NUM_RETURN_SEQS = 10
    MAX_NEW_TOKENS = 32

    base_output = "recall/data/pseudo_query"
    os.makedirs(base_output, exist_ok=True)
    output_path = os.path.join(
        base_output,
        f"rank{rank}.jsonl" if world_size > 1 else "rank.jsonl",
    )

    with open(output_path, "w", encoding="utf-8") as f_out:
        for batch in tqdm(
            dataloader,
            desc="Generating",
            dynamic_ncols=True,
            disable=not is_main,
        ):
            # QueryEvalCollator returns (inputs, labels)
            inputs, labels = batch
            input_ids = inputs["input_ids"].to(device, non_blocking=True)
            attention_mask = inputs["attention_mask"].to(device, non_blocking=True)

            with torch.no_grad():
                gen = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    top_k=TOP_K,
                    num_return_sequences=NUM_RETURN_SEQS,
                )
                # gen: [B*R, L] -> [B, R, L]
                batch_size = input_ids.size(0)
                gen = gen.view(batch_size, NUM_RETURN_SEQS, -1).cpu()

            for item_id, seqs in zip(labels, gen):
                item_id_int = int(item_id)
                for seq in seqs:
                    query = tokenizer.decode(seq, skip_special_tokens=True)
                    rec = {"item_id": item_id_int, "pseudo_query": query}
                    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if is_main:
        print(f"[Rank {rank}] Done. Saved pseudo queries to {output_path}")

    # Optional: barrier + merge hint
    # Auto merge after all ranks finish
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        dist.barrier()
        if is_main:
            merged_path = os.path.join(base_output, "pseudo_query.all.jsonl")
            print(f"[Rank 0] Merging into {merged_path} ...")
    
            with open(merged_path, "w", encoding="utf-8") as fout:
                for r in range(world_size):
                    part_path = os.path.join(base_output, f"rank{r}.jsonl")
                    with open(part_path, "r", encoding="utf-8") as fin:
                        for line in fin:
                            fout.write(line)
            print("[Rank 0] Merge completed.")
            for r in range(world_size):
                os.remove(os.path.join(base_output, f"rank{r}.jsonl"))



if __name__ == "__main__":
    main()
