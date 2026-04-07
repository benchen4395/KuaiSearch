## built-in
import math,logging,json,random,functools,os
import types
os.environ["TOKENIZERS_PARALLELISM"]='true'
os.environ["WANDB_IGNORE_GLOBS"]='*.bin' ## not upload ckpt to wandb cloud

## third-party
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
import transformers
from transformers import (
    BertTokenizer,
    BertModel,
)
transformers.logging.set_verbosity_error()
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import random_split
def set_seed(seed: int = 42):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_yaml_file(file_path):
    import yaml  
    with open(file_path, "r") as file:  
        config = yaml.safe_load(file)  
    return config  


def get_linear_scheduler(
    optimizer,
    warmup_steps,
    total_training_steps,
    steps_shift=0,
    last_epoch=-1,
):
    from torch.optim.lr_scheduler import LambdaLR
    """Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        current_step += steps_shift
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            1e-7,
            float(total_training_steps - current_step) / float(max(1, total_training_steps - warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)
    
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    ## adding args here for more control from CLI is possible
    parser.add_argument("--config_file",default='config/train_dpr_nq.yaml')
    parser.add_argument("--share_encoder", type=lambda x: x.lower() == 'true', default=None, help="Whether to share encoder for query and doc")
    args = parser.parse_args()

    yaml_config = get_yaml_file(args.config_file)
    args_dict = {k:v for k,v in vars(args).items() if v is not None}
    yaml_config.update(args_dict)
    args = types.SimpleNamespace(**yaml_config)
    return args

class DualEncoder(nn.Module):
    def __init__(self, base_model: str, share_encoder: bool = False):

        super().__init__()
        self.share_encoder = share_encoder
        self.query_encoder = BertModel.from_pretrained(base_model, add_pooling_layer=False)
        if share_encoder:
            self.doc_encoder = self.query_encoder
        else:
            self.doc_encoder = BertModel.from_pretrained(base_model, add_pooling_layer=False)
    def forward(
        self,
        query_input_ids,
        query_attention_mask,
        query_token_type_ids,
        doc_input_ids,
        doc_attention_mask,
        doc_token_type_ids,
    ):
        CLS_POS = 0
        query_embedding = self.query_encoder(
            input_ids=query_input_ids,
            attention_mask=query_attention_mask,
            token_type_ids=query_token_type_ids,
        ).last_hidden_state[:, CLS_POS, :]

        doc_embedding = self.doc_encoder(
            input_ids=doc_input_ids,
            attention_mask=doc_attention_mask,
            token_type_ids=doc_token_type_ids,
        ).last_hidden_state[:, CLS_POS, :]

        return query_embedding, doc_embedding


def calculate_dpr_loss(matching_score,labels):
    return F.nll_loss(input=F.log_softmax(matching_score,dim=1),target=labels)



class DPRDataset(torch.utils.data.Dataset):
    def __init__(self,file_path):
        self.data = json.load(open(file_path))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.data[idx]

    @staticmethod
    def collate_fn(samples,tokenizer,args):
        
        # prepare query input
        queries = [x['query'] for x in samples]
        query_inputs = tokenizer(queries,max_length=args.max_length,padding=True,truncation=True,return_tensors='pt')
        
        # prepare document input
        ## select the first positive document
        ## passage = title + document
        positive_passages = [random.choice(x['positive_item']) for x in samples]
        positive_docs = [x['text'] for x in positive_passages]

            ## random choose one negative document
        negative_passages = [ random.choice(x['negative_item']) 
                                 for x in samples ]

        negative_docs = [x["text"] for x in negative_passages]
        docs = positive_docs + negative_docs

        doc_inputs = tokenizer(docs,max_length=args.max_length,padding=True,truncation=True,return_tensors='pt')

        return {
            'query_input_ids':query_inputs.input_ids,
            'query_attention_mask':query_inputs.attention_mask,
            'query_token_type_ids':query_inputs.token_type_ids,

            "doc_input_ids":doc_inputs.input_ids,
            "doc_attention_mask":doc_inputs.attention_mask,
            "doc_token_type_ids":doc_inputs.token_type_ids,
        }

def validate(model, dataloader, accelerator):
    model.eval()
    total_loss = 0.0
    total_cnt = 0

    for batch in dataloader:
        with torch.no_grad():
            query_emb, doc_emb = model(**batch)

        query_num = query_emb.size(0)

        # 如果你想在 eval 时也做多卡 all_gather（和训练同逻辑），可以按训练那段来：
        if accelerator.use_distributed and accelerator.num_processes > 1:
            single_device_query_num = query_emb.size(0)
            single_device_doc_num = doc_emb.size(0)

            # all_gather docs
            doc_list = [torch.zeros_like(doc_emb) for _ in range(accelerator.num_processes)]
            dist.all_gather(tensor_list=doc_list, tensor=doc_emb.contiguous())
            doc_list[dist.get_rank()] = doc_emb
            doc_emb = torch.cat(doc_list, dim=0)

            # all_gather queries
            query_list = [torch.zeros_like(query_emb) for _ in range(accelerator.num_processes)]
            dist.all_gather(tensor_list=query_list, tensor=query_emb.contiguous())
            query_list[dist.get_rank()] = query_emb
            query_emb = torch.cat(query_list, dim=0)

            # labels 的写法与训练对齐
            labels = torch.cat([
                torch.arange(single_device_query_num) + gpu_idx * single_device_doc_num
                for gpu_idx in range(accelerator.num_processes)
            ], dim=0).to(query_emb.device)
        else:
            # 单机/单卡 eval：batch 内 in-batch negative
            labels = torch.arange(query_emb.size(0), device=query_emb.device)

        scores = torch.matmul(query_emb, doc_emb.T)  # [Q, D]
        loss = calculate_dpr_loss(scores, labels)

        total_loss += loss.item() * len(labels)
        total_cnt += len(labels)

    # 分布式下再聚合  loss（都是标量或小 list，不会炸内存）
    if accelerator.use_distributed and accelerator.num_processes > 1:

        loss_from_all = [None for _ in range(accelerator.num_processes)]
        cnt_from_all = [None for _ in range(accelerator.num_processes)]
        dist.all_gather_object(loss_from_all, total_loss)
        dist.all_gather_object(cnt_from_all, total_cnt)
        total_loss = sum(loss_from_all)
        total_cnt = sum(cnt_from_all)

    avg_loss = total_loss / total_cnt

    return avg_loss

def train(args):
    set_seed(args.seed)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision='no',
        kwargs_handlers=[kwargs]
    )

    LOG_DIR = args.save_dir

    tokenizer = BertTokenizer.from_pretrained(args.base_model)
    share_encoder = getattr(args, 'share_encoder', False)
    dual_encoder = DualEncoder(base_model=args.base_model, share_encoder=share_encoder)
    dual_encoder.train()


    full_dataset = DPRDataset(args.train_file)
    
    n_total = len(full_dataset)
    n_dev = max(1, int(n_total * 0.01))
    n_train = n_total - n_dev
    
    train_dataset, dev_dataset = random_split(
        full_dataset,
        [n_train, n_dev],
        generator=torch.Generator().manual_seed(42)
    )
    collate_fn = functools.partial(DPRDataset.collate_fn,tokenizer=tokenizer,args=args)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=args.per_device_train_batch_size,shuffle=True,collate_fn=collate_fn,num_workers=4,pin_memory=True)
    
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset,batch_size=args.per_device_eval_batch_size,shuffle=False,collate_fn=collate_fn,num_workers=4,pin_memory=True)

    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in dual_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in dual_encoder.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters,lr=args.lr, eps=args.adam_eps)
    
    dual_encoder, optimizer, train_dataloader, dev_dataloader = accelerator.prepare(
        dual_encoder, optimizer, train_dataloader, dev_dataloader,
    )
    
    NUM_UPDATES_PER_EPOCH = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    MAX_TRAIN_STEPS = NUM_UPDATES_PER_EPOCH * args.max_train_epochs
    MAX_TRAIN_EPOCHS = math.ceil(MAX_TRAIN_STEPS / NUM_UPDATES_PER_EPOCH)
    TOTAL_TRAIN_BATCH_SIZE = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    lr_scheduler = get_linear_scheduler(optimizer,warmup_steps=args.warmup_steps,total_training_steps=MAX_TRAIN_STEPS)

    logger.info("***** Running training *****")
    logger.info(f"  Num train examples = {len(train_dataset)}")
    logger.info(f"  Num dev examples = {len(dev_dataset)}")
    logger.info(f"  Num Epochs = {MAX_TRAIN_EPOCHS}")
    logger.info(f"  Per device train batch size = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {TOTAL_TRAIN_BATCH_SIZE}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {MAX_TRAIN_STEPS}")
    logger.info(f"  Per device eval batch size = {args.per_device_eval_batch_size}")
    progress_bar = tqdm(range(MAX_TRAIN_STEPS), disable=not accelerator.is_local_main_process,ncols=100)
    best_val_loss = float("inf")
    patience_counter = 0
    early_stop=False
    for epoch in range(MAX_TRAIN_EPOCHS):
        set_seed(args.seed+epoch)
        progress_bar.set_description(f"epoch: {epoch+1}/{MAX_TRAIN_EPOCHS}")
        for step,batch in enumerate(train_dataloader):
            with accelerator.accumulate(dual_encoder):
                with accelerator.autocast():
                    query_embedding,doc_embedding  = dual_encoder(**batch)
                    single_device_query_num,_ = query_embedding.shape
                    single_device_doc_num,_ = doc_embedding.shape
                    if accelerator.use_distributed:
                        doc_list = [torch.zeros_like(doc_embedding) for _ in range(accelerator.num_processes)]
                        dist.all_gather(tensor_list=doc_list, tensor=doc_embedding.contiguous())
                        doc_list[dist.get_rank()] = doc_embedding
                        doc_embedding = torch.cat(doc_list, dim=0)

                        query_list = [torch.zeros_like(query_embedding) for _ in range(accelerator.num_processes)]
                        dist.all_gather(tensor_list=query_list, tensor=query_embedding.contiguous())
                        query_list[dist.get_rank()] = query_embedding
                        query_embedding = torch.cat(query_list, dim=0)

                    matching_score = torch.matmul(query_embedding,doc_embedding.permute(1,0))
                    labels = torch.cat([torch.arange(single_device_query_num) + gpu_index * single_device_doc_num for gpu_index in range(accelerator.num_processes)],dim=0).to(matching_score.device)
                    loss = calculate_dpr_loss(matching_score,labels=labels)

                accelerator.backward(loss)

                ## one optimization step
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    progress_bar.set_postfix(loss=f"{loss:.4f}",lr=f"{lr_scheduler.get_last_lr()[0]:6f}")
                    accelerator.clip_grad_norm_(dual_encoder.parameters(), args.max_grad_norm)
                    if not accelerator.optimizer_step_was_skipped:
                        lr_scheduler.step()
                                  
                optimizer.step()
                optimizer.zero_grad()
        val_loss = validate(dual_encoder, dev_dataloader, accelerator)
        dual_encoder.train()
        if accelerator.is_local_main_process:
            logger.info(f"[Epoch {epoch+1}] val_loss={val_loss:.4f} (best={best_val_loss:.4f}, patience={patience_counter}/{args.patience})")
        
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                logger.info(f"[BEST] New best model at epoch {epoch+1}, saving...")
        
                unwrapped = accelerator.unwrap_model(dual_encoder)
                save_dir_q = os.path.join(LOG_DIR, "best", "query_encoder")
                save_dir_d = os.path.join(LOG_DIR, "best", "doc_encoder")
        
                os.makedirs(save_dir_q, exist_ok=True)
                os.makedirs(save_dir_d, exist_ok=True)
        
                unwrapped.query_encoder.save_pretrained(save_dir_q)
                unwrapped.doc_encoder.save_pretrained(save_dir_d)
                tokenizer.save_pretrained(save_dir_q)
                tokenizer.save_pretrained(save_dir_d)
        
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    logger.info(
                        f"[EarlyStop] Stop at epoch {epoch+1}, "
                        f"best_val_loss={best_val_loss:.4f}"
                    )
                    early_stop = True
        
        accelerator.wait_for_everyone()
        if accelerator.use_distributed and accelerator.num_processes > 1:
            # 用 object 方式广播
            obj_list = [early_stop]
            dist.broadcast_object_list(obj_list, src=0)
            early_stop = obj_list[0]
        if early_stop:
            break

def main():
    args = parse_args()
    train(args)

if __name__ == '__main__':
    main()