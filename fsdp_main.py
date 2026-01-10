import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import functools
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from types import SimpleNamespace
from collections import OrderedDict
from datasets import load_from_disk
from transformers import GPT2TokenizerFast
from torch.nn.attention import SDPBackend
import wandb
from dotenv import load_dotenv
import pynvml

load_dotenv()

def get_all_ranks_gpu_metrics(local_rank, world_size, device):
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**2
    except Exception:
        util, mem = 0.0, 0.0

    local_metrics = torch.tensor([float(util), float(mem)], device=device)
    gathered_metrics = [torch.zeros(2, device=device) for _ in range(world_size)]
    
    dist.all_gather(gathered_metrics, local_metrics)
    
    metrics_dict = {}
    if dist.get_rank() == 0:
        for rank_idx, data in enumerate(gathered_metrics):
            metrics_dict[f"gpu_util/rank_{rank_idx}"] = data[0].item()
            metrics_dict[f"gpu_mem_mb/rank_{rank_idx}_mb"] = data[1].item()
            
    return metrics_dict
    
class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len):
        super(EmbeddingLayer, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)

    def forward(self, x):
        seq_len = x.size(1)
        positions = (
            torch.arange(seq_len, dtype=torch.long, device=x.device)
            .unsqueeze(0)
            .expand_as(x)
        )
        token_embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding(positions)
        embeddings = token_embeddings + position_embeddings
        return embeddings


class AttentionLayer(nn.Module):
    def __init__(self, dmodel, heads):
        super(AttentionLayer, self).__init__()
        self.ln = nn.LayerNorm(dmodel)
        self.heads = heads
        self.input_projection = nn.Linear(dmodel, 3 * dmodel, bias=False)
        self.output_projection = nn.Linear(dmodel, dmodel, bias=False)

    def forward(self, x, attention_mask):
        x = self.ln(x)
        projected = self.input_projection(x)
        batch, seq_len = x.shape[:-1]
        q_chunk, k_chunk, v_chunk = torch.chunk(projected, chunks=3, dim=-1)
        query = q_chunk.view(batch, seq_len, self.heads, -1).transpose(1, 2)
        key = k_chunk.view(batch, seq_len, self.heads, -1).transpose(1, 2)
        value = v_chunk.view(batch, seq_len, self.heads, -1).transpose(1, 2)

        with torch.nn.attention.sdpa_kernel(
            [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]
        ):
            attention_output = F.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=attention_mask,
                is_causal=True,
            )
        output = self.output_projection(attention_output.transpose(1, 2).flatten(-2))
        return output


def FeedForward(dmodel):
    return nn.Sequential(
        OrderedDict(
            [
                ("ff_layernorm", nn.LayerNorm(dmodel)),
                ("pre_relu", nn.Linear(dmodel, 4 * dmodel, bias=True)),
                ("relu", nn.ReLU()),
                ("post_relu", nn.Linear(4 * dmodel, dmodel, bias=True)),
            ]
        )
    )


class Block(nn.Module):
    def __init__(self, dmodel, heads):
        super().__init__()
        self.attention_layer = AttentionLayer(dmodel, heads)
        self.feed_forward_layer = FeedForward(dmodel)

    def forward(self, x, attention_mask):
        out_attention = self.attention_layer(x, attention_mask)
        x = x + out_attention
        out_feed_forward = self.feed_forward_layer(x)
        x = x + out_feed_forward
        return x


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding_layer = EmbeddingLayer(
            config.vocab_size, config.d_model, config.max_len
        )
        self.blocks = nn.ModuleList(
            [Block(config.d_model, config.num_heads) for _ in range(config.num_layers)]
        )
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        output = self.embedding_layer(input_ids)
        for block in self.blocks:
            output = block(output, attention_mask)
        output = self.head(output)
        return output


def collate_tokenize(tokenizer, sequence_length, data):
    text_batch = [element["text"] for element in data]
    tokenized = tokenizer(
        text_batch,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=sequence_length + 1,
    )
    input_ids = tokenized["input_ids"]
    tokenized["input_ids"] = input_ids[:, :-1]
    tokenized["target_ids"] = input_ids[:, 1:]
    tokenized["attention_mask"] = tokenized["attention_mask"][:, :-1]
    return tokenized


def get_dataloader(batch_size, sequence_length, rank, world_size, split="train"):
    if split == "train":
        hf_dataset = load_from_disk(
            "/net/tscratch/people/plgjkrajewski/datasets/c4/train"
        )
    else:
        hf_dataset = load_from_disk(
            "/net/tscratch/people/plgjkrajewski/datasets/c4/validation"
        )

    sampler = DistributedSampler(
        hf_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=(split == "train"),
        seed=42,
    )

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataloader = DataLoader(
        hf_dataset,
        batch_size=batch_size,
        collate_fn=functools.partial(collate_tokenize, tokenizer, sequence_length),
        sampler=sampler,
        pin_memory=True,
        num_workers=2,
    )
    return dataloader


def setup():
    dist.init_process_group("nccl")


def cleanup():
    dist.destroy_process_group()


def calculate_valid_loss(model, valid_dataloader, device, validation_steps):
    model.eval()
    valid_losses = []
    
    for _, batch in zip(range(validation_steps), valid_dataloader):
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            attention_mask = batch["attention_mask"]
            
            outputs = model(input_ids)
            
            mask_loss = F.cross_entropy(
                outputs.flatten(0, -2),
                target_ids.reshape(-1).long(),
                reduction="none",
            )
            mask_loss = mask_loss[attention_mask.reshape(-1) == 1]
            loss = mask_loss.mean().item()
            valid_losses.append(loss)
            
    if len(valid_losses) > 0:
        local_mean_valid_loss = sum(valid_losses) / len(valid_losses)
    else:
        local_mean_valid_loss = 0.0

    loss_tensor = torch.tensor(local_mean_valid_loss, device=device)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    
    world_size = int(os.environ["WORLD_SIZE"])
    mean_valid_loss = loss_tensor.item() / world_size

    model.train()
    return mean_valid_loss

def train_model(config, args):
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    dataloader = get_dataloader(
        config.batch_size, config.seq_length, global_rank, world_size, "train"
    )
    valid_dataloader = get_dataloader(
        config.batch_size,
        config.seq_length,
        global_rank,
        world_size,
        split="validation",
    )

    validation_steps  = int(
        1e06 // (config.batch_size * config.seq_length * world_size) 
    )

    bf16_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    model = Transformer(config)

    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=20000
    )

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=bf16_policy,
        device_id=device,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
    )

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    model.train()

    if global_rank == 0:
        job_name = os.getenv("SLURM_JOB_NAME", "local-run")

        job_id = os.getenv("SLURM_JOB_ID", "")
        if job_id:
            job_name = f"{job_name}-{job_id}"

        wandb.init(project=os.getenv("WANDB_PROJECT"), name=job_name, config=vars(args))
        print(f"Starting FSDP Training on {world_size} GPUs...")

    for i, batch in enumerate(dataloader):
        if i >= config.train_steps:
            break

        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        attention_mask = batch["attention_mask"]

        optimizer.zero_grad()
        outputs = model(input_ids)

        mask_loss = F.cross_entropy(
            outputs.flatten(0, -2),
            target_ids.reshape(-1).long(),
            reduction="none",
        )
        mask_loss = mask_loss[attention_mask.reshape(-1) == 1]
        loss = mask_loss.mean()
        
        if i % config.log_val_loss_freq == 0:     
            val_loss = calculate_valid_loss(model, valid_dataloader, device, validation_steps)
            if global_rank == 0:
                print(f"Step:{i}, Valid loss:{val_loss}")
                wandb.log({"valid_loss": val_loss}, step=i)
        if i % config.log_train_loss_freq == 0:
            all_gpu_stats = get_all_ranks_gpu_metrics(local_rank, world_size, device)  
            if global_rank == 0:
                print(f"Step:{i}, Train Loss:{loss}")
                wandb.log({"train_loss": loss.item(),**all_gpu_stats}, step=i)
                

        loss.backward()
        optimizer.step()

    if global_rank == 0:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--dmodel", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_training_steps", type=int, default=1000)
    args = parser.parse_args()

    config = SimpleNamespace(
        train_steps=args.n_training_steps,
        vocab_size=50257,
        max_len=256,
        d_model=args.dmodel,
        num_heads=args.n_heads,
        num_layers=args.n_layers,
        learning_rate=1e-4,
        seq_length=256,
        batch_size=args.batch_size,
        log_train_loss_freq=25,
        log_val_loss_freq=100,
    )

    setup()
    train_model(config, args)
    cleanup()


if __name__ == "__main__":
    main()
