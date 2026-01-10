from functools import partial
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from types import SimpleNamespace
from torch.optim import AdamW
import torch.nn.functional as F
from torch.nn.attention import SDPBackend
from collections import OrderedDict
from datasets import load_dataset, load_from_disk
from transformers import GPT2TokenizerFast
import argparse
import wandb
import os
from dotenv import load_dotenv

load_dotenv()

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len):
        super(EmbeddingLayer, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)

    def forward(self, x):
        # x: (batch_size, seq_len)
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
    def __init__(
        self,
        dmodel,
        heads,
    ):
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


def FeedForward(
    dmodel,
):
    return nn.Sequential(
        OrderedDict(
            [
                ("ff_layernorm", nn.LayerNorm(dmodel)),
                (
                    "pre_relu",
                    nn.Linear(
                        dmodel,
                        4 * dmodel,
                        bias=True,
                    ),
                ),
                ("relu", nn.ReLU()),
                (
                    "post_relu",
                    nn.Linear(
                        4 * dmodel,
                        dmodel,
                        bias=True,
                    ),
                ),
            ]
        )
    )


class Block(nn.Module):

    def __init__(
        self,
        dmodel,
        heads,
    ):
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


def get_dataloader(
    batch_size,
    sequence_length,
    split="train",
    buffer_size=10000,
    seed=42,
    num_workers=2,
):
    if split == "train":
        hf_dataset = load_from_disk(
            "/net/tscratch/people/plgjkrajewski/datasets/c4/train"
        )
    else:
        hf_dataset = load_from_disk(
            "/net/tscratch/people/plgjkrajewski/datasets/c4/validation"
        )
    hf_dataset = hf_dataset.to_iterable_dataset(num_shards=64)
    hf_dataset = hf_dataset.shuffle(buffer_size=buffer_size, seed=seed)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataloader = DataLoader(
        hf_dataset,
        batch_size=batch_size,
        collate_fn=partial(collate_tokenize, tokenizer, sequence_length),
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )
    return dataloader


def calculate_valid_loss(model, valid_dataloader, device, validation_steps):
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
            mean_valid_loss = sum(valid_losses) / validation_steps
    return mean_valid_loss


def train_model(config, device):
    dataloader = get_dataloader(config.batch_size, config.seq_length)
    valid_dataloader = get_dataloader(
        config.batch_size, config.seq_length, split="validation"
    )
    validation_steps = int(
        1e06 // (config.batch_size * config.seq_length)
    )  # we want to evaluate on 1M tokens
    model = Transformer(config)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    model.train()
    
    wandb.watch(model, log="all", log_freq=config.log_train_loss_freq)

    for i, batch in zip(range(config.train_steps), dataloader):
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

        if i % config.log_train_loss_freq == 0:
            print(f"Step:{i}, Train Loss:{loss}")
            wandb.log({"train_loss": loss.item()}, step=i)

        if i % config.log_train_loss_freq == 0:
            val_loss = calculate_valid_loss(model, valid_dataloader, device, validation_steps)
            print(f"Step:{i}, Valid loss:{val_loss}")
            wandb.log({"valid_loss": val_loss}, step=i)

        loss.backward()
        optimizer.step()

    print(
        f"Final valid loss:{calculate_valid_loss(model, valid_dataloader, device, validation_steps)}"
    )


def main():
    parser = argparse.ArgumentParser(description="Transformer Training Script")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--dmodel", type=int, default=256, help="Embedding dimension size")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--n_training_steps", type=int, default=1000, help="Total training steps")
    
    args = parser.parse_args()

    job_name = os.getenv("SLURM_JOB_NAME", "local-run")

    job_id = os.getenv("SLURM_JOB_ID", "")
    if job_id:
        job_name = f"{job_name}-{job_id}"
        
    wandb.init(
        project=os.getenv("WANDB_PROJECT"),
        name=job_name,
        config=vars(args)
    )

    config = SimpleNamespace(
        train_steps=args.n_training_steps,
        vocab_size=50257,
        max_len=256,
        d_model=args.dmodel,
        num_heads=args.n_heads,
        num_layers=args.n_layers,
        learning_rate=1e-4,
        dropout=0.0,
        seq_length=256,
        batch_size=args.batch_size,
        log_train_loss_freq=50,
        log_valid_loss_freq=50
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print(f"Device type is: {device}. Remember to train on GPU.")
        
    train_model(config, device)
    wandb.finish()


if __name__ == "__main__":
    main()
