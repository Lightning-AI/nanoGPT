"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ lightning run model --accelerator=cuda --precision=bf16 train_fabric.py

To run with DDP on 4 gpus on 1 node, example:
$ lightning run model --accelerator=cuda --precision=bf16 --devices=4 train_fabric.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ lightning run model --accelerator=cuda --precision=bf16 --devices=8 --num_nodes=2 --node_rank=0 --main_address=123.456.123.456 --main_port=1234 train_fabric.py
- Run on the worker node:
$ lightning run model --accelerator=cuda --precision=bf16 --devices=8 --num_nodes=2 --node_rank=1 --main_address=123.456.123.456 --main_port=1234 train_fabric.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)

Try also --strategy="deepspeed" with devices > 1.
"""

import os
import time
import math
import pickle

import numpy as np
import torch
from lightning.fabric import Fabric

from model import GPTConfig, GPT

from torch.profiler import profile, record_function, ProfilerActivity


# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 1 # used to simulate larger batch sizes
batch_size = 4 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 50 # total number of training iterations
weight_decay = 1e-2
beta1 = 0.9
beta2 = 0.95
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# system
compile = False # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

fabric = Fabric()

master_process = fabric.global_rank == 0

if master_process:
    os.makedirs(out_dir, exist_ok=True)

fabric.seed_everything(1337 + fabric.global_rank)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

# poor man's data loader, TODO evaluate need for actual DataLoader
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    x, y = fabric.to_device((x, y))
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    vocab_size = meta['vocab_size']
    print(f"vocab_size = {vocab_size} (from {meta_path})")
else:
    print(f"vocab_size not found in {meta_path}, using GPT-2 default of 50257")
    vocab_size = 50257

# model init
model_args = dict(n_layer = n_layer, n_head = n_head, n_embd = n_embd, block_size = block_size, dropout = dropout, vocab_size = vocab_size)
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)


if block_size < model.config.block_size:
    model.crop_block_size(block_size)


# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2))
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])


# setup according to the precision, accelerator and strategy passed in
# to the Fabric constructor, that is:
# 1. move model and optimizer to the chosen device
# 2. prepare the model for the chosen precision
# 3. wrap the model according to the chosen strategy
model, optimizer = fabric.setup(model, optimizer)

timings = []
# with profile(activities=[ProfilerActivity.CUDA]) as prof:

while True:
    t0 = time.perf_counter()

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    optimizer.zero_grad(set_to_none=True)
    for micro_step in range(gradient_accumulation_steps):
        X, Y = get_batch('train')
        with fabric.no_backward_sync(model, enabled=(micro_step < gradient_accumulation_steps - 1)):
            # with record_function("forward"):
            logits, loss = model(X, Y)
            # with record_function("backward"):
            fabric.backward(loss)

    # with record_function("optimizer_step"):
    optimizer.step()

    t1 = time.perf_counter()
    timings.append(t1 - t0)

    # termination conditions
    if iter_num > max_iters:
        break

    iter_num += 1

fabric.print("iter time", torch.mean(torch.tensor(timings)).item())
# fabric.print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))
