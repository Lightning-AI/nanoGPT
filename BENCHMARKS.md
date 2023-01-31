# Benchmarks - Fabric vs. PyTorch

## Performance

Date: Jan 19, 2023

To show that Fabric does not add any noticeable overhead, we benchmarked all code by recording the iteration speed under different settings.
Results below were obtained on a 8 x A100 GPU Lambda server. All experiments with PyTorch 2.0 nightly unless specified otherwise.

| Experiment                 | PyTorch | Fabric |
|----------------------------|---------|--------|
| Single GPU                 | 159ms   | 163ms  |
| DDP (4 GPUs)               | 168 ms  | 171 ms |
| DeepSpeed (4 GPUs) Stage 1 |         | 156 ms |
| DeepSpeed (4 GPUs) Stage 2 |         | 160 ms |



## Convergence Parity

We ran the PyTorch code and the Fabric code with the same parameters and compared the train/val loss curves in this [W&B Report](https://wandb.ai/justusschock/gpt2-fabric/reports/NanoGPT-with-Fabric--VmlldzozNDU2MTU1).

