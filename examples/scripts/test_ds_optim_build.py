import deepspeed

deepspeed.ops.adam.cpu_adam.CPUAdamBuilder().load()
deepspeed.ops.adam.fused_adam.FusedAdamBuilder().load()
