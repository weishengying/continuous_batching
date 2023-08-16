本仓库用于快速验证一些想法，如实现 continuous batching， paged_attention, tokne_attention等。

代码调度上完全参考 vllm，后续会考虑借鉴 light-llm 的调度

模型上以 gpt2 为例子，目前 model_exectutor/models/gpt2_model 实现的 gpt2 模型完全仿照 HF 实现，为了适配 

continuous batching 的输入，模型结构会有一定的修改。