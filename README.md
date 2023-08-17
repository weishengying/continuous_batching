本仓库用于学习和快速验证一些想法，如实现 continuous batching， paged_attention, tokne_attention等。

代码调度上完全参考 vllm，后续会考虑借鉴 light-llm 的调度

模型上以 gpt2 为例子，目前 model_exectutor/models/目录下 实现的 gpt2_model.py 中的 gpt2 模型完全仿照 HF 实现，计算流程是一样的。

gpt2.py 中的模型为了支持 continuous batching 的输入，重新定义了模型的计算流程。

运行 `python llm_test.py` 便是使用 continuous batching，输入不需要 padding
运行 `python gpt2_model_test.py` 是 static bathing，如果是多 batch, 需要 tokenizer padding

另外， gpt2.py 的实现是最基础最简单的实现， attention 计算效率很低，后面会参考 vllm 的 multi_query_attention 和 single_query_attention 做优化，此外，cache kv 管理上，没有使用 vllm 类似的分快管理，因此可能会有很多显存碎片。