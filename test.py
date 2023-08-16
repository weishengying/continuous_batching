from vllm import LLM
import torch
from vllm.config import SchedulerConfig
from transformers import AutoTokenizer, AutoConfig
from vllm.sampling_params import SamplingParams

from vllm.model_executor.models.gpt2_model import GPT2LMHeadModel
torch.manual_seed(10)

gpt2_config = AutoConfig.from_pretrained('/mnt/weishengying/gpt2')
model = GPT2LMHeadModel(gpt2_config).cuda().eval()

def load_weight(model):
    from transformers import GPT2LMHeadModel
    gpt2_model = GPT2LMHeadModel.from_pretrained('/mnt/weishengying/gpt2')
    gpt2_named_parameters = {}
    for name, param in gpt2_model.named_parameters():
        gpt2_named_parameters[name] = param
        
    for name, param in model.named_parameters():
        if name == "lm_head.weight":
            param.data.copy_(gpt2_named_parameters["transformer.wte.weight"])
        else:
            param.data.copy_(gpt2_named_parameters[name])

load_weight(model)

tokenizer = AutoTokenizer.from_pretrained('/mnt/weishengying/gpt2', trust_remote_code=True)

scheduler_config = SchedulerConfig(max_num_batched_tokens=512, max_num_seqs=4, max_model_len=50)
sampling_params = SamplingParams(temperature=0, max_tokens=512)
llm = LLM(model=model, tokenizer=tokenizer, scheduler_config=scheduler_config, log_stats=False)

prompt="Life is a fucking movie"
outputs = llm.generate(prompt, sampling_params, use_tqdm=False)

for output in outputs:
    print(output.outputs[0].text)


