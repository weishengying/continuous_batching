from gpt2_model import GPT2LMHeadModel
from transformers import AutoConfig, AutoTokenizer
import torch

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

prompt="Life is a fucking movie"
max_len = 50
input_ids = tokenizer(prompt, return_tensors="pt")['input_ids'].cuda()
while torch.numel(input_ids) < max_len:
    # forward step
    output = model.forward(input_ids = input_ids, use_cache=False)
    # greedy sample
    next_token = torch.argmax(output.logits[:,-1,:], dim=-1)
    if next_token == gpt2_config.eos_token_id: # eos_token
        break
    # concat input
    input_ids = torch.cat([input_ids, next_token[:, None]], dim=-1)
print("output_text: ", tokenizer.batch_decode(input_ids.cpu().numpy()))
