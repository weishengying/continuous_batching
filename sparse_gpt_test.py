from vllm.model_executor.models.sparse_gpt2_model import GPT2LMHeadModel
from transformers import AutoConfig, AutoTokenizer
import torch
import time

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

prompt=["Title: The Beauty of Nature. Introduction: Nature is a magnificent gift that surrounds us every day. From the gentle breeze rustling through the trees to the vibrant colors of blooming flowers, nature has a unique way of captivating our senses and bringing joy to our lives. Body:One of the most awe-inspiring aspects of nature is its diverse landscapes. From the majestic mountains to the vast oceans, each place offers its own beauty and charm. The towering peaks covered in glistening snow, the crystal-clear lakes reflecting the sky above, and the golden beaches kissed by the gentle waves all remind us of the wonders that exist beyond our daily routines. Moreover, nature provides a sense of tranquility and inner peace. Taking a stroll through a lush green forest, listening to the birds singing their melodic tunes, and feeling the warmth of the sun on your skin can rejuvenate both the body and soul. In nature, we can find solace, free our minds from worries, and reconnect with ourselves. Furthermore, nature is a teacher like no other. Observing the delicate balance of ecosystems, the perseverance of plants to grow against all odds, and the symbiotic relationships between different species, we learn invaluable lessons about resilience, harmony, and coexistence. Nature reminds us of the preciousness of life and the importance of cherishing and protecting our environment. Conclusion: In a world filled with technological advancements and fast-paced lifestyles, it is essential to take a step back and appreciate the beauty that nature offers us. Whether it's a breathtaking sunset, a radiant rainbow after a storm, or simply the gentle touch of a breeze, nature has the power to uplift our spirits and remind us of the wonders of life." * 3]
prompt=["a " * 900]
# max_len = 50
# input_ids = tokenizer(prompt, return_tensors="pt")['input_ids'].cuda()
# while torch.numel(input_ids) < max_len:
#     # forward step
#     output = model.forward(input_ids = input_ids, use_cache=False)
#     # greedy sample
#     next_token = torch.argmax(output.logits[:,-1,:], dim=-1)
#     if next_token == gpt2_config.eos_token_id: # eos_token
#         break
#     # concat input
#     input_ids = torch.cat([input_ids, next_token[:, None]], dim=-1)
# print("output_text: ", tokenizer.batch_decode(input_ids.cpu().numpy()))


max_len = 10
cnt = 0
start = time.time()
repeat_times = 1
for _ in range(repeat_times):
    input_ids = tokenizer(prompt, return_tensors="pt")['input_ids'].cuda()
    print(input_ids.size())
    cache_kv = None
    generate = None
    while cnt < max_len:
        # forward step
        output = model.forward(input_ids = input_ids, past_key_values=cache_kv) #(logits, cache_kvs)
        # greedy sample
        next_token = torch.argmax(output[0][:,-1,:], dim=-1)
        cnt += 1
        # cache_kv
        cache_kv = output[1]
        # concat output
        if generate is not None:
            generate = torch.cat([generate, next_token[:, None]], dim=-1)
        else:
            generate = next_token[:, None]
        # get last new token as input
        input_ids = generate[:, -1].unsqueeze(-1)
    output_text = tokenizer.batch_decode(generate.cpu().numpy())
    print("output_text: ", output_text)
    cnt = 0

end = time.time()
print(f"time: {(end - start) / repeat_times * 1000} ms")
