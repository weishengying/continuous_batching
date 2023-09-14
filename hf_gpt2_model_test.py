from vllm.model_executor.models.hf_gpt2_model import GPT2LMHeadModel
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
prompt = ['Chinese cuisine comprise cuisines originating from China. Because of the Chinese diaspora and historical power of the country, Chinese cuisine has profoundly influenced many other cuisines in Asia and beyond, with modifications made to cater to local palates. Chinese food staples such as rice, soy sauce, noodles, tea, chili oil, and tofu, and utensils such as chopsticks and the wok, can now be found worldwide. The world\'s earliest eating establishments recognizable as restaurants in the modern sense first emerged in Song dynasty China during the 11th and 12th centuries. Street food became an integral aspect of Chinese food culture during the Tang dynasty, and the street food culture of much of Southeast Asia was established by coolie workers imported from China during the late 19th century. The preferences for seasoning and cooking techniques of Chinese provinces depend on differences in social class, religion, historical background, and ethnic groups. Geographic features including mountains, rivers, forests, and deserts also have a strong effect on the local available ingredients, considering that the climate of China varies from tropical in the south to subarctic in the northeast. Imperial royal and noble preference also plays a role in the change of Chinese cuisine. Because of imperial expansion and trading, ingredients and cooking techniques from other cultures have been integrated into Chinese cuisines over time. There are numerous regional, religious, and ethnic styles of Chinese cuisine found within China and abroad. Chinese cuisine is highly diverse and most frequently categorised into provincial divisions, although these province-level classifications consist of many more styles within themselves. The most praised Four Great Traditions in Chinese cuisine are Chuan, Lu, Yue, and Huaiyang, representing cuisines of West, North, South, and East China, respectively.']

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


max_len = 100
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
