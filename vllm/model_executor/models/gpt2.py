from typing import Optional, Tuple, Union, Dict, List
from vllm.model_executor.input_metadata import InputMetadata
import torch
from torch import nn

from .activations import ACT2FN

class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x


class GPT2Attention(nn.Module):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
                1, max_positions, max_positions
            ),
        )

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx

        self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

    def _attn(self, query, key, value, attention_mask=None):
        #q,k,v (head, num_tokens, head_dim)
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.tensor(
                value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)


        # if only "normal" attention layer implements causal mask
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, key_length - query_length : key_length, :key_length].to(torch.bool)
        # print(f"causal_mask: {causal_mask}")
        mask_value = torch.finfo(attn_weights.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights, mask_value)


        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, value)

        return attn_output #(head, num_tokens, head_dim)

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape) # (num_tokens, head, head_features)
        return tensor.permute(1, 0, 2)  # (head, num_tokens, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        # input: (head, num_tokens, head_dim)
        tensor = tensor.permute(1, 0, 2).contiguous() # (num_tokens, head, head_dim)
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape) # (num_tokens, embed)

    def forward(
        self,
        input_metadata: InputMetadata,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Dict[int, Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor, Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
        #q,k,v (num_tokens, embed)
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=-1)

        query = self._split_heads(query, self.num_heads, self.head_dim) # (head, num_tokens, head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim) # (head, num_tokens, head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim) # (head, num_tokens, head_dim)

        # 按不同 seq_id 的 num_tokens 切分 q k v
        present_qkv: Dict[int, List(torch.Tensor)] = {} # seq_id --> qkv
        seq_ids = []
        for seq_id, _ in input_metadata.seq_groups:
            seq_ids.extend(seq_id)
        kv_lens = []
        for prompt_len in input_metadata.prompt_lens:
            kv_lens.append(prompt_len)
        for i in range(input_metadata.num_generation_tokens):
            kv_lens.append(1)
        assert len(seq_ids) == len(kv_lens)
        
        shift = 0
        for seq_id, kv_len in zip(seq_ids, kv_lens):
            present_qkv[seq_id] = [query[:, shift : shift + kv_len, :],
                                    key[:, shift : shift + kv_len, :], 
                                    value[:, shift : shift + kv_len, :]]
            shift += kv_len
            input_metadata.sample_pos[seq_id] = shift - 1
        assert shift == query.size()[-2]
        
        for seq_id in seq_ids:
            if layer_past[seq_id] is not None:
                past_key, past_value = layer_past[seq_id]
                present_qkv[seq_id][1] = torch.cat((past_key, present_qkv[seq_id][1]), dim=-2)
                present_qkv[seq_id][2] = torch.cat((past_value, present_qkv[seq_id][2]), dim=-2)

        atten_outputs = []
        for seq_id in present_qkv.keys():
            query, key, value = present_qkv[seq_id]
            attn_output = self._attn(query, key, value, attention_mask) 
            atten_outputs.append(attn_output)
        attn_output = torch.cat(atten_outputs, dim=-2) # (head, num_tokens, head_dim)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)

        present_kv: Dict[int, Tupe(torch.Tensor)] = {} 
        for seq_id in present_qkv.keys():
            present_kv[seq_id] = tuple(present_qkv[seq_id][1:])
        outputs = (attn_output, present_kv)

        return outputs


class GPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GPT2Block(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPT2MLP(inner_dim, config)

    def forward(
        self,
        input_metadata: InputMetadata,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        residual = hidden_states # (num_tokens, embed)
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            input_metadata,
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
        ) #(hidden_state, dict(int, (cache_k_tensor, cache_v_tensor)))
        attn_output = attn_outputs[0] 
        cache_kv = attn_outputs[1]
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        outputs = (hidden_states, cache_kv)

        return outputs


class GPT2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

    def forward(
        self,
        input_metadata: InputMetadata,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Dict[int, Tuple[Tuple[torch.Tensor]]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]:
        input_shape = input_ids.size()

        device = input_ids.device

        if past_key_values is None:
            past_key_values = dict() # seq_id --> cache kv
            
        for seq_ids, _ in input_metadata.seq_groups:
            assert len(seq_ids) == 1 # assert greedy sample
            if seq_ids[0] not in list(past_key_values.keys()):
                past_key_values[seq_ids[0]] = tuple([None] * len(self.h))

        deleted_seq_id = []
        for seq_id in past_key_values.keys():
            if seq_id  not in input_metadata.seq_data.keys():
                deleted_seq_id.append(seq_id)
        for seq_id in deleted_seq_id:
            del past_key_values[seq_id] # 删除已经完成的 seq 的 cache kv

        inputs_embeds = self.wte(input_ids) # (num_tokens, embed)
        position_embeds = self.wpe(position_ids) # (num_tokens, embed)
        hidden_states = inputs_embeds + position_embeds # (num_tokens, embed)

        output_shape = input_shape + (hidden_states.size(-1),)

        cache_kvs = dict() # seq_id --> cache kv
        for seq_id in past_key_values.keys():
            cache_kvs[seq_id] = ()

        for i, block in enumerate(self.h):
            layer_past = {}
            for seq_id in past_key_values.keys():
                layer_past[seq_id] = past_key_values[seq_id][i]
            outputs = block(
                input_metadata,
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
            )
            hidden_states = outputs[0] # output: (hidden_states, cache_kv)
            for seq_id in cache_kvs.keys():
                cache_kvs[seq_id] = cache_kvs[seq_id] + (outputs[1][seq_id], ) # {seq_id : (cache_k, cache_v), (cache_k, cache_v), ...}

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)

        return (hidden_states, cache_kvs) #(hidden_states, presents)


class GPT2LMHeadModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(
        self,
        input_metadata: InputMetadata,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Dict[int, Tuple[Tuple[torch.Tensor]]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Dict[int, Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]]:
        # input_ids : (num_tokens)
        transformer_outputs = self.transformer(
            input_metadata,
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = transformer_outputs[0] # (num_tokens, embed)

        lm_logits = self.lm_head(hidden_states) # (num_tokens, vocab_size)

        return (lm_logits, transformer_outputs[1]) # logits, cache_kvs



