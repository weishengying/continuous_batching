from typing import Optional, Tuple, Union

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
                1, 1, max_positions, max_positions
            ),
            # DIFFERENT 
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
            # self.c_attn = get_lora_linear(2 * self.embed_dim, self.embed_dim, config)
            # self.q_attn = get_lora_linear(self.embed_dim, self.embed_dim, config)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
            # self.c_attn = get_lora_linear(3 * self.embed_dim, self.embed_dim, config)

        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)
        # self.c_proj = get_lora_linear(self.embed_dim, self.embed_dim, config)


        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.num_heads, self.head_dim, self.pruned_heads)
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # Update hyper params
        self.split_size = (self.split_size // self.num_heads) * (self.num_heads - len(heads))
        self.num_heads = self.num_heads - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn_first(self, query, key, value, attention_mask=None, head_mask=None):
        # (batch, head, seq_length, head_features)
        batch_size = query.size(0)
        num_head = query.size(1)
        query_len = query.size(2)
        d_head = query.size(3)
        block_len = 512
        num_blocks = query_len // block_len
        num_global = 16
        global_blocks_len = block_len // num_global

        # (batch_size, num_head, num_blocks, block_len, d_head)
        first_q = query[..., :query_len-(query_len % block_len), :].view(
            batch_size, num_head, num_blocks, block_len, d_head)
        first_k_local = key[..., :query_len-(query_len % block_len), :].view(
            batch_size, num_head, num_blocks, block_len, d_head)
        first_v_local = value[..., :query_len-(query_len % block_len), :].view(
            batch_size, num_head, num_blocks, block_len, d_head)

        mask_num_head = attention_mask.size(1)
        mask_batch_size = attention_mask.size(0)
        # (batch_size, num_head, num_blocks, block_len, block_len)
        first_mask_local = attention_mask[..., :query_len-(query_len % block_len), :query_len-(query_len % block_len)].view(
            mask_batch_size, mask_num_head, num_blocks, block_len, num_blocks, block_len)
        first_mask_local = torch.diagonal(first_mask_local, dim1=2, dim2=-2).transpose(-2, -1).transpose(-2, -3)

        # batch_size, num_head, num_blocks, block_len, global_num_blocks
        mask_stride = attention_mask[..., :query_len-(query_len % block_len), :query_len-(query_len % global_blocks_len)].view(
            mask_batch_size, mask_num_head, num_blocks, block_len, -1, global_blocks_len)[...,-1]

        # (batch_size, num_head, num_blocks, block_len, num_global_blocks + block_len)
        first_mask = torch.cat((mask_stride, first_mask_local), dim=-1)

        # (batch_size, num_head, 1, num_global_blocks, d_head)
        k_stride = key[..., :query_len-(query_len % global_blocks_len), :].view(
            batch_size, num_head, -1, global_blocks_len, d_head)[:, :, :, -1:, :].transpose(-2,-3)
        v_stride = value[..., :query_len-(query_len % global_blocks_len), :].view(
            batch_size, num_head, -1, global_blocks_len, d_head)[:, :, :, -1:, :].transpose(-2,-3)

        # (batch_size, num_head, num_blocks, num_global_blocks + block_len, d_head)
        first_k = torch.cat((torch.tile(k_stride, (1, 1, num_blocks, 1, 1)), first_k_local), dim=-2)
        first_v = torch.cat((torch.tile(v_stride, (1, 1, num_blocks, 1, 1)), first_v_local), dim=-2)

        # (batch_size, num_head, num_blocks, block_len, d_head)
        # (batch_size, num_head, num_blocks, block_len, num_global_blocks + block_len)
        attn_output, attn_weights = self._attn(query=first_q,
                                               key=first_k,
                                               value=first_v,
                                               attention_mask=first_mask,
                                               head_mask=head_mask)
        attn_output = attn_output.view(batch_size, num_head, num_blocks*block_len, d_head)

        return attn_output, attn_weights

    def _attn_second(self, query, key, value, attention_mask=None, head_mask=None):
        # (batch, head, seq_length, head_features)
        batch_size = query.size(0)
        num_head = query.size(1)
        query_len = query.size(2)
        d_head = query.size(3)
        block_len = 512
        remain_len = query_len % block_len
        num_blocks = query_len // block_len
        num_global = 16
        global_blocks_len = block_len // num_global

        # (batch_size, num_head, remain_len, d_head)
        second_q = query[..., -remain_len:, :].contiguous().view(
            batch_size, num_head, -1, d_head)
        second_k_local = key[..., -remain_len:, :].contiguous().view(
            batch_size, num_head, -1, d_head)
        second_v_local = value[..., -remain_len:, :].contiguous().view(
            batch_size, num_head, -1, d_head)

        mask_num_head = attention_mask.size(1)
        mask_batch_size = attention_mask.size(0)
        # (batch_size, num_head, remain_len, remain_len)
        first_mask_local = attention_mask[..., -remain_len:, -remain_len:]

        # (batch_size, num_head, remain_len, global_num_blocks)
        mask_stride = attention_mask[..., -remain_len:, :query_len-(query_len % global_blocks_len)].contiguous().view(
            mask_batch_size, mask_num_head, remain_len, -1, global_blocks_len)[..., -1]

        # (batch_size, num_head, remain_len, num_global_blocks + remain_len)
        second_mask = torch.cat((mask_stride, first_mask_local), dim=-1)

        # (batch_size, num_head, num_global_blocks, d_head)
        k_stride = key[..., :query_len-(query_len % global_blocks_len), :].contiguous().view(
            batch_size, num_head, -1, global_blocks_len, d_head)[:, :, :, -1, :]
        v_stride = value[..., :query_len-(query_len % global_blocks_len), :].contiguous().view(
            batch_size, num_head, -1, global_blocks_len, d_head)[:, :, :, -1, :]

        # (batch_size, num_head, num_global_blocks + remain_len, d_head)
        second_k = torch.cat((k_stride, second_k_local), dim=-2)
        second_v = torch.cat((v_stride, second_v_local), dim=-2)

        # (batch_size, num_head, remain_len, d_head)
        # (batch_size, num_head, remain_len, num_global_blocks + remain_len)
        attn_output, attn_weights = self._attn(query=second_q,
                                               key=second_k,
                                               value=second_v,
                                               attention_mask=second_mask,
                                               head_mask=head_mask)

        return attn_output, attn_weights

    def _sparse_attn_without_kv_cache(self, query, key, value, attention_mask, head_mask):
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length: key_length, :key_length].bool()
        if attention_mask is not None:
            attention_mask = torch.where(causal_mask, attention_mask,
                                         attention_mask + self.masked_bias.to(attention_mask.dtype))
        else:
            attention_mask = torch.where(causal_mask, 0.0, self.masked_bias.to(query.dtype))
        query_len = query.size(2)
        block_len = 512
        remain_len = query_len % block_len
        num_blocks = query_len // block_len

        if num_blocks > 0 and remain_len > 0:
            attn_output_first, _ = self._attn_first(query, key, value, attention_mask, head_mask)
            attn_output_second, _ = self._attn_second(query, key, value, attention_mask, head_mask)
            return torch.cat((attn_output_first, attn_output_second), dim=-2).contiguous(), None
        elif num_blocks > 0:
            attn_output_first, _ = self._attn_first(query, key, value, attention_mask, head_mask)
            return attn_output_first, None
        else:
            attn_output_second, _ = self._attn_second(query, key, value, attention_mask, head_mask)
            return attn_output_second, None

    def _sparse_attn_with_kv_cache(self, query, key, value, attention_mask, head_mask):
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length: key_length, :key_length].bool()
        if attention_mask is not None:
            attention_mask = torch.where(causal_mask, attention_mask,
                                         attention_mask + self.masked_bias.to(attention_mask.dtype))
        else:
            attention_mask = torch.where(causal_mask, 0.0, self.masked_bias.to(query.dtype))
        batch_size = query.size(0)
        num_head = query.size(1)
        query_len = key.size(2)
        d_head = query.size(3)
        block_len = 512
        remain_len = query_len % block_len
        num_blocks = query_len // block_len
        num_global = 16
        global_blocks_len = block_len // num_global

        # (batch_size, num_head, 1, d_head)
        second_q = query.view(batch_size, num_head, 1, d_head)

        # (batch_size, num_head, remain_len, d_head)
        second_k_local = key[..., -remain_len:, :].contiguous().view(
            batch_size, num_head, -1, d_head)
        second_v_local = value[..., -remain_len:, :].contiguous().view(
            batch_size, num_head, -1, d_head)

        mask_num_head = attention_mask.size(1)
        mask_batch_size = attention_mask.size(0)
        # (batch_size, num_head, 1, remain_len)
        first_mask_local = attention_mask[..., -remain_len:]

        # (batch_size, num_head, 1, global_num_blocks)
        mask_stride = attention_mask[..., :query_len-(query_len % global_blocks_len)].contiguous().view(
            mask_batch_size, mask_num_head, 1, -1, global_blocks_len)[..., -1]

        # (batch_size, num_head, 1, num_global_blocks + remain_len)
        second_mask = torch.cat((mask_stride, first_mask_local), dim=-1)

        # (batch_size, num_head, num_global_blocks, d_head)
        k_stride = key[..., :query_len-(query_len % global_blocks_len), :].contiguous().view(
            batch_size, num_head, -1, global_blocks_len, d_head)[:, :, :, -1, :]
        v_stride = value[..., :query_len-(query_len % global_blocks_len), :].contiguous().view(
            batch_size, num_head, -1, global_blocks_len, d_head)[:, :, :, -1, :]

        # (batch_size, num_head, num_global_blocks + remain_len, d_head)
        second_k = torch.cat((k_stride, second_k_local), dim=-2)
        second_v = torch.cat((v_stride, second_v_local), dim=-2)

        # (batch_size, num_head, 1, d_head)
        # (batch_size, num_head, remain_len, num_global_blocks + remain_len)
        attn_output, attn_weights = self._attn(query=second_q,
                                               key=second_k,
                                               value=second_v,
                                               attention_mask=second_mask)

        return attn_output, None

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # (batch, head, seq_length, head_features)
        '''
        print('*' * 50)
        print('query', query.shape)
        print('key', key.shape)
        print('value', value.shape)
        print('attention_mask', attention_mask.shape)
        print('head_mask', head_mask)
        import time
        time.sleep(1)
        '''

        # (batch, head, seq_length, seq_length)
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / (value.size(-1) ** 0.5)

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        # if causal_mask is not None:
        #     # if only "normal" attention layer implements causal mask
        #     # query_length, key_length = query.size(-2), key.size(-2)
        #     # causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()
        #     attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None, head_mask=None):
        # Use `torch.baddbmm` (a bit more efficient w/ alpha param for scaling -- from Megatron-LM)
        bsz, num_heads, q_seq_len, dk = query.size()
        _, _, k_seq_len, _ = key.size()

        # Preallocate attn_weights for `baddbmm`
        attn_weights = torch.empty(bsz * num_heads, q_seq_len, k_seq_len, dtype=torch.float32, device=query.device)

        # Compute Scale Factor
        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value.size(-1)) ** 0.5

        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= float(self.layer_idx + 1)

        # Upcast (turn off autocast) and reorder (Scale K by 1 / root(dk))
        if is_amp_available:
            with autocast(enabled=False):
                q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
                attn_weights = torch.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
                attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)
        else:
            q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
            attn_weights = torch.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
            attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()
            attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op if otherwise
        if attn_weights.dtype != torch.float32:
            raise RuntimeError("Error with upcasting, attn_weights does not have dtype torch.float32")
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        #print('********************** attttttttttention ************************')
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        present = (key, value)


        # if self.reorder_and_upcast_attn:
        #     attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        if query.size(2) == 1:
            attn_output, attn_weights = self._sparse_attn_with_kv_cache(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._sparse_attn_without_kv_cache(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)

        return outputs  # a, present, (attentions)


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
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
        )
        attn_output = attn_outputs[0] #(hidden_state, (cache_k, cache_v))
        cache_kv = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        outputs = (hidden_states,) + cache_kv

        return outputs  # (hidden_states, (cache_kv))


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
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]:
        if input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        else:
            raise ValueError("You have to specify either input_ids")

        device = input_ids.device

        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = ()

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
            )
            hidden_states = outputs[0] # output: (hidden_states, (cache_k, cache_v))
            presents = presents + (outputs[1],) # ((cache_k, cache_v), (cache_k, cache_v), ...)

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)

        return (hidden_states, presents) #(hidden_states, presents)


class GPT2LMHeadModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]:

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        return (lm_logits, transformer_outputs[1]) #logits, cache_kvs

