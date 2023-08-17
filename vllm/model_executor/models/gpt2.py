from vllm.model_executor.input_metadata import InputMetadata


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
        self.register_buffer("masked_bias", torch.tensor(-1e4))

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
        self.is_cross_attention = is_cross_attention

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.pruned_heads = set()

    def single_query_cached_kv_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        input_metadata: InputMetadata,
    ):
        attn_outputs = ()
        for i in len(input_metadata.num_generation_tokens):
            attn_weights = torch.matmul(query[i:i+1], key[i:i+1].transpose(-1, -2))

            if self.scale_attn_weights:
                attn_weights = attn_weights / torch.tensor(
                    value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
                )

            if not self.is_cross_attention:
                causal_mask = self.bias[0 : 1, :1].to(torch.bool)
                # print(f"causal_mask: {causal_mask}")
                mask_value = torch.finfo(attn_weights.dtype).min
                # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
                # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
                mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
                attn_weights = torch.where(causal_mask, attn_weights, mask_value)

            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

            attn_output = torch.matmul(attn_weights, value[shift:input_metadata.prompt_lens[i]])

            attn_outputs.append(attn_output)
        
        attn_output = torch.cat(attn_outputs, dim = -2) # (head, num_generation_tokens, head_dim)

    def multi_query_kv_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        attn_outputs = ()
        shift = 0
        for i in len(input_metadata.num_prompts):
            attn_weights = torch.matmul(query[shift:input_metadata.prompt_lens[i]], 
                                        key[shift:input_metadata.prompt_lens[i]].transpose(-1, -2))
            shift = input_metadata.prompt_lens[i]

            if self.scale_attn_weights:
                attn_weights = attn_weights / torch.tensor(
                    value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
                )

            if not self.is_cross_attention:
                query_length, key_length = query[shift:input_metadata.prompt_lens[i]].size(-2), \
                                            key[shift:input_metadata.prompt_lens[i]].size(-2)
                causal_mask = self.bias[:, key_length - query_length : key_length, :key_length].to(torch.bool)
                # print(f"causal_mask: {causal_mask}")
                mask_value = torch.finfo(attn_weights.dtype).min
                # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
                # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
                mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
                attn_weights = torch.where(causal_mask, attn_weights, mask_value)

            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

            attn_output = torch.matmul(attn_weights, value[shift:input_metadata.prompt_lens[i]])

            attn_outputs.append(attn_output)
        
        attn_output = torch.cat(attn_outputs, dim = -2) # (head, num_prompt_tokens, head_dim)


    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size) #(token_num, head, head_features)
        tensor = tensor.view(new_shape)
        return tensor.permute(1, 0, 2)  # (head, token_num, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        # tensor: (head, num_tokens, head_dim)
        tensor = tensor.permute(1, 0, 2).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape) #(num_tokens, embed)

    def forward(
        self,
        input_metadata : InputMetadata,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=-1) #(token_num, embed)

        query = self._split_heads(query, self.num_heads, self.head_dim) # (head, token_num, head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim) # (head, token_num, head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim) # (head, token_num, head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None
        
        # Compute the attention op for prompts.
        num_prompt_tokens = input_metadata.num_prompt_tokens
        if num_prompt_tokens > 0:
            attn_output = self.multi_query_kv_attention(
                            query[:num_prompt_tokens],
                            key[:num_prompt_tokens],
                            value[:num_prompt_tokens],
                            input_metadata,
                        )

        num_valid_tokens = input_metadata.num_valid_tokens
        # Compute the attention op for generation tokens.
        if num_valid_tokens - num_prompt_tokens > 0:
            attn_output= self.single_query_cached_kv_attention(
                            output[num_prompt_tokens:num_valid_tokens],
                            query[num_prompt_tokens:num_valid_tokens], key_cache,
                            value_cache, input_metadata)
                
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

        if config.add_cross_attention:
            self.crossattention = GPT2Attention(config, is_cross_attention=True, layer_idx=layer_idx)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPT2MLP(inner_dim, config)

    def forward(
        self,
        input_metadata: InputMetadata,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states) # (num_tokens, embed)
        attn_outputs = self.attn(
            input_metadata,
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            use_cache=use_cache,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)


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
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        # input_ids:(num_tokens)
        device = input_ids.device

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.h))


        inputs_embeds = self.wte(input_ids) #(num_tokens, embed)
        position_embeds = self.wpe(position_ids) #(num_tokens, embed)
        hidden_states = inputs_embeds + position_embeds #(num_tokens, embed)

        input_shape = input_ids.size()
        output_shape = input_shape + (hidden_states.size(-1),) #(num_tokens, embed)

        presents = () if use_cache else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)


            outputs = block(
                input_metadata,
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                use_cache=use_cache,
            )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
        )


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
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_metadata,
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        return CausalLMOutputWithCrossAttentions(
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
        )


