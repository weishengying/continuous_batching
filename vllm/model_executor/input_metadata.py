from typing import Dict, List, Tuple

import torch

from vllm.sampling_params import SamplingParams
from vllm.sequence import SequenceData


class InputMetadata:
    """Metadata for input sequences. Used for PagedAttention.

    Args:
        seq_groups: List of (seq_ids, sampling_params).
        seq_data: Seq_id -> SequenceData.
        prompt_lens: Lengths of prompts.
        context_lens: the length of attention context for each generation token.
        max_context_len: The maximum context length.
    """

    def __init__(
        self,
        seq_groups: List[Tuple[List[int], SamplingParams]],
        seq_data: Dict[int, SequenceData],
        prompt_lens: List[int],
        context_lens: torch.Tensor,
        max_context_len: int,
    ) -> None:
        self.seq_groups = seq_groups
        self.seq_data = seq_data
        self.prompt_lens = prompt_lens
        self.context_lens = context_lens
        self.max_context_len = max_context_len

        self.num_prompts = len(prompt_lens)
        self.num_prompt_tokens = sum(prompt_lens)
        self.num_generation_tokens = context_lens.shape[0]
        self.num_valid_tokens = self.num_prompt_tokens + self.num_generation_tokens

        assert context_lens.shape[0] == self.num_generation_tokens


    def __repr__(self) -> str:
        # Print only useful metadata.
        return (f'InputMetadata('
                f'num_valid_tokens={self.num_valid_tokens}, '
                f'num_prompt_tokens={self.num_prompt_tokens}, '
                f'num_prompts={self.num_prompts}, '
                f'prompt_lens={self.prompt_lens}, '
                f'num_generation_tokens={self.num_generation_tokens}, '
                f'context_lens={self.context_lens}, '
                f'max_context_len={self.max_context_len})')