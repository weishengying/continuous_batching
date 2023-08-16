import time
from functools import partial
from typing import Any, List, Optional, TYPE_CHECKING

from vllm.config import SchedulerConfig
from vllm.core.scheduler import Scheduler
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.transformers_utils.tokenizer import detokenize_incrementally
                                               
from vllm.utils import Counter
from vllm.model_executor.input_metadata import InputMetadata
from vllm.sequence import SequenceGroupMetadata
from vllm.sequence import SequenceOutputs

from typing import Dict, List, Optional, Tuple
import torch

logger = init_logger(__name__)


class LLMEngine:
    def __init__(
        self,
        model,
        tokenizer,
        scheduler_config: SchedulerConfig,
        log_stats: bool,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.scheduler_config = scheduler_config
        self.log_stats = log_stats
        self.seq_counter = Counter()

        # Create the scheduler.
        self.scheduler = Scheduler(scheduler_config, log_stats)

    def add_request(
        self,
        request_id: str,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
    ) -> None:
        """Add a request to the engine's request pool.

        The request is added to the request pool and will be processed by the
        scheduler as `engine.step()` is called. The exact scheduling policy is
        determined by the scheduler.

        Args:
            request_id: The unique ID of the request.
            prompt: The prompt string. Can be None if prompt_token_ids is
                provided.
            sampling_params: The sampling parameters for text generation.
            prompt_token_ids: The token IDs of the prompt. If None, we
                use the tokenizer to convert the prompts to token IDs.
            arrival_time: The arrival time of the request. If None, we use
                the current time.
        """
        if arrival_time is None:
            arrival_time = time.time()
        if prompt_token_ids is None:
            assert prompt is not None
            prompt_token_ids = self.tokenizer.encode(prompt)

        # Create the sequences.
        seqs: List[Sequence] = []
        for _ in range(sampling_params.best_of):
            seq_id = next(self.seq_counter)
            seq = Sequence(seq_id, prompt, prompt_token_ids)
            seqs.append(seq)

        # Create the sequence group.
        seq_group = SequenceGroup(request_id, seqs, sampling_params,
                                  arrival_time)

        # Add the sequence group to the scheduler.
        self.scheduler.add_seq_group(seq_group)

    def abort_request(self, request_id: str) -> None:
        """Aborts a request with the given ID.

        Args:
            request_id: The ID of the request to abort.
        """
        self.scheduler.abort_seq_group(request_id)

    def get_num_unfinished_requests(self) -> int:
        """Gets the number of unfinished requests."""
        return self.scheduler.get_num_unfinished_seq_groups()

    def has_unfinished_requests(self) -> bool:
        """Returns True if there are unfinished requests."""
        return self.scheduler.has_unfinished_seqs()

    def step(self, model_kwargs) -> List[RequestOutput]:
        """Performs one decoding iteration and returns newly generated results.

        This function performs one decoding iteration of the engine. It first
        schedules the sequences to be executed in the next iteration and the
        token blocks to be swapped in/out/copy. Then, it executes the model
        and updates the scheduler with the model outputs. Finally, it decodes
        the sequences and returns the newly generated results.
        """
        (seq_group_metadata_list, scheduler_outputs,
         ignored_seq_groups) = self.scheduler.schedule()
        if ((not seq_group_metadata_list) and scheduler_outputs.is_empty()
                and (not ignored_seq_groups)):
            # Nothing to do.
            return []

        # Execute the model.
        tokens_tensor, positions_tensor, input_metadata = self._prepare_inputs(seq_group_metadata_list)
        
        outputs = self.model.forward(input_ids = tokens_tensor, past_key_values = model_kwargs["past_key_values"])
        
        model_kwargs["past_key_values"] = outputs.past_key_values
        next_token_logits = outputs.logits[-1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1)

        output: Dict[int, SequenceOutputs] = {}
        for seq_ids, _ in input_metadata.seq_groups:
            assert len(seq_ids) == 1
            seq_id = seq_ids[0]
            output[seq_id] = SequenceOutputs(seq_id, seq_id, next_tokens.item(),None)

        # Update the scheduler with the model outputs.
        seq_groups = self.scheduler.update(output)

        # Decode the sequences.
        self._decode_sequences(seq_groups)
        # Stop the sequences that meet the stopping criteria.
        self._stop_sequences(seq_groups)
        # Free the finished sequence groups.
        self.scheduler.free_finished_seq_groups()

        # Create the outputs.
        request_outputs: List[RequestOutput] = []
        for seq_group in seq_groups + ignored_seq_groups:
            request_output = RequestOutput.from_seq_group(seq_group)
            request_outputs.append(request_output)
        return request_outputs

    def _decode_sequences(self, seq_groups: List[SequenceGroup]) -> None:
        """Decodes the sequence outputs."""
        for seq_group in seq_groups:
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                new_token, new_output_text = detokenize_incrementally(
                    self.tokenizer,
                    seq.output_tokens,
                    seq.get_last_token_id(),
                    skip_special_tokens=True,
                )
                if new_token is not None:
                    seq.output_tokens.append(new_token)
                    seq.output_text = new_output_text

    def _stop_sequences(self, seq_groups: List[SequenceGroup]) -> None:
        """Stop the finished sequences."""
        for seq_group in seq_groups:
            sampling_params = seq_group.sampling_params
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                # Check if the sequence has generated a stop string.
                stopped = False
                for stop_str in sampling_params.stop:
                    if seq.output_text.endswith(stop_str):
                        # Truncate the output text so that the stop string is
                        # not included in the output.
                        seq.output_text = seq.output_text[:-len(stop_str)]
                        self.scheduler.free_seq(
                            seq, SequenceStatus.FINISHED_STOPPED)
                        stopped = True
                        break
                if stopped:
                    continue

                # Check if the sequence has reached max_seq_len.
                if seq.get_len() > self.scheduler_config.max_model_len:
                    self.scheduler.free_seq(
                        seq, SequenceStatus.FINISHED_LENGTH_CAPPED)
                    continue
                # Check if the sequence has reached max_tokens.
                if seq.get_output_len() == sampling_params.max_tokens:
                    self.scheduler.free_seq(
                        seq, SequenceStatus.FINISHED_LENGTH_CAPPED)
                    continue
                # Check if the sequence has generated the EOS token.
                if not sampling_params.ignore_eos:
                    if seq.get_last_token_id() == self.tokenizer.eos_token_id:
                        self.scheduler.free_seq(
                            seq, SequenceStatus.FINISHED_STOPPED)
                        continue
    
    def _prepare_inputs(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata]:
        seq_groups: List[Tuple[List[int], SamplingParams]] = []
        input_tokens: List[int] = []
        input_positions: List[int] = []

        # Add prompt tokens.
        prompt_lens: List[int] = []
        for seq_group_metadata in seq_group_metadata_list:
            if not seq_group_metadata.is_prompt:
                continue

            seq_ids = list(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params
            seq_groups.append((seq_ids, sampling_params))

            # Use any sequence in the group.
            seq_id = seq_ids[0]

            seq_data = seq_group_metadata.seq_data[seq_id]
            prompt_tokens = seq_data.get_token_ids()
            prompt_len = len(prompt_tokens)
            prompt_lens.append(prompt_len)

            input_tokens.extend(prompt_tokens)
            # NOTE(woosuk): Here we assume that the first token in the prompt
            # is always the first token in the sequence.
            input_positions.extend(range(len(prompt_tokens)))

        # Add generation tokens.
        max_context_len = 0
        max_num_blocks_per_seq = 0
        context_lens: List[int] = []
        for seq_group_metadata in seq_group_metadata_list:
            if seq_group_metadata.is_prompt:
                continue

            seq_ids = list(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params
            seq_groups.append((seq_ids, sampling_params))

            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                generation_token = seq_data.get_last_token_id()
                input_tokens.append(generation_token)

                context_len = seq_data.get_len()
                position = context_len - 1
                input_positions.append(position)

                max_context_len = max(max_context_len, context_len)

                context_lens.append(context_len)

        # Optimization: Pad the input length to be a multiple of 8.
        # This is required for utilizing the Tensor Cores in NVIDIA GPUs.
        # input_tokens = _pad_to_alignment(input_tokens, multiple_of=8)
        # input_positions = _pad_to_alignment(input_positions, multiple_of=8)

        # Convert to tensors.
        tokens_tensor = torch.cuda.LongTensor(input_tokens)
        positions_tensor = torch.cuda.LongTensor(input_positions)
        context_lens_tensor = torch.cuda.IntTensor(context_lens)

        seq_data: Dict[int, SequenceData] = {}
        for seq_group_metadata in seq_group_metadata_list:
            seq_data.update(seq_group_metadata.seq_data)

        input_metadata = InputMetadata(
            seq_groups=seq_groups,
            seq_data=seq_data,
            prompt_lens=prompt_lens,
            context_lens=context_lens_tensor,
            max_context_len=max_context_len,
        )
        return tokens_tensor, positions_tensor, input_metadata



def _pad_to_alignment(x: List[int], multiple_of: int) -> List[int]:
    return x + [0] * ((-len(x)) % multiple_of)