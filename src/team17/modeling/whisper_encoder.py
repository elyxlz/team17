import torch
import torch.nn as nn
import transformers
import transformers.activations
import transformers.modeling_outputs
import transformers.models
from transformers.models.whisper import modeling_whisper as whisper


class ModifiedWhisperEncoder(
    whisper.WhisperEncoder, transformers.modeling_utils.ModuleUtilsMixin
):
    """
    Encoder portion of OpenAI's Whisper model.

    This implementation is a slightly modified version of HF Transformers' Whisper Encoder, with only a few fixes:
    1. base_model_prefix updated to allow for doing `.from_pretrained` directly on the encoder
    2. allow less than 30 second of audio padding to be passed in:
        - relaxed ValueError check for `input_features` length to be less than or equal to `expected_seq_length` instead of strictly equal
        - embed_pos is now sliced to match the length of `inputs_embeds`

    Original: https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py
    """

    base_model_prefix = "model.encoder"
    _no_split_modules = ["WhisperEncoderLayer"]
    audio_streaming_mask = None

    def init_latency_mask(self, audio_latency_block_size: int, dtype: torch.dtype):
        if audio_latency_block_size is None:
            self.audio_streaming_mask = None
            return

        # maximum sequence length
        max_seqlen = (
            self.config.max_source_positions
            * self.conv1.stride[0]
            * self.conv2.stride[0]
        )
        assert (
            max_seqlen > 0
        ), f"maximum sequence length must be positive, got {max_seqlen}"
        assert (
            max_seqlen % audio_latency_block_size == 0
        ), f"audio_latency_block_size {audio_latency_block_size} must divide {max_seqlen} evenly."
        # Given the block size, we calculate number of blocks.
        audio_latency_nblocks = max_seqlen // audio_latency_block_size
        audio_streaming_mask = (
            torch.tril(
                torch.ones(audio_latency_nblocks, audio_latency_nblocks),
                diagonal=0,
            )
            .repeat_interleave(audio_latency_block_size, dim=0)
            .repeat_interleave(audio_latency_block_size, dim=1)
        )
        audio_streaming_mask = (1.0 - audio_streaming_mask) * torch.finfo(dtype).min
        audio_streaming_mask = audio_streaming_mask[None, None, :, :]
        self.register_buffer(
            "audio_streaming_mask", audio_streaming_mask, persistent=False
        )

    def forward(
        self,
        input_features,
        audio_len=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        expected_seq_length = (
            self.config.max_source_positions
            * self.conv1.stride[0]
            * self.conv2.stride[0]
        )
        if input_features.shape[-1] > expected_seq_length:
            raise ValueError(
                f"Whisper expects the mel input features to be of length {expected_seq_length} or less, but found {input_features.shape[-1]}. Make sure to pad the input mel features to {expected_seq_length}."
            )

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.embed_positions.weight[: inputs_embeds.size(-2)]

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # Create attention mask based on audio lengths to mask out padding tokens
        # For each sample in batch:
        # - Convert raw audio length to feature length after convolutions
        # - Create boolean mask that is True for valid positions and False for padding
        # - Convert to extended attention mask format expected by transformer layers
        #   (1.0 for positions to attend to, large negative for positions to ignore)
        # This masking ensures consistent behavior between training and inference
        # by preventing the model from attending to padding tokens in both cases
        attention_mask = None
        if audio_len is not None:
            audio_feature_len = self._get_feat_extract_output_lengths(audio_len)
            max_seq_len = hidden_states.shape[1]
            attention_mask = torch.arange(max_seq_len, device=hidden_states.device)[
                None, :
            ].lt(audio_feature_len.view(-1, 1))
            attention_mask = self.get_extended_attention_mask(
                attention_mask,
                None,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

        if self.audio_streaming_mask is not None:
            seqlen = hidden_states.size(-2)
            if attention_mask is not None:
                attention_mask = torch.minimum(
                    self.audio_streaming_mask[:, :, :seqlen, :seqlen], attention_mask
                )  # merge
            else:
                attention_mask = self.audio_streaming_mask[:, :, :seqlen, :seqlen]
            attention_mask = attention_mask.to(hidden_states.dtype)

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert (
                head_mask.size()[0] == (len(self.layers))
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:  # skip the layer
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                        output_attentions,
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(
                            head_mask[idx] if head_mask is not None else None
                        ),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, encoder_states, all_attentions]
                if v is not None
            )
        return transformers.modeling_outputs.BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )
