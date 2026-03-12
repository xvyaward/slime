# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from mbridge.core import register_model
from .llm_bridge import LLMBridge


@register_model("exaone4")
class EXAONE4Bridge(LLMBridge):
    """
    Bridge implementation for EXAONE4 models.

    This class extends LLMBridge to provide specific configurations and
    optimizations for EXAONE4 models, handling the conversion between
    Hugging Face EXAONE4 format and Megatron-Core.
    """
    # HF Exaone4 text model structure:
    #   model/ lm_head
    #     ㄴ model.embed_tokens
    #     ㄴ model.layers
    #     ㄴ model.norm
    # Keys on the left are Megatron-Core param names; values are HF param names.
    _DIRECT_MAPPING = {
        "embedding.word_embeddings.weight": "model.embed_tokens.weight",
        "decoder.final_layernorm.weight": "model.norm.weight",
        "output_layer.weight": "lm_head.weight",
    }

    _ATTENTION_MAPPING = {
        # attention projections
        "self_attention.linear_qkv.weight": [
            "model.layers.{layer_number}.self_attn.q_proj.weight",
            "model.layers.{layer_number}.self_attn.k_proj.weight",
            "model.layers.{layer_number}.self_attn.v_proj.weight",
        ],
        "self_attention.linear_proj.weight": [
            "model.layers.{layer_number}.self_attn.o_proj.weight"
        ],
        # exaone4 uses input_layernorm.weight key on MCore side, mapping to post_attention_layernorm on HF
        "input_layernorm.weight": [
            "model.layers.{layer_number}.post_attention_layernorm.weight"
        ],
        # # backward-compat: some specs expose LN via linear_qkv.layer_norm_weight
        # "self_attention.linear_qkv.layer_norm_weight": [
        #     "model.layers.{layer_number}.post_attention_layernorm.weight"
        # ],
        # QK LayerNorm
        "self_attention.q_layernorm.weight": [
            "model.layers.{layer_number}.self_attn.q_norm.weight"
        ],
        "self_attention.k_layernorm.weight": [
            "model.layers.{layer_number}.self_attn.k_norm.weight"
        ],
    }

    _MLP_MAPPING = {
        "mlp.linear_fc1.weight": [
            "model.layers.{layer_number}.mlp.gate_proj.weight",
            "model.layers.{layer_number}.mlp.up_proj.weight",
        ],
        # exaone4 uses pre_mlp_layernorm on MCore side; maps to post_feedforward_layernorm on HF
        "pre_mlp_layernorm.weight": [
            "model.layers.{layer_number}.post_feedforward_layernorm.weight"
        ],
        # # backward-compat: some specs use mlp.linear_fc1.layer_norm_weight
        # "mlp.linear_fc1.layer_norm_weight": [
        #     "model.layers.{layer_number}.post_feedforward_layernorm.weight"
        # ],
        "mlp.linear_fc2.weight": [
            "model.layers.{layer_number}.mlp.down_proj.weight"
        ],
    }

    def _adjust_mapping_for_shared_weights(self):
        if getattr(self.hf_config, "tie_word_embeddings", False):
            self._DIRECT_MAPPING["output_layer.weight"] = "model.embed_tokens.weight"

    def _get_hf_shared_weight_keys(self):
        if getattr(self.hf_config, "tie_word_embeddings", False):
            return ["model.embed_tokens.weight"]
        return []

    def _build_config(self):
        """
        Build the configuration for EXAONE4 models.

        Configures EXAONE4-specific parameters such as QK layer normalization.
        EXAONE4 uses layer normalization on query and key tensors.

        Returns:
            TransformerConfig: Configuration object for EXAONE4 models
        """
        # Derive hybrid-attention settings from HF config (auto) with override support

        base = dict(
            qk_layernorm=True,
            ln_reorder=True,
            add_qkv_bias=False,
            add_bias_linear=False,
            cp_comm_type="a2a",
        )

        return self._build_base_config(**base)
