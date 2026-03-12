MODEL_ARGS=(
   --swiglu
   --num-layers 30
   --hidden-size 2048
   --ffn-hidden-size 4096
   --num-attention-heads 32
   --group-query-attention
   --num-query-groups 8
   --use-rotary-position-embeddings
   --disable-bias-linear
   --normalization "RMSNorm"
   --norm-epsilon 1e-5
   --rotary-base "${MODEL_ARGS_ROTARY_BASE:-1000000}"
   --use-rope-scaling
   --rope-scaling-factor 16.0
   --vocab-size 102400
   --kv-channels 64
   --qk-layernorm
   --ln-reorder
)