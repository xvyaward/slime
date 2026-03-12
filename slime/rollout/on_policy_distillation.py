import aiohttp
import logging
import torch

from slime.utils.types import Sample

logger = logging.getLogger(__name__)


async def reward_func(args, sample, **kwargs):
    input_ids = sample.tokens

    if getattr(args, "opd_teacher_prompt_replace", None) is not None:
        target_str, replace_str = args.opd_teacher_prompt_replace
        
        # Cache tokenizer on args to avoid repeated loading
        if not hasattr(args, "_cached_tokenizer"):
            from transformers import AutoTokenizer
            args._cached_tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        tokenizer = args._cached_tokenizer
        
        # Separate prompt and response tokens
        prompt_len = len(sample.tokens) - sample.response_length
        prompt_tokens = sample.tokens[:prompt_len]
        response_tokens = sample.tokens[prompt_len:]
        
        # Decode prompt, replace, and re-encode
        prompt_text = tokenizer.decode(prompt_tokens)
        new_prompt_text = prompt_text.replace(target_str, replace_str)
        new_prompt_tokens = tokenizer.encode(new_prompt_text)

        # Log modification once per batch or infrequently
        if not hasattr(args, "_logged_opd_replacement"):
             logger.info(f"\n[OPD Prompt Modification]\nOriginal: {prompt_text}\nModified: {new_prompt_text}\n")
             args._logged_opd_replacement = True
        
        # Re-stitch
        input_ids = new_prompt_tokens + response_tokens

    payload = {
        # "text": sample.prompt + sample.response,
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
        "top_logprobs_num": 5,
    }
    session_kwargs = {}
    async with aiohttp.ClientSession(**session_kwargs) as session:
        async with session.post(args.rm_url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Process rewards from teacher model and extract teacher log probabilities.

    This function:
    1. Extracts teacher log-probs from the reward response (which contains sglang's logprob output)
    2. Trims them to match the response length
    3. Stores them in sample.teacher_log_probs for OPD KL penalty computation
    4. Returns scalar rewards (0.0 for pure distillation) compatible with GRPO/PPO

    Note: The reward_func calls the teacher server which returns token-level log-probs.
    For pure on-policy distillation without task rewards, we return 0.0 for each sample.
    The actual learning signal comes from the OPD KL penalty applied in compute_advantages_and_returns.
    """
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    response_lengths = [sample.response_length for sample in samples]

    # Extract teacher log-probs from the sglang response
    teacher_log_probs = [
        torch.tensor([item[0] for item in reward["meta_info"]["input_token_logprobs"][1:]], dtype=torch.float32)
        for reward in raw_rewards
    ]
    teacher_log_probs = [
        t_log_prob[-response_length:]
        for t_log_prob, response_length in zip(teacher_log_probs, response_lengths, strict=False)
    ]

    for sample, t_log_probs in zip(samples, teacher_log_probs, strict=False):
        sample.teacher_log_probs = t_log_probs

    # Return scalar rewards for GRPO/PPO advantage estimator
    # For pure on-policy distillation, we use 0.0 as the task reward.
    # The learning signal comes entirely from the OPD KL penalty.
    # If you have task rewards, you can add them here.
    
    # scalar_rewards = [0.0] * len(samples)

    think_bonus_coef = getattr(args, "opd_think_bonus_coef", 0.0)
    think_token_id = getattr(args, "opd_think_token_id", -1)
    opd_think_kl_scale = getattr(args, "opd_think_kl_scale", 0.0)
    
    scalar_rewards = []
    for reward, length, sample in zip(raw_rewards, response_lengths, samples, strict=False):
        base_reward = 0.0
        think_bonus = 0.0
        
        # default kl weights
        kl_opts = [1.0] * length
        
        if (think_bonus_coef > 0 or opd_think_kl_scale > 0) and think_token_id != -1:
            top_logprobs = reward["meta_info"].get("input_top_logprobs", [])
            top_logprobs_response = top_logprobs[-length:]
            
            for i, step_preds in enumerate(top_logprobs_response):
                if step_preds is None:
                    continue
                for logprob, token_id, _ in step_preds:
                    if token_id == think_token_id:
                        prob = torch.exp(torch.tensor(logprob, dtype=torch.float32)).item()
                        if think_bonus_coef > 0:
                            think_bonus += think_bonus_coef * prob
                        if opd_think_kl_scale > 0:
                            kl_opts[i] = 1.0 + opd_think_kl_scale * prob
        
        sample.teacher_kl_weights = kl_opts
        scalar_rewards.append(base_reward + think_bonus)

    return scalar_rewards, scalar_rewards
