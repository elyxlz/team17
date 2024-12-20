from ultravox.evaluation import (
    eval_types,
    gpt_eval_boolq,
    gpt_eval_conv,
    gpt_eval_instruct,
    string_based,
    wer,
)

METRIC_REGISTRY = {
    "asr": wer.evaluate_answer_asr,
    "boolq": gpt_eval_boolq.evaluate_answer_boolq,
    "instruct": gpt_eval_instruct.evaluate_answer_instruct,
    "conversation": gpt_eval_conv.evaluate_conversation_response,
    "exact_match_last_word": string_based.match_last_word,
    "bleu": string_based.bleu,
}


def evaluate_answer(sample: eval_types.Sample, metric: str) -> eval_types.Result:
    if metric in METRIC_REGISTRY:
        return METRIC_REGISTRY[metric](sample)
    else:
        raise ValueError(f"Unknown metric: {metric}")
