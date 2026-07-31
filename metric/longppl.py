import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase



ALPHA = -1e9
BETA = -2.8
TRUNC_LEN = 1


SHORT_CONTEXT_BATCH_SIZE = 512


def _get_pad_token_id(tokenizer: PreTrainedTokenizerBase) -> int:
    if tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)
    if tokenizer.eos_token_id is not None:
        return int(tokenizer.eos_token_id)
    if tokenizer.bos_token_id is not None:
        return int(tokenizer.bos_token_id)
    return 0


def _target_log_probabilities(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
) -> torch.Tensor:
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)


def _calculate_full_context_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )


    prediction_logits = outputs.logits[:, :-1, :]
    target_ids = input_ids[:, 1:]
    full_log_probs = _target_log_probabilities(prediction_logits, target_ids)
    return full_log_probs[0].detach().cpu()


def _calculate_short_context_log_probs(
    model: PreTrainedModel,
    input_ids_cpu: torch.Tensor,
    device: torch.device,
    trunc_len: int,
    batch_size: int,
    pad_token_id: int,
) -> torch.Tensor:

    if trunc_len < 1:
        raise ValueError(f"trunc_len must be at least 1, but got {trunc_len}.")
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, but got {batch_size}.")

    sequence_length = int(input_ids_cpu.numel())
    if sequence_length < 2:
        return torch.empty(0, dtype=torch.float32)

    all_short_log_probs: List[torch.Tensor] = []
    target_positions = list(range(1, sequence_length))

    for batch_start in range(0, len(target_positions), batch_size):
        batch_positions = target_positions[batch_start : batch_start + batch_size]
        windows: List[torch.Tensor] = []

        for target_position in batch_positions:
            context_start = max(0, target_position - trunc_len)
            # Include the target token at the end of the window. The logit at
            # the penultimate real position predicts this target token.
            windows.append(input_ids_cpu[context_start : target_position + 1])

        lengths_cpu = torch.tensor([window.numel() for window in windows], dtype=torch.long)
        padded_cpu = pad_sequence(
            windows,
            batch_first=True,
            padding_value=pad_token_id,
        )
        attention_mask_cpu = (
            torch.arange(padded_cpu.size(1), dtype=torch.long).unsqueeze(0)
            < lengths_cpu.unsqueeze(1)
        ).long()

        padded = padded_cpu.to(device, non_blocking=True)
        attention_mask = attention_mask_cpu.to(device, non_blocking=True)
        lengths = lengths_cpu.to(device, non_blocking=True)

        with torch.inference_mode():
            outputs = model(
                input_ids=padded,
                attention_mask=attention_mask,
                use_cache=False,
            )

        row_indices = torch.arange(padded.size(0), device=device)
        prediction_positions = lengths - 2
        target_positions_in_window = lengths - 1

        prediction_logits = outputs.logits[row_indices, prediction_positions, :]
        target_ids = padded[row_indices, target_positions_in_window]
        batch_log_probs = _target_log_probabilities(prediction_logits, target_ids)
        all_short_log_probs.append(batch_log_probs.detach().cpu())

        del outputs, prediction_logits, batch_log_probs, padded, attention_mask, lengths

    return torch.cat(all_short_log_probs, dim=0)


def calculate_longppl(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    instruction: str,
    alpha: float = ALPHA,
    beta: float = BETA,
    trunc_len: int = TRUNC_LEN,
    short_context_batch_size: int = SHORT_CONTEXT_BATCH_SIZE,
) -> Dict[str, Any]:

    if not isinstance(instruction, str):
        instruction = str(instruction)

    encoded = tokenizer(
        instruction,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=False,
    )
    input_ids_cpu = encoded["input_ids"][0].detach().cpu()
    sequence_length = int(input_ids_cpu.numel())

    if sequence_length < 2:
        return {
            "longppl": None,
            "average_key_nll": None,
            # Backward-compatible alias for the old program. Despite its old
            # name, this quantity is an average negative log probability.
            "average_log_prob": None,
            "n_scored_tokens": 0,
            "n_key_tokens": 0,
            "key_token_ratio": 0.0,
            "token_log_probs": [],
            "token_metrics": [],
        }

    device = next(model.parameters()).device
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask", torch.ones_like(encoded["input_ids"])).to(device)

    full_log_probs = _calculate_full_context_log_probs(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    short_log_probs = _calculate_short_context_log_probs(
        model=model,
        input_ids_cpu=input_ids_cpu,
        device=device,
        trunc_len=trunc_len,
        batch_size=short_context_batch_size,
        pad_token_id=_get_pad_token_id(tokenizer),
    )

    if full_log_probs.numel() != short_log_probs.numel():
        raise RuntimeError(
            "Full-context and short-context token counts differ: "
            f"{full_log_probs.numel()} vs. {short_log_probs.numel()}."
        )


    lsd = full_log_probs - short_log_probs
    key_mask = (lsd > alpha) & (full_log_probs > beta)

    key_full_log_probs = full_log_probs[key_mask]
    n_scored_tokens = int(full_log_probs.numel())
    n_key_tokens = int(key_mask.sum().item())

    if n_key_tokens > 0:
        average_key_nll = float((-key_full_log_probs).mean().item())
        longppl = float(math.exp(average_key_nll))
    else:
        average_key_nll = None
        longppl = None

    target_token_ids = input_ids_cpu[1:].tolist()
    target_tokens = tokenizer.convert_ids_to_tokens(target_token_ids)

    token_log_probs: List[Tuple[str, float]] = []
    token_metrics: List[Dict[str, Any]] = []

    for offset, token_id in enumerate(target_token_ids):
        token = target_tokens[offset]
        full_log_prob = float(full_log_probs[offset].item())
        short_log_prob = float(short_log_probs[offset].item())
        token_lsd = float(lsd[offset].item())
        is_key_token = bool(key_mask[offset].item())

        token_log_probs.append((token, full_log_prob))

        token_metrics.append(
            {
                "position": offset + 1,
                "token_id": int(token_id),
                "token": token,
                "text": tokenizer.decode([token_id], skip_special_tokens=False),
                "full_log_prob": full_log_prob,
                "short_log_prob": short_log_prob,
                "full_nll": -full_log_prob,
                "short_nll": -short_log_prob,
                "lcl": full_log_prob,
                "lsd": token_lsd,
                "is_key_token": is_key_token,
            }
        )

    return {
        "longppl": longppl,
        "average_key_nll": average_key_nll,
        "average_log_prob": average_key_nll,
        "n_scored_tokens": n_scored_tokens,
        "n_key_tokens": n_key_tokens,
        "key_token_ratio": n_key_tokens / n_scored_tokens,
        "token_log_probs": token_log_probs,
        "token_metrics": token_metrics,
    }


def process_file(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    input_file_path: str,
    output_file_path: str,
    alpha: float = ALPHA,
    beta: float = BETA,
    trunc_len: int = TRUNC_LEN,
    short_context_batch_size: int = SHORT_CONTEXT_BATCH_SIZE,
) -> None:
    output_dir = os.path.dirname(os.path.abspath(output_file_path))
    os.makedirs(output_dir, exist_ok=True)

    processed = 0
    skipped = 0

    with open(input_file_path, "r", encoding="utf-8") as f_in, open(
        output_file_path, "w", encoding="utf-8"
    ) as f_out:
        for line_number, line in enumerate(f_in, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                instruction = data.get("instruction")
                if instruction is None:
                    skipped += 1
                    print(
                        f"[Warning] {input_file_path}:{line_number} has no "
                        "'instruction' field; skipped."
                    )
                    continue

                instruction = str(instruction)
                metrics = calculate_longppl(
                    model=model,
                    tokenizer=tokenizer,
                    instruction=instruction,
                    alpha=alpha,
                    beta=beta,
                    trunc_len=trunc_len,
                    short_context_batch_size=short_context_batch_size,
                )

                result = {
                    "instruction": instruction,
                    "alpha": alpha,
                    "beta": beta,
                    "trunc_len": trunc_len,
                    **metrics,
                }
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                f_out.flush()
                processed += 1

                if processed % 100 == 0:
                    print(
                        f"[Progress] {input_file_path}: processed {processed} "
                        f"instructions."
                    )

            except Exception as exc:
                skipped += 1
                error_result = {
                    "line_number": line_number,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                f_out.write(json.dumps(error_result, ensure_ascii=False) + "\n")
                f_out.flush()
                print(
                    f"[Error] {input_file_path}:{line_number}: "
                    f"{type(exc).__name__}: {exc}"
                )

    print(
        f"[Done] {input_file_path} -> {output_file_path}; "
        f"processed={processed}, skipped_or_failed={skipped}."
    )


def main(
    model_path: str,
    input_file_paths: Sequence[str],
    output_file_paths: Sequence[str],
    alpha: float = ALPHA,
    beta: float = BETA,
    trunc_len: int = TRUNC_LEN,
    short_context_batch_size: int = SHORT_CONTEXT_BATCH_SIZE,
    device_name: Optional[str] = None,
) -> None:
    if len(input_file_paths) != len(output_file_paths):
        raise ValueError(
            "input_file_paths and output_file_paths must contain the same "
            f"number of paths, but got {len(input_file_paths)} and "
            f"{len(output_file_paths)}."
        )

    if device_name is None:
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    if device.type == "cuda":
        torch_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
    else:
        torch_dtype = torch.float32
        print("CUDA is not available; using CPU.")

    print(f"Loading LLM from: {model_path}")
    print(f"Device: {device}; dtype: {torch_dtype}")
    print(
        "LongPPL parameters: "
        f"alpha={alpha}, beta={beta}, trunc_len={trunc_len}, "
        f"short_context_batch_size={short_context_batch_size}"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.bos_token_id is not None:
            tokenizer.pad_token = tokenizer.bos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.use_cache = False

    for input_file, output_file in zip(input_file_paths, output_file_paths):
        process_file(
            model=model,
            tokenizer=tokenizer,
            input_file_path=input_file,
            output_file_path=output_file,
            alpha=alpha,
            beta=beta,
            trunc_len=trunc_len,
            short_context_batch_size=short_context_batch_size,
        )


if __name__ == "__main__":
    MODEL_PATH = "path"
    INPUT_FILE_PATHS = ["file.jsonl"]
    OUTPUT_FILE_PATHS = ["file_longppl.jsonl"]

    main(
        model_path=MODEL_PATH,
        input_file_paths=INPUT_FILE_PATHS,
        output_file_paths=OUTPUT_FILE_PATHS,
        alpha=ALPHA,
        beta=BETA,
        trunc_len=TRUNC_LEN,
        short_context_batch_size=SHORT_CONTEXT_BATCH_SIZE,
        device_name="cuda:0",
    )
