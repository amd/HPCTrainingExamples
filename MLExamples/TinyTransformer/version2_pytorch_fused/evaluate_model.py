#!/usr/bin/env python3
"""
Tiny LLaMA V2: Trained-Model Evaluation

Loads a checkpoint saved by tiny_llama_v2.py (--save-checkpoints) and inspects how useful its
predictions actually are:

- Validation perplexity on a fresh sample of held-out Wikipedia text.
- Free-form greedy-decoded completions for open prompts (qualitative fluency check).
- Top-k next-token predictions right after a prompt (more legible than long greedy runs given
  the tiny BPE vocabulary).
- A quantitative sentence-completion test: for each prompt, the model scores a plausible vs.
  implausible candidate continuation (average log-probability per token); a "useful" model
  should consistently prefer the plausible one.

Usage:
    # Full benchmark suite (perplexity, fixed prompts, completion test) - the default
    python evaluate_model.py --checkpoint quality_runs/final_run/20260902_.../best_model.pt \
        --dataset wikipedia --wiki-num-docs 8000

    # Try your own prompt (one-shot)
    python evaluate_model.py --checkpoint <path> --wiki-num-docs 8000 \
        --prompt "The Roman Empire was"

    # Interactively type prompts and see completions
    python evaluate_model.py --checkpoint <path> --wiki-num-docs 8000 --interactive

    # Sample instead of greedy decoding (reduces repetition loops on a lightly-trained model)
    python evaluate_model.py --checkpoint <path> --wiki-num-docs 8000 \
        --interactive --temperature 0.8
"""

import argparse
import sys
from typing import Any, List, Optional, Tuple

try:
    import readline  # enables backspace/arrow-key editing in --interactive mode
except ImportError:
    pass

import torch
import torch.nn.functional as F

from tiny_llama_v2 import FusionConfig, TinyLlamaConfig, TinyLlamaV2, build_dataset_from_args, evaluate, generate
from wiki_dataset import add_dataset_args, safe_exit

# Open-ended prompts for qualitative greedy-decoding samples.
FREEFORM_PROMPTS = [
    "The history of",
    "In the early 20th century,",
    "The capital of France is",
    "Water is made of",
]

# (prompt, plausible continuation, implausible continuation) triples for the quantitative
# sentence-completion test. A model that has learned real structure should score the plausible
# continuation with a higher average per-token log-probability.
COMPLETION_TESTS: List[Tuple[str, str, str]] = [
    ("The capital of France is", " Paris", " a banana"),
    ("Water is made of hydrogen and", " oxygen", " the moon"),
    ("The sun rises in the", " east", " kitchen"),
    ("Two plus two equals", " four", " Wednesday"),
    ("The largest planet in the solar system is", " Jupiter", " a sandwich"),
]


@torch.no_grad()
def top_k_next_tokens(model: torch.nn.Module, dataset: Any, max_seq_len: int, device: torch.device,
                       prompt: str, k: int = 5) -> List[Tuple[str, float]]:
    """Return the model's top-k next-token predictions (decoded, with probability) after `prompt`."""
    model.eval()
    ids = dataset.encode(prompt) or [0]
    input_ids = torch.tensor([ids[-max_seq_len:]], dtype=torch.long, device=device)
    logits = model(input_ids)['logits'][0, -1, :]
    probs = F.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, k)
    model.train()
    return [(dataset.decode([tid.item()]), p.item()) for tid, p in zip(top_ids, top_probs)]


@torch.no_grad()
def score_continuation(model: torch.nn.Module, dataset: Any, max_seq_len: int, device: torch.device,
                        prompt: str, continuation: str) -> float:
    """Average log-probability per token the model assigns to `continuation` following `prompt`.

    Higher (less negative) = the model considers this continuation more likely.
    """
    prompt_ids = dataset.encode(prompt)
    cont_ids = dataset.encode(continuation)
    if not cont_ids:
        return float('-inf')

    ids = (prompt_ids + cont_ids)[-max_seq_len:]
    n_cont = min(len(cont_ids), len(ids))

    model.eval()
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    log_probs = F.log_softmax(model(input_ids)['logits'][0], dim=-1)
    model.train()

    total_lp, count = 0.0, 0
    for i in range(len(ids) - n_cont, len(ids)):
        if i == 0:
            continue  # no preceding context to predict this position from
        total_lp += log_probs[i - 1, ids[i]].item()
        count += 1

    return total_lp / count if count > 0 else float('-inf')


def run_single_prompt(model: torch.nn.Module, dataset: Any, config: TinyLlamaConfig, device: torch.device,
                       prompt: str, max_new_tokens: int, temperature: Optional[float]) -> None:
    """Print a completion plus the top-5 next-token predictions for one custom prompt."""
    sample = generate(model, dataset, config.max_seq_len, device, prompt, max_new_tokens, temperature=temperature)
    print(f"Prompt:     {prompt!r}")
    print(f"Completion: {sample!r}")
    preds = top_k_next_tokens(model, dataset, config.max_seq_len, device, prompt, k=5)
    formatted = ", ".join(f"{tok!r} ({p:.1%})" for tok, p in preds)
    print(f"Top-5 next tokens right after the prompt: {formatted}")


def run_interactive(model: torch.nn.Module, dataset: Any, config: TinyLlamaConfig, device: torch.device,
                     max_new_tokens: int, temperature: Optional[float]) -> None:
    """Read prompts from stdin in a loop and print completions until the user quits."""
    print("\nInteractive completion mode - type a prompt and press Enter.")
    print("Type 'quit', 'exit', or press Ctrl-D to stop.\n")
    while True:
        try:
            prompt = input(">>> ").strip()
        except EOFError:
            print()
            break
        if not prompt:
            continue
        if prompt.lower() in ('quit', 'exit'):
            break
        run_single_prompt(model, dataset, config, device, prompt, max_new_tokens, temperature)
        print()


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained Tiny LLaMA V2 checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to a best_model.pt checkpoint')
    add_dataset_args(parser)
    parser.set_defaults(dataset='wikipedia')
    parser.add_argument('--eval-batches', type=int, default=30, help='Validation batches for perplexity')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for perplexity evaluation')
    parser.add_argument('--generate-tokens', type=int, default=50, help='Tokens to generate per prompt')
    parser.add_argument('--temperature', type=float, default=None,
                        help='Sampling temperature for generation, e.g. 0.7-1.0 (default: greedy/deterministic)')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Generate a completion for this custom prompt and exit '
                             '(skips the full benchmark suite below)')
    parser.add_argument('--interactive', action='store_true',
                        help='Interactively type prompts and see completions (skips the full benchmark suite)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print("TINY LLAMA V2 - MODEL EVALUATION")
    print("=" * 80)
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = TinyLlamaConfig(**ckpt['config'])
    fusion_config = FusionConfig(**ckpt['fusion_config'])
    print(f"   Trained for {ckpt['step']} steps, checkpoint val_loss={ckpt['val_loss']:.4f}")
    print(f"   Vocab size: {config.vocab_size} | Hidden dim: {config.hidden_dim} | "
          f"Layers: {config.n_layers} | Seq len: {config.max_seq_len}")

    print(f"\nRebuilding dataset ({args.dataset}) to match training tokenizer/vocab ...")
    dataset = build_dataset_from_args(args, config, vocab_size_explicit=True)

    model = TinyLlamaV2(config, fusion_config).to(device)
    state_dict = ckpt['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in state_dict):
        # Checkpoint was saved from a torch.compile()-wrapped model (train_job.sbatch enables
        # this by default) - torch.compile wraps the module and prefixes every state_dict key
        # with "_orig_mod.". Strip it so it loads into this plain (uncompiled) instance.
        state_dict = {k.removeprefix('_orig_mod.'): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {total_params:,} parameters")

    if args.prompt or args.interactive:
        # User wants to try prompts themselves - skip the fixed benchmark suite below.
        if args.prompt:
            print()
            run_single_prompt(model, dataset, config, device, args.prompt, args.generate_tokens, args.temperature)
        if args.interactive:
            run_interactive(model, dataset, config, device, args.generate_tokens, args.temperature)
        return

    print("\n" + "-" * 70)
    print("VALIDATION PERPLEXITY")
    print("-" * 70)
    val_loss, val_ppl = evaluate(model, dataset, device, args.batch_size, args.eval_batches)
    print(f"Validation loss: {val_loss:.4f} | Perplexity: {val_ppl:.2f} "
          f"(random-guess perplexity would be ~{config.vocab_size})")

    print("\n" + "-" * 70)
    print("FREE-FORM COMPLETIONS (greedy decoding)")
    print("-" * 70)
    for prompt in FREEFORM_PROMPTS:
        sample = generate(model, dataset, config.max_seq_len, device, prompt, args.generate_tokens)
        print(f"Prompt: {prompt!r}\n   -> {sample!r}\n")

    print("-" * 70)
    print("TOP-5 NEXT-TOKEN PREDICTIONS")
    print("-" * 70)
    for prompt in FREEFORM_PROMPTS[:3]:
        preds = top_k_next_tokens(model, dataset, config.max_seq_len, device, prompt, k=5)
        formatted = ", ".join(f"{tok!r} ({p:.1%})" for tok, p in preds)
        print(f"Prompt: {prompt!r}\n   -> {formatted}\n")

    print("-" * 70)
    print("QUANTITATIVE SENTENCE-COMPLETION TEST")
    print("(higher / less negative avg log-prob per token = model considers it more likely)")
    print("-" * 70)
    correct = 0
    for prompt, good, bad in COMPLETION_TESTS:
        good_score = score_continuation(model, dataset, config.max_seq_len, device, prompt, good)
        bad_score = score_continuation(model, dataset, config.max_seq_len, device, prompt, bad)
        picked_plausible = good_score > bad_score
        correct += int(picked_plausible)
        verdict = "PASS" if picked_plausible else "FAIL"
        print(f"[{verdict}] {prompt!r}")
        print(f"   plausible   {good!r:20s} avg logp = {good_score:7.3f}")
        print(f"   implausible {bad!r:20s} avg logp = {bad_score:7.3f}")
    print(f"\nScore: {correct}/{len(COMPLETION_TESTS)} completion tests preferred the plausible continuation")


if __name__ == "__main__":
    main()
    if 'datasets' in sys.modules:
        # Bypass normal interpreter teardown: streaming the Wikipedia dataset spins up
        # background aiohttp/fsspec threads that can otherwise trigger a spurious
        # 'PyGILState_Release' fatal error on exit (see wiki_dataset.safe_exit).
        safe_exit(0)
