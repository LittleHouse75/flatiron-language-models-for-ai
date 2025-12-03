"""
Training utilities shared by all seq2seq models:
- Training loop
- Validation loop
- ROUGE evaluation
- Early stopping
- Checkpointing (best model)
- Returns a pandas DataFrame of metrics
"""

import torch
from tqdm.auto import tqdm
import pandas as pd
import evaluate
from pathlib import Path

rouge_metric = evaluate.load("rouge")


def evaluate_on_validation(
    model,
    loader,
    tokenizer,
    device,
    max_target_len,
):
    """
    Compute validation loss + ROUGE.
    """
    model.eval()
    total_loss = 0
    steps = 0

    preds = []
    refs = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Loss
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            total_loss += loss.item()
            steps += 1

            # Generation: rely on the model's generation_config
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=model.generation_config,
            )

            preds.extend(
                tokenizer.batch_decode(generated, skip_special_tokens=True)
            )

            # Restore padding for label decoding
            labels_dec = torch.where(
                labels != -100,
                labels,
                tokenizer.pad_token_id,
            )
            refs.extend(
                tokenizer.batch_decode(labels_dec, skip_special_tokens=True)
            )

    avg_loss = total_loss / max(1, steps)
    rouge_scores = rouge_metric.compute(predictions=preds, references=refs)

    return avg_loss, rouge_scores


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    tokenizer,
    device,
    epochs,
    max_target_len,
    checkpoint_dir: str = None,
    patience: int = 2,
    grad_accum_steps: int = 1,
):
    """
    Full seq2seq training loop with:
    - gradient accumulation
    - early stopping
    - checkpointing best model

    Returns a pandas DataFrame of metrics.
    """

    history = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    # Prepare checkpoint directory
    if checkpoint_dir is not None:
        ckpt_dir = Path(checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        step = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch in progress:
            step += 1

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            # Gradient accumulation
            (loss / grad_accum_steps).backward()

            if step % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_train_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.3f}"})

        # Handle any leftover grads if len(train_loader) % grad_accum_steps != 0
        if step % grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_train_loss = total_train_loss / len(train_loader)

        # ---- VALIDATION ----
        val_loss, rouge = evaluate_on_validation(
            model=model,
            loader=val_loader,
            tokenizer=tokenizer,
            device=device,
            max_target_len=max_target_len,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "rouge1": rouge["rouge1"],
            "rouge2": rouge["rouge2"],
            "rougeL": rouge["rougeL"],
            "rougeLsum": rouge["rougeLsum"],
        }
        history.append(epoch_record)

        print(f"\nEpoch {epoch} complete.")
        print(f"  Train loss: {avg_train_loss:.4f}")
        print(f"  Val loss:   {val_loss:.4f}")

        # ---- CHECKPOINT ----
        if checkpoint_dir is not None and val_loss < best_val_loss:
            print("  ✓ New best model — saving checkpoint.")

            best_val_loss = val_loss
            epochs_without_improvement = 0

            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)

        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s).")

        # ---- EARLY STOPPING ----
        if epochs_without_improvement >= patience:
            print("\n🛑 Early stopping triggered!")
            break

    return pd.DataFrame(history)