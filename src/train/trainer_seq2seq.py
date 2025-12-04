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
from pathlib import Path

from torch.cuda.amp import autocast, GradScaler

# Use our standardized ROUGE computation
from src.eval.rouge_eval import compute_rouge_from_lists


def evaluate_on_validation(
    model,
    loader,
    tokenizer,
    device,
    max_target_len,
    num_beams: int = 4,
    no_repeat_ngram_size: int = 3,
    length_penalty: float = 1.0,
    min_length: int = 1,
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

            # Get decoder attention mask if available
            decoder_attention_mask = batch.get("decoder_attention_mask")
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(device)

            # Loss
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            total_loss += loss.item()
            steps += 1

            # Generation with consistent config
            generation_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_length": max_target_len,
                "num_beams": num_beams,
                "no_repeat_ngram_size": no_repeat_ngram_size,
                "early_stopping": True,
                "length_penalty": length_penalty,
                "min_length": min_length,
            }
            
            generated = model.generate(**generation_kwargs)

            preds.extend(
                tokenizer.batch_decode(generated, skip_special_tokens=True)
            )


            # Restore padding for label decoding
            # We need to replace -100 (ignore index) with a valid token ID
            # so the tensor can be decoded properly.
            if tokenizer.pad_token_id is not None:
                replacement_token_id = tokenizer.pad_token_id
            elif tokenizer.eos_token_id is not None:
                replacement_token_id = tokenizer.eos_token_id
            else:
                # This should never happen with properly configured tokenizers,
                # but we handle it gracefully
                raise ValueError(
                    "Tokenizer has neither pad_token_id nor eos_token_id set. "
                    "Cannot decode labels safely. Please configure the tokenizer "
                    "with at least one of these special tokens."
                )

            labels_dec = torch.where(
                labels != -100,
                labels,
                replacement_token_id,
            )
            refs.extend(
                tokenizer.batch_decode(labels_dec, skip_special_tokens=True)
            )


    avg_loss = total_loss / max(1, steps)
    rouge_scores = compute_rouge_from_lists(predictions=preds, references=refs)


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
    use_amp: bool = False,
    num_beams: int = 4,
    no_repeat_ngram_size: int = 3,
    length_penalty: float = 1.0,
    min_length: int = 1,
):
    """
    Full seq2seq training loop with:
    - gradient accumulation
    - early stopping
    - checkpointing best model
    - reloading best model before returning  # <-- NEW

    Returns a pandas DataFrame of metrics.
    """

    history = []
    best_val_loss = float("inf")
    best_epoch = None  # <-- NEW: Track which epoch was best
    epochs_without_improvement = 0
    scaler = GradScaler() if use_amp else None

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

            # Get decoder attention mask if available
            decoder_attention_mask = batch.get("decoder_attention_mask")
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(device)

            # Training step
            if use_amp:
                with autocast():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        decoder_attention_mask=decoder_attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss
                scaler.scale(loss / grad_accum_steps).backward()
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_attention_mask=decoder_attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                (loss / grad_accum_steps).backward()

            # Optimizer step
            if step % grad_accum_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad()

            total_train_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.3f}"})

        # Handle any leftover grads if len(train_loader) % grad_accum_steps != 0
        if step % grad_accum_steps != 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
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
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=length_penalty,
            min_length=min_length,
        )

        # Check if this epoch is meaningfully better than the previous best
        # We use a small tolerance to ignore floating-point noise
        IMPROVEMENT_THRESHOLD = 1e-6
        improvement = best_val_loss - val_loss
        is_best = improvement > IMPROVEMENT_THRESHOLD

        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "rouge1": rouge["rouge1"],
            "rouge2": rouge["rouge2"],
            "rougeL": rouge["rougeL"],
            "rougeLsum": rouge["rougeLsum"],
            "improved": is_best,  # was: "is_best"
        }
        history.append(epoch_record)

        print(f"\nEpoch {epoch} complete.")
        print(f"  Train loss: {avg_train_loss:.4f}")
        print(f"  Val loss:   {val_loss:.4f}")

        # ---- CHECKPOINT & EARLY STOPPING TRACKING ----
        # IMPORTANT: We must track improvement SEPARATELY from checkpointing.
        # Otherwise, early stopping breaks when checkpoint_dir is None.
        
        if is_best:
            # Update best tracking regardless of whether we're saving
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            print("  ✓ New best validation loss!")
            
            # Only save checkpoint if directory is configured
            if checkpoint_dir is not None:
                print(f"    Saving checkpoint to {ckpt_dir}")
                model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                
                # Explicitly save generation config
                if hasattr(model, 'generation_config'):
                    model.generation_config.save_pretrained(ckpt_dir)
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s).")

        # ---- EARLY STOPPING ----
        if epochs_without_improvement >= patience:
            print("\n🛑 Early stopping triggered!")
            break

    # Reload best checkpoint before returning
    if checkpoint_dir is not None and best_epoch is not None:
        print(f"\n📦 Reloading best model from epoch {best_epoch}...")
        
        try:
            # Determine model class from the existing model
            model_class = model.__class__
            
            # Load fresh model from checkpoint
            reloaded_model = model_class.from_pretrained(ckpt_dir)
            reloaded_model.to(device)
            
            # Verify the state dicts are compatible before loading
            model_keys = set(model.state_dict().keys())
            reloaded_keys = set(reloaded_model.state_dict().keys())
            
            if model_keys != reloaded_keys:
                missing = model_keys - reloaded_keys
                unexpected = reloaded_keys - model_keys
                
                error_msg = "State dict mismatch!\n"
                if missing:
                    error_msg += f"  Missing keys: {list(missing)[:5]}...\n"
                if unexpected:
                    error_msg += f"  Unexpected keys: {list(unexpected)[:5]}...\n"
                
                raise ValueError(error_msg)
            
            # Copy weights to existing model (preserves the reference)
            # strict=True ensures all keys match exactly
            model.load_state_dict(reloaded_model.state_dict(), strict=True)
            
            # Clean up
            del reloaded_model
            
            print("  ✓ Best model weights restored successfully.")
            
            # Verify by checking a sample of weights
            print(f"    Verified: Model is now from epoch {best_epoch}")
            
        except Exception as e:
            # This is now a CRITICAL error, not just a warning
            print(f"\n❌ CRITICAL: Could not reload best weights!")
            print(f"   Error: {e}")
            print(f"\n   The model in memory has weights from epoch {epoch} (last epoch),")
            print(f"   but the BEST model was from epoch {best_epoch}.")
            print(f"\n   To use the best model, manually load it:")
            print(f"   >>> model = {model_class.__name__}.from_pretrained('{ckpt_dir}')")
            print(f"\n   Continuing with last-epoch weights for now...")
            
            # Add a flag to the history so this is visible
            if history:
                history[-1]["reload_failed"] = True
    
    # Create the history DataFrame
    history_df = pd.DataFrame(history)

    # =================================================================
    # Determine which epoch's weights are currently in memory
    # =================================================================
    if checkpoint_dir is not None and best_epoch is not None:
        reload_failed = history[-1].get("reload_failed", False) if history else False
        
        if reload_failed:
            weights_epoch = epoch  # Last epoch (reload failed)
            weights_note = "RELOAD FAILED - weights are from last epoch, not best"
        else:
            weights_epoch = best_epoch
            weights_note = "Best epoch weights loaded successfully"
    else:
        weights_epoch = epoch
        weights_note = "No checkpointing - weights are from last epoch"

    # =================================================================
    # Save metadata to a separate JSON file (not duplicated in every row)
    # =================================================================
    training_metadata = {
        "weights_epoch": weights_epoch,
        "weights_note": weights_note,
        "best_epoch": best_epoch if best_epoch is not None else -1,
        "total_epochs_run": epoch,
        "early_stopped": epochs_without_improvement >= patience,
    }
    
    if checkpoint_dir is not None:
        import json
        metadata_path = Path(checkpoint_dir) / "training_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(training_metadata, f, indent=2)
        print(f"  Saved training metadata to: {metadata_path}")

    # Keep attrs for in-memory use (these don't persist to CSV, but that's OK now)
    history_df.attrs["weights_epoch"] = weights_epoch
    history_df.attrs["weights_note"] = weights_note
    history_df.attrs["training_metadata"] = training_metadata

    return history_df
