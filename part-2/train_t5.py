import os
import argparse
import pickle
import shutil
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import numpy as np
import wandb

from t5_utils import initialize_model, initialize_optimizer_and_scheduler, save_model, load_model_from_checkpoint, setup_wandb
from transformers import T5TokenizerFast
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records, set_random_seeds

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0


def _generation_kwargs(args, tokenizer):
    return dict(
        max_length=args.max_gen_length,
        num_beams=args.num_beams,
        early_stopping=True,
        repetition_penalty=args.repetition_penalty,
        length_penalty=args.length_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )


def get_args():
    '''
    Arguments for training. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(description='T5 training loop')

    # Model hyperparameters
    parser.add_argument('--finetune', action='store_true', help="Whether to finetune T5 or not")
    
    # Training hyperparameters
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW"],
                        help="What optimizer to use")
    parser.add_argument('--learning_rate', type=float, default=2.5e-4,
                        help="~2e-4–3e-4 is typical for T5-small fine-tuning; lower can reduce dev oscillation.")
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--label_smoothing', type=float, default=0.08,
                        help="Regularizes the CE loss; often helps generalization on seq2seq.")
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help="If >0, clip global grad norm after backward (0 disables).")

    parser.add_argument('--scheduler_type', type=str, default="cosine", choices=["none", "cosine", "linear"],
                        help="Whether to use a LR scheduler and what type to use if so")
    parser.add_argument('--num_warmup_epochs', type=int, default=2,
                        help="How many epochs to warm up the learning rate for if using a scheduler")
    parser.add_argument('--max_n_epochs', type=int, default=40,
                        help="How many epochs to train the model for")
    parser.add_argument('--patience_epochs', type=int, default=15,
                        help="If validation performance stops improving, how many epochs should we wait before stopping?")

    parser.add_argument('--use_wandb', action='store_true',
                        help="If set, we will use wandb to keep track of experiments")
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help="How should we name this experiment?")

    # Data hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)

    # Decoding (dev/test generation)
    parser.add_argument('--num_beams', type=int, default=10,
                        help="Beam size for model.generate (larger is slower but often better F1).")
    parser.add_argument('--max_gen_length', type=int, default=384,
                        help="Should be >= max SQL token length used in load_data (384).")
    parser.add_argument('--repetition_penalty', type=float, default=1.0,
                        help="1.0 is often safer for structured SQL; >1 can hurt valid repeats.")
    parser.add_argument('--length_penalty', type=float, default=1.0)

    parser.add_argument(
        '--keep_previous_artifacts',
        action='store_true',
        help="Do not delete this experiment's old checkpoints/results before training (default: clear so writes never conflict).",
    )

    args = parser.parse_args()
    return args


def _safe_remove(path):
    if path and os.path.isfile(path):
        os.remove(path)


def clear_experiment_artifacts(args):
    """
    Remove prior outputs for this experiment_name so new runs can always write.
    Does not touch shared files like records/dev_gt_records.pkl.
    """
    model_type = "ft" if args.finetune else "scr"
    name = args.experiment_name
    checkpoint_dir = os.path.join("checkpoints", f"{model_type}_experiments", name)
    if os.path.isdir(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)

    os.makedirs("results", exist_ok=True)
    os.makedirs("records", exist_ok=True)
    os.makedirs(os.path.join("checkpoints", f"{model_type}_experiments"), exist_ok=True)

    for split in ("dev", "test"):
        _safe_remove(os.path.join("results", f"t5_{model_type}_{name}_{split}.sql"))
        _safe_remove(os.path.join("records", f"t5_{model_type}_{name}_{split}.pkl"))

def ensure_results_dirs_and_dev_gt():
    os.makedirs("results", exist_ok=True)
    os.makedirs("records", exist_ok=True)
    if not os.path.exists("records/dev_gt_records.pkl"):
        from utils import compute_records
        with open("data/dev.sql") as f:
            dev_queries = [q.strip() for q in f.readlines()]
        records, error_msgs = compute_records(dev_queries)
        payload = (records, error_msgs)
        with open("records/dev_gt_records.pkl", "wb") as f:
            pickle.dump(payload, f)
    if not os.path.exists("records/ground_truth_dev.pkl"):
        shutil.copy("records/dev_gt_records.pkl", "records/ground_truth_dev.pkl")

def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    if not getattr(args, "keep_previous_artifacts", False):
        clear_experiment_artifacts(args)
        print(
            f"Cleared previous checkpoints/results for experiment '{args.experiment_name}' "
            f"(use --keep_previous_artifacts to skip)."
        )

    best_f1 = -1
    epochs_since_improvement = 0

    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/dev_gt_records.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{args.experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{args.experiment_name}_dev.pkl')
    for epoch in range(args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}: Average train loss was {tr_loss}")

        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(args, model, dev_loader,
                                                                         gt_sql_path, model_sql_path,
                                                                         gt_record_path, model_record_path)
        print(f"Epoch {epoch}: Dev loss: {eval_loss}, Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}")
        print(f"Epoch {epoch}: {error_rate*100:.2f}% of the generated outputs led to SQL errors")

        if args.use_wandb:
            result_dict = {
                'train/loss' : tr_loss,
                'dev/loss' : eval_loss,
                'dev/record_f1' : record_f1,
                'dev/record_em' : record_em,
                'dev/sql_em' : sql_em,
                'dev/error_rate' : error_rate,
            }
            wandb.log(result_dict, step=epoch)

        if record_f1 > best_f1:
            best_f1 = record_f1
            epochs_since_improvement = 0
            save_model(checkpoint_dir, model)
            print(f"  -> New best Record F1 {best_f1:.6f}; saved checkpoints/{model_type}_experiments/{args.experiment_name}/best_model.pt")
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= args.patience_epochs:
            break

def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(train_loader):
        optimizer.zero_grad()
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        logits = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_input,
        )['logits']

        non_pad = decoder_targets != PAD_IDX
        loss = criterion(logits[non_pad], decoder_targets[non_pad])
        loss.backward()
        if args.max_grad_norm and args.max_grad_norm > 0:
            clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        if scheduler is not None: 
            scheduler.step()

        with torch.no_grad():
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / total_tokens
        
def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    '''
    You must implement the evaluation loop to be using during training. We recommend keeping track
    of the model loss on the SQL queries, the metrics compute_metrics returns (save_queries_and_records should be helpful)
    and the model's syntax error rate. 

    To compute non-loss metrics, you will need to perform generation with the model. Greedy decoding or beam search
    should both provide good results. If you find that this component of evaluation takes too long with your compute,
    we found the cross-entropy loss (in the evaluation set) to be well (albeit imperfectly) correlated with F1 performance.
    '''
    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
    model.eval()
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    total_loss = 0.0
    total_tokens = 0
    all_predictions = []

    with torch.no_grad():
        for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(dev_loader):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            decoder_input = decoder_input.to(DEVICE)
            decoder_targets = decoder_targets.to(DEVICE)

            logits = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_input,
            )["logits"]

            non_pad = decoder_targets != PAD_IDX
            loss = criterion(logits[non_pad], decoder_targets[non_pad])
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

            generated = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                **_generation_kwargs(args, tokenizer),
            )
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            all_predictions.extend(decoded)

    avg_loss = total_loss / max(total_tokens, 1)
    save_queries_and_records(all_predictions, model_sql_path, model_record_path)
    sql_em, record_em, record_f1, error_msgs = compute_metrics(
        gt_sql_pth, model_sql_path, gt_record_path, model_record_path
    )
    n = len(error_msgs)
    error_rate = sum(1 for m in error_msgs if m) / n if n else 0.0
    return avg_loss, record_f1, record_em, sql_em, error_rate
        
def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    '''
    You must implement inference to compute your model's generated SQL queries and its associated 
    database records. Implementation should be very similar to eval_epoch.
    '''
    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
    model.eval()
    all_predictions = []

    with torch.no_grad():
        for encoder_input, encoder_mask, _ in tqdm(test_loader):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)

            generated = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                **_generation_kwargs(args, tokenizer),
            )
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            all_predictions.extend(decoded)

    save_queries_and_records(all_predictions, model_sql_path, model_record_path)

def main():
    set_random_seeds(42)
    ensure_results_dirs_and_dev_gt()

    # Get key arguments
    args = get_args()
    if args.use_wandb:
        # Recommended: Using wandb (or tensorboard) for result logging can make experimentation easier
        setup_wandb(args)

    # Load the data and the model
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)
    model = initialize_model(args)
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

    # Train 
    train(args, model, train_loader, dev_loader, optimizer, scheduler)

    # Evaluate
    model = load_model_from_checkpoint(args)
    model.eval()
    
    # Dev set
    experiment_name = args.experiment_name
    model_type = 'ft' if args.finetune else 'scr'
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/dev_gt_records.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_dev.pkl')
    dev_loss, dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(args, model, dev_loader,
                                                                                    gt_sql_path, model_sql_path,
                                                                                    gt_record_path, model_record_path)
    print(f"Dev set results: Loss: {dev_loss}, Record F1: {dev_record_f1}, Record EM: {dev_record_em}, SQL EM: {dev_sql_em}")
    print(f"Dev set results: {dev_error_rate*100:.2f}% of the generated outputs led to SQL errors")

    # Test set
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_test.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_test.pkl')
    test_inference(args, model, test_loader, model_sql_path, model_record_path)

if __name__ == "__main__":
    main()
