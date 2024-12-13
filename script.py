import argparse
import os
import torch
from transformers import MarianMTModel, MarianTokenizer
from torch.utils.data import DataLoader, Dataset
import random
import sacrebleu
from nltk.translate.meteor_score import single_meteor_score
from sacrebleu.metrics import TER

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset Class for Monolingual Data
class MonolingualDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128, fraction=0.001):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        selected_size = int(len(lines) * fraction)
        self.lines = random.sample(lines, selected_size)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx].strip()
        encoding = self.tokenizer(line, return_tensors="pt", max_length=self.max_length, padding="max_length", truncation=True)
        return encoding["input_ids"].squeeze(), encoding["attention_mask"].squeeze()

# Helper Function: Add Noise to Monolingual Data
def add_noise(input_ids, tokenizer, noise_prob=0.1):
    noisy_input_ids = input_ids.clone()
    vocab_size = tokenizer.vocab_size
    for i in range(noisy_input_ids.size(0)):
        for j in range(noisy_input_ids.size(1)):
            if random.random() < noise_prob:
                noisy_input_ids[i, j] = random.randint(0, vocab_size - 1)  # Replace with random token ID
    return noisy_input_ids

# Helper Function: Back-Translation Step
def back_translate(source_ids, source_mask, source_lang, target_lang, models, tokenizers):
    # Forward translation
    model_fwd = models[f"{source_lang}-{target_lang}"]
    tokenizer_fwd = tokenizers[f"{source_lang}-{target_lang}"]
    with torch.no_grad():
        generated_ids = model_fwd.generate(source_ids, attention_mask=source_mask)
    target_texts = tokenizer_fwd.batch_decode(generated_ids, skip_special_tokens=True)

    # Back translation
    tokenizer_bwd = tokenizers[f"{target_lang}-{source_lang}"]
    model_bwd = models[f"{target_lang}-{source_lang}"]
    target_ids = tokenizer_bwd(target_texts, return_tensors="pt", padding="max_length", truncation=True).input_ids.to(device)
    with torch.no_grad():
        generated_back_ids = model_bwd.generate(target_ids)
    return generated_back_ids

# Training Function
def train_monolingual(models, tokenizers, source_file, source_lang, target_lang, epochs=1, batch_size=8):
    dataset = MonolingualDataset(source_file, tokenizers[f"{source_lang}-{target_lang}"])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(
        list(models[f"{source_lang}-{target_lang}"].parameters()) +
        list(models[f"{target_lang}-{source_lang}"].parameters()), lr=1e-5
    )

    for epoch in range(epochs):
        total_loss = 0
        for step, (input_ids, attention_mask) in enumerate(dataloader):
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            noisy_input_ids = add_noise(input_ids, tokenizers[f"{source_lang}-{target_lang}"])

            # Train forward translation (denoising autoencoder)
            model_fwd = models[f"{source_lang}-{target_lang}"]
            outputs = model_fwd(input_ids=noisy_input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            total_loss += loss.item()

            # Train back-translation
            generated_back_ids = back_translate(input_ids, attention_mask, source_lang, target_lang, models, tokenizers)
            model_bwd = models[f"{target_lang}-{source_lang}"]
            outputs = model_bwd(input_ids=generated_back_ids, labels=input_ids)
            loss += outputs.loss
            total_loss += loss.item()

            # Update model weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss / len(dataloader):.4f}")

    # Save models
    models[f"{source_lang}-{target_lang}"].save_pretrained(f"marianmt_{source_lang}_{target_lang}")
    tokenizers[f"{source_lang}-{target_lang}"].save_pretrained(f"marianmt_{source_lang}_{target_lang}")
    print(f"Saved forward model: marianmt_{source_lang}_{target_lang}/")

    models[f"{target_lang}-{source_lang}"].save_pretrained(f"marianmt_{target_lang}_{source_lang}")
    tokenizers[f"{target_lang}-{source_lang}"].save_pretrained(f"marianmt_{target_lang}_{source_lang}")
    print(f"Saved backward model: marianmt_{target_lang}_{source_lang}/")

# Evaluation Function
def evaluate_parallel(test_file, models, tokenizers, source_lang, target_lang, fraction=1.0):
    with open(test_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Select a fraction of the dataset
    selected_size = int(len(lines) * fraction)
    lines = random.sample(lines, selected_size)

    predictions, references = [], []
    for line in lines:
        source, reference = line.strip().split(" ||| ")
        tokenizer = tokenizers[f"{source_lang}-{target_lang}"]
        model = models[f"{source_lang}-{target_lang}"]
        inputs = tokenizer(source, return_tensors="pt", max_length=128, padding="max_length", truncation=True).to(device)
        translated_tokens = model.generate(**inputs)
        predictions.append(tokenizer.decode(translated_tokens[0], skip_special_tokens=True))
        references.append(reference)

    # Compute metrics
    bleu_score = sacrebleu.corpus_bleu(predictions, [references]).score
    meteor_scores = [single_meteor_score(ref.split(), pred.split()) for ref, pred in zip(references, predictions)]
    avg_meteor_score = sum(meteor_scores) / len(meteor_scores)
    ter_metric = TER()
    ter_score = ter_metric.corpus_score(predictions, [references]).score

    print(f"BLEU: {bleu_score:.2f}, METEOR: {avg_meteor_score:.2f}, TER: {ter_score:.2f}")

# CLI Implementation
def main():
    parser = argparse.ArgumentParser(description="Machine Translation CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Train Command
    train_parser = subparsers.add_parser("train", help="Train translation models using monolingual corpora")
    train_parser.add_argument("--source_file", required=True, help="Path to monolingual source language corpus")
    train_parser.add_argument("--source_lang", required=True, help="Source language code (e.g., en, fr, es)")
    train_parser.add_argument("--target_lang", required=True, help="Target language code (e.g., en, fr, es)")
    train_parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")

    # Evaluate Command
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate translation models using parallel corpus")
    evaluate_parser.add_argument("--test_file", required=True, help="Path to parallel test dataset")
    evaluate_parser.add_argument("--source_lang", required=True, help="Source language code")
    evaluate_parser.add_argument("--target_lang", required=True, help="Target language code")
    evaluate_parser.add_argument("--model_dir", required=True, help="Directory of the trained model")
    evaluate_parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of the dataset to use for testing")

    args = parser.parse_args()

    # Load models and tokenizers
    models, tokenizers = {}, {}
    lang_pairs = [("en", "fr"), ("en", "es"), ("fr", "es")]
    for src, tgt in lang_pairs:
        models[f"{src}-{tgt}"] = MarianMTModel.from_pretrained(f"Helsinki-NLP/opus-mt-{src}-{tgt}").to(device)
        models[f"{tgt}-{src}"] = MarianMTModel.from_pretrained(f"Helsinki-NLP/opus-mt-{tgt}-{src}").to(device)
        tokenizers[f"{src}-{tgt}"] = MarianTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{src}-{tgt}")
        tokenizers[f"{tgt}-{src}"] = MarianTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{tgt}-{src}")

    if args.command == "train":
        train_monolingual(models, tokenizers, args.source_file, args.source_lang, args.target_lang, epochs=args.epochs)

    elif args.command == "evaluate":
        # Load models and tokenizers from the specified directory
        model_dir = args.model_dir
        models[f"{args.source_lang}-{args.target_lang}"] = MarianMTModel.from_pretrained(model_dir).to(device)
        tokenizers[f"{args.source_lang}-{args.target_lang}"] = MarianTokenizer.from_pretrained(model_dir)
        
        evaluate_parallel(args.test_file, models, tokenizers, args.source_lang, args.target_lang, fraction=args.fraction)

if __name__ == "__main__":
    main()
