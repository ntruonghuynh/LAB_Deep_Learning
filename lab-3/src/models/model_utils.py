"""
Model utilities for Hugging Face transformers.

Functions for loading models, tokenizers, and demonstrating tokenization.
"""

from typing import Dict, Any, List, Optional
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
import torch


def load_sentiment_pipeline(
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
):
    """
    Load a pre-trained sentiment analysis pipeline.

    Args:
        model_name: Name of the pre-trained model from Hugging Face Hub

    Returns:
        Sentiment analysis pipeline
    """
    sentiment_analyzer = pipeline("sentiment-analysis", model=model_name)
    print(f"✓ Loaded sentiment analysis pipeline: {model_name}")
    return sentiment_analyzer


def demonstrate_tokenization(
    text: str,
    tokenizer_name: str = "distilbert-base-uncased"
) -> Dict[str, Any]:
    """
    Demonstrate the tokenization process step by step.

    Args:
        text: Input text to tokenize
        tokenizer_name: Name of the tokenizer

    Returns:
        Dictionary containing tokenization results at each step
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Step 1: Original text
    print(f"\n{'='*60}")
    print("Tokenization Process")
    print(f"{'='*60}")
    print(f"Original text:\n  {text}\n")

    # Step 2: Tokenize to subwords
    tokens = tokenizer.tokenize(text)
    print(f"Tokens ({len(tokens)}):\n  {tokens}\n")

    # Step 3: Convert tokens to IDs
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    print(f"Token IDs:\n  {token_ids}\n")

    # Step 4: Full encoding with special tokens
    encoded = tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    print(f"Full encoding:")
    print(f"  input_ids: {encoded['input_ids'][0].tolist()}")
    print(f"  attention_mask: {encoded['attention_mask'][0].tolist()}\n")

    # Step 5: Decode back to text
    decoded = tokenizer.decode(encoded['input_ids'][0])
    print(f"Decoded text:\n  {decoded}")
    print(f"{'='*60}\n")

    return {
        'original': text,
        'tokens': tokens,
        'token_ids': token_ids,
        'encoded': encoded,
        'decoded': decoded
    }


def load_model_and_tokenizer(
    model_name: str = "distilbert-base-uncased",
    num_labels: int = 2
):
    """
    Load a pre-trained model and tokenizer for finetuning.

    Args:
        model_name: Name of the pre-trained model
        num_labels: Number of output labels

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"✓ Model loaded successfully")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")

    return model, tokenizer


def analyze_model_architecture(model) -> Dict[str, Any]:
    """
    Analyze model architecture and parameter counts.

    Args:
        model: Hugging Face model

    Returns:
        Dictionary of architecture statistics
    """
    stats = {
        'total_params': sum(p.numel() for p in model.parameters()),
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'layers': {},
        'model_size_mb': sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
    }

    # Count parameters by layer
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        stats['layers'][name] = params

    return stats


def print_model_info(model, tokenizer) -> None:
    """
    Print detailed model information.

    Args:
        model: Hugging Face model
        tokenizer: Hugging Face tokenizer
    """
    stats = analyze_model_architecture(model)

    print(f"\n{'='*60}")
    print("Model Information")
    print(f"{'='*60}")
    print(f"Model class: {model.__class__.__name__}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"Model config: {model.config.name_or_path}")
    print(f"\nParameter counts:")
    print(f"  Total parameters: {stats['total_params']:,}")
    print(f"  Trainable parameters: {stats['trainable_params']:,}")
    print(f"  Model size: {stats['model_size_mb']:.2f} MB")

    print(f"\nParameters by layer:")
    for layer_name, param_count in stats['layers'].items():
        percentage = (param_count / stats['total_params']) * 100
        print(f"  {layer_name:20s}: {param_count:12,} ({percentage:5.2f}%)")

    print(f"{'='*60}\n")


def test_model_output(
    model,
    tokenizer,
    text: str,
    label_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Test model output on a sample text.

    Args:
        model: Hugging Face model
        tokenizer: Hugging Face tokenizer
        text: Input text
        label_names: Optional list of label names

    Returns:
        Dictionary with predictions
    """
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Get prediction
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=-1).item()

    result = {
        'text': text,
        'logits': logits[0].tolist(),
        'probabilities': probs[0].tolist(),
        'predicted_class': predicted_class,
    }

    if label_names:
        result['predicted_label'] = label_names[predicted_class]

    # Print results
    print(f"\nPrediction for: \"{text}\"")
    print(f"  Predicted class: {predicted_class}")
    if label_names:
        print(f"  Predicted label: {label_names[predicted_class]}")
    print(f"  Probabilities:")
    for i, prob in enumerate(probs[0].tolist()):
        label = label_names[i] if label_names else f"Class {i}"
        print(f"    {label}: {prob:.4f}")

    return result


def compare_tokenizers(
    text: str,
    tokenizer_names: List[str]
) -> None:
    """
    Compare how different tokenizers handle the same text.

    Args:
        text: Input text
        tokenizer_names: List of tokenizer names to compare
    """
    print(f"\n{'='*60}")
    print(f"Tokenizer Comparison")
    print(f"{'='*60}")
    print(f"Text: {text}\n")

    for name in tokenizer_names:
        tokenizer = AutoTokenizer.from_pretrained(name)
        tokens = tokenizer.tokenize(text)
        encoded = tokenizer(text)

        print(f"{name}:")
        print(f"  Tokens ({len(tokens)}): {tokens}")
        print(f"  Token IDs: {encoded['input_ids']}")
        print()

    print(f"{'='*60}\n")
