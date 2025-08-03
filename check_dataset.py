"""
Quick script to check the structure of the HuggingFace dataset.
"""

from datasets import load_dataset

print("Loading dataset to check structure...")
dataset = load_dataset("Synthyra/homodimer_benchmark", split="train", streaming=True)

# Get first example
first_example = next(iter(dataset))
print(f"\nDataset columns: {list(first_example.keys())}")
print(f"\nFirst example:")
for key, value in first_example.items():
    if isinstance(value, str) and len(value) > 50:
        print(f"  {key}: {value[:50]}... (length: {len(value)})")
    else:
        print(f"  {key}: {value}")

# Try a few more examples
print("\nChecking a few more examples:")
for i, example in enumerate(dataset.take(5)):
    print(f"\nExample {i+1}:")
    for key, value in example.items():
        if isinstance(value, str) and len(value) > 30:
            print(f"  {key}: {value[:30]}... (length: {len(value)})")
        else:
            print(f"  {key}: {value}")