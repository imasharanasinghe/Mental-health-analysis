import sys
import os

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from src.data import preprocess, load_dataset

# Load and preprocess the dataset
df = preprocess(load_dataset('data/dataset.csv'))

# Get rows with valid severity data
valid_severity = df.dropna(subset=['Severity'])

print("=== Severity Prediction Information ===")
print(f"Total records: {len(df)}")
print(f"Records with severity data: {len(valid_severity)}")
print(f"Percentage with data: {(len(valid_severity) / len(df)) * 100:.1f}%")
print()

print("Severity distribution:")
severity_counts = valid_severity['Severity'].value_counts()
print(severity_counts)
print()

print("Severity categories (ordered):")
print(valid_severity['Severity'].cat.categories.tolist())
print()

# Show some examples for each severity level
print("=== Sample Inputs for Each Severity Level ===")

# None severity (best mental health)
none_examples = valid_severity[valid_severity['Severity'] == 'None'].head(1)
if len(none_examples) > 0:
    example = none_examples.iloc[0]
    print("1. For None Severity (Best Mental Health):")
    print(f"   Age: {int(example['Age'])}")
    print(f"   Sleep_Hours: {example['Sleep_Hours']}")
    print(f"   Work_Hours: {example['Work_Hours']}")
    print(f"   Physical_Activity_Hours: {example['Physical_Activity_Hours']}")
    print(f"   Social_Media_Usage: {example['Social_Media_Usage']}")
    print()

# Low severity
low_examples = valid_severity[valid_severity['Severity'] == 'Low'].head(1)
if len(low_examples) > 0:
    example = low_examples.iloc[0]
    print("2. For Low Severity:")
    print(f"   Age: {int(example['Age'])}")
    print(f"   Sleep_Hours: {example['Sleep_Hours']}")
    print(f"   Work_Hours: {example['Work_Hours']}")
    print(f"   Physical_Activity_Hours: {example['Physical_Activity_Hours']}")
    print(f"   Social_Media_Usage: {example['Social_Media_Usage']}")
    print()

# Medium severity
medium_examples = valid_severity[valid_severity['Severity'] == 'Medium'].head(1)
if len(medium_examples) > 0:
    example = medium_examples.iloc[0]
    print("3. For Medium Severity:")
    print(f"   Age: {int(example['Age'])}")
    print(f"   Sleep_Hours: {example['Sleep_Hours']}")
    print(f"   Work_Hours: {example['Work_Hours']}")
    print(f"   Physical_Activity_Hours: {example['Physical_Activity_Hours']}")
    print(f"   Social_Media_Usage: {example['Social_Media_Usage']}")
    print()

# High severity
high_examples = valid_severity[valid_severity['Severity'] == 'High'].head(1)
if len(high_examples) > 0:
    example = high_examples.iloc[0]
    print("4. For High Severity:")
    print(f"   Age: {int(example['Age'])}")
    print(f"   Sleep_Hours: {example['Sleep_Hours']}")
    print(f"   Work_Hours: {example['Work_Hours']}")
    print(f"   Physical_Activity_Hours: {example['Physical_Activity_Hours']}")
    print(f"   Social_Media_Usage: {example['Social_Media_Usage']}")
    print()