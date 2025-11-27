from collections import defaultdict

# Initialize entity counter
entity_counter = defaultdict(int)

# Read all triplet files (train/valid/test)
for file_path in ["./wn18rr/WN18RR/train.tsv"]:
    with open(file_path, 'r') as f:
        for line in f:
            head, relation, tail = line.strip().split('\t')
            entity_counter[head] += 1
            entity_counter[tail] += 1

# Sort entities by occurrence count in ascending order (low frequency first)
sorted_entities = sorted(entity_counter.items(), key=lambda x: x[1])

# User input: percentage of entities to extract from the tail (e.g., last 20%)
target_percent = 0.001  # Modify this to adjust percentage
n_total = len(sorted_entities)
n_low_freq = int(n_total * target_percent)  # Round down

# Extract last N% low frequency entities
low_freq_entities = sorted_entities[:n_low_freq]  # Already sorted in ascending order, take first n_low_freq

# Output statistics
print(f"Total entities: {n_total}")
print(f"Last {target_percent*100:.0f}% entity count: {n_low_freq}")
print(f"Lowest frequency entity example: {low_freq_entities[0][0]} (occurrence count={low_freq_entities[0][1]})")

# Save results to file
with open("low_frequency_entities_wn18rr.txt", 'w') as f:
    for entity, count in low_freq_entities:
        f.write(f"{entity}\t{count}\n")