import json

# Assume triplet file is train.tsv and entity type file is entity2type.txt
triples_file = './fb15k-237/FB15k-237/train.tsv'
entity_type_file = './fb15k-237/FB15k-237/entity2type.txt'
output_file = 'relation_entity_types.json'  # Output file

# Read entity types and build mapping
entity_to_type = {}
with open(entity_type_file, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            entity_id, entity_type = parts
            entity_to_type[entity_id] = entity_type

# Count head and tail entity types for each relation
relation_types = {}

with open(triples_file, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 3:
            head_entity, relation, tail_entity = parts
            
            # Get head and tail entity types
            head_type = entity_to_type.get(head_entity, None)
            tail_type = entity_to_type.get(tail_entity, None)
            
            # Initialize relation type dictionary
            if relation not in relation_types:
                relation_types[relation] = {
                    "head_entity_types": set(),
                    "tail_entity_types": set()
                }
            
            # Count head entity types
            if head_type:
                relation_types[relation]["head_entity_types"].add(head_type)
            
            # Count tail entity types
            if tail_type:
                relation_types[relation]["tail_entity_types"].add(tail_type)

# Convert sets to lists for JSON serialization
for relation in relation_types:
    relation_types[relation]["head_entity_types"] = list(relation_types[relation]["head_entity_types"])
    relation_types[relation]["tail_entity_types"] = list(relation_types[relation]["tail_entity_types"])

# Write results to JSON file
with open(output_file, 'w', encoding='utf-8') as out_file:
    json.dump(relation_types, out_file, ensure_ascii=False, indent=4)

print(f"Results have been written to {output_file}.")
'''
'''
import json

# Assume triplet file is train.tsv and entity type file is entity2type.txt
triples_file = './fb15k-237/FB15k-237/train.tsv'
entity_type_file = './fb15k-237/FB15k-237/entity2type.txt'
output_file = 'relation_entity_types.json'  # Output file

# Read entity types and build mapping
entity_to_type = {}
with open(entity_type_file, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            entity_id, entity_type = parts
            entity_to_type[entity_id] = entity_type

# Count head and tail entity types for each relation
relation_types = {}

with open(triples_file, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 3:
            head_entity, relation, tail_entity = parts
            
            # Get head and tail entity types
            head_type = entity_to_type.get(head_entity, None)
            tail_type = entity_to_type.get(tail_entity, None)
            
            # Initialize relation type dictionary
            if relation not in relation_types:
                relation_types[relation] = {
                    "head_entity_types": set(),
                    "tail_entity_types": set(),
                    "head_entity_type_counts": {},  # Count dictionary
                    "tail_entity_type_counts": {}   # Count dictionary
                }
            
            # Count head entity types
            if head_type:
                relation_types[relation]["head_entity_types"].add(head_type)
                # Count head entity types
                if head_type not in relation_types[relation]["head_entity_type_counts"]:
                    relation_types[relation]["head_entity_type_counts"][head_type] = 0
                relation_types[relation]["head_entity_type_counts"][head_type] += 1
            
            # Count tail entity types
            if tail_type:
                relation_types[relation]["tail_entity_types"].add(tail_type)
                # Count tail entity types
                if tail_type not in relation_types[relation]["tail_entity_type_counts"]:
                    relation_types[relation]["tail_entity_type_counts"][tail_type] = 0
                relation_types[relation]["tail_entity_type_counts"][tail_type] += 1

# Convert sets to lists for JSON serialization
for relation in relation_types:
    relation_types[relation]["head_entity_types"] = list(relation_types[relation]["head_entity_types"])
    relation_types[relation]["tail_entity_types"] = list(relation_types[relation]["tail_entity_types"])

# Write results to JSON file
with open(output_file, 'w', encoding='utf-8') as out_file:
    json.dump(relation_types, out_file, ensure_ascii=False, indent=4)

print(f"Results have been written to {output_file}.")


'''
# Specify the entity type to find
target_entity_type = "City"

# Read JSON file
with open('relation_entity_types.json', 'r', encoding='utf-8') as in_file:
    relation_types = json.load(in_file)

# Find relations with specified entity type as head entity
matching_relations = []

for relation, types in relation_types.items():
    if target_entity_type in types["head_entity_types"] and types["head_entity_type_counts"].get(target_entity_type, 0) > 2:
        matching_relations.append(relation)

# Output results
if matching_relations:
    print(f"Relations with '{target_entity_type}' as head entity:")
    for relation in matching_relations:
        print(relation)
else:
    print(f"No relations found with '{target_entity_type}' as head entity.")
'''