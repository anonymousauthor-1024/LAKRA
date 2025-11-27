"""
Generate entity2id.txt and relation2id.txt files
Extract all unique entities and relations from triplets file, and assign IDs starting from 0
"""

def generate_id_mappings(input_file, entity_output_file, relation_output_file):
    """
    Generate entity and relation ID mapping files from triplets file
    
    Args:
        input_file: Input triplets file path (format: head_entity\trelation\ttail_entity)
        entity_output_file: Output entity mapping file path
        relation_output_file: Output relation mapping file path
    """
    entities = set()
    relations = set()
    
    # First scan: collect all unique entities and relations
    print(f"Reading file: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) != 3:
                print(f"Warning: Line {line_num} has incorrect format, skipping: {line}")
                continue
            
            head, relation, tail = parts
            entities.add(head)
            entities.add(tail)
            relations.add(relation)
    
    # Sort to ensure consistency (optional)
    #entities = sorted(entities)
    #relations = sorted(relations)
    
    print(f"\nStatistics:")
    print(f"- Unique entities: {len(entities)}")
    print(f"- Unique relations: {len(relations)}")
    
    # Generate entity2id.txt
    print(f"\nGenerating: {entity_output_file}")
    with open(entity_output_file, 'w', encoding='utf-8') as f:
        f.write(f"{len(entities)}\n")  # First line: total number of entities
        for entity_id, entity in enumerate(entities):
            f.write(f"{entity}\t{entity_id}\n")
    
    print(f"✓ Generated {entity_output_file}, containing {len(entities)} entities")
    
    # Generate relation2id.txt
    print(f"Generating: {relation_output_file}")
    with open(relation_output_file, 'w', encoding='utf-8') as f:
        f.write(f"{len(relations)}\n")  # First line: total number of relations
        for relation_id, relation in enumerate(relations):
            f.write(f"{relation}\t{relation_id}\n")
    
    print(f"✓ Generated {relation_output_file}, containing {len(relations)} relations")
    
    # Display some examples
    print(f"\nEntity mapping examples (first 5):")
    for i, entity in enumerate(entities[:5]):
        print(f"  {entity} -> {i}")
    
    print(f"\nRelation mapping examples (first 5):")
    for i, relation in enumerate(relations[:5]):
        print(f"  {relation} -> {i}")
    
    return entities, relations


if __name__ == "__main__":
    import sys
    
    # Can specify file paths via command line arguments, or use default paths
    if len(sys.argv) >= 4:
        input_file = sys.argv[1]
        entity_output = sys.argv[2]
        relation_output = sys.argv[3]
    else:
        # Default paths - please modify according to actual situation
        input_file = "./train.tsv"
        entity_output = "./entity2id.txt"
        relation_output = "./relation2id.txt"
        
        print("Using default paths:")
        print(f"  Input file: {input_file}")
        print(f"  Entity mapping output: {entity_output}")
        print(f"  Relation mapping output: {relation_output}")
        print("\nTip: You can specify file paths via command line arguments:")
        print("  python generate_id_mappings.py <input_file> <entity_output> <relation_output>\n")
    
    try:
        entities, relations = generate_id_mappings(input_file, entity_output, relation_output)
        print("\n✅ All files generated successfully!")
    except FileNotFoundError as e:
        print(f"\n❌ Error: File not found {e.filename}")
        print("Please check if the file path is correct")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
