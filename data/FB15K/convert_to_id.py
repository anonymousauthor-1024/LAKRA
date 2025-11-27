"""
Convert triplets file to ID format
Convert entities and relations to IDs based on entity2id.txt and relation2id.txt
"""

def load_mappings(entity_file, relation_file):
    """
    Load entity and relation mapping dictionaries
    
    Args:
        entity_file: entity2id.txt file path
        relation_file: relation2id.txt file path
    
    Returns:
        entity2id: Entity to ID mapping dictionary
        relation2id: Relation to ID mapping dictionary
    """
    entity2id = {}
    relation2id = {}
    
    # Load entity mapping
    print(f"Loading entity mapping: {entity_file}")
    with open(entity_file, 'r', encoding='utf-8') as f:
        num_entities = int(f.readline().strip())
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                entity, entity_id = parts
                entity2id[entity] = int(entity_id)
    
    print(f"✓ Loaded {len(entity2id)} entity mappings")ty mappings")
    
    # Load relation mapping
    print(f"Loading relation mapping: {relation_file}")
    with open(relation_file, 'r', encoding='utf-8') as f:
        num_relations = int(f.readline().strip())
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                relation, relation_id = parts
                relation2id[relation] = int(relation_id)
    
    print(f"✓ Loaded {len(relation2id)} relation mappings")
    
    return entity2id, relation2id


def convert_triples_to_id(input_file, output_file, entity2id, relation2id):
    """
    Convert triplets file to ID format
    
    Args:
        input_file: Input file (format: head_entity\trelation\ttail_entity)
        output_file: Output file (format: num_triples\nhead_id relation_id tail_id)
        entity2id: Entity to ID mapping
        relation2id: Relation to ID mapping
    """
    print(f"\nConverting: {input_file} -> {output_file}")
    
    triples = []
    skipped = 0
    
    # Read and convert triplets
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) != 3:
                print(f"Warning: Line {line_num} has incorrect format, skipping")
                skipped += 1
                continue
            
            head, relation, tail = parts
            
            # Check if all entities and relations are in mappings
            if head not in entity2id:
                print(f"Warning: Line {line_num}, head entity '{head}' not in mapping, skipping")
                skipped += 1
                continue
            
            if tail not in entity2id:
                print(f"Warning: Line {line_num}, tail entity '{tail}' not in mapping, skipping")
                skipped += 1
                continue
            
            if relation not in relation2id:
                print(f"Warning: Line {line_num}, relation '{relation}' not in mapping, skipping")
                skipped += 1
                continue
            
            # Convert to IDs
            head_id = entity2id[head]
            relation_id = relation2id[relation]
            tail_id = entity2id[tail]
            
            triples.append((head_id, relation_id, tail_id))
            
            # Show progress every 100k lines
            if line_num % 100000 == 0:
                print(f"  Processed {line_num} lines...")lines...")
    
    # Write to output file
    print(f"Writing {len(triples)} triplets to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        # First line: number of triplets
        f.write(f"{len(triples)}\n")
        
        # Write all triplets (format: head_id relation_id tail_id)
        for head_id, relation_id, tail_id in triples:
            f.write(f"{head_id} {tail_id} {relation_id}\n")
    
    print(f"✓ Generated {output_file}")
    print(f"  - Successfully converted: {len(triples)} triplets")
    if skipped > 0:
        print(f"  - Skipped: {skipped} lines")
    
    return len(triples)


def main():
    import sys
    import os
    
    # Can be specified via command line argument, or use default path
    if len(sys.argv) >= 2:
        data_dir = sys.argv[1]
    else:
        data_dir = "./"
        print(f"Using default data directory: {data_dir}\n")
        print("Tip: You can specify data directory via command line argument:")
        print("  python convert_to_id.py <data_directory>\n")
    
    # File paths
    entity_file = os.path.join(data_dir, "entity2id.txt")
    relation_file = os.path.join(data_dir, "relation2id.txt")
    
    # Check if mapping files exist
    if not os.path.exists(entity_file):
        print(f"❌ Error: File not found {entity_file}")
        print("Please run generate_id_mappings.py first to generate mapping files")
        return
    
    if not os.path.exists(relation_file):
        print(f"❌ Error: File not found {relation_file}")
        print("Please run generate_id_mappings.py first to generate mapping files")
        return
    
    try:
        # Load mappings
        entity2id, relation2id = load_mappings(entity_file, relation_file)
        
        # Convert all files
        files_to_convert = [
            ("train.tsv", "train2id.txt"),
            #("dev.tsv", "valid2id.txt"),
            #("test.tsv", "test2id.txt")
        ]
        
        total_triples = 0
        for input_name, output_name in files_to_convert:
            input_file = os.path.join(data_dir, input_name)
            output_file = os.path.join(data_dir, output_name)
            
            if not os.path.exists(input_file):
                print(f"⚠ Skipping: File not found {input_file}")
                continue
            
            num_triples = convert_triples_to_id(input_file, output_file, entity2id, relation2id)
            total_triples += num_triples
        
        print(f"\n✅ All files converted successfully!")
        print(f"Total {total_triples} triplets converted")ts converted")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: File not found {e.filename}")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
