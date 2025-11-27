import json
from openai import OpenAI

def load_relation_entity_types(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_subgraph_nodes_from_file(entity, sparse_nodes_dir='../sparse_nodes/FB15K-237'):
    """
    Read subgraph nodes for specified entity from sparse_nodes folder
    
    Args:
        entity: Target entity ID
        sparse_nodes_dir: sparse_nodes folder path
    
    Returns:
        subgraph_nodes: Set of subgraph node IDs
    """
    import os
    
    # Build file path (handle special characters in entity ID)
    safe_entity_id = entity.replace('/', '_').replace('\\', '_')
    entity_file = os.path.join(os.path.dirname(__file__), sparse_nodes_dir, f"{safe_entity_id}.txt")
    
    if not os.path.exists(entity_file):
        print(f"Warning: Entity file not found: {entity_file}")
        return set()
    
    # Read nodes from file
    subgraph_nodes = set()
    in_nodes_section = False
    
    with open(entity_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Detect start of nodes section
            if 'Nodes in Subgraph:' in line:
                in_nodes_section = True
                continue
            
            # Detect start of triplets section (end of nodes section)
            if 'Triples in Subgraph:' in line:
                break
            
            # Skip separator lines
            if line.startswith('='):
                continue
            
            # In nodes section, parse nodes
            if in_nodes_section and line:
                parts = line.split('\t')
                if len(parts) >= 1:
                    node_id = parts[0]
                    subgraph_nodes.add(node_id)
    
    return subgraph_nodes

def get_tail_entitys_by_relation(data, relation, entity_id):
    candidate_entities_types = []
    entity_dict = {}
    if relation not in data:
        #relation = '/' + relation  # Add / character before relation string
        return entity_dict
    
    # Get subgraph nodes for entity_id
    subgraph_nodes = get_subgraph_nodes_from_file(entity_id, '../sparse_nodes/FB15K-237')
    if not subgraph_nodes:
        print(f"Warning: No subgraph nodes found for entity {entity_id}")
        return entity_dict
        
    # Add constraint of 10 or more
    for type in data[relation]['tail_entity_types']:
        if data[relation]["tail_entity_type_counts"][type] >= 10:
            candidate_entities_types.append(type)
    if len(candidate_entities_types) < 1:
        candidate_entities_types = data[relation]['tail_entity_types']
    for type in candidate_entities_types:
        entity_dict[type] = {}
        file_path = 'fb15k-237/FB15k-237/entity2type_with_names_and_descriptions.txt'
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:  # Ensure enough parts in line
                    candidate_entity_id = parts[0]
                    entity_type = parts[1]
                    entity_name = parts[2]
                    entity_desc = parts[3]
                    
                    # Check if entity type matches AND entity is in subgraph nodes
                    if entity_type == type and candidate_entity_id in subgraph_nodes:
                        entity_dict[type][entity_name] = entity_desc
    
    return entity_dict

def get_head_entitys_by_relation(data, relation, entity_id):
    candidate_entities_types = []
    entity_dict = {}
    if relation not in data:
         relation = '/' + relation  # Add / character before relation string
    
    # Get subgraph nodes for entity_id
    subgraph_nodes = get_subgraph_nodes_from_file(entity_id, '../sparse_nodes/FB15K-237')
    if not subgraph_nodes:
        print(f"Warning: No subgraph nodes found for entity {entity_id}")
        return entity_dict
    
    # Add constraint of 10 or more
    for type in data[relation]['head_entity_types']:
        if data[relation]["head_entity_type_counts"][type] >= 10:
            candidate_entities_types.append(type)
    if len(candidate_entities_types) < 1:
        candidate_entities_types = data[relation]['head_entity_types']
    for type in candidate_entities_types:
        entity_dict[type] = {}
        file_path = 'fb15k-237/FB15k-237/entity2type_with_names_and_descriptions.txt'
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:  # Ensure enough parts in line
                    candidate_entity_id = parts[0]
                    entity_type = parts[1]
                    entity_name = parts[2]
                    entity_desc = parts[3]
                    
                    # Check if entity type matches AND entity is in subgraph nodes
                    if entity_type == type and candidate_entity_id in subgraph_nodes:
                        entity_dict[type][entity_name] = entity_desc
    
    return entity_dict


def get_relations_by_entity_type(data, entity_type):
    head_relations = []
    tail_relations = []

    for relation, details in data.items():
        # Check head entity type
        # Prevent too many relations, add constraint to only include candidates greater than 10
        if entity_type in details['head_entity_types'] and details['head_entity_type_counts'][entity_type] > 5:
            head_relations.append(relation)
        # Check tail entity type
        if entity_type in details['tail_entity_types'] and details['tail_entity_type_counts'][entity_type] > 5:
            tail_relations.append(relation)
        
    if len(head_relations) < 1:
         for relation, details in data.items():
            # Check head entity type
            # Prevent too many relations, add constraint to only include candidates greater than 10
            if entity_type in details['head_entity_types']:
                head_relations.append(relation)

    if len(tail_relations) < 1:
        for relation, details in data.items():
            if entity_type in details['tail_entity_types']:
                tail_relations.append(relation)

    return head_relations, tail_relations

'''
# Example usage
file_path = 'relation_entity_types.json'  # Replace with your file path
data = load_relation_entity_types(file_path)

entity_type = 'Concept'  # Replace with your entity type to query
head_relations, tail_relations = get_relations_by_entity_type(data, entity_type)

print(f"Relations for '{entity_type}' as head entity: {head_relations}")
print(f"Relations for '{entity_type}' as tail entity: {tail_relations}")
'''

def load_tsv(file_path):
    triplets = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')  # Assume file is tab-separated
            if len(parts) == 3:  # Ensure head entity, relation and tail entity exist
                head_entity, relation, tail_entity = parts
                triplets.append((head_entity, relation, tail_entity))
    return triplets

def load_entity_mapping(file_path):
    entity_mapping = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')  # Assume file is tab-separated
            if len(parts) == 2:  # Ensure ID and name exist
                entity_id, entity_name = parts
                entity_mapping[entity_id] = entity_name
    return entity_mapping

def get_triplets_by_entity(triplets, entity):
    """
    Get triplets related to specified entity
    
    Args:
        triplets: Original triplet list (used when use_sparse_nodes=False)
        entity: Target entity ID
        use_sparse_nodes: Whether to read k-hop subgraph from sparse_nodes folder (default True)
        sparse_nodes_dir: sparse_nodes folder path (relative to current file)
    
    Returns:
        related_triplets: List of related triplets
    """
    sparse_nodes_dir='../sparse_nodes/FB15K-237'
        # Read k-hop subgraph for corresponding entity from sparse_nodes folder
    import os
    
    # Build file path (handle special characters in entity ID)
    safe_entity_id = entity.replace('/', '_').replace('\\', '_')
    entity_file = os.path.join(os.path.dirname(__file__), sparse_nodes_dir, f"{safe_entity_id}.txt")
    
    # Read triplets from file
    related_triplets = []
    in_triples_section = False
    
    with open(entity_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Detect start of triplets section
            if 'Triples in Subgraph:' in line:
                in_triples_section = True
                continue
            
            # Skip separator lines
            if line.startswith('='):
                continue
            
            # In triplets section, parse triplets
            if in_triples_section and line:
                parts = line.split('\t')
                if len(parts) == 3:
                    head_entity, relation, tail_entity = parts
                    related_triplets.append((head_entity, relation, tail_entity))
    
    print(f"Loaded {len(related_triplets)} triplets from {entity_file}")
    return related_triplets
  

def get_triplets_by_relations(triplets, relations):
    related_triplets = []
    
    for relation in relations:
        for head_entity, rel, tail_entity in triplets:
            if rel == relation:
                related_triplets.append((head_entity, rel, tail_entity))
                break  # After finding one matching triplet, break out of inner loop

    return related_triplets

def get_some_triplets_by_relation(triplets, possible_relation):
    if not possible_relation.startswith('/'):
        possible_relation = '/' + possible_relation
    related_triplets = []
    count = 0
    for head_entity, relation, tail_entity in triplets:
        # Tentatively select 5 example triplets for possible relations
        if relation == possible_relation and count < 6:
            related_triplets.append((head_entity, relation, tail_entity))
            count = count + 1
    return related_triplets

def map_triplets_to_names(triplets, entity_mapping):
    mapped_triplets = []
    for head_entity, relation, tail_entity in triplets:
        head_name = entity_mapping.get(head_entity, head_entity)  # Keep ID if name not found
        tail_name = entity_mapping.get(tail_entity, tail_entity)  # Keep ID if name not found
        mapped_triplets.append((head_name, relation, tail_name))
    return mapped_triplets

'''
# Example usage
tsv_file_path = 'fb15k-237/FB15k-237/train.tsv'  # Replace with your file path
entity_mapping_file_path = 'entity2text.txt'  # Replace with your file path

triplets = load_tsv(tsv_file_path)
entity_mapping = load_entity_mapping(entity_mapping_file_path)

entity = '/m/0fbvqf'  # Replace with your entity to query
related_triplets = get_triplets_by_entity(triplets, entity)

mapped_triplets = map_triplets_to_names(related_triplets, entity_mapping)

print(f"Triplets related to entity '{entity}' (mapped to names): {mapped_triplets}")
'''

def convert_triplets_to_natural_language(triplets):
    client = OpenAI(
    # This is the default and can be omitted
    api_key="sk-3056ef7bf8864448b1694f76a52134c1",
    base_url="https://api.deepseek.com",
    )
    # Build input prompt
    prompt = "The following are some triplets in a knowledge graph, please convert them into natural language sentences and only return the sentences without any additional information:\n"
    for head, relation, tail in triplets:
        prompt += f"({head}, {relation}, {tail})\n"
    
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model="deepseek-chat",
        temperature=0.1
    )
    
    # Get natural language description
    natural_language_output = chat_completion.choices[0].message.content
    return natural_language_output

'''
# Example usage
triplets = [
    ('/m/0fbvqf', '/award/award_category/winners', '/m/04bd8y'),
    ('/m/017s11', '/award/award_nominee/award_nominations', '/m/02hxhz'),
    ('/m/05b__vr', '/award/award_winner/awards_won', '/m/064nh4k'),
    # Add more triplets...
]

natural_language_output = convert_triplets_to_natural_language(triplets)
print(natural_language_output)
'''
