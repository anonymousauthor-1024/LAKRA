import json
from openai import OpenAI

def load_relation_entity_types(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_subgraph_nodes_from_file(entity, sparse_nodes_dir='../sparse_nodes'):
    """
    从sparse_nodes文件夹中读取指定实体的子图节点
    
    Args:
        entity: 目标实体ID
        sparse_nodes_dir: sparse_nodes文件夹路径
    
    Returns:
        subgraph_nodes: 子图节点ID的集合
    """
    import os
    
    # 构建文件路径（处理实体ID中的特殊字符）
    safe_entity_id = entity.replace('/', '_').replace('\\', '_')
    entity_file = os.path.join(os.path.dirname(__file__), sparse_nodes_dir, f"{safe_entity_id}.txt")
    
    if not os.path.exists(entity_file):
        print(f"Warning: Entity file not found: {entity_file}")
        return set()
    
    # 读取文件中的节点
    subgraph_nodes = set()
    in_nodes_section = False
    
    with open(entity_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # 检测节点部分的开始
            if 'Nodes in Subgraph:' in line:
                in_nodes_section = True
                continue
            
            # 检测三元组部分的开始（节点部分结束）
            if 'Triples in Subgraph:' in line:
                break
            
            # 跳过分隔线
            if line.startswith('='):
                continue
            
            # 在节点部分，解析节点
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
        #relation = '/' + relation  # 在 relation 字符串前加上 / 字符
        return entity_dict
    
    # 获取entity_id的子图节点
    subgraph_nodes = get_subgraph_nodes_from_file(entity_id)
    if not subgraph_nodes:
        print(f"Warning: No subgraph nodes found for entity {entity_id}")
        return entity_dict
        
    #加个10以上的限制
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
                if len(parts) >= 4:  # 确保行中有足够的部分
                    candidate_entity_id = parts[0]
                    entity_type = parts[1]
                    entity_name = parts[2]
                    entity_desc = parts[3]
                    
                    # 检查实体类型是否匹配 AND 实体是否在子图节点中
                    if entity_type == type and candidate_entity_id in subgraph_nodes:
                        entity_dict[type][entity_name] = entity_desc
    
    return entity_dict

def get_head_entitys_by_relation(data, relation, entity_id):
    candidate_entities_types = []
    entity_dict = {}
    if relation not in data:
         relation = '/' + relation  # 在 relation 字符串前加上 / 字符
    
    # 获取entity_id的子图节点
    subgraph_nodes = get_subgraph_nodes_from_file(entity_id)
    if not subgraph_nodes:
        print(f"Warning: No subgraph nodes found for entity {entity_id}")
        return entity_dict
    
    #加个10以上的限制
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
                if len(parts) >= 4:  # 确保行中有足够的部分
                    candidate_entity_id = parts[0]
                    entity_type = parts[1]
                    entity_name = parts[2]
                    entity_desc = parts[3]
                    
                    # 检查实体类型是否匹配 AND 实体是否在子图节点中
                    if entity_type == type and candidate_entity_id in subgraph_nodes:
                        entity_dict[type][entity_name] = entity_desc
    
    return entity_dict


def get_relations_by_entity_type(data, entity_type):
    head_relations = []
    tail_relations = []

    for relation, details in data.items():
        # 检查头实体类型
        #防止关系过多，这里增加了一个约束大于10的才加入候选关系
        if entity_type in details['head_entity_types'] and details['head_entity_type_counts'][entity_type] > 5:
            head_relations.append(relation)
        # 检查尾实体类型
        if entity_type in details['tail_entity_types'] and details['tail_entity_type_counts'][entity_type] > 5:
            tail_relations.append(relation)
        
    if len(head_relations) < 1:
         for relation, details in data.items():
            # 检查头实体类型
            #防止关系过多，这里增加了一个约束大于10的才加入候选关系
            if entity_type in details['head_entity_types']:
                head_relations.append(relation)

    if len(tail_relations) < 1:
        for relation, details in data.items():
            if entity_type in details['tail_entity_types']:
                tail_relations.append(relation)

    return head_relations, tail_relations

'''
# 示例调用
file_path = 'relation_entity_types.json'  # 替换为您的文件路径
data = load_relation_entity_types(file_path)

entity_type = 'Concept'  # 替换为您要查询的实体类型
head_relations, tail_relations = get_relations_by_entity_type(data, entity_type)

print(f"作为头实体 '{entity_type}' 的关系: {head_relations}")
print(f"作为尾实体 '{entity_type}' 的关系: {tail_relations}")
'''

def load_tsv(file_path):
    triplets = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')  # 假设文件是以制表符分隔的
            if len(parts) == 3:  # 确保有头实体、关系和尾实体
                head_entity, relation, tail_entity = parts
                triplets.append((head_entity, relation, tail_entity))
    return triplets

def load_entity_mapping(file_path):
    entity_mapping = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')  # 假设文件是以制表符分隔的
            if len(parts) == 2:  # 确保有编号和名称
                entity_id, entity_name = parts
                entity_mapping[entity_id] = entity_name
    return entity_mapping

def get_triplets_by_entity(triplets, entity):
    """
    获取与指定实体相关的三元组
    
    Args:
        triplets: 原始三元组列表（当use_sparse_nodes=False时使用）
        entity: 目标实体ID
        use_sparse_nodes: 是否从sparse_nodes文件夹读取k跳子图（默认True）
        sparse_nodes_dir: sparse_nodes文件夹路径（相对于当前文件的路径）
    
    Returns:
        related_triplets: 相关的三元组列表
    """
    sparse_nodes_dir='../sparse_nodes'
        # 从sparse_nodes文件夹读取对应实体的k跳子图
    import os
    
    # 构建文件路径（处理实体ID中的特殊字符）
    safe_entity_id = entity.replace('/', '_').replace('\\', '_')
    entity_file = os.path.join(os.path.dirname(__file__), sparse_nodes_dir, f"{safe_entity_id}.txt")
    
    # 读取文件中的三元组
    related_triplets = []
    in_triples_section = False
    
    with open(entity_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # 检测三元组部分的开始
            if 'Triples in Subgraph:' in line:
                in_triples_section = True
                continue
            
            # 跳过分隔线
            if line.startswith('='):
                continue
            
            # 在三元组部分，解析三元组
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
                break  # 找到一个匹配的三元组后，跳出内层循环

    return related_triplets

def get_some_triplets_by_relation(triplets, possible_relation):
    if not possible_relation.startswith('/'):
        possible_relation = '/' + possible_relation
    related_triplets = []
    count = 0
    for head_entity, relation, tail_entity in triplets:
        #暂定选5个可能关系的示例三元组
        if relation == possible_relation and count < 6:
            related_triplets.append((head_entity, relation, tail_entity))
            count = count + 1
    return related_triplets

def map_triplets_to_names(triplets, entity_mapping):
    mapped_triplets = []
    for head_entity, relation, tail_entity in triplets:
        head_name = entity_mapping.get(head_entity, head_entity)  # 如果没有找到名称，保留编号
        tail_name = entity_mapping.get(tail_entity, tail_entity)  # 如果没有找到名称，保留编号
        mapped_triplets.append((head_name, relation, tail_name))
    return mapped_triplets

'''
# 示例调用
tsv_file_path = 'fb15k-237/FB15k-237/train.tsv'  # 替换为您的文件路径
entity_mapping_file_path = 'entity2text.txt'  # 替换为您的文件路径

triplets = load_tsv(tsv_file_path)
entity_mapping = load_entity_mapping(entity_mapping_file_path)

entity = '/m/0fbvqf'  # 替换为您要查询的实体
related_triplets = get_triplets_by_entity(triplets, entity)

mapped_triplets = map_triplets_to_names(related_triplets, entity_mapping)

print(f"与实体 '{entity}' 相关的三元组（映射到名称）: {mapped_triplets}")
'''

def convert_triplets_to_natural_language(triplets):
    client = OpenAI(
    # This is the default and can be omitted
    #api_key="sk-7BKjBWTe16dDJPMhWTRAKPfh1frydPcCAZyp3cIGcbxTNZdU",
    #base_url="https://xiaoai.plus/v1",
    api_key="sk-3056ef7bf8864448b1694f76a52134c1",
    base_url="https://api.deepseek.com",
    #base_url="https://open.bigmodel.cn/api/paas/v4/",
    #api_key="62240b4c5fdadad03787e9dc5e7fd240.ZAkYTBf7UhunTFlj",
    )
    # 构建输入提示
    prompt = "The following are some triplets in a knowledge graph, please convert them into natural language sentences and only return the sentences without any additional information:\n"
    for head, relation, tail in triplets:
        prompt += f"({head}, {relation}, {tail})\n"
    
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model="deepseek-chat",
        #model="gpt-3.5-turbo",
        #model="glm-4-flash",
        temperature=0.1
    )
    
    # 获取自然语言描述
    natural_language_output = chat_completion.choices[0].message.content
    return natural_language_output

'''
# 示例调用
triplets = [
    ('/m/0fbvqf', '/award/award_category/winners', '/m/04bd8y'),
    ('/m/017s11', '/award/award_nominee/award_nominations', '/m/02hxhz'),
    ('/m/05b__vr', '/award/award_winner/awards_won', '/m/064nh4k'),
    # 添加更多三元组...
]

natural_language_output = convert_triplets_to_natural_language(triplets)
print(natural_language_output)
'''
