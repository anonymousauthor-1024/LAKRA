import json

# 假设三元组文件为 train.tsv，实体类型文件为 entity2type.txt
triples_file = './fb15k-237/FB15k-237/train.tsv'
entity_type_file = './fb15k-237/FB15k-237/entity2type.txt'
output_file = 'relation_entity_types.json'  # 输出文件

# 读取实体类型并建立映射
entity_to_type = {}
with open(entity_type_file, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            entity_id, entity_type = parts
            entity_to_type[entity_id] = entity_type

# 统计每个关系的头实体和尾实体类型
relation_types = {}

with open(triples_file, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 3:
            head_entity, relation, tail_entity = parts
            
            # 获取头实体和尾实体的类型
            head_type = entity_to_type.get(head_entity, None)
            tail_type = entity_to_type.get(tail_entity, None)
            
            # 初始化关系类型字典
            if relation not in relation_types:
                relation_types[relation] = {
                    "head_entity_types": set(),
                    "tail_entity_types": set()
                }
            
            # 统计头实体类型
            if head_type:
                relation_types[relation]["head_entity_types"].add(head_type)
            
            # 统计尾实体类型
            if tail_type:
                relation_types[relation]["tail_entity_types"].add(tail_type)

# 将集合转换为列表以便于 JSON 序列化
for relation in relation_types:
    relation_types[relation]["head_entity_types"] = list(relation_types[relation]["head_entity_types"])
    relation_types[relation]["tail_entity_types"] = list(relation_types[relation]["tail_entity_types"])

# 将结果写入 JSON 文件
with open(output_file, 'w', encoding='utf-8') as out_file:
    json.dump(relation_types, out_file, ensure_ascii=False, indent=4)

print(f"Results have been written to {output_file}.")
'''
'''
import json

# 假设三元组文件为 train.tsv，实体类型文件为 entity2type.txt
triples_file = './fb15k-237/FB15k-237/train.tsv'
entity_type_file = './fb15k-237/FB15k-237/entity2type.txt'
output_file = 'relation_entity_types.json'  # 输出文件

# 读取实体类型并建立映射
entity_to_type = {}
with open(entity_type_file, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            entity_id, entity_type = parts
            entity_to_type[entity_id] = entity_type

# 统计每个关系的头实体和尾实体类型
relation_types = {}

with open(triples_file, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 3:
            head_entity, relation, tail_entity = parts
            
            # 获取头实体和尾实体的类型
            head_type = entity_to_type.get(head_entity, None)
            tail_type = entity_to_type.get(tail_entity, None)
            
            # 初始化关系类型字典
            if relation not in relation_types:
                relation_types[relation] = {
                    "head_entity_types": set(),
                    "tail_entity_types": set(),
                    "head_entity_type_counts": {},  # 计数字典
                    "tail_entity_type_counts": {}   # 计数字典
                }
            
            # 统计头实体类型
            if head_type:
                relation_types[relation]["head_entity_types"].add(head_type)
                # 计数头实体类型
                if head_type not in relation_types[relation]["head_entity_type_counts"]:
                    relation_types[relation]["head_entity_type_counts"][head_type] = 0
                relation_types[relation]["head_entity_type_counts"][head_type] += 1
            
            # 统计尾实体类型
            if tail_type:
                relation_types[relation]["tail_entity_types"].add(tail_type)
                # 计数尾实体类型
                if tail_type not in relation_types[relation]["tail_entity_type_counts"]:
                    relation_types[relation]["tail_entity_type_counts"][tail_type] = 0
                relation_types[relation]["tail_entity_type_counts"][tail_type] += 1

# 将集合转换为列表以便于 JSON 序列化
for relation in relation_types:
    relation_types[relation]["head_entity_types"] = list(relation_types[relation]["head_entity_types"])
    relation_types[relation]["tail_entity_types"] = list(relation_types[relation]["tail_entity_types"])

# 将结果写入 JSON 文件
with open(output_file, 'w', encoding='utf-8') as out_file:
    json.dump(relation_types, out_file, ensure_ascii=False, indent=4)

print(f"Results have been written to {output_file}.")


'''
# 指定要查找的实体类型
target_entity_type = "City"

# 读取 JSON 文件
with open('relation_entity_types.json', 'r', encoding='utf-8') as in_file:
    relation_types = json.load(in_file)

# 找到以指定实体类型作为头实体的关系
matching_relations = []

for relation, types in relation_types.items():
    if target_entity_type in types["head_entity_types"] and types["head_entity_type_counts"].get(target_entity_type, 0) > 2:
        matching_relations.append(relation)

# 输出结果
if matching_relations:
    print(f"Relations with '{target_entity_type}' as head entity:")
    for relation in matching_relations:
        print(relation)
else:
    print(f"No relations found with '{target_entity_type}' as head entity.")
'''