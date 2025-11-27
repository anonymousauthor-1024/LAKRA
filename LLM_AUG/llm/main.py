from openai import OpenAI
from utils_ import *
import re
import requests

def fix_json_format(json_str):
    """
    尝试修复常见的JSON格式问题
    """
    if not json_str:
        return None
    
    # 移除可能的多余字符
    json_str = json_str.strip()
    
    # 处理格式: {"{\"key1\",\"key2\",\"key3\":value}
    # 这种格式需要转换为: {"(key1,key2,key3)": value}
    pattern = r'\{\"([^\"]+)\",\"([^\"]+)\",\"([^\"]+)\":(\d+)\}'
    match = re.search(pattern, json_str)
    if match:
        key1, key2, key3, value = match.groups()
        fixed_json = f'{{"({key1}, {key2}, {key3})": {value}}}'
        return fixed_json
    
    # 处理格式: {"key1","key2","key3":value} (缺少外层引号)
    pattern2 = r'\{\"([^\"]+)\",\"([^\"]+)\",\"([^\"]+)\":(\d+)\}'
    match2 = re.search(pattern2, json_str)
    if match2:
        key1, key2, key3, value = match2.groups()
        fixed_json = f'{{"({key1}, {key2}, {key3})": {value}}}'
        return fixed_json
    
    # 处理您提到的具体格式: {"{\"Kandy\",\"/location/administrative_division/first_level_division_of\",\"Sri Lanka\":1}
    # 这种格式的问题是缺少了外层的引号和右括号
    pattern3 = r'\{\"([^\"]+)\",\"([^\"]+)\",\"([^\"]+)\":(\d+)\}'
    if re.search(pattern3, json_str):
        # 尝试找到三个部分：实体1, 关系, 实体2
        parts = re.findall(r'\"([^\"]+)\"', json_str)
        if len(parts) >= 3:
            # 找到数字值
            value_match = re.search(r':(\d+)', json_str)
            if value_match:
                value = value_match.group(1)
                fixed_json = f'{{"({parts[0]}, {parts[1]}, {parts[2]})": {value}}}'
                return fixed_json
    
    # 处理其他可能的格式问题
    try:
        # 尝试替换常见的格式问题
        fixed = json_str
        
        # 如果字符串以 { 开始但没有正确结束，尝试添加 }
        if fixed.startswith('{') and not fixed.endswith('}'):
            fixed += '}'
        
        # 修复键名缺少引号的情况
        fixed = re.sub(r'(\w+):', r'"\1":', fixed)
        
        # 修复值缺少引号的情况（如果值不是数字）
        fixed = re.sub(r':\s*([^"\d\{\}\[\],\s][^,\}\]]*)', r': "\1"', fixed)
        
        # 验证修复后的JSON是否有效
        json.loads(fixed)
        return fixed
    except:
        pass
    
    # 如果都无法修复，返回None
    return None

def get_possible_relations(entity_name, entity_desc, related_triplets, nl, candidate_relations, example_triplets):
    client = OpenAI(
    # This is the default and can be omitted
    api_key="sk-3056ef7bf8864448b1694f76a52134c1",
    base_url="https://api.deepseek.com",
    )

    prompt = "The following is information related to a knowledge graph:\n"
    str1 = f"Entity: {entity_name}\n"
    #print(str1, '\n')
    str2 = f"Entity description: {entity_desc}\n"
    #print(str2, '\n')
    str3 = f"Known triplets related to entity {entity_name}: "
    for h, r, t in related_triplets:
        str3 += f"({h}, {r}, {t})\n"
    #print(str3, '\n')

    str4 = f"Converting above known triplets into natural language: {nl}\n"
    #print(str4, '\n')
    str5 = f"Candidate relations in a knowledge graph: "
    for relation in candidate_relations:
        str5 += f"{relation},\n"
    #print(str5, '\n')
    str6 = f"Some known triplets corresponding to candidate relations: "
    for head, relation, tail in example_triplets:
        str6 += f"({head}, {relation}, {tail})\n"
    #print(str6, '\n')
    str7 = f"Based on the above information, if the entity {entity_name} is used as the head entity, what are the possible relationships that could form valid triplets? Please ensure that possible relationships are selected only from the candidate relations in a knowledge graph. Please return strictly JSON format where the key is possible relations and the value is number 1. If no possible relationships exist, strictly return an empty JSON format: {{}}."
    prompt = prompt + str1 + str2 + str3 + str4 + str5 + str6 + str7
    #print(str7, '\n')
    print(prompt, '\n')
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model="deepseek-chat",
        response_format={"type": "json_object"},
        temperature=0.2,
        #extra_body={"enable_thinking": False}
    )

    output = chat_completion.choices[0].message.content
    print(output)
    return output

def get_possible_relations_tail(entity_name, entity_desc, related_triplets, nl, candidate_relations, example_triplets):
    client = OpenAI(
    # This is the default and can be omitted
    api_key="sk-3056ef7bf8864448b1694f76a52134c1",
    base_url="https://api.deepseek.com",
    )

    prompt = "The following is information related to a knowledge graph:\n"
    str1 = f"Entity: {entity_name}\n"
    #print(str1, '\n')
    str2 = f"Entity description: {entity_desc}\n"
    #print(str2, '\n')
    str3 = f"Known triplets related to entity {entity_name}: "
    for h, r, t in related_triplets:
        str3 += f"({h}, {r}, {t})\n"
    #print(str3, '\n')

    str4 = f"Converting above known triplets into natural language: {nl}\n"
    #print(str4, '\n')
    str5 = f"Candidate relations in knowledge graph: "
    for relation in candidate_relations:
        str5 += f"{relation},\n"
    #print(str5, '\n')
    str6 = f"Known triplets corresponding to candidate relations: "
    for head, relation, tail in example_triplets:
        str6 += f"({head}, {relation}, {tail})\n"
    #print(str6, '\n')

    str7 = f"Consider the triple ({entity_name}, {possible_relation}, ?). Based on the information provided for {entity_name}, infer the appropriate tail entity or entities from the candidate list above. Ensure that each inferred triple is well justified by the available evidence.If no possible relationships exist, strictly return an empty JSON format: {{}}"
    prompt = prompt + str1 + str2 + str3 + str4 + str5 + str6 + str7
    #print(str7, '\n')
    print(prompt, '\n')
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model="deepseek-chat",
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    output = chat_completion.choices[0].message.content
    print(output)
    return output

def get_possible_tail_entity(entity_name, entity_desc, related_triplets, nl, possible_relation, example_triplets, name_list):
    client = OpenAI(
    # This is the default and can be omitted
    api_key="sk-3056ef7bf8864448b1694f76a52134c1",
    base_url="https://api.deepseek.com",
    )

    prompt = "The following is information related to knowledge graph:\n"
    str1 = f"Entity: {entity_name}\n"
    #print(str1, '\n')
    str2 = f"Entity description: {entity_desc}\n"
    #print(str2, '\n')
    str3 = f"Some known context triplets related to entity {entity_name}: "
    for h, r, t in related_triplets:
        str3 += f"({h}, {r}, {t})\n"
    #print(str3, '\n')

    str4 = f"Converting above known triplets into natural language: {nl}\n"
    #print(str4, '\n')

    str5 = f"Known triplets corresponding to relation {possible_relation}: "
    for h, r, t in example_triplets:
        str5 += f"({h}, {r}, {t})\n"

    str6 = f"Now, consider the triplet ({entity_name}, {possible_relation}, ?). Based on the information provided for {entity_name}, infer the appropriate tail entity or entities from the candidate list above. Ensure that each inferred triple is well justified by the available evidence.  If they exist, return strictly: {{\"(head,relation,tail)\": value}} where the key is possible triplets and the value should be number 1. If none exist, strictly return an empty JSON format: {{}}. No extra information. Please ensure that the possible tail entity is selected only from the candidate entities. Please exclude given known triplets and only return new triplets. \n"

    #print(str6, '\n')
    str7 = f"Candidate entities:\n"
    for name in name_list:
        str7 += f"{name}\n"
    
    prompt = prompt + str1 + str2 + str3 + str4 + str5 + str6 + str7
    print(prompt, '\n')
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model="deepseek-chat",
        response_format={"type": "json_object"},
        temperature=0.2,
        #extra_body={"enable_thinking": False}
    )

    output = chat_completion.choices[0].message.content

    return output

def get_possible_head_entity(entity_name, entity_desc, related_triplets, nl, possible_relation, example_triplets, name_list):
    client = OpenAI(
    # This is the default and can be omitted
    api_key="sk-3056ef7bf8864448b1694f76a52134c1",
    base_url="https://api.deepseek.com",
    )

    prompt = "The following is information related to a knowledge graph:\n"
    str1 = f"Entity: {entity_name}\n"
    #print(str1, '\n')
    str2 = f"Entity description: {entity_desc}\n"
    #print(str2, '\n')
    str3 = f"Known context triplets related to entity {entity_name}: "
    for h, r, t in related_triplets:
        str3 += f"({h}, {r}, {t})\n"
    #print(str3, '\n')

    str4 = f"Converting above known triplets into natural language: {nl}\n"
    #print(str4, '\n')

    str5 = f"Known triplets corresponding to relation {possible_relation}: "
    for h, r, t in example_triplets:
        str5 += f"({h}, {r}, {t})\n"
    #print(str5, '\n')

    str6 = f"Now, consider the triplet (?, {possible_relation}, {entity_name}). Based on the information provided for {entity_name}, infer the appropriate head entity or entities from the candidate list above. Ensure that each inferred triple is well justified by the available evidence.  If they exist, return strictly: {{\"(head,relation,tail)\": value}} where the key is possible triplets and the value should be number 1. If none exist, strictly return an empty JSON format: {{}}. No extra information. Please ensure that the possible tail entity is selected only from the candidate entities. Please exclude given known triplets and only return new triplets. \n"
    #print(str6, '\n')
    str7 = f"Candidate entities:\n"
    for name in name_list:
       str7 += f"{name}\n"
    #print(str7)
    
    prompt = prompt + str1 + str2 + str3 + str4 + str5 + str6 + str7
    print(prompt, '\n')
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model="deepseek-chat",
        #model="gpt-3.5-turbo",
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    output = chat_completion.choices[0].message.content
    print(output)

    return output


# 定义文件路径
file_path = 'LLM_AUG/fb15k-237/low_frequency_entities_fb15k237.txt'
aug_triplet_file = "LLM_AUG/generated_triplets/aug_triplets_tail_fb15k237.txt"
# 打开文件并逐行读取
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        parts = line.strip().split('\t')  # 假设文件是以制表符分隔的

        if len(parts) != 5:
            continue

        entity_id = parts[0]
        entity_type = parts[2]
        entity_name = parts[3]
        entity_desc = parts[4]

        tsv_file_path = 'fb15k-237/FB15k-237/train.tsv'  # 替换为您的文件路径
        entity_mapping_file_path = 'fb15k-237/FB15k-237/entity2text.txt'  # 替换为您的文件路径

        #找到和该稀疏实体相关的三元组
        triplets = load_tsv(tsv_file_path)
        entity_mapping = load_entity_mapping(entity_mapping_file_path)
        related_triplets = get_triplets_by_entity(triplets, entity_id)
        mapped_triplets = map_triplets_to_names(related_triplets, entity_mapping)
        print(f"与实体 '{entity_name}' 相关的三元组（映射到名称）: {mapped_triplets}")

        #将相关三元组转化为自然语言
        natural_language_output = convert_triplets_to_natural_language(mapped_triplets)
        #print(natural_language_output)

        #找出该实体作为头实体和尾实体可能拥有的关系
        file_path = 'LLM_AUG/fb15k-237/relation_entity_types.json'  # 替换为您的文件路径
        data = load_relation_entity_types(file_path)
        head_relations, tail_relations = get_relations_by_entity_type(data, entity_type)

        print(f"作为头实体 '{entity_type}' 的关系: {head_relations}")
        print(f"作为尾实体 '{entity_type}' 的关系: {tail_relations}")

        #给LLM的关系示例，让LLM了解关系
        example_triplets = get_triplets_by_relations(triplets, head_relations)
        #example_triplets = get_triplets_by_relations(triplets, tail_relations)
        example_mapped_triplets = map_triplets_to_names(example_triplets, entity_mapping)
        #当实体作为头实体时的获得到的可能的候选关系
        #当实体作为尾实体时的获得到的可能的候选关系
        possible_relations = get_possible_relations(entity_name, entity_desc, mapped_triplets, natural_language_output, head_relations, example_mapped_triplets)
        # 解析 JSON 字符串
        try:
            parsed_json = json.loads(possible_relations)
        except json.JSONDecodeError as e:
            print(f"关系解析JSON错误: {e}")
            print(f"原始输出: {possible_relations}")
            
            # 尝试修复JSON格式
            fixed_json = fix_json_format(possible_relations)
            if fixed_json:
                try:
                    parsed_json = json.loads(fixed_json)
                    print(f"修复后的JSON: {fixed_json}")
                except json.JSONDecodeError:
                    print("关系JSON修复失败，跳过此实体")
                    continue
            else:
                print("无法修复关系JSON格式，跳过此实体")
                continue
        
        possible_relations_list = list(set(parsed_json.keys()))

        #possible_relations_list = ["/people/ethnicity/languages_spoken"]

        
        #接下来根据头实体和可能的关系，获取候选尾实体
        #局限在k跳上下问
        #首先得到各个关系的候选实体有哪些再让大模型选择合理的
        for possible_relation in possible_relations_list:
            #可能的尾实体，这里需要做个处理
            possible_entity_dict = get_tail_entitys_by_relation(data, possible_relation, entity_id)
            #possible_entity_dict = get_head_entitys_by_relation(data, possible_relation)
            #如果尾实体过多，将尾实体划分，每一份最多500个实体大概5000个tokens
            count = 0
            current_names = []  # 用于存储当前的 name 列表
            sub_lists = []  # 用于存储划分后的子列表
            for k, v in possible_entity_dict.items():
                for name, desc in v.items():
                    #if k == "Country" or k == "State":
                    #    current_names.append(f"{name}")
                    #else:
                    #    current_names.append(f"{name}    entity_type:{k}")
                    current_names.append(f"{name}    entity_type:{k}")
                    count += 1

                    if count >= 500:
                        # 将当前的 names 列表添加到子列表中
                        sub_lists.append(current_names)
                        # 重置计数器和当前名称列表
                        count = 0
                        current_names = []
            # 处理剩余的 names
            if current_names:
                sub_lists.append(current_names)
                
            example_possible_triplets = get_some_triplets_by_relation(triplets, possible_relation)
            mapped_example_possible_triplets = map_triplets_to_names(example_possible_triplets, entity_mapping)

            for name_list in sub_lists:
                #possible_tail_entity = get_possible_tail_entity(entity_name, entity_desc, mapped_triplets, natural_language_output, possible_relation, mapped_example_possible_triplets, name_list)
                possible_head_entity = get_possible_head_entity(entity_name, entity_desc, mapped_triplets, natural_language_output, possible_relation, mapped_example_possible_triplets, name_list)
                print(possible_head_entity)
                
                # 尝试解析JSON，如果失败则尝试修复格式
                try:
                    #complete_triplet = json.loads(possible_tail_entity)
                    complete_triplet = json.loads(possible_head_entity)
                except json.JSONDecodeError as e:
                    print(f"JSON解析错误: {e}")
                    #print(f"原始输出: {possible_tail_entity}")
                    print(f"原始输出: {possible_head_entity}")
                    
                    # 尝试修复常见的JSON格式问题
                    #fixed_json = fix_json_format(possible_tail_entity)
                    fixed_json = fix_json_format(possible_head_entity)
                    if fixed_json:
                        try:
                            complete_triplet = json.loads(fixed_json)
                            print(f"修复后的JSON: {fixed_json}")
                        except json.JSONDecodeError:
                            print("JSON修复失败，跳过此次结果")
                            continue
                    else:
                        print("无法修复JSON格式，跳过此次结果")
                        continue
                
                # 记录complete_triplet到log文件
                log_file = "LLM_AUG/fb15k-237/complete_triplet_log.txt"
                with open(log_file, 'a', encoding='utf-8') as log:
                    log.write(f"Entity: {entity_name} ({entity_id})\n")
                    log.write(f"Relation: {possible_relation}\n")
                    log.write(f"Complete triplet: {complete_triplet}\n")
                    log.write("-" * 50 + "\n")

                if not complete_triplet:
                    continue
                with open(aug_triplet_file, 'a', encoding='utf-8') as file:  
                    for triplet, _ in complete_triplet.items():
                        file.write(f"{triplet}\n")
        
        with open(aug_triplet_file, 'a', encoding='utf-8') as file:  
            file.write(f"{entity_id} {entity_name}\n")
        '''
        example_triplets_tail = get_triplets_by_relations(triplets, tail_relations)
        example_mapped_triplets_tail = map_triplets_to_names(example_triplets_tail, entity_mapping)
        #获取当实体作为尾实体时可能的关系
        possible_relations_tail = get_possible_relations_tail(entity_name, entity_desc, mapped_triplets, natural_language_output, tail_relations, example_mapped_triplets_tail)

        parsed_json_tail = json.loads(possible_relations_tail)
        possible_relations_list_tail = list(set(parsed_json.keys()))

        #接下来根据尾实体和可能的关系，获取候选头实体
        #首先得到各个关系的候选实体有哪些再让大模型选择合理的
        for possible_relation in possible_relations_list_tail:
            possible_entity_dict = get_head_entitys_by_relation(data, possible_relation)
            example_possible_triplets = get_some_triplets_by_relation(triplets, possible_relation)
            mapped_example_possible_triplets = map_triplets_to_names(example_possible_triplets, entity_mapping)
            possible_tail_entity = get_possible_tail_entity(entity_name, entity_desc, mapped_triplets, natural_language_output, possible_relation, mapped_example_possible_triplets, possible_entity_dict)

            complete_triplet = json.loads(possible_tail_entity)

            aug_triplet_file = "LLM_AUG/fb15k-237/aug_triplets.txt"
            with open(aug_triplet_file, 'a', encoding='utf-8') as file:  
                for triplet, _ in complete_triplet.items():
                    file.write(f"{triplet}\n")
        '''
        '''
        with open(aug_triplet_file, 'a', encoding='utf-8') as file:  
            file.write(f"=========={entity_id}  {entity_name}========\n")
        '''


