from openai import OpenAI
from utils_ import *
import re
import requests

def fix_json_format(json_str):
    """
    Attempt to fix common JSON format issues
    """
    if not json_str:
        return None
    
    # Remove possible extra characters
    json_str = json_str.strip()
    
    # Handle format: {"{\"key1\",\"key2\",\"key3\":value}
    # This format needs to be converted to: {"(key1,key2,key3)": value}
    pattern = r'\{\"([^\"]+)\",\"([^\"]+)\",\"([^\"]+)\":(\d+)\}'
    match = re.search(pattern, json_str)
    if match:
        key1, key2, key3, value = match.groups()
        fixed_json = f'{{"({key1}, {key2}, {key3})": {value}}}'
        return fixed_json
    
    # Handle format: {"key1","key2","key3":value} (missing outer quotes)
    pattern2 = r'\{\"([^\"]+)\",\"([^\"]+)\",\"([^\"]+)\":(\d+)\}'
    match2 = re.search(pattern2, json_str)
    if match2:
        key1, key2, key3, value = match2.groups()
        fixed_json = f'{{"({key1}, {key2}, {key3})": {value}}}'
        return fixed_json
    
    # Handle specific format mentioned: {"{\"Kandy\",\"/location/administrative_division/first_level_division_of\",\"Sri Lanka\":1}
    # The problem with this format is missing outer quotes and right bracket
    pattern3 = r'\{\"([^\"]+)\",\"([^\"]+)\",\"([^\"]+)\":(\d+)\}'
    if re.search(pattern3, json_str):
        # Try to find three parts: entity1, relation, entity2
        parts = re.findall(r'\"([^\"]+)\"', json_str)
        if len(parts) >= 3:
            # Find numeric value
            value_match = re.search(r':(\d+)', json_str)
            if value_match:
                value = value_match.group(1)
                fixed_json = f'{{"({parts[0]}, {parts[1]}, {parts[2]})": {value}}}'
                return fixed_json
    
    # Handle other possible format issues
    try:
        # Try to replace common format issues
        fixed = json_str
        
        # If string starts with { but doesn't end properly, try adding }
        if fixed.startswith('{') and not fixed.endswith('}'):
            fixed += '}'
        
        # Fix missing quotes for key names
        fixed = re.sub(r'(\w+):', r'"\1":', fixed)
        
        # Fix missing quotes for values (if value is not a number)
        fixed = re.sub(r':\s*([^"\d\{\}\[\],\s][^,\}\]]*)', r': "\1"', fixed)
        
        # Verify if fixed JSON is valid
        json.loads(fixed)
        return fixed
    except:
        pass
    
    # If unable to fix, return None
    return None

def get_possible_relations(entity_name, entity_desc, related_triplets, nl, candidate_relations, example_triplets):
    client = OpenAI(
    # This is the default and can be omitted
    api_key="sk-3056ef7bf8864448b1694f76a521mask",
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
    api_key="sk-3056ef7bf8864448b1694f76a521mask",
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
    api_key="sk-3056ef7bf8864448b1694f76a521mask",
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
    api_key="sk-3056ef7bf8864448b1694f76a521mask",
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


# Define file paths
file_path = 'LLM_AUG/fb15k-237/low_frequency_entities_fb15k237.txt'
aug_triplet_file = "LLM_AUG/generated_triplets/aug_triplets_tail_fb15k237.txt"
# Open file and read line by line
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        parts = line.strip().split('\t')  # Assume file is tab-separated

        if len(parts) != 5:
            continue

        entity_id = parts[0]
        entity_type = parts[2]
        entity_name = parts[3]
        entity_desc = parts[4]

        tsv_file_path = 'fb15k-237/FB15k-237/train.tsv'  # Replace with your file path
        entity_mapping_file_path = 'fb15k-237/FB15k-237/entity2text.txt'  # Replace with your file path

        # Find triplets related to this sparse entity
        triplets = load_tsv(tsv_file_path)
        entity_mapping = load_entity_mapping(entity_mapping_file_path)
        related_triplets = get_triplets_by_entity(triplets, entity_id)
        mapped_triplets = map_triplets_to_names(related_triplets, entity_mapping)
        print(f"Triplets related to entity '{entity_name}' (mapped to names): {mapped_triplets}")

        # Convert related triplets to natural language
        natural_language_output = convert_triplets_to_natural_language(mapped_triplets)
        #print(natural_language_output)

        # Find possible relations when entity acts as head or tail entity
        file_path = 'LLM_AUG/fb15k-237/relation_entity_types.json'  # Replace with your file path
        data = load_relation_entity_types(file_path)
        head_relations, tail_relations = get_relations_by_entity_type(data, entity_type)

        print(f"Relations for '{entity_type}' as head entity: {head_relations}")
        print(f"Relations for '{entity_type}' as tail entity: {tail_relations}")

        # Provide relation examples for LLM to understand relations
        example_triplets = get_triplets_by_relations(triplets, head_relations)
        #example_triplets = get_triplets_by_relations(triplets, tail_relations)
        example_mapped_triplets = map_triplets_to_names(example_triplets, entity_mapping)
        # Get possible candidate relations when entity acts as head entity
        # Get possible candidate relations when entity acts as tail entity
        possible_relations = get_possible_relations(entity_name, entity_desc, mapped_triplets, natural_language_output, head_relations, example_mapped_triplets)
        # Parse JSON string
        try:
            parsed_json = json.loads(possible_relations)
        except json.JSONDecodeError as e:
            print(f"Relation JSON parsing error: {e}")
            print(f"Original output: {possible_relations}")
            
            # Try to fix JSON format
            fixed_json = fix_json_format(possible_relations)
            if fixed_json:
                try:
                    parsed_json = json.loads(fixed_json)
                    print(f"Fixed JSON: {fixed_json}")
                except json.JSONDecodeError:
                    print("Failed to fix relation JSON, skipping this entity")
                    continue
            else:
                print("Unable to fix relation JSON format, skipping this entity")
                continue
        
        possible_relations_list = list(set(parsed_json.keys()))

        #possible_relations_list = ["/people/ethnicity/languages_spoken"]

        
        # Next, get candidate tail entities based on head entity and possible relations
        # Constrained within k-hop neighborhood
        # First get candidate entities for each relation, then let LLM choose reasonable ones
        for possible_relation in possible_relations_list:
            # Possible tail entities, need to process this
            possible_entity_dict = get_tail_entitys_by_relation(data, possible_relation, entity_id)
            #possible_entity_dict = get_head_entitys_by_relation(data, possible_relation)
            # If too many tail entities, divide them, max 500 entities per batch (approximately 5000 tokens)
            count = 0
            current_names = []  # Store current name list
            sub_lists = []  # Store divided sublists
            for k, v in possible_entity_dict.items():
                for name, desc in v.items():
                    #if k == "Country" or k == "State":
                    #    current_names.append(f"{name}")
                    #else:
                    #    current_names.append(f"{name}    entity_type:{k}")
                    current_names.append(f"{name}    entity_type:{k}")
                    count += 1

                    if count >= 500:
                        # Add current names list to sublists
                        sub_lists.append(current_names)
                        # Reset counter and current name list
                        count = 0
                        current_names = []
            # Handle remaining names
            if current_names:
                sub_lists.append(current_names)
                
            example_possible_triplets = get_some_triplets_by_relation(triplets, possible_relation)
            mapped_example_possible_triplets = map_triplets_to_names(example_possible_triplets, entity_mapping)

            for name_list in sub_lists:
                #possible_tail_entity = get_possible_tail_entity(entity_name, entity_desc, mapped_triplets, natural_language_output, possible_relation, mapped_example_possible_triplets, name_list)
                possible_head_entity = get_possible_head_entity(entity_name, entity_desc, mapped_triplets, natural_language_output, possible_relation, mapped_example_possible_triplets, name_list)
                print(possible_head_entity)
                
                # Try to parse JSON, if failed then try to fix format
                try:
                    #complete_triplet = json.loads(possible_tail_entity)
                    complete_triplet = json.loads(possible_head_entity)
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error: {e}")
                    #print(f"Original output: {possible_tail_entity}")
                    print(f"Original output: {possible_head_entity}")
                    
                    # Try to fix common JSON format issues
                    #fixed_json = fix_json_format(possible_tail_entity)
                    fixed_json = fix_json_format(possible_head_entity)
                    if fixed_json:
                        try:
                            complete_triplet = json.loads(fixed_json)
                            print(f"Fixed JSON: {fixed_json}")
                        except json.JSONDecodeError:
                            print("Failed to fix JSON, skipping this result")
                            continue
                    else:
                        print("Unable to fix JSON format, skipping this result")
                        continue
                
                # Log complete_triplet to log file
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
        # Get possible relations when entity acts as tail entity
        possible_relations_tail = get_possible_relations_tail(entity_name, entity_desc, mapped_triplets, natural_language_output, tail_relations, example_mapped_triplets_tail)

        parsed_json_tail = json.loads(possible_relations_tail)
        possible_relations_list_tail = list(set(parsed_json.keys()))

        # Next, get candidate head entities based on tail entity and possible relations
        # First get candidate entities for each relation, then let LLM choose reasonable ones
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


