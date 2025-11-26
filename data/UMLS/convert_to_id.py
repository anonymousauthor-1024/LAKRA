"""
将三元组文件转换为ID格式
根据 entity2id.txt 和 relation2id.txt 将实体和关系转换为ID
"""

def load_mappings(entity_file, relation_file):
    """
    加载实体和关系的映射字典
    
    Args:
        entity_file: entity2id.txt 文件路径
        relation_file: relation2id.txt 文件路径
    
    Returns:
        entity2id: 实体到ID的映射字典
        relation2id: 关系到ID的映射字典
    """
    entity2id = {}
    relation2id = {}
    
    # 加载实体映射
    print(f"正在加载实体映射: {entity_file}")
    with open(entity_file, 'r', encoding='utf-8') as f:
        num_entities = int(f.readline().strip())
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                entity, entity_id = parts
                entity2id[entity] = int(entity_id)
    
    print(f"✓ 已加载 {len(entity2id)} 个实体映射")
    
    # 加载关系映射
    print(f"正在加载关系映射: {relation_file}")
    with open(relation_file, 'r', encoding='utf-8') as f:
        num_relations = int(f.readline().strip())
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                relation, relation_id = parts
                relation2id[relation] = int(relation_id)
    
    print(f"✓ 已加载 {len(relation2id)} 个关系映射")
    
    return entity2id, relation2id


def convert_triples_to_id(input_file, output_file, entity2id, relation2id):
    """
    将三元组文件转换为ID格式
    
    Args:
        input_file: 输入文件 (格式: 头实体\t关系\t尾实体)
        output_file: 输出文件 (格式: 三元组数量\n头实体id 关系id 尾实体id)
        entity2id: 实体到ID的映射
        relation2id: 关系到ID的映射
    """
    print(f"\n正在转换: {input_file} -> {output_file}")
    
    triples = []
    skipped = 0
    
    # 读取并转换三元组
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) != 3:
                print(f"警告: 第 {line_num} 行格式不正确，跳过")
                skipped += 1
                continue
            
            head, relation, tail = parts
            
            # 检查是否所有实体和关系都在映射中
            if head not in entity2id:
                print(f"警告: 第 {line_num} 行，头实体 '{head}' 不在映射中，跳过")
                skipped += 1
                continue
            
            if tail not in entity2id:
                print(f"警告: 第 {line_num} 行，尾实体 '{tail}' 不在映射中，跳过")
                skipped += 1
                continue
            
            if relation not in relation2id:
                print(f"警告: 第 {line_num} 行，关系 '{relation}' 不在映射中，跳过")
                skipped += 1
                continue
            
            # 转换为ID
            head_id = entity2id[head]
            relation_id = relation2id[relation]
            tail_id = entity2id[tail]
            
            triples.append((head_id, relation_id, tail_id))
            
            # 每处理10万行显示进度
            if line_num % 100000 == 0:
                print(f"  已处理 {line_num} 行...")
    
    # 写入输出文件
    print(f"正在写入 {len(triples)} 个三元组到 {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        # 第一行写入三元组数量
        f.write(f"{len(triples)}\n")
        
        # 写入所有三元组 (格式: 头实体id 关系id 尾实体id)
        for head_id, relation_id, tail_id in triples:
            f.write(f"{head_id} {tail_id} {relation_id}\n")
    
    print(f"✓ 已生成 {output_file}")
    print(f"  - 成功转换: {len(triples)} 个三元组")
    if skipped > 0:
        print(f"  - 跳过: {skipped} 行")
    
    return len(triples)


def main():
    import sys
    import os
    
    # 可以通过命令行参数指定，或使用默认路径
    if len(sys.argv) >= 2:
        data_dir = sys.argv[1]
    else:
        data_dir = "./"
        print(f"使用默认数据目录: {data_dir}\n")
        print("提示: 可以通过命令行参数指定数据目录:")
        print("  python convert_to_id.py <data_directory>\n")
    
    # 文件路径
    entity_file = os.path.join(data_dir, "entity2id.txt")
    relation_file = os.path.join(data_dir, "relation2id.txt")
    
    # 检查映射文件是否存在
    if not os.path.exists(entity_file):
        print(f"❌ 错误: 找不到文件 {entity_file}")
        print("请先运行 generate_id_mappings.py 生成映射文件")
        return
    
    if not os.path.exists(relation_file):
        print(f"❌ 错误: 找不到文件 {relation_file}")
        print("请先运行 generate_id_mappings.py 生成映射文件")
        return
    
    try:
        # 加载映射
        entity2id, relation2id = load_mappings(entity_file, relation_file)
        
        # 转换所有文件
        files_to_convert = [
            ("train.tsv", "train2id_aug.txt"),
            #("dev.tsv", "valid2id.txt"),
            #("test.tsv", "test2id.txt")
        ]
        
        total_triples = 0
        for input_name, output_name in files_to_convert:
            input_file = os.path.join(data_dir, input_name)
            output_file = os.path.join(data_dir, output_name)
            
            if not os.path.exists(input_file):
                print(f"⚠ 跳过: 找不到文件 {input_file}")
                continue
            
            num_triples = convert_triples_to_id(input_file, output_file, entity2id, relation2id)
            total_triples += num_triples
        
        print(f"\n✅ 所有文件转换完成!")
        print(f"总共转换了 {total_triples} 个三元组")
        
    except FileNotFoundError as e:
        print(f"\n❌ 错误: 找不到文件 {e.filename}")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
