"""
生成 entity2id.txt 和 relation2id.txt 文件
从三元组文件中提取所有唯一的实体和关系，并分配从0开始的ID
"""

def generate_id_mappings(input_file, entity_output_file, relation_output_file):
    """
    从三元组文件生成实体和关系的ID映射文件
    
    Args:
        input_file: 输入的三元组文件路径 (格式: 头实体\t关系\t尾实体)
        entity_output_file: 输出的实体映射文件路径
        relation_output_file: 输出的关系映射文件路径
    """
    entities = set()
    relations = set()
    
    # 第一遍扫描：收集所有唯一的实体和关系
    print(f"正在读取文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) != 3:
                print(f"警告: 第 {line_num} 行格式不正确，跳过: {line}")
                continue
            
            head, relation, tail = parts
            entities.add(head)
            entities.add(tail)
            relations.add(relation)
    
    # 排序以保证一致性（可选）
    #entities = sorted(entities)
    #relations = sorted(relations)
    
    print(f"\n统计信息:")
    print(f"- 唯一实体数: {len(entities)}")
    print(f"- 唯一关系数: {len(relations)}")
    
    # 生成 entity2id.txt
    print(f"\n正在生成: {entity_output_file}")
    with open(entity_output_file, 'w', encoding='utf-8') as f:
        f.write(f"{len(entities)}\n")  # 第一行写入实体总数
        for entity_id, entity in enumerate(entities):
            f.write(f"{entity}\t{entity_id}\n")
    
    print(f"✓ 已生成 {entity_output_file}，包含 {len(entities)} 个实体")
    
    # 生成 relation2id.txt
    print(f"正在生成: {relation_output_file}")
    with open(relation_output_file, 'w', encoding='utf-8') as f:
        f.write(f"{len(relations)}\n")  # 第一行写入关系总数
        for relation_id, relation in enumerate(relations):
            f.write(f"{relation}\t{relation_id}\n")
    
    print(f"✓ 已生成 {relation_output_file}，包含 {len(relations)} 个关系")
    
    # 显示前几个示例
    print(f"\n实体映射示例 (前5个):")
    for i, entity in enumerate(entities[:5]):
        print(f"  {entity} -> {i}")
    
    print(f"\n关系映射示例 (前5个):")
    for i, relation in enumerate(relations[:5]):
        print(f"  {relation} -> {i}")
    
    return entities, relations


if __name__ == "__main__":
    import sys
    
    # 可以通过命令行参数指定文件路径，或使用默认路径
    if len(sys.argv) >= 4:
        input_file = sys.argv[1]
        entity_output = sys.argv[2]
        relation_output = sys.argv[3]
    else:
        # 默认路径 - 请根据实际情况修改
        input_file = "./train.tsv"
        entity_output = "./entity2id.txt"
        relation_output = "./relation2id.txt"
        
        print("使用默认路径:")
        print(f"  输入文件: {input_file}")
        print(f"  实体映射输出: {entity_output}")
        print(f"  关系映射输出: {relation_output}")
        print("\n提示: 可以通过命令行参数指定文件路径:")
        print("  python generate_id_mappings.py <input_file> <entity_output> <relation_output>\n")
    
    try:
        entities, relations = generate_id_mappings(input_file, entity_output, relation_output)
        print("\n✅ 所有文件生成完成!")
    except FileNotFoundError as e:
        print(f"\n❌ 错误: 找不到文件 {e.filename}")
        print("请检查文件路径是否正确")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
