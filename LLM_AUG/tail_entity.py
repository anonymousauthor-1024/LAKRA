from collections import defaultdict

# 初始化实体计数器
entity_counter = defaultdict(int)

# 读取所有三元组文件（train/valid/test）
for file_path in ["./wn18rr/WN18RR/train.tsv"]:
    with open(file_path, 'r') as f:
        for line in f:
            head, relation, tail = line.strip().split('\t')
            entity_counter[head] += 1
            entity_counter[tail] += 1

# 将实体按出现次数升序排序（低频在前）
sorted_entities = sorted(entity_counter.items(), key=lambda x: x[1])

# 用户输入：需要提取后百分之多少的实体（例如后20%）
target_percent = 0.001  # 修改此处调整百分比
n_total = len(sorted_entities)
n_low_freq = int(n_total * target_percent)  # 向下取整

# 提取后N%的低频实体k
low_freq_entities = sorted_entities[:n_low_freq]  # 已按升序排列，直接取前n_low_freq个

# 输出统计信息
print(f"实体总数: {n_total}")
print(f"后{target_percent*100:.0f}%实体数量: {n_low_freq}")
print(f"最低频实体示例: {low_freq_entities[0][0]} (出现次数={low_freq_entities[0][1]})")

# 将结果保存到文件
with open("low_frequency_entities_wn18rr.txt", 'w') as f:
    for entity, count in low_freq_entities:
        f.write(f"{entity}\t{count}\n")