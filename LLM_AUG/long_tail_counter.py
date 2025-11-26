from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def load_dataset_stats(file_path):
    """加载数据集统计信息"""
    entity_counter = defaultdict(int)
    relation_counter = defaultdict(int)
    
    with open(file_path, 'r') as f:
        for line in f:
            head, relation, tail = line.strip().split('\t')
            entity_counter[head] += 1
            entity_counter[tail] += 1
            relation_counter[relation] += 1
    
    entity_counts = list(entity_counter.values())
    relation_counts = list(relation_counter.values())
    
    return entity_counts, relation_counts

# 加载两个数据集
fb15k_entity_counts, fb15k_relation_counts = load_dataset_stats("./fb15k-237/FB15k-237/train.tsv")
wn18rr_entity_counts, wn18rr_relation_counts = load_dataset_stats("./wn18rr/WN18RR/train.tsv")


# 为了兼容后面的代码，保持原有变量名
entity_counts = wn18rr_entity_counts
relation_counts = wn18rr_relation_counts

# 创建左右两个子图显示长尾分布对比
plt.figure(figsize=(15, 6))

# 左图：FB15K-237长尾分布
plt.subplot(1, 2, 1)
sorted_fb15k_entity = sorted(fb15k_entity_counts, reverse=True)
plt.plot(sorted_fb15k_entity, color='blue', linewidth=2)
plt.yscale('log')
plt.xlabel("Rank (by degree)")
plt.ylabel("Degree (log scale)")
plt.title("FB15K-237 Entity Degree (Long-Tail Distribution)")
plt.grid(True, alpha=0.3)

# 右图：WN18RR长尾分布
plt.subplot(1, 2, 2)
sorted_wn18rr_entity = sorted(wn18rr_entity_counts, reverse=True)
plt.plot(sorted_wn18rr_entity, color='red', linewidth=2)
plt.yscale('log')
plt.xlabel("Rank (by degree)")
plt.ylabel("Degree (log scale)")
plt.title("WN18RR Entity Degree (Long-Tail Distribution)")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

def min_max_normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val)

def gini_coefficient(counts):
    # 对数据进行归一化
    normalized_counts = min_max_normalize(counts)
    sorted_counts = np.sort(normalized_counts)
    n = len(sorted_counts)
    index = np.arange(1, n+1)
    return (np.sum((2 * index - n - 1) * sorted_counts)) / (n * np.sum(sorted_counts))

def top_n_percentage(counts, n_percent=0.2):
    counts_sorted = sorted(counts, reverse=True)
    total = sum(counts_sorted)
    n = int(len(counts_sorted) * n_percent)
    top_sum = sum(counts_sorted[:n])
    return top_sum / total

# 计算两个数据集的统计指标并显示对比
print("=== 数据集对比统计 ===")
print("\n--- FB15K-237 ---")
gini_entity_fb15k = gini_coefficient(fb15k_entity_counts)
gini_relation_fb15k = gini_coefficient(fb15k_relation_counts)
print(f"Entity Gini: {gini_entity_fb15k:.3f}, Relation Gini: {gini_relation_fb15k:.3f}")
print(f"Top 20% Entities: {top_n_percentage(fb15k_entity_counts, 0.8)*100:.1f}%")
print(f"Top 20% Relations: {top_n_percentage(fb15k_relation_counts, 0.2)*100:.1f}%")

print("\n--- WN18RR ---")
gini_entity = gini_coefficient(entity_counts)
gini_relation = gini_coefficient(relation_counts)
print(f"Entity Gini: {gini_entity:.3f}, Relation Gini: {gini_relation:.3f}")
#指度数占比
print(f"Top 80% Entities: {top_n_percentage(entity_counts, 0.8)*100:.1f}%")
print(f"Top 80% Relations: {top_n_percentage(relation_counts, 0.2)*100:.1f}%")