"""
提取2跳子图并统计节点个数
从train.tsv中提取指定实体为起点的2跳子图
支持将子图存储到Neo4j数据库
"""

from collections import defaultdict, deque
import time
from neo4j import GraphDatabase
import argparse
import os


def load_entity_names(entity_file):
    """
    加载实体ID到名称的映射
    """
    entity2name = {}
    print(f"Loading entity names from {entity_file}...")
    
    with open(entity_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                # 使用split分割，限制最多分割1次（处理名称中可能包含空格的情况）
                parts = line.split(None, 1)
                if len(parts) == 2:
                    entity_id, entity_name = parts
                    entity2name[entity_id] = entity_name
    
    print(f"Loaded {len(entity2name)} entity names")
    return entity2name


def load_train_graph(train_file):
    """
    加载训练集知识图谱，构建图结构
    返回邻接表结构和三元组列表
    """
    graph = defaultdict(set)  # 使用set避免重复边
    triples = []  # 存储原始三元组
    
    print("Loading train graph...")
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 3:
                continue
            
            head, relation, tail = parts
            # 无向图：添加双向边
            graph[head].add(tail)
            graph[tail].add(head)
            # 保存原始三元组
            triples.append((head, relation, tail))
    
    print(f"Graph loaded. Total entities: {len(graph)}, Total triples: {len(triples)}")
    return graph, triples


def load_low_freq_entities(low_freq_file):
    """
    从低频实体文件中读取实体列表
    支持两种格式：
    1. entity\tcount 格式（如 low_frequency_entities_wn18rr.txt）
    2. 三元组格式（如 low_freq_triplets_in_test.txt）
    """
    entities = []
    
    print(f"Loading low frequency entities from {low_freq_file}...")
    with open(low_freq_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            
            if len(parts) == 2:
                # entity\tcount 格式
                entity, count = parts
                entities.append(entity)
            elif len(parts) == 3:
                # 三元组格式
                head, relation, tail = parts
                if head not in entities:
                    entities.append(head)
                if tail not in entities:
                    entities.append(tail)
    
    print(f"Total low frequency entities: {len(entities)}")
    return entities


def extract_2hop_subgraph(graph, triples, start_entity):
    """
    使用BFS提取从start_entity出发的2跳子图
    返回子图中所有节点的集合和相关的三元组
    """
    if start_entity not in graph:
        return set(), []
    
    # BFS遍历
    visited = set()
    queue = deque([(start_entity, 0)])  # (entity, hop_distance)
    visited.add(start_entity)
    
    subgraph_nodes = set()
    subgraph_nodes.add(start_entity)
    
    while queue:
        current_entity, hop_distance = queue.popleft()
        
        # 只扩展到2跳
        if hop_distance < 2:
            # 遍历所有邻居
            for neighbor in graph[current_entity]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    subgraph_nodes.add(neighbor)
                    queue.append((neighbor, hop_distance + 1))
                else:
                    # 即使访问过，只要在2跳内，也要加入子图
                    subgraph_nodes.add(neighbor)
    
    # 提取子图中的所有三元组
    subgraph_triples = []
    for head, relation, tail in triples:
        if head in subgraph_nodes and tail in subgraph_nodes:
            subgraph_triples.append((head, relation, tail))
    
    return subgraph_nodes, subgraph_triples


class Neo4jConnector:
    """
    Neo4j数据库连接器
    """
    def __init__(self, uri, user, password, database=None, entity2name=None):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database  # 指定数据库名称，None表示使用默认数据库
        self.entity2name = entity2name if entity2name else {}
        if database:
            print(f"Using database: {database}")
    
    def close(self):
        self.driver.close()
    
    def clear_database(self):
        """清空数据库"""
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("Database cleared.")
    
    def create_subgraph(self, nodes, triples):
        """
        将子图存储到Neo4j
        nodes: 节点集合
        triples: 三元组列表 [(head, relation, tail), ...]
        """
        with self.driver.session(database=self.database) as session:
            # 创建节点（包含实体ID和实体名称）
            print(f"Creating {len(nodes)} nodes...")
            for node in nodes:
                entity_name = self.entity2name.get(node, node)  # 如果找不到名称，使用ID
                session.run(
                    "MERGE (e:Entity {id: $id, name: $name})",
                    id=node, name=entity_name
                )
            
            # 创建关系
            print(f"Creating {len(triples)} relationships...")
            batch_size = 1000
            for i in range(0, len(triples), batch_size):
                batch = triples[i:i + batch_size]
                for head, relation, tail in batch:
                    # 使用MERGE确保关系不重复
                    session.run(
                        """
                        MATCH (h:Entity {id: $head})
                        MATCH (t:Entity {id: $tail})
                        MERGE (h)-[r:RELATION {type: $relation}]->(t)
                        """,
                        head=head, tail=tail, relation=relation
                    )
                if (i + batch_size) % 5000 == 0:
                    print(f"  Processed {i + batch_size}/{len(triples)} relationships...")
            
            print("Subgraph successfully stored in Neo4j!")


def extract_and_save_individual_subgraphs(graph, triples, low_freq_file, output_dir, entity2name, k_hop=2):
    """
    为每个低频实体提取k-hop子图，并保存到单独的文件中
    
    Args:
        graph: 图的邻接表表示
        triples: 所有三元组列表
        low_freq_file: 低频实体文件路径
        output_dir: 输出文件夹路径
        entity2name: 实体ID到名称的映射
        k_hop: 子图的跳数（默认2跳）
    """
    # 加载低频实体
    low_freq_entities = load_low_freq_entities(low_freq_file)
    
    # 创建输出文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    print(f"\nExtracting {k_hop}-hop subgraphs for {len(low_freq_entities)} entities...")
    
    total_entities = len(low_freq_entities)
    processed = 0
    
    for entity in low_freq_entities:
        # 提取子图
        subgraph_nodes, subgraph_triples = extract_khop_subgraph_general(graph, triples, entity, k_hop)
        
        # 构建输出文件名（使用实体ID）
        entity_name = entity2name.get(entity, entity)
        # 清理文件名中的特殊字符
        safe_entity_id = entity.replace('/', '_').replace('\\', '_')
        output_file = os.path.join(output_dir, f"{safe_entity_id}.txt")
        
        # 保存子图信息到文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Entity: {entity}\n")
            f.write(f"Entity Name: {entity_name}\n")
            f.write(f"K-hop: {k_hop}\n")
            f.write(f"Subgraph Nodes: {len(subgraph_nodes)}\n")
            f.write(f"Subgraph Triples: {len(subgraph_triples)}\n")
            f.write("\n" + "=" * 50 + "\n")
            f.write("Nodes in Subgraph:\n")
            f.write("=" * 50 + "\n")
            
            for node in sorted(subgraph_nodes):
                node_name = entity2name.get(node, node)
                f.write(f"{node}\t{node_name}\n")
            
            f.write("\n" + "=" * 50 + "\n")
            f.write("Triples in Subgraph:\n")
            f.write("=" * 50 + "\n")
            
            for head, relation, tail in subgraph_triples:
                head_name = entity2name.get(head, head)
                tail_name = entity2name.get(tail, tail)
                f.write(f"{head}\t{relation}\t{tail}\n")
                f.write(f"  {head_name} -> [{relation}] -> {tail_name}\n")
        
        processed += 1
        if processed % 10 == 0 or processed == total_entities:
            print(f"Processed {processed}/{total_entities} entities...")
    
    print(f"\nAll subgraphs saved to: {output_dir}")
    print(f"Total files created: {processed}")


def extract_khop_subgraph_general(graph, triples, start_entity, k_hop):
    """
    使用BFS提取从start_entity出发的k-hop子图
    返回子图中所有节点的集合和相关的三元组
    """
    if start_entity not in graph:
        return set(), []
    
    # BFS遍历
    visited = set()
    queue = deque([(start_entity, 0)])  # (entity, hop_distance)
    visited.add(start_entity)
    
    subgraph_nodes = set()
    subgraph_nodes.add(start_entity)
    
    while queue:
        current_entity, hop_distance = queue.popleft()
        
        # 只扩展到k跳
        if hop_distance < k_hop:
            # 遍历所有邻居
            for neighbor in graph[current_entity]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    subgraph_nodes.add(neighbor)
                    queue.append((neighbor, hop_distance + 1))
                else:
                    # 即使访问过，只要在k跳内，也要加入子图
                    subgraph_nodes.add(neighbor)
    
    # 提取子图中的所有三元组
    subgraph_triples = []
    for head, relation, tail in triples:
        if head in subgraph_nodes and tail in subgraph_nodes:
            subgraph_triples.append((head, relation, tail))
    
    return subgraph_nodes, subgraph_triples


def batch_extract_subgraphs(graph, triples, low_freq_file, output_file):
    """
    批量提取低频实体的子图并统计
    """
    # 加载低频实体
    low_freq_entities = load_low_freq_entities(low_freq_file)
    
    # 对每个实体提取2跳子图并统计
    print("\nExtracting 2-hop subgraphs...")
    entity_subgraph_stats = {}
    
    total_entities = len(low_freq_entities)
    processed = 0
    
    for entity in sorted(low_freq_entities):
        subgraph_nodes, subgraph_triples = extract_2hop_subgraph(graph, triples, entity)
        entity_subgraph_stats[entity] = len(subgraph_nodes)
        
        processed += 1
        if processed % 100 == 0:
            print(f"Processed {processed}/{total_entities} entities...")
    
    # 保存统计结果
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Entity\tSubgraph_Node_Count\n")
        f.write("=" * 50 + "\n")
        
        for entity in sorted(entity_subgraph_stats.keys()):
            node_count = entity_subgraph_stats[entity]
            f.write(f"{entity}\t{node_count}\n")
        
        f.write("\n" + "=" * 50 + "\n")
        f.write("Summary Statistics\n")
        f.write("=" * 50 + "\n")
        
        node_counts = list(entity_subgraph_stats.values())
        f.write(f"Total entities processed: {len(entity_subgraph_stats)}\n")
        f.write(f"Average subgraph size: {sum(node_counts) / len(node_counts):.2f}\n")
        f.write(f"Min subgraph size: {min(node_counts)}\n")
        f.write(f"Max subgraph size: {max(node_counts)}\n")
        f.write(f"Median subgraph size: {sorted(node_counts)[len(node_counts)//2]}\n")
    
    print("\n" + "=" * 50)
    print("Statistics Summary:")
    print("=" * 50)
    print(f"Total entities processed: {len(entity_subgraph_stats)}")
    print(f"Average subgraph size: {sum(node_counts) / len(node_counts):.2f}")
    print(f"Min subgraph size: {min(node_counts)}")
    print(f"Max subgraph size: {max(node_counts)}")
    print(f"Median subgraph size: {sorted(node_counts)[len(node_counts)//2]}")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Extract 2-hop subgraph and store to Neo4j')
    parser.add_argument('--train_file', type=str, default='train.tsv', 
                        help='Path to training file')
    parser.add_argument('--entity_name_file', type=str, default='entity2text.txt',
                        help='Path to entity name mapping file')
    parser.add_argument('--entity', type=str, default=None,
                        help='Specific entity ID to extract subgraph')
    parser.add_argument('--low_freq_file', type=str, default='low_frequency_entities_wn18rr.txt',
                        help='Path to low frequency entities file (for batch mode)')
    parser.add_argument('--output_file', type=str, default='entity_2hop_subgraph_stats.txt',
                        help='Path to output statistics file (for stats mode)')
    parser.add_argument('--output_dir', type=str, default='sparse_nodes',
                        help='Directory to save individual entity subgraph files')
    parser.add_argument('--k_hop', type=int, default=2,
                        help='Number of hops for subgraph extraction (default: 2)')
    parser.add_argument('--mode', type=str, default='individual', choices=['individual', 'stats'],
                        help='Mode: individual (save each entity to separate file) or stats (generate statistics)')
    parser.add_argument('--neo4j_uri', type=str, default='bolt://localhost:7687',
                        help='Neo4j connection URI')
    parser.add_argument('--neo4j_user', type=str, default='neo4j',
                        help='Neo4j username')
    parser.add_argument('--neo4j_password', type=str, default='password',
                        help='Neo4j password')
    parser.add_argument('--neo4j_database', type=str, default=None,
                        help='Neo4j database name (default: neo4j default database)')
    parser.add_argument('--store_neo4j', action='store_true',
                        help='Store subgraph to Neo4j database')
    parser.add_argument('--clear_db', action='store_true',
                        help='Clear Neo4j database before storing')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    # 1. 加载知识图谱
    graph, triples = load_train_graph(args.train_file)
    
    # 2. 加载实体名称映射
    entity2name = load_entity_names(args.entity_name_file)
    
    # 3. 根据模式选择操作
    if args.entity:
        # 单实体模式：提取指定实体的子图
        entity_name = entity2name.get(args.entity, args.entity)
        print(f"\n=== Extracting 2-hop subgraph for entity: {args.entity} ({entity_name}) ===")
        subgraph_nodes, subgraph_triples = extract_2hop_subgraph(graph, triples, args.entity)
        
        print(f"\nSubgraph Statistics:")
        print(f"  Nodes: {len(subgraph_nodes)}")
        print(f"  Triples: {len(subgraph_triples)}")
        
        # 显示前10个节点（带名称）
        print(f"\nSample nodes (first 10):")
        for i, node in enumerate(list(subgraph_nodes)[:10]):
            node_name = entity2name.get(node, node)
            print(f"  {i+1}. {node} ({node_name})")
        
        # 显示前10个三元组（带名称）
        print(f"\nSample triples (first 10):")
        for i, (h, r, t) in enumerate(subgraph_triples[:10]):
            h_name = entity2name.get(h, h)
            t_name = entity2name.get(t, t)
            print(f"  {i+1}. ({h}, {r}, {t})")
            print(f"       {h_name} -> [{r}] -> {t_name}")
        
        # 存储到Neo4j
        if args.store_neo4j:
            print(f"\n=== Storing subgraph to Neo4j ===")
            neo4j_conn = Neo4jConnector(
                args.neo4j_uri, 
                args.neo4j_user, 
                args.neo4j_password, 
                args.neo4j_database,
                entity2name
            )
            
            if args.clear_db:
                neo4j_conn.clear_database()
            
            neo4j_conn.create_subgraph(subgraph_nodes, subgraph_triples)
            neo4j_conn.close()
    else:
        # 批量模式：根据mode参数选择处理方式
        if args.mode == 'individual':
            print(f"\n=== Individual mode: Extracting {args.k_hop}-hop subgraphs for each entity ===")
            extract_and_save_individual_subgraphs(
                graph, triples, args.low_freq_file, args.output_dir, entity2name, args.k_hop
            )
        else:  # stats mode
            print(f"\n=== Stats mode: Processing all low frequency entities ===")
            batch_extract_subgraphs(graph, triples, args.low_freq_file, args.output_file)
            print(f"Results saved to: {args.output_file}")
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()

#python extract_khop_subgraph.py --train_file data/WN18RR/train.tsv --entity_name_file data/WN18RR/entity2text.txt --low_freq_file low_frequency_entities_wn18rr.txt --output_dir sparse_nodes --k_hop 3