#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Efficient calculation of shortest path distances between head and tail entities
in test.tsv based on train.tsv knowledge graph (undirected).
"""

import time
from collections import defaultdict, deque
from typing import Dict, Set, List, Tuple
import sys


def load_train_graph(train_file: str) -> Dict[str, Set[str]]:
    """
    Load training knowledge graph as an undirected graph.
    Returns adjacency list representation.
    """
    print(f"Loading train graph from {train_file}...")
    graph = defaultdict(set)
    
    with open(train_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 50000 == 0:
                print(f"  Processed {line_num} lines...")
            
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                head, relation, tail = parts[0], parts[1], parts[2]
                # Build undirected graph (ignore relation direction)
                graph[head].add(tail)
                graph[tail].add(head)
    
    print(f"Graph loaded: {len(graph)} entities, {sum(len(v) for v in graph.values()) // 2} edges")
    return graph


def bfs_shortest_path(graph: Dict[str, Set[str]], start: str, end: str, max_depth: int = 10) -> int:
    """
    BFS to find shortest path distance between start and end entities.
    Returns distance, or -1 if no path exists within max_depth.
    """
    if start == end:
        return 0
    
    if start not in graph or end not in graph:
        return -1
    
    visited = {start}
    queue = deque([(start, 0)])
    
    while queue:
        current, dist = queue.popleft()
        
        # Early termination if max depth exceeded
        if dist >= max_depth:
            return -1
        
        for neighbor in graph[current]:
            if neighbor == end:
                return dist + 1
            
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    
    return -1  # No path found


def calculate_test_distances(test_file: str, graph: Dict[str, Set[str]], max_depth: int = 10) -> List[int]:
    """
    Calculate shortest path distances for all triples in test.tsv.
    Returns list of distances (-1 for no path).
    """
    print(f"\nCalculating distances for test triples from {test_file}...")
    distances = []
    
    with open(test_file, 'r', encoding='utf-8') as f:
        test_lines = f.readlines()
    
    total = len(test_lines)
    print(f"Total test triples: {total}")
    
    for idx, line in enumerate(test_lines, 1):
        if idx % 1000 == 0:
            print(f"  Processed {idx}/{total} ({idx*100/total:.1f}%)")
        
        parts = line.strip().split('\t')
        if len(parts) >= 3:
            head, relation, tail = parts[0], parts[1], parts[2]
            dist = bfs_shortest_path(graph, head, tail, max_depth)
            distances.append(dist)
        else:
            distances.append(-1)
    
    print(f"Distance calculation completed!")
    return distances


def analyze_statistics(distances: List[int]) -> None:
    """
    Analyze and print statistics of distance distribution.
    """
    print("\n" + "="*70)
    print("SHORTEST PATH DISTANCE STATISTICS")
    print("="*70)
    
    total = len(distances)
    dist_counts = defaultdict(int)
    
    for dist in distances:
        dist_counts[dist] += 1
    
    # Sort by distance
    sorted_dists = sorted(dist_counts.keys())
    
    print(f"\nTotal test triples: {total}")
    print("\nDistance Distribution:")
    print("-" * 70)
    print(f"{'Distance':<15} {'Count':<15} {'Percentage':<15} {'Cumulative %'}")
    print("-" * 70)
    
    cumulative = 0
    for dist in sorted_dists:
        count = dist_counts[dist]
        percentage = count * 100.0 / total
        cumulative += percentage
        
        if dist == -1:
            dist_label = "No path / >10"
        elif dist == 0:
            dist_label = "0 (same)"
        else:
            dist_label = str(dist)
        
        print(f"{dist_label:<15} {count:<15} {percentage:<14.2f}% {cumulative:<.2f}%")
    
    print("-" * 70)
    
    # Additional statistics
    valid_distances = [d for d in distances if d >= 0]
    if valid_distances:
        avg_dist = sum(valid_distances) / len(valid_distances)
        print(f"\nConnected triples: {len(valid_distances)} ({len(valid_distances)*100/total:.2f}%)")
        print(f"Average distance (for connected): {avg_dist:.2f}")
        print(f"Min distance: {min(valid_distances)}")
        print(f"Max distance: {max(valid_distances)}")
    
    disconnected = dist_counts[-1]
    print(f"Disconnected triples: {disconnected} ({disconnected*100/total:.2f}%)")
    
    print("\n" + "="*70)


def save_results(distances: List[int], output_file: str) -> None:
    """
    Save distances to output file.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("distance\n")
        for dist in distances:
            f.write(f"{dist}\n")
    print(f"\nResults saved to: {output_file}")


def main():
    # File paths
    train_file = "train.tsv"
    test_file = "train.tsv"
    output_file = "train_shortest_distances.txt"
    
    # Start timing
    start_time = time.time()
    
    # Load train graph
    graph = load_train_graph(train_file)
    load_time = time.time()
    print(f"Graph loading time: {load_time - start_time:.2f} seconds")
    
    # Calculate distances for test set
    distances = calculate_test_distances(test_file, graph, max_depth=10)
    calc_time = time.time()
    print(f"Distance calculation time: {calc_time - load_time:.2f} seconds")
    
    # Analyze and print statistics
    analyze_statistics(distances)
    
    # Save results
    save_results(distances, output_file)
    
    # Total time
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
