import heapq
import math

# ---------------------------
# Euclidean distance heuristic for A*
# ---------------------------
def heuristic(u, v, pos):
    # si u o v son tuplas, son coordenadas directas
    if isinstance(u, tuple):
        x1, y1 = u
    else:
        x1, y1 = pos[u]

    if isinstance(v, tuple):
        x2, y2 = v
    else:
        x2, y2 = pos[v]

    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


# ---------------------------
# A* algorithm - FIXED for dict of dicts
# ---------------------------
def astar(graph, positions, start, goal):
    queue = [(0 + heuristic(start, goal, positions), 0, start, [])]  # (f_score, g_score, node, path)
    g_scores = {start: 0}
    visited = set()

    while queue:
        f, g, node, path = heapq.heappop(queue)
        current_path = path + [node]
        
        if node == goal:
            return g, current_path
        
        if node in visited:
            continue
        visited.add(node)
        
        # FIXED: Iterate through dictionary items instead of list of tuples
        for neighbor, weight in graph.get(node, {}).items():
            if neighbor in visited:
                continue
            tentative_g = g + weight
            if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                g_scores[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal, positions)
                heapq.heappush(queue, (f_score, tentative_g, neighbor, current_path))
    
    return float('inf'), []  # No path found

# ---------------------------
# Expected distance score function - FIXED
# ---------------------------
def expected_shortest_path_score(graph, positions, landmarks, weights):
    total_distance = 0
    n = len(landmarks)
    pairs_processed = 0
    
    for i in range(n):
        for j in range(i+1, n):
            if weights[i] > 0 and weights[j] > 0:  # Solo considerar landmarks con peso > 0
                d, _ = astar(graph, positions, landmarks[i], landmarks[j])
                if d < float('inf'):
                    total_distance += d * weights[i] * weights[j]
                    pairs_processed += 1
    
    # Evitar división por cero
    if pairs_processed == 0:
        return float('inf')
    
    return total_distance / pairs_processed  # Promedio para normalizar

def euclid(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# ---------------------------
# example usage with dict of dicts
# ---------------------------
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import math
    import random
    import pickle

    # ---------- Load real data ----------
    barrios = pd.read_csv("Poblacion/barrios_barcelona_geocoded.csv")

    with open("metro_graph_with_penalty.pkl", "rb") as f:
        metro_origin = pickle.load(f)

    metro = {node: dict(edges) for node, edges in metro_origin.items()}

    positions_pre = pd.read_json("stations.json")
    positions = {}
    ref_name = random.choice(list(positions_pre.keys()))
    lat0 = positions_pre[ref_name]['lan']
    lon0 = positions_pre[ref_name]['lon']

    for name, coords in positions_pre.items():
        lat, lon = coords['lan'], coords['lon']
        x = (lon - lon0) * 111.0 * math.cos(math.radians(lat0))
        y = (lat - lat0) * 111.0
        positions[name] = (x, y)

    barrios_positions = {}
    for _, row in barrios.iterrows():
        name = row['Nom_Barri']
        lat = row['lat']
        lon = row['lon']
        x = (lon - lon0) * 111.0 * math.cos(math.radians(lat0))
        y = (lat - lat0) * 111.0
        barrios_positions[name] = (x, y)

    landmarks = list(barrios_positions.keys())
    pos_landmarks = [barrios_positions[name] for name in landmarks]
    weights = [row['poblacion'] for _, row in barrios.iterrows()]

    # Sample n landmarks based on population weights
    n = 50
    weights_array = np.array(weights)
    probabilities = weights_array / weights_array.sum()
    sample_indices = np.random.choice(len(landmarks), size=n, replace=False, p=probabilities)
    landmarks = [landmarks[i] for i in sample_indices]
    pos_landmarks = [barrios_positions[landmarks[i]] for i in range(n)]
    weights = [weights[i] for i in sample_indices]
    weights = [w / sum(weights) for w in weights]  # normalize

    import genetico2 as gen
    # ---------- Generate graph (replace mock function) ----------
    metro, positions = gen.generar_grafo_inicio_vectorized(metro, pos_landmarks, positions, landmarks)

    # ---------- Compute score ----------
    score = expected_shortest_path_score(metro, positions, landmarks, weights)
    print(f"Expected Shortest Path Score: {score}")
