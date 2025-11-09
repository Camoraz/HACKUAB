import heuristica as hr
import random
import math
import copy
import numpy as np

# ---------------------------
# parámetros globales
# ---------------------------
DIST_MAX = 3      # distancia máxima para conectar con landmark
WALK_SPEED = 0.01         # m/s, para calcular weight caminando
METRO_SPEED = 1      # m/s, para calcular weight en metro
ESPERA_METRO = 10.0     # segundos de espera promedio
MAX_ESTACIONES = 10
PERTURBATION_SCALE = 2.5 # inicial para mutación
ANNEALING_RATE = 0.995

WEIGHT_CONVENIENCIA = 3
WEIGHT_RECTITUD = 0.012
WEIGHT_HOMOGENEIDAD = 0.1
WEIGHT_LONGITUD = 0.001

def euclid(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def relative_to_absolute(relative_line):
    """
    Convert relative line representation to absolute coordinates.
    First element is absolute, rest are relative to previous.
    """
    if not relative_line:
        return []
    
    absolute_line = [relative_line[0]]  # First station is absolute
    for i in range(1, len(relative_line)):
        prev_x, prev_y = absolute_line[i-1]
        dx, dy = relative_line[i]
        absolute_line.append((prev_x + dx, prev_y + dy))
    
    return absolute_line

def absolute_to_relative(absolute_line):
    """
    Convert absolute coordinates to relative representation.
    First element remains absolute, rest are relative to previous.
    """
    if not absolute_line:
        return []
    
    relative_line = [absolute_line[0]]  # First station remains absolute
    for i in range(1, len(absolute_line)):
        prev_x, prev_y = absolute_line[i-1]
        curr_x, curr_y = absolute_line[i]
        relative_line.append((curr_x - prev_x, curr_y - prev_y))
    
    return relative_line

def generar_grafo_inicio(metro, puntos, positions, names):
    """
    Generate graph from relative line representation.
    """
    
    graph = copy.deepcopy(metro)
    pos = copy.deepcopy(positions)

    for i, name in enumerate(names):
        if name not in graph:
            graph[name] = {}
            pos[name] = puntos[i]


    for i, p_new in enumerate(puntos):
        n_new = names[i]
        for n_old, p_old in positions.items():
            dist = euclid(p_new, p_old)
            if (dist <= DIST_MAX or True):
                weight = ESPERA_METRO + dist / WALK_SPEED
                graph[n_new][n_old] = weight
                graph[n_old][n_new] = weight

    return graph, pos

def generar_grafo_inicio_vectorized(metro, puntos, positions, names):
    graph = copy.deepcopy(metro)
    pos = copy.deepcopy(positions)

    n_new = len(names)
    old_names = list(positions.keys())
    n_old = len(old_names)

    for i, name in enumerate(names):
        if name not in graph:
            graph[name] = {}
            pos[name] = puntos[i]

    old_positions = np.array([positions[name] for name in old_names])  # shape (V, 2)
    new_positions = np.array(puntos)  # shape (n, 2)

    # Compute all pairwise distances at once
    diff = new_positions[:, None, :] - old_positions[None, :, :]  # shape (n_new, n_old, 2)
    dists = np.linalg.norm(diff, axis=2)  # shape (n_new, n_old)

    for i, n_name in enumerate(names):
        for j, old_name in enumerate(old_names):
            print(f"{i}, {j}")
            weight = ESPERA_METRO + dists[i, j] / WALK_SPEED
            graph[n_name][old_name] = weight
            graph[old_name][n_name] = weight

    return graph, pos

def generar_grafo(metro, relative_line, positions):
    """
    Generate graph from relative line representation.
    """
    # Convert relative line to absolute coordinates
    absolute_line = relative_to_absolute(relative_line)
    
    graph = copy.deepcopy(metro)
    pos = copy.deepcopy(positions)

    nombres_linea = [f"L{i}" for i in range(len(absolute_line))]

    for i, name in enumerate(nombres_linea):
        if name not in graph:
            graph[name] = {}
            pos[name] = absolute_line[i]

    for i in range(len(absolute_line) - 1):
        p1, p2 = absolute_line[i], absolute_line[i+1]
        n1, n2 = nombres_linea[i], nombres_linea[i+1]
        dist = euclid(p1, p2)
        weight = dist / METRO_SPEED

        graph[n1][n2] = weight
        graph[n2][n1] = weight

    for i, p_new in enumerate(absolute_line):
        n_new = nombres_linea[i]
        for n_old, p_old in positions.items():
            dist = euclid(p_new, p_old)
            if dist <= DIST_MAX:
                weight = ESPERA_METRO + dist / WALK_SPEED
                graph[n_new][n_old] = weight
                graph[n_old][n_new] = weight

    return graph, pos

def evaluar_grafo(metro, relative_line, positions, landmarks, weights):
    """
    Evaluate graph with relative line representation.
    """
    graph, pos_new = generar_grafo(metro, relative_line, positions)
    score = hr.expected_shortest_path_score(graph, pos_new, landmarks, weights)
    return score

def cruzar(relative_line1, relative_line2, generation=1):
    """
    Crossover for relative line representation.
    """
    assert len(relative_line1) == len(relative_line2), "Parents must have same number of stations"
    
    # Convert to absolute for crossover
    absolute_line1 = relative_to_absolute(relative_line1)
    absolute_line2 = relative_to_absolute(relative_line2)
    
    hijo_absolute = []
    
    # Crossover first station (absolute)
    x1, y1 = absolute_line1[0]
    x2, y2 = absolute_line2[0]
    alpha = 0.5
    d_x = abs(x1 - x2)
    d_y = abs(y1 - y2)
    
    x_child = np.random.uniform(min(x1, x2) - alpha*d_x, max(x1, x2) + alpha*d_x)
    y_child = np.random.uniform(min(y1, y2) - alpha*d_y, max(y1, y2) + alpha*d_y)
    
    # Gaussian noise with decaying variance
    sigma = 1.0 * (ANNEALING_RATE ** generation)
    x_child += np.random.normal(0, sigma)
    y_child += np.random.normal(0, sigma)
    
    hijo_absolute.append((x_child, y_child))
    
    # Crossover relative displacements
    for i in range(1, len(absolute_line1)):
        # Get relative displacements from both parents
        dx1, dy1 = relative_line1[i]
        dx2, dy2 = relative_line2[i]
        
        # BLX-alpha crossover on relative displacements
        alpha = 0.5
        d_dx = abs(dx1 - dx2)
        d_dy = abs(dy1 - dy2)
        
        dx_child = np.random.uniform(min(dx1, dx2) - alpha*d_dx, max(dx1, dx2) + alpha*d_dx)
        dy_child = np.random.uniform(min(dy1, dy2) - alpha*d_dy, max(dy1, dy2) + alpha*d_dy)
        
        # Gaussian noise with decaying variance
        sigma = 1.0 * (ANNEALING_RATE ** generation)
        dx_child += np.random.normal(0, sigma)
        dy_child += np.random.normal(0, sigma)
        
        hijo_absolute.append((hijo_absolute[-1][0] + dx_child, hijo_absolute[-1][1] + dy_child))
    
    # Convert back to relative representation
    return absolute_to_relative(hijo_absolute)

def path_unstraightness(relative_line):
    """
    Calculate how "unstraight" a path is using relative representation.
    """
    absolute_line = relative_to_absolute(relative_line)
    
    if len(absolute_line) < 3:
        return 0.0  # 2 points or less are always straight
    
    total_angle_change = 0.0
    
    for i in range(1, len(absolute_line)-1):
        x0, y0 = absolute_line[i-1]
        x1, y1 = absolute_line[i]
        x2, y2 = absolute_line[i+1]
        
        # Vector from previous to current
        v1 = (x1 - x0, y1 - y0)
        # Vector from current to next
        v2 = (x2 - x1, y2 - y1)
        
        # Compute angle between v1 and v2 using dot product
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        mag1 = math.hypot(*v1)
        mag2 = math.hypot(*v2)
        if mag1 == 0 or mag2 == 0:
            angle = 0.0
        else:
            # Clamp cos_theta to [-1, 1] to avoid numerical errors
            cos_theta = max(-1.0, min(1.0, dot / (mag1 * mag2)))
            angle = math.acos(cos_theta)  # angle in radians
        total_angle_change += angle**2
    
    return total_angle_change

def path_spacing_penalty(relative_line):
    """
    Calculate spacing penalty using relative representation.
    """
    absolute_line = relative_to_absolute(relative_line)
    
    if len(absolute_line) < 2:
        return 0.0  # no segments, no penalty
    
    # Compute distances between consecutive stations
    distances = []
    for i in range(len(absolute_line) - 1):
        x1, y1 = absolute_line[i]
        x2, y2 = absolute_line[i+1]
        dist = math.hypot(x2 - x1, y2 - y1)
        distances.append(dist)
    
    # Standard deviation of distances
    return np.std(distances)

def path_length(relative_line):
    """
    Calculate total length of a path using relative representation.
    """
    absolute_line = relative_to_absolute(relative_line)
    
    if len(absolute_line) < 2:
        return 0.0  # no segments, no length
    
    total_length = 0.0
    for i in range(len(absolute_line) - 1):
        x1, y1 = absolute_line[i]
        x2, y2 = absolute_line[i+1]
        total_length += math.hypot(x2 - x1, y2 - y1)
    
    return total_length


# ---------------------------
# algoritmo genetico
# ---------------------------
def genetico(metro, positions, landmarks, weights,
             population_size=50, generations=100,
             tournament_k=3, elitism=2, seed=None):
    """
    Genetic algorithm to optimize a line (list of MAX_ESTACIONES (x,y) tuples).
    Now uses relative representation: first station absolute, rest relative to previous.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Determine bounding box for initialization from existing positions
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Add a margin so new stations can lie slightly outside existing nodes
    margin_x = max(1.0, (max_x - min_x) * 0.2)
    margin_y = max(1.0, (max_y - min_y) * 0.2)
    low_x = min_x - margin_x
    high_x = max_x + margin_x
    low_y = min_y - margin_y
    high_y = max_y + margin_y

    line_length = MAX_ESTACIONES

    # Helper: random individual in relative representation
    def random_line(line_length=MAX_ESTACIONES):
        """
        Generate a line in relative representation.
        First station is absolute, rest are relative displacements.
        """
        # Start somewhere randomly
        x, y = random.uniform(low_x, high_x), random.uniform(low_y, high_y)
        line = [(x, y)]  # First station is absolute
        
        # Initial random direction
        angle = random.uniform(0, 2*math.pi)
        
        for _ in range(1, line_length):
            # Small random turn to create gentle curve
            angle += random.uniform(-math.pi/6, math.pi/6)
            
            # Move roughly 0.5 units in that direction with some randomness
            step = 0.5 + 0.2 * random.random()
            dx = step * math.cos(angle)
            dy = step * math.sin(angle)
            
            line.append((dx, dy))  # Relative displacement
        
        return line

    # Helper: evaluate individual (higher is better)
    def fitness(relative_line):
        graph, pos_new = generar_grafo(metro, relative_line, positions)
        score = WEIGHT_CONVENIENCIA * hr.expected_shortest_path_score(graph, pos_new, landmarks, weights)
        score += WEIGHT_RECTITUD * path_unstraightness(relative_line)
        score += WEIGHT_HOMOGENEIDAD * path_spacing_penalty(relative_line)
        score += WEIGHT_LONGITUD * path_length(relative_line)
        return -score

    # Initialize population
    population = [random_line() for _ in range(population_size)]
    fitnesses = [fitness(ind) for ind in population]

    # Track best
    best_idx = int(np.argmax(fitnesses))
    best_individual = copy.deepcopy(population[best_idx])
    best_score = fitnesses[best_idx]

    # Main loop
    for gen in range(1, generations + 1):
        new_population = []

        # Elitism: keep top-N
        idx_sorted = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)
        for i in idx_sorted[:elitism]:
            new_population.append(copy.deepcopy(population[i]))

        # Produce rest of population
        while len(new_population) < population_size:
            # Tournament selection for parents
            def tournament_select():
                aspirants = random.sample(range(population_size), k=min(tournament_k, population_size))
                best = max(aspirants, key=lambda i: fitnesses[i])
                return population[best]

            parent1 = tournament_select()
            parent2 = tournament_select()

            # Ensure parents lengths match (they should)
            assert len(parent1) == len(parent2) == line_length

            # Crossover -> child (cruzar already applies BLX + gaussian noise with annealing)
            child = cruzar(parent1, parent2, generation=gen)

            # Optional local perturbation: small gaussian scaled by PERTURBATION_SCALE * decay
            # This is separate from cruzar's internal noise; keep it small.
            decay = ANNEALING_RATE ** gen
            perturb_scale = PERTURBATION_SCALE * decay
            
            # Perturb first station (absolute)
            x0, y0 = child[0]
            child[0] = (x0 + np.random.normal(0, perturb_scale), 
                        y0 + np.random.normal(0, perturb_scale))
            
            # Perturb relative displacements
            for i in range(1, len(child)):
                dx, dy = child[i]
                child[i] = (dx + np.random.normal(0, perturb_scale),
                           dy + np.random.normal(0, perturb_scale))

            new_population.append(child)

        # Replace population and recompute fitnesses
        population = new_population
        fitnesses = [fitness(ind) for ind in population]

        # Update best
        gen_best_idx = int(np.argmax(fitnesses))
        gen_best_score = fitnesses[gen_best_idx]
        if gen_best_score > best_score:
            best_score = gen_best_score
            best_individual = copy.deepcopy(population[gen_best_idx])

        # (optional) you can print progress here
        print(f"Gen {gen}: best_score = {best_score:.4f}")

    return best_individual, best_score

# ---------------------------
# ejemplo de uso
# ---------------------------

if __name__ == "__main__":

    import pandas as pd
    import math
    import random
    import pickle

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

    n = 5

    # Convert to NumPy array for weighted sampling
    weights_array = np.array(weights)
    probabilities = weights_array / weights_array.sum()

    # Sample n landmarks based on population weights
    sample_indices = np.random.choice(len(landmarks), size=n, replace=False, p=probabilities)
    landmarks = [landmarks[i] for i in sample_indices]
    pos_landmarks = [barrios_positions[landmarks[i]] for i in range(n)]
    weights = [weights[i] for i in sample_indices]

    # Normalize weights (plain Python)
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    metro, positions = generar_grafo_inicio_vectorized(metro, pos_landmarks, positions, landmarks)

    # ejecutar genetico
    best_relative_line, best_score = genetico(metro, positions, landmarks, weights, population_size=20, generations=1500)
    
    # Convert to absolute for display
    best_absolute_line = relative_to_absolute(best_relative_line)
    
    print(f"Mejor linea encontrada (relative): {best_relative_line}")
    print(f"Mejor linea encontrada (absolute): {best_absolute_line}")
    print(f"Score: {best_score:.4f}")

    import plotting as plt
    
    metro, pos_neww = generar_grafo(metro, best_relative_line, positions)
    plt.plot_simple_solution(metro, pos_neww, landmarks, weights, best_absolute_line, best_score)
