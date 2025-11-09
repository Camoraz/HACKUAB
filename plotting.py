import matplotlib.pyplot as plt

def plot_simple_solution(metro, positions, landmarks, weights, best_line, best_score=None):
    """
    Visualización simple del grafo original con la nueva línea superpuesta
    Compatible con metro como dict-of-dicts.
    """
    plt.figure(figsize=(10, 8))
    
    # 1. Dibujar conexiones del metro original (líneas grises)
    for node, edges in metro.items():
        x1, y1 = positions[node]
        for neighbor, weight in edges.items():  # <-- dict-of-dicts
            x2, y2 = positions[neighbor]
            plt.plot([x1, x2], [y1, y2], 'gray', linewidth=2, alpha=0.7)
    
    # 2. Dibujar nodos del metro original (círculos azules)
    for node, (x, y) in positions.items():
        if node in metro:  # Solo nodos que están en el grafo del metro
            plt.plot(x, y, 'bo', markersize=10, label='Estación Metro' if node == list(metro.keys())[0] else "")
            plt.text(x, y, f'  {node}', fontsize=9, verticalalignment='center')
    
    # 3. Dibujar landmarks (triángulos rojos)
    for i, lm in enumerate(landmarks):
        x, y = positions[lm]
        plt.plot(x, y, 'r^', markersize=12, label='Landmark' if i == 0 else "")
        plt.text(x, y, f'  {lm}({weights[i]})', fontsize=9, color='red', verticalalignment='center')
    
    # 4. Dibujar nueva línea (línea verde con puntos)
    if best_line:
        line_x = [coord[0] for coord in best_line]
        line_y = [coord[1] for coord in best_line]
        
        # Dibujar línea
        plt.plot(line_x, line_y, 'g-', linewidth=3, alpha=0.8, label='Nueva Línea')
        
        # Dibujar estaciones
        plt.plot(line_x, line_y, 'go', markersize=8, label='Nueva Estación')
        
        # Numerar estaciones
        for i, (x, y) in enumerate(best_line):
            plt.text(x, y, f'  {i+1}', fontsize=8, color='green', verticalalignment='center')
    
    # Configuración final
    plt.xlabel('Coordenada X')
    plt.ylabel('Coordenada Y')
    
    title = 'Sistema de Metro con Nueva Línea'
    if best_score is not None:
        title += f' (Score: {best_score:.2f})'
    plt.title(title)
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Guardar a PNG en vez de mostrar
    output_file = "metro_plot.png"
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Gráfico guardado en {output_file}")
