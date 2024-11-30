import numpy as np
from aima3.logic import *
from aima3.csp import backtracking_search, AC3, mrv, lcv, mac, CSP
import random

# Fonction de validation des contraintes (les cases adjacentes doivent avoir des valeurs différentes)
def sudoku_constraint(A, a, B, b):
    """Les cases adjacentes (A, B) doivent avoir des valeurs différentes"""
    return a != b

def degree(assignment, csp):
    """Retourne la variable avec le plus grand nombre de voisins non assignés."""
    unassigned_vars = [var for var in csp.variables if var not in assignment]
    return max(unassigned_vars, key=lambda var: len([n for n in csp.neighbors[var] if n not in assignment]))

def mrv_degree(assignment, csp):
    """Retourne la variable avec le moins de valeurs restantes, puis le plus grand nombre de voisins non assignés"""
    unassigned_vars = [var for var in csp.variables if var not in assignment]
    mrv_vars = [var for var in unassigned_vars if len(csp.domains[var]) == min(len(csp.domains[v]) for v in unassigned_vars)]
    return max(mrv_vars, key=lambda var: len([n for n in csp.neighbors[var] if n not in assignment]))

# Dictionnaire pour maper les stratégies en fonctions
heuristics_dict = {
    "mrv": mrv,  # Minimum Remaining Values
    "degree": degree,  # Degree heuristic
    "mrv_degree": mrv_degree  # Minimum Remaining Values + Degree heuristic
}

value_orders_dict = {
    "lcv": lcv,  # Least Constraining Value
    "random": lambda var, assignment, csp: sorted(csp.choices(var), key=lambda _: random.random())
}

inference_methods_dict = {
    "ac3": lambda csp, var, value, assignment, removals: AC3(csp),  # Corrigé pour que AC3 soit utilisé correctement dans le contexte
    "mac": mac  # Maintain Arc-Consistency
}

def solve_sudoku_csp(grid, variable_heuristic, value_order, inference_method):
    """Résoudre le Sudoku avec les heuristiques et stratégies passées"""
    
    # Afficher la grille d'entrée pour vérifier qu'elle est bien reçue
    print("Grille initiale:")
    print(np.array(grid))  # Afficher la grille sous forme de tableau NumPy

    # Initialiser les variables et domaines pour CSP
    variables = [(i, j) for i in range(9) for j in range(9)]
    domains = {(i, j): [grid[i][j]] if grid[i][j] != 0 else list(range(1, 10)) for i in range(9) for j in range(9)}

    # Initialiser les voisins
    neighbors = {}
    for i in range(9):
        for j in range(9):
            neighbors[(i, j)] = set()
            for x in range(9):
                if x != i:
                    neighbors[(i, j)].add((x, j))  # Voisin dans la même colonne
                if x != j:
                    neighbors[(i, j)].add((i, x))  # Voisin dans la même ligne
            start_row, start_col = 3 * (i // 3), 3 * (j // 3)
            for r in range(start_row, start_row + 3):
                for c in range(start_col, start_col + 3):
                    if (r, c) != (i, j):
                        neighbors[(i, j)].add((r, c))

    # Créer un CSP
    csp = CSP(variables, domains, neighbors, sudoku_constraint)

    # Appliquer les heuristiques et stratégies passées
    solution = backtracking_search(
        csp,
        select_unassigned_variable=heuristics_dict[variable_heuristic],  # Mapper à la fonction
        order_domain_values=value_orders_dict[value_order],  # Mapper à la fonction
        inference=inference_methods_dict[inference_method]  # Mapper à la fonction
    )

    solved_grid = np.zeros((9, 9), dtype=int)

    if solution is None:
        print("Failed to solve the Sudoku puzzle")
    
    else:
        # Créer la grille résolue        
        for (i, j), value in solution.items():
            solved_grid[i, j] = value

    return solved_grid

# Block de code à fins de débogage
if __name__ == "__main__":
    # Tableau NumPy représentant une grille de Sudoku
    instance = np.array([
    [4, 0, 0, 0, 0, 0, 8, 0, 5],
    [0, 3, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 7, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 0, 0, 6, 0],
    [0, 0, 0, 0, 8, 0, 4, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 6, 0, 3, 0, 7, 0],
    [5, 0, 0, 2, 0, 0, 0, 0, 0],
    [1, 0, 4, 0, 0, 0, 0, 0, 0]
    ])
    variable_heuristic = "mrv"
    value_order = "lcv"
    inference_method = "mac"


# Appel de la fonction avec les arguments passés depuis C#
solved_grid = solve_sudoku_csp(instance, variable_heuristic, value_order, inference_method)

# Afficher la grille résolue
print("\nGrille résolue:")
print(solved_grid)
