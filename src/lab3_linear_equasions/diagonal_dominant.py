import numpy as np

def print_matrix_and_vector(matrix, vector, message="Current state:"):
    print(f"\n{message}")
    print("Matrix A:")
    print(matrix)
    print("\nVector b:")
    print(vector)

def check_diagonal_dominance(matrix):
    n = len(matrix)
    for i in range(n):
        diagonal = abs(matrix[i][i])
        row_sum = sum(abs(matrix[i][j]) for j in range(n) if j != i)
        print(f"\nRow {i+1} check:")
        print(f"Diagonal element: |{diagonal:.3f}|")
        print(f"Sum of other elements: {row_sum:.3f}")
        print(f"Is dominant? {diagonal > row_sum}")
        if diagonal <= row_sum:
            return False
    return True

def make_diagonally_dominant_gauss(matrix_a, vector_b):
    n = len(matrix_a)
    matrix = matrix_a.copy()
    vector = vector_b.copy()
    
    print_matrix_and_vector(matrix, vector, "Starting system:")
    print("\nChecking initial diagonal dominance:")
    initial_dominant = check_diagonal_dominance(matrix)
    
    if not initial_dominant:
        print("\nMatrix is not diagonally dominant. Applying Gaussian elimination...")
        
        # For each row
        for i in range(n):
            # Find the largest absolute value in the current column (pivot)
            pivot_row = i
            max_val = abs(matrix[i][i])
            for k in range(i + 1, n):
                if abs(matrix[k][i]) > max_val:
                    max_val = abs(matrix[k][i])
                    pivot_row = k
            
            # Swap rows if necessary
            if pivot_row != i:
                print(f"\nSwapping row {i+1} with row {pivot_row+1}")
                matrix[i], matrix[pivot_row] = matrix[pivot_row].copy(), matrix[i].copy()
                vector[i], vector[pivot_row] = vector[pivot_row], vector[i]
                print_matrix_and_vector(matrix, vector, "After row swap:")
            
            # Eliminate the largest off-diagonal element in each row
            for j in range(n):
                if i != j and abs(matrix[j][i]) > abs(matrix[j][j])/2:
                    # Calculate multiplier
                    multiplier = matrix[j][i] / matrix[i][i]
                    print(f"\nEliminating element at row {j+1}, column {i+1}")
                    print(f"Using multiplier: {multiplier:.3f}")
                    
                    # Subtract the multiplied row
                    matrix[j] = matrix[j] - multiplier * matrix[i]
                    vector[j] = vector[j] - multiplier * vector[i]
                    
                    print_matrix_and_vector(matrix, vector, "After elimination:")

    print("\nFinal diagonal dominance check:")
    final_dominant = check_diagonal_dominance(matrix)
    
    return matrix, vector

# Initial system
matrix_a = np.array([
    [2.12, 0.42, 1.34, 0.88],
    [0.42, 3.95, 1.87, 0.43],
    [1.34, 1.87, 2.98, 0.46],
    [0.88, 0.43, 0.46, 4.44]
])

vector_b = np.array([11.172, 0.115, 0.009, 9.349])

# Make the matrix diagonally dominant using Gaussian elimination
new_matrix, new_vector = make_diagonally_dominant_gauss(matrix_a, vector_b)

print("\nFinal system:")
print_matrix_and_vector(new_matrix, new_vector, "Final result:")