import numpy as np

def create_matrix_random():
    rows = np.random.randint(2, 6)
    cols = np.random.randint(2, 6)
    matrix = np.random.randint(-10, 10, size=(rows, cols))
    print(f"\nВипадково створена матриця ({rows}x{cols}):\n", matrix)
    return matrix


def create_matrix_random_with_size():
    rows = int(input("Введіть кількість рядків: "))
    cols = int(input("Введіть кількість стовпців: "))
    matrix = np.random.randint(-10, 10, size=(rows, cols))
    print(f"\nВипадкова матриця ({rows}x{cols}):\n", matrix)
    return matrix


def create_matrix_manual():
    rows = int(input("Введіть кількість рядків: "))
    cols = int(input("Введіть кількість стовпців: "))
    print("Введіть елементи матриці по рядках:")
    elements = []
    for i in range(rows):
        row = list(map(float, input(f"Рядок {i+1}: ").split()))
        while len(row) != cols:
            print(f"❗️Кількість елементів має бути {cols}")
            row = list(map(float, input(f"Рядок {i+1}: ").split()))
        elements.append(row)
    matrix = np.array(elements)
    print(f"\nВведена матриця:\n", matrix)
    return matrix


def get_basis(A):
    U, S, Vt = np.linalg.svd(A)
    rank = np.linalg.matrix_rank(A)
    basis = Vt[:rank].T  
    print("\nБазис матриці (по стовпцях):\n", np.round(basis, 3))
    return basis


def matrix_operations(A, B=None):
    print("\n=== ОПЕРАЦІЇ НАД МАТРИЦЯМИ ===")

    print("\n Транспонована матриця Aᵀ:\n", A.T)

    print(" Норма матриці ||A|| =", np.linalg.norm(A))

    if A.shape[0] == A.shape[1]:
        det = np.linalg.det(A)
        print("\n Визначник det(A) =", round(det, 3))
        if det != 0:
            inv = np.linalg.inv(A)
            print(" Обернена матриця A⁻¹:\n", np.round(inv, 3))
        else:
            print(" Обернена матриця не існує (det = 0)")
    else:
        print(" Обернена матриця не обчислюється (A не квадратна).")

    dim = min(A.shape)
    I = np.eye(dim)
    print(f"\n Одинична матриця розмірності {dim}x{dim}:\n", I)

    get_basis(A)

    if B is not None:
        print("\n=== ОПЕРАЦІЇ НАД ДВОМА МАТРИЦЯМИ ===")

        if A.shape == B.shape:
            print("\n Сума матриць A + B =\n", A + B)
            print("\n Покомпонентне множення A ⊙ B =\n", A * B)

        if A.shape[1] == B.shape[0]:
            print("\n Матричний добуток A × B =\n", np.dot(A, B))
        else:
            print("\n Добуток A×B неможливий (несумісні розміри).")

    try:
        scalar = float(input("\nВведіть скаляр для множення: "))
        print(f"\n A × {scalar} =\n", A * scalar)
    except ValueError:
        print(" Некоректне значення скаляра — пропускаємо операцію.")


def main():
    print("=== СТВОРЕННЯ МАТРИЦІ ===")
    print("1 — Повністю випадкова матриця")
    print("2 — Випадкова матриця із заданими розмірами")
    print("3 — Ручне введення матриці")
    choice = input("Оберіть спосіб створення матриць (1/2/3): ")

    if choice == "1":
        A = create_matrix_random()
        B = create_matrix_random()
    elif choice == "2":
        A = create_matrix_random_with_size()
        B = create_matrix_random_with_size()
    elif choice == "3":
        A = create_matrix_manual()
        B = create_matrix_manual()
    else:
        print(" Невірний вибір.")
        return

    print("\nМатриця A:\n", A)
    print("Матриця B:\n", B)
    matrix_operations(A, B)


if __name__ == "__main__":
    main()
