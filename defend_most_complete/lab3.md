
У цій роботі розглядаються ітераційні методи, які дозволяють отримати розв'язок системи із заданою точністю. Зокрема:

- Метод Якобі (простої ітерації)
- Метод Гаусса-Зейделя

Оскільки матриця не має діагональної переваги, було виконано одну ітерацію методом Гаусса для отримання діагональної переваги.

## Функція для отримання діагональної переваги

## Метод Гаусса-Зейделя
> Основна ідея методу Зейделя полягає в ітеративному покращенні наближення до розв'язку системи лінійних рівнянь шляхом послідовного уточнення кожної компоненти вектора розв'язку. На кожній ітерації метод обчислює нові значення компонент одну за одною, використовуючи при цьому найновіші доступні значення інших компонент. Щойно обчислене нове значення компоненти відразу застосовується для обчислення наступних компонент, не чекаючи завершення повної ітерації. Це дозволяє методу швидше наближатися до точного розв'язку, оскільки кожне нове обчислення базується на найактуальнішій інформації. Процес продовжується до досягнення заданої точності або вичерпання максимальної кількості ітерацій. Метод також включає обчислення вектора нев'язки для оцінки якості поточного наближення


```python
def seidel(A, b, tolerance=1e-6, max_iterations=10000):
    global seidel_b
    n = len(A)
    # Ініціалізація вектора розв'язку нулями
    x = np.zeros_like(b, dtype=np.double)
    converge = False
    
    for k in range(max_iterations):
        if converge:
            break
        print(f"Ітерація №{k+1}")
        x_new = np.copy(x)
        
        for i in range(n):
            # Обчислення суми для вже оновлених компонент
            # Використовуємо нові значення x_new для j < i
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            
            # Обчислення суми для ще не оновлених компонент
            # Використовуємо старі значення x для j > i
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            
            # Оновлення i-ї компоненти вектора розв'язку
            # Ізоляція змінної x[i]:
            # 1. Початкове рівняння: A[i][i] * x[i] + sum(A[i][j] * x[j] для j != i) = b[i]
            # 2. Переносимо всі члени, крім A[i][i] * x[i], в праву частину
            # 3. Ділимо обидві частини на A[i][i], щоб отримати x[i] окремо
            # Ізоляція потрібна для:
            # - Спрощення обчислень: розраховуємо кожну змінну окремо
            # - Застосування ітераційного процесу: використовуємо попередні наближення
            # - Забезпечення збіжності: за певних умов це гарантує збіжність методу
            x_new[i] = (b[i] - s1 - s2) / A[i][i]
        
        print("Наближення")
        print(x_new.reshape((-1, 1)))
        
        print("Вектор нев'язки")
        # Обчислення вектора нев'язки
        # r = b - Ax, де r - вектор нев'язки
        # Показує, наскільки поточне рішення "не задовольняє" систему рівнянь
        vector = b - np.dot(A, x_new.reshape((-1, 1)))
        
        # Перевірка умови збіжності
        # Порівнюємо нове наближення зі старим
        converge = np.allclose(x, x_new, atol=tolerance, rtol=0.)
        print(vector)
        
        seidel_b = vector
        x = x_new
    
    return x.reshape((-1, 1))
```

## Метод Якобі
> Основна ідея методу Якобі полягає в ітеративному розв'язанні системи лінійних рівнянь шляхом послідовного уточнення кожної компоненти вектора розв'язку. На кожній ітерації метод обчислює нові значення всіх компонент одночасно, використовуючи значення з попередньої ітерації. Ключовим аспектом є розділення матриці системи на діагональну та позадіагональну частини, що дозволяє ізолювати кожну змінну і виразити її через інші. Нове наближення обчислюється за формулою, яка враховує вплив недіагональних елементів та ізолює діагональні. Процес повторюється до досягнення заданої точності або вичерпання максимальної кількості ітерацій. На кожному кроці обчислюється вектор нев'язки, який показує, наскільки поточне наближення відрізняється від точного розв'язку. Важливою особливістю методу Якобі є те, що всі компоненти оновлюються одночасно на основі значень з попередньої ітерації, що робить метод придатним для паралельних обчислень. 
```python
def jacobi(A, b, tolerance=1e-6, max_iterations=10000):
    global jacobi_b
    # Ініціалізація вектора розв'язку нулями
    # Це початкове наближення, з якого почнеться ітераційний процес
    x = np.zeros_like(b, dtype=np.double)
    
    # Розділення матриці A на діагональну та позадіагональну частини
    # T містить всі елементи A, крім діагональних
    # Це потрібно для ізоляції змінних у методі Якобі
    T = A - np.diag(np.diagonal(A))
    
    for k in range(max_iterations):
        print(f"Ітерація №{k+1}")
        # Зберігаємо попереднє наближення для перевірки збіжності
        x_old = x
        
        # Обчислення нового наближення
        # Формула: x = D^(-1) * (b - R * x), де D - діагональ A, R - решта A
        # 1. np.dot(T, x) обчислює вплив недіагональних елементів
        # 2. b - np.dot(T, x) ізолює вплив діагональних елементів
        # 3. Ділення на діагональ A (np.diagonal(A)) дає нове наближення
        # Ізоляція потрібна для:
        # - Незалежного обчислення кожної змінної
        # - Можливості паралельних обчислень
        # - Забезпечення збіжності за певних умов
        x = (b - np.dot(T, x)) / np.diagonal(A).reshape((-1, 1))
        
        print("Наближення")
        print(x.reshape((-1, 1)))
        
        print("Вектор нев'язки")
        # Обчислення вектора нев'язки
        # r = b - Ax, де r - вектор нев'язки
        # b - "бажаний результат" системи
        # Ax - фактичний результат при поточному наближенні x
        # Різниця показує, наскільки поточне рішення "не задовольняє" систему
        vector = b - np.dot(A, x.reshape((-1, 1)))
        print(vector)
        
        # Перевірка умови збіжності
        # Порівнюємо нове наближення зі старим
        # Якщо різниця менша за задану точність, вважаємо, що метод зійшовся
        if np.allclose(x_old, x, atol=tolerance, rtol=0.):
            break
        
        jacobi_b = vector
    
    return x
```

## Порівняння методів

> Основна різниця між методом Якобі та методом Зейделя полягає в стратегії оновлення компонент вектора розв'язку під час ітераційного процесу. Метод Якобі оновлює всі компоненти одночасно на кожній ітерації, використовуючи лише значення з попередньої ітерації. Це робить його більш придатним для паралельних обчислень, але може уповільнювати збіжність. Натомість метод Зейделя оновлює компоненти послідовно, використовуючи найновіші доступні значення для обчислення кожної наступної компоненти. Це часто призводить до швидшої збіжності, оскільки кожне нове обчислення базується на найактуальнішій інформації. Метод Зейделя зазвичай збігається швидше, але більш чутливий до порядку рівнянь у системі. Метод Якобі може бути стабільнішим у деяких випадках і легше піддається паралелізації. Вибір між цими методами залежить від конкретної задачі, структури матриці системи та доступних обчислювальних ресурсів. Обидва методи мають свої переваги і застосовуються в різних сценаріях чисельного аналізу та розв'язання систем лінійних рівнянь.

Обидва методи - Якобі та Гаусса-Зейделя - базуються на принципі ізоляції змінних, проте реалізують цей принцип по-різному:

1. **Метод Якобі:**
   - Ізолює всі змінні одночасно
   - Використовує лише дані з попередньої ітерації
   - Дозволяє паралельні обчислення
   - Простіше реалізувати з використанням векторизованих операцій

2. **Метод Гаусса-Зейделя:**
   - Ізолює змінні послідовно
   - Використовує оновлені значення на поточній ітерації
   - Часто збігається швидше
   - Зазвичай реалізується з використанням циклів

Ця відмінність у підході до ізоляції призводить до низки наслідків. По-перше, метод Якобі дозволяє обчислювати всі нові значення паралельно, тоді як у методі Гаусса-Зейделя обчислення нових значень залежить від попередніх, що ускладнює паралелізацію. По-друге, метод Гаусса-Зейделя часто збігається швидше, оскільки використовує найсвіжіші доступні дані.

Незважаючи на ці відмінності, обидва методи ефективно використовують принцип ізоляції змінних для розв'язання систем лінійних рівнянь, кожен зі своїми перевагами в певних ситуаціях.