
---

# lab2
- Що таке **СЛАР**;
  - Система лінійних алгебраічних рівнянь
- Яке завдання 2ї лабораторної роботи?
  - Для (симетричної за моїм варіантом) матриці вирішити СЛАР методом квадратичного кореня.
- Як дізнатись, що матриця симетрична:
   - Матриця - ця ж транспонована матриця повинно дорівнювати 0.
- Що таке **факторизація**
  - **Розкладання**. 
- **Метод квадратного кореня** - окремий випадок розкладання матриці на (LU) нижню та верхню трикутну для додатньо визначених матриць. 
- **Додатньо визначена матриця** - симетрична матриця для якої власні значення > 0;
- **Власні значення** це корені характеристичного рівняння, тобто значння  при яких A - eigenvals * np.eye() - I  
- **Власні вектори** - це вектори, при множенні яких на матрицю А - не міняють свій напрямок (лише довжину).
- Для яких **матриць** працює метод **Холецького**?
   - Симетричних та додатньо-визначених (власні значення > 0)
- Покзати у коді де кожен крок **знаходження** елементів **матриць** **множників**.
```python
if i == j: # діагональні елементи  
    sum = np.sum([u[k, i]**2 for k in range(i)]) 
    u[i, i] = np.sqrt(a[i, i] - sum) 
# діагональний елемент - квадратний корінь різниці між елементом матриці та сумою попередніх елементів того ж стовпця
else: # Верхня трикнутна матриця 
    sum = np.sum([u[k, i] * u[k, j] for k in range(i)])
    u[i, j] = (a[i, j] - sum) / u[i, i]
# Віднімаємо попередні елементи щоб виокремити i, j, прибираючи попередні елементи 
```
- Чому необхідний **прямий** та **зворотній** **ходи**?  
   - Зворотній хід потрібен аби розв'язати простіші системи Uy = b; Ux=y; Прямий - розкладає матрицю А.
- Показати у коді де знахдяться **y** та **x**; що які елементи задіюються.
```python
# T'y = b
y = np.zeros_like(b)
for i in range(n):
    sum = np.sum([u[k, i] * y[k] for k in range(i)]) 
    y[i] = (b[i] - sum) / u[i, i] 
# знаходимо кожен y віднімаючи суму добутку вже відомих множників ділячи на діагональний елемент (коофіцієнт y)
``` 
```python
x = np.zeros_like(b)
for i in range(n-1, -1, -1):
    sum = np.sum([u[i, k] * x[k] for k in range(i+1, n)])       
    x[i] = (y[i] - sum) / u[i, i]
# знаходимо кожен х віднімаючи суму добутку вже відомих множників ділячи на діагональний елемент (коофіцієнт х)
```
Це має сенс оскільки U та U.T це множники матриці А. 
- Що таке **вектор** **нев'язки**?
   - Різниця між правою частиною рівнянь (вектор б) та добутком матриці на отриманий розв'язок. ця різниця повина дорівнювати 0. 
- Система розв'язується у **два** **етапи**: спочатку U^T * y = b, потім U * x = y. Чому такий підхід ефективніший за пряме розв'язання системи? 
   - Розв'язок трикутних систем - послідовне перемноження скалярів. Пряме розв'язання потребувало б множення матриць, що має складність O(n^2). 

- Чому **розклад** Холецького **існує** тільки для **симетричних** **додатно** **визначених** **матриць**? 
    - Симетричність гарантує що елементи матриці зійдуться при множенні розкладених матриць (U, U.T). Всі діагональні елементи будуть додатніми ( не буде ділення на 0 ).
- Чому **діагональні** **елементи** u[i,i] обчислюються через **квадратний** **корінь** і які проблеми це може створювати?
    - При множенні розкладених матриць (U на U.T) діагональні елементи утворюються як суми квадратів елементів відповідного рядка. Тому небхідно брати корінт. (наприклад, a22 = l21² + l22²) 
- Як пов'язані **визначник** **матриці** A та **добуток** **діагональних** елементів матриці U?
    - Визначеник дорівнює добутку діагональних елементів матриці U.
- Чому у[i,j] = 0 для i < j (верхня трикутність) є оптимальним з точки зору обчислень?
     - Бо матриця симетрична, Нижня трикутна матриця є транспонованою верхньою 
- Чому розклад Холецького має вигляд **A = U^T * U**, а не просто** A = U * U**?
    - Множення U.T * U гарантує симетричність матриці. 
- Як пов'язані визначник матриці A і елементи матриці U?
    - Мтриця А додатньо визначена лише коли власні значення додатні, тобто є умовою для розкладу.

---
# lab3
- В чому полягає завдання лабораторної робти;
     - Вирішення СЛАР ітераційними методами якобі та зейделя.
     - Для мого варіанту необхідно було звести матрицю до такої, що має діагональну перевагу. 
     - Діагональна перевага визначається тим, що діагональний елемент більший за суму недіагональних елементів того ж рядку. 
     - Ітераційним методами є такі, що отримають розв'язок системи поступово лише із заданою точністю.

- Метод Якобі полягає у перетворенні вихідної системи лінійних рівнянь Ax=b до еквівалентного вигляду x=Bx+c шляхом одночасного обчислення наближень коренів за значеннями попередньої ітерації.
- Ітераційна матриця визначає як поточні корені впливають на наступні наближення. (B) Містить всі змінні частини системи.  
- Ітераційний вектор - (С) константна частина. 
- у чому полягає метод **зейделя?**
    - Модифікаці методу якобі де одразу використовуютьс нові значення для обчислення наступних наближенб.
```python
# Кожен наступний корень уточнюється попереднім? 
for i in range(n):
# сума вже обчислених компонент
 sum1 = sum(a[i,j] * x_new[j] for j in range(i))
# сума ще не обчислених
sum2 = sum(a[i,j] * x[j] for j in range(i+1, n))
# ділення виражає х з рівняння. (Bx+C)
x_new[i] = (b[i] - sum1 - sum2) / a[i,i]
```
- Що таке матриця С? 
  - Матриця С містить 1/aii, щоб одним множенням поділити всі рівняння на їх діагональні елементи і виразити кожне x.
- Чому необхідна діагональна перевага для методу простої ітерації (якобі)? 
    - діагональна перевага забезпечує збіжність процесу, тому що всі коофіцієнти B? будуть менші за одиницю, що не дозволяє рішеню розхитуватись
-  Який критерій закінчення ітераційного процесу;
    - Різниця між наближеннями менше заданої точності 
- Чому діагональне переважання гарантує збіжність?
    Коли ділимо рівняння на діагональний елемент (який найбільший), всі коефіцієнти ітераційної матриці B стають за модулем менше 1, через що кожна наступна ітерація дає менші зміни ніж попередні, тобто процес "затухає". У інакшому випадку процкес би розгойдувався. 
- Чому метод Зейделя збігається швидше за Якобі?
    - Метод якобі чекає ітерації щоб використати нові значення, коли метод зейделя робить це одразу. 
- Чому розбиття на L, D, U працює?
  - Це розклад матриці на верхню трикутну, діагональну та нижню трикутну матриці. 
- Чому використання нових значень прискорює збіжність?
    -   Нові значення точніші за попередні наближення; 
    -   За одну ітерацію якобі використовує лише старі наближення, у той час як зейделя і нові, і старі. 
- Який метод стійкіший до похибок округлення?
   - Метод якобі, значення х обчислюються незалежно один від одного. Тобто похибкба не накопичується в межах однієї ітерації. 
- Чому порядок обходу змінних важливий?
  - збіжність методу прискорюється послідовним обчисленням через матрицю коофіцієнтів (C). Інший порядок погіршить збіжність або унеможливить через накопичення похибки. 
- Що таке критерій збіжності?
  - Критерій що показує чи метод може збігтись, для якобі/зейделя - матриця повинна бути діагонально переважною.
- Як перевірити точність розв'язку?
  - Узявши різницю розв'зку помноженого на початкову матрицю з шуканим b (вектор нев'язки)
- Чому у методі Якобі використовується матричний підхід (iteration_matrix), а в методі Зейделя - поелементний?
    - Метод зейделя обраховує нові наближення послідовно, аби зменшити кількість ітерацій, тоді як у методі якобі наближення обраховуються незалежно кожноого циклу.
Як обчислюється матриця ітерацій у методі Якобі і чому вона має таку форму?
-  Як впливає перестановка рівнянь на збіжність методів?
   - Ніяк

# lab4


Що таке форма Фробеніуса (Frobenius)?
    - Канонічна форма матриці що зберігає її характеристичний многочлен у останньому її рядку.
Що таке власні значення (eigenvalues)?
    - Власні значення це корені характеристичного рівняння, тобто значння  при яких A - eigenvals * np.eye() - I  
    - Власні вектори - це вектори, при множенні яких на матрицю А - не міняють свій напрямок (лише довжину).
- Характеристичне рівняння - рівняння, корені якого є власними значеннями. `det(A - λ * I) = 0`


Матриця суміжності S - накопичує загальне перетворення

```python
# -mat_a_i[idx,j] / mat_a_i[idx, idx-1]  # Коефіцієнти для обнулення елементів
Обчислює число, на яке треба помножити рядок щоб при відніманні отримати нуль під головною діагоналлю

# 1 / mat_a_i[idx, idx-1]   
mat_a_i = np.linalg.inv(mat_m) @ mat_a_i @ mat_m  # Власне перетворення подібності
Нормалізуючий коефіцієнт; Робить елемент піддіагоналі рівним 1

#  mat_a_i = np.linalg.inv(mat_m) @ mat_a_i @ mat_m
# M⁻¹AM - перетворення подібності
# M⁻¹ (зліва) - обнуляє елементи під діагоналлю
# M (справа) - зберігає структуру матриці в новому базисі

Перетворення подібності - це представлення тієї ж лінійної трансформації в іншому базисі, яке зберігає власні значення та змінює власні вектори множенням на матрицю переходу.
```

- Яка математична суть цього перетворення?
   - Перетворення подібності - це операція виду B = M⁻¹AM, де:
   - A - початкова матриця
   - M - невироджена матриця перетворення
   - B - результуюча матриця
   - M⁻¹ - обернена матриця до M
   - Зберігає власні значення матриці
   - Змінює власні вектори (якщо v - власний вектор A, то Mv буде власним вектором B)
   - Зберігає слід матриці (сума діагональних елементів)
   - Зберігає визначник
```
mat_y = np.array([
    [l**3, l**2, l, 1] for l in lambda_values
])
```

- Ці степені з'являються природно із форми Фробеніуса, бо кожен наступний рядок у власному векторі множиться на матрицю зсуву, що дає степінь λ на один менше ніж у попередньому рядку, починаючи з λⁿ⁻¹

- Якщо матриця має кратні власні значення, як це вплине на процес знаходження власних векторів?
    - Якщо матриця має кратні (повторювані) власні значення, для знаходження повного набору власних векторів, що відповідають кожному власному значенню, необхідно застосовувати метод узагальнених власних векторів, який полягає в розв'язанні системи рівнянь (A - λI)^j x = 0, де j = 1, 2, ..., k, і k - кратність власного значення λ.

- При обчисленні Y-матриці власних векторів, чому важливо використовувати саме таку степеневу послідовність?

     - Ця степенева послідовність виникає з самої структури форми Фробеніуса - матриця-компаньйон просто зсуває елементи вниз при множенні на вектор, і останній рядок використовує коефіцієнти характеристичного многочлена, тому коли ми множимо таку матрицю на власний вектор, кожен елемент автоматично стає степенем власного значення на один менше за попередній.

- Чому метод Данилевського може бути чисельно нестійким?
     - Який критерій можна використовувати для оцінки точності отриманих результатів?

- Чому власні вектори, що відповідають різним власним значенням, лінійно незалежні?
     - Якщо два власних вектори відповідають різним власним значенням, вони мусять бути лінійно незалежними, бо інакше один вектор при множенні на матрицю давав би одразу два різних результати, що неможливо.


- Як нормалізація власних векторів впливає на результат?
    - Нормалізація власних векторів (ділення на їх довжину) не змінює їхні напрямки та властивості як власних векторів, а лише приводить їх до однакової довжини 1, що спрощує порівняння та подальші обчислення.

- Чому такий порядок операцій?
        ```
        mat_a_i = np.linalg.inv(mat_m) @ mat_a_i @ mat_m
        Яка математична суть цього перетворення подібності?
        Чому важливий порядок множення матриць?
        ```
     - При перетворенні подібності M⁻¹AM порядок множення критичний, бо:
        1. M⁻¹ зліва - переводить рядки в новий базис (обнуляє елементи) 
        2. M справа - переводить стовпці назад (зберігає структуру)

        Якщо змінити порядок, отримаємо інше перетворення, яке не дасть потрібну форму Фробеніуса.


- Як пов'язані власні значення матриці з її 
діагоналізацією?
     - Матрицю можна діагоналізувати (представити як P⁻¹AP = D, де D - діагональна) тоді і тільки тоді, коли вона має повний набір лінійно незалежних власних векторів, при цьому власні значення з'являться на діагоналі матриці D, а стовпці P будуть відповідними власними векторами.
- Які є альтернативні методи знаходження власних значень?
    - QR-алгоритм (найпоширеніший в практиці)
    - Степеневий метод (для найбільшого власного значення)
-   Як власні значення пов'язані зі стійкістю динамічних систем?
    - Система стійка, якщо всі власні значення мають від'ємну дійсну частину
    - Чим далі власні значення від уявної осі, тим стійкіша система
-  Які властивості повинна мати результуюча матриця?
    - Одиниці на піддіагоналі
    - Нулі під піддіагоналлю
    - Коефіцієнти характеристичного многочлена в останньому рядку
    - Ті ж самі власні значення, що й вихідна матриця
-   Чому степені йдуть від найвищої до 1?
-   Як це пов'язано з характеристичним поліномом?
     - Степені йдуть від найвищої (λⁿ⁻¹) до 1 тому що це відповідає структурі характеристичного полінома det(A - λI) = λⁿ + aₙ₋₁λⁿ⁻¹ + ... + a₁λ + a₀. У формі Фробеніуса останній рядок містить коефіцієнти цього полінома, і коли ми множимо матрицю на власний вектор з такими степенями, отримуємо рівняння характеристичного полінома, корені якого є власними значеннями.

-  Що таке матриця Ax, Mx, матриця суміжності
    - Матриця А - початкова матриця, для якої шукаємо власні значення і вектори

    - Матриця М (Mi) - матриця перетворення на кожному кроці алгоритму для приведення до форми Фробеніуса:
    - Створює одиниці на піддіагоналі
    - Обнуляє елементи під піддіагоналлю

    - Матриця суміжності S - накопичує всі перетворення:
    - S = M₁ * M₂ * ... * Mₙ 
    - Використовується для знаходження власних векторів початкової матриці
    - Зберігає інформацію про всі виконані перетворення подібності
У чому полягає метод данилевського;
-  Що таке характеристичне рівняння
    Характеристичне рівняння det(A - λI) = 0:
    - отримується з визначника матриці (A - λI)
    - де λ - невідома (власні значення)
    - I - одинична матриця
    - розв'язки (корені) цього рівняння - власні значення матриці А
    
    У формі Фробеніуса коефіцієнти характеристичного рівняння з'являються в останньому рядку матриці, що спрощує його знаходження: λⁿ + aₙ₋₁λⁿ⁻¹ + ... + a₁λ + a₀ = 0
-  Чому метод Данилевського працює?
    - Послідовні перетворення подібності зберігають власні значення
    - Приводить матрицю до форми, де характеристичний поліном легко зчитується

# lab5

Про що лабраторна робота; 

- У чому поглягає теорема ш турма; # метод поліномів штурма 
    Кількість дійсних коренів P(x) на інтервалі [a,b] дорівнює різниці між кількістю змін знаків у послідовності {P₀(a), P₁(a), ..., Pₘ(a)} та кількістю змін знаків у послідовності {P₀(b), P₁(b), ..., Pₘ(b)}.
    Властивості послідовності Штурма:

    Сусідні члени послідовності не можуть мати спільних коренів
    Якщо Pᵢ(c) = 0 для деякого i > 0, то Pᵢ₋₁(c) та Pᵢ₊₁(c) мають протилежні знаки

- Правило Гюа:
     Верхня межа додатних коренів многочлена дорівнює 1 плюс найбільше з відношень модулів від'ємних коефіцієнтів до коефіцієнта при старшому члені. Для від'ємних коренів - робимо заміну x на -x і беремо результат з протилежним знаком.

- Послідовність Штурма:
    Послідовність многочленів {f₀, f₁, f₂, ...}, де f₀ - вихідний многочлен, f₁ - його похідна, кожен наступний - остача від ділення попередніх з протилежним знаком.

- Правило гюа
  - Верхня межа додатних коренів = 1 + max(|від'ємні коефіцієнти|/|старший коефіцієнт|)і

- метод бісекції у двох словах
    Ділимо відрізок навпіл і відкидаємо ту частину, де функція має однакові знаки на кінцях.

- метод хорд у двох словах
  - З'єднуємо точки графіка прямою лінією і беремо точку перетину з віссю X як наближення до кореня.
  
- метод ньютона у двох словах
    - Проводимо дотичну до графіка функції і беремо точку її перетину з віссю X як наближення (xᵢ₊₁ = xᵢ - f(xᵢ)/f'(xᵢ)).

- Чому множимо для перевірки проміжку 
    - Множення значень функції (f(c) * f(a) < 0 або f(c) * f(b) > 0) використовується для перевірки знаків функції на кінцях відрізка.
    Це базується на теоремі Больцано-Коші: якщо неперервна функція має різні знаки на кінцях відрізка, то всередині відрізка є корінь.
    Розглянемо детальніше:

    Якщо f(c) * f(a) < 0 - значить значення мають різні знаки (один "+", інший "-"), тому корінь між ними
    Якщо f(c) * f(a) > 0 - значить значення мають однакові знаки, корінь в іншій частині


- який метод приводить до меншої кількості ітерацій і чим це зумовлено.
  - Метод Ньютона має найшвидшу (квадратичну) збіжність, бо використовує похідну функції, що дає більше інформації про її поведінку. В той час як метод бісекції просто ділить відрізок навпіл (лінійна збіжність), а метод хорд використовує лише значення функції на кінцях відрізка (збіжність порядку ≈1.618).


- Чому метод бісекції завжди збігається, але повільно?
    Причина гарантованої збіжності: через теорему Больцано-Коші - якщо функція неперервна і має різні знаки на кінцях відрізка, то при діленні навпіл ми завжди зберігаємо відрізок з коренем всередині. Причина повільної збіжності: довжина відрізка зменшується лише вдвічі на кожному кроці, незалежно від форми функції чи близькості до кореня.

як оцінити кількість ітерацій для досягнення заданої точності?

- Чому важливо, щоб функція змінювала знак на кінцях відрізка?
    -  Це гарантує наявність кореня на відрізку (за теоремою Больцано-Коші) - якщо функція неперервна і змінює знак, вона обов'язково перетне вісь X.
 


- Які недоліки методу при пошуку кратних коренів?
    -  При кратних коренях функція не змінює знак (тільки дотикається до осі X), тому методи, що базуються на зміні знаку (бісекція, хорди), можуть не працювати.

- Чому швидкість збіжності методу хорд лінійна?
    Швидкість лінійна бо використовуємо лише значення функції (без похідної), і наближення залежить від форми кривої - січна не враховує як швидко змінюється функція біля кореня.


- Як геометрично інтерпретується метод хорд?
    - Проводимо пряму (хорду) через точки (a, f(a)) і (b, f(b)), і беремо точку перетину цієї прямої з віссю OX як наступне наближення. Потім повторюємо процес, замінюючи один з кінців відрізка на знайдену точку (залежно від знаку функції).

- Чому важливо правильно вибирати початкові наближення
    При поганому виборі початкової точки метод може розбігатися (якщо дотична майже паралельна осі OX) або "стрибати" між різними коренями. Правильний вибір початкової точки (де графік функції "добре поводиться" і знак другої похідної не змінюється) забезпечує швидку збіжність до потрібного кореня.

- Чому збіжність квадратична при простих коренях?
    При простому корені похідна не дорівнює нулю, і дотична добре апроксимує функцію - помилка наближення пропорційна квадрату відстані до кореня (x_n+1 - r ≈ C(x_n - r)²), тому на кожній ітерації кількість вірних знаків подвоюється.

- Як теорема про проміжне значення пов'язана з методом бісекції?
    теорема гарантує існування кореня, якщо функція неперервна і змінює знак - це основа роботи методу бісекції.

- Як теорема про нерухому точку пов'язана з ітераційними методами?
    Ітераційні методи перетворюють рівняння f(x)=0 в x=φ(x), шукаючи нерухому точку відображення φ.

- Чому саме лінійна апроксимація використовується в методі хорд?
    Найпростіша апроксимація, що зберігає значення функції в двох точках.

- Як ряд Тейлора пов'язаний з методом Ньютона?
        Метод Ньютона використовує лінійний член ряду Тейлора для апроксимації функції.

- Як визначити, що метод розбігається?
    Якщо послідовність наближень не зменшує похибку або відхиляється від області пошуку.

- Що робити, якщо функція не диференційована?
    Використовуються методи, що не потребують похідної (бісекція, хорди).

- Недиференційована функція — це функція, яка не має похідної в певній точці або на деякому проміжку.

- Чому саме дотична використовується в методі Ньютона, а не інші апроксимації?
    Дотична дає найкращу лінійну апроксимацію функції у точці (це випливає з означення похідної)
    Вона враховує і значення функції, і швидкість її зміни (похідну), що забезпечує квадратичну збіжність біля простого кореня. Інші лінійні апроксимації (як у методі хорд) дають повільнішу збіжність, бо не використовують інформацію про похідну.




# lab7
- що таке квадратура; - це метод наближеного обчислення визначених інтегралів шляхом заміни складної функції простішою.

що таке поліноми лежандра ;  - спеціальні ортогональні поліноми на [-1,1], де кожен наступний має вищий степінь за попередній.

Інтерполяційні поліноми у двох словах; - це поліноми, що проходять через задані точки функції для її наближення.

квадратура  трапецій у двох словах;   - метод наближення інтеграла сумою площ трапецій, де криву замінюють ламаною.

з чого складаєься оцінка похибки формули трапецій - Оцінка похибки трапецій включає крок інтегрування в квадраті, довжину відрізка, максимум модуля другої похідної та коефіцієнт 1/12.

що таке ортогональність  ( зокрема у поліномах лежандра ) - Ортогональність у поліномах Лежандра означає, що інтеграл добутку будь-яких двох різних поліномів на [-1,1] дорівнює нулю.


ДЕ У КВАДРАТУРІ ГАУСА ЗАМІНА ЗМІННОЇ   x = (b + a) / 2 z*(b-a)/2
    - Заміна x = ((b-a)/2)z + (b+a)/2 в квадратурі Гауса переводить довільний інтервал [a,b] в стандартний [-1,1], де визначені поліноми Лежандра.

з чого складається похибка квадратури гауса
    - Похибка Гауса включає факторіали m і 2m, довжину інтервалу в степені (2m+1), максимум модуля 2m-ї похідної та знаменник (2m+1).

- чому квадратура гауса ефективна до 2m-1 та до чого взагалі похідна в похибці 
    При m точках квадратура точна для поліномів степеня 2m-1, а похідна порядку 2m з'являється як перший член, що не інтегрується точно.

- що таке вагові коофіцієнти у квадратурі гауса 
    Вагові коефіцієнти - це множники при значеннях функції в вузлах квадратури, які обчислюються з умови точності для поліномів.

- Чому саме поліноми Лежандра використовуються в квадратурі Гауса, а не інші ортогональні поліноми? Які переваги це дає?
    Поліноми Лежандра використовуються бо вони оптимальні для рівномірної ваги на [-1,1] і дають найменшу похибку серед усіх квадратур з тим же числом вузлів.

- Чому квадратура Гауса точна для поліномів степеня до 2m-1, а не просто m?
    Точність до степеня 2m-1 досягається завдяки m умовам від вузлів та m умовам від оптимального вибору цих вузлів.

Як співвідносяться локальна і глобальна похибки в методі трапецій?
    локальна похибка виникає на кожному окремому відрізку розбиття і пропорційна h³ та значенню другої похідної.
    Глобальна похибка є сумою всіх локальних похибок і пропорційна h², оскільки кількість відрізків пропорційна 1/h.
    При зменшенні кроку h локальні похибки зменшуються швидше, ніж зростає їх кількість.

Чому в оцінці похибки квадратури Гауса з'являється факторіал (2m)! і як це пов'язано з рядом Тейлора?
    Факторіал (2m)! з'являється через розклад функції в ряд Тейлора до члена з похідною порядку 2m.
    Цей член відповідає першому "неточному" доданку в квадратурі Гауса.
    Величина факторіалу показує, наскільки швидко спадають коефіцієнти в розкладі Тейлора гладкої функції.

Як впливає погана обумовленість на точність обчислень при великих значеннях m?
    При збільшенні m зростає число обумовленості матриці системи для вагових коефіцієнтів.
    Це призводить до накопичення помилок округлення при обчисленні вузлів і ваг.
    В результаті реальна точність може бути гіршою за теоретичну при великих m.


Як вибрати оптимальне значення m для квадратури Гауса з урахуванням і точності, і обчислювальної складності?
    Треба враховувати баланс між бажаною точністю та обчислювальними витратами.
    При збільшенні m швидко зростає складність обчислення вузлів і ваг.
    Практично часто достатньо m ≤ 20 для досягнення високої точності.


У яких випадках метод трапецій може бути кращим вибором, ніж квадратура Гауса?
    Простота реалізації та модифікації методу трапецій.
    Краща робота з негладкими функціями чи функціями з особливостями.
    Легше оцінювати похибку і контролювати точність.
    Простіше адаптивне уточнення в областях з великою похибкою.

- Як би змінилась ефективність методів при інтегруванні осцилюючих функцій?
    Метод трапецій потребує значно більше вузлів для досягнення тієї ж точності через часті зміни знаку функції.
    Квадратура Гауса краще справляється з осциляціями, але її ефективність також падає.

- Як можна адаптивно вибирати вузли інтегрування для покращення точності?
    Можна розбивати інтервал на підінтервали, де похибка перевищує допустиму.
    В областях з більшою кривизною або осциляціями варто використовувати більше вузлів.
    Для методу трапецій це простіше реалізувати, ніж для квадратури Гауса.

- Чи можливо оцінити похибку без використання похідних високих порядків?
    Можна використовувати порівняння результатів при різній кількості вузлів.
    Правило Рунге дозволяє оцінити похибку через порівняння двох наближень.
    Можливе використання статистичних методів оцінки похибки.


Як би змінився алгоритм для обчислення кратних інтегралів?
    Необхідне застосування методу послідовно по кожній змінній.
    Кількість точок зростає експоненційно з розмірністю.
    Потрібні спеціальні стратегії вибору вузлів для зменшення обчислювальної складності.

Яка асимптотична складність обчислення вузлів і вагових коефіцієнтів у квадратурі Гауса?
    Пошук коренів полінома Лежандра має складність O(m²).
    Обчислення вагових коефіцієнтів також потребує O(m²) операцій.
    Загальна складність підготовки квадратури - O(m²).

Як стабільність алгоритму залежить від вибору точок інтегрування?
    Рівномірний розподіл точок (як у методі трапецій) дає передбачувану поведінку похибки, але може бути неоптимальним.

Які існують методи прискорення збіжності для погано обумовлених інтегралів?
    Вузли Гауса оптимізовані для мінімізації похибки, але можуть призводити до нестабільності при великих m через погану обумовленість.

Як теорема про середнє значення інтеграла пов'язана з вибором вузлів у квадратурних формулах?
    Можна використовувати заміну змінних для згладжування особливостей функції.
    Екстраполяція Річардсона дозволяє підвищити порядок точності.
    Адаптивне розбиття області інтегрування з урахуванням поведінки функції покращує збіжність.

Чому саме ортогональні поліноми дають оптимальні точки для інтегрування?
    Ортогональність забезпечує максимальну лінійну незалежність базисних функцій.
    Корені ортогональних поліномів розташовані оптимально для мінімізації похибки інтегрування.
    Властивості ортогональності дозволяють досягти точності для поліномів максимально можливого степеня.

Як пов'язані квадратурні формули з інтерполяцією Лагранжа?
    Квадратурні формули можна розглядати як інтеграл від інтерполяційного полінома.
    Вагові коефіцієнти пов'язані з інтегралами від базисних поліномів Лагранжа.
    Точність квадратури обмежена точністю відповідної інтерполяції.

Які переваги та недоліки методу Гауса порівняно з іншими методами чисельного інтегрування?
    Гаус дає найвищу точність при фіксованій кількості вузлів.
    Складніша реалізація і обчислення вузлів та ваг.
    Менш гнучкий для адаптивного уточнення порівняно з простішими методами.

Які модифікації потрібні для стохастичних інтегралів?
    Потрібно враховувати випадковий характер підінтегральної функції.
    Можливе використання методів Монте-Карло з відповідними вагами.
    Необхідні спеціальні методи оцінки похибки для випадкових величин.

# lab8

що таке проблема коші -  це задача пошуку розв'язку диференціального рівняння при заданих початкових умовах. Тобто маємо рівняння y' = f(x,y) і початкову умову y(x₀) = y₀, потрібно знайти функцію y(x).

метод ейлера - найпростіший чисельний метод розв'язку ЗДР, де наступне значення обчислюється як y[n+1] = y[n] + h*f(x[n],y[n]). По суті, це лінійна апроксимація функції в точці.

реверснутий метод ейлера  - а використовує значення похідної в наступній точці: y[n+1] = y[n] + h*f(x[n+1],y[n+1]). Це неявний метод, стійкіший за звичайний метод Ейлера для жорстких систем.
Коефіцієнти в методі Ейлера отримуються з розкладу Тейлора:

```
у методі ейлера лінійна апроксимація точки 
 y[n+1] = y[n] + h*f(x[n],y[n])
точка * значення функції 

це призводило до нестабільності розв'язку - рішення "розхитувалось"
реверснутий метод ейлера  - а використовує значення похідної в наступній точці 

це у свою чергу вже призводило до затухання, тобто рішення втрачало енергію, менше осцилювало. 
 y[n+1] = y[n] + h*f(x[n+1],y[n+1]).

І Ідея рунге-кутта 2? порядку полягала у тому, щоб узяти 
ейлера, реверсивного ейлера та поєднати їх, аби "затухання" та "розхитування" системи компенсували одне одного, усуваючи нестабільність та уточнюючи розв'язок 

ідея рунге кутта 3го порядку у тому, щоб між ними порахувати ще одну точку

а 4го порядку, як у лабораторній, у тому, щоб брати дві проміжні точки, оскільки вони мають у двічі більшу точність ( у двічі менший степсайз ) ніж у у коофіцієнту k1 та k4

 
        k1 = dt * f(t,          x)
        k2 = dt * f(t + dt / 2, x + k1 / 2)
        k3 = dt * f(t + dt / 2, x + k2 / 2)
        k4 = dt * f(t + dt,     x + k3)
        cur_step_size_x = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
ось з формули рк4 ти бачиш схожість до методів ейлера та зважену суму усіх коофіцієнтів
усі з цих коофіцієнтів відповідна похідна 
(згадуємо як виводити таблицю похідних а саме (f(x) + f(x+h))/h, відповідно h у нас 1, а значення перемножується на степсайз, тобто отримуємо проміжні степи
```

чому похибка tau має сенс, з чого вона складається

чим відрізняється адаптивний rk4 від фіксованого - 
Як вирігується розмір кроку, що таке правило рунге 

чому методу адамса потрібно три точки на початку - а потрібно три точки на початку, бо він використовує інтерполяційний поліном для прогнозування наступного значення. Більше точок дає точнішу інтерполяцію.

Інтерполяці - наближення функції між відомими точками. Екстраполяція - прогнозування поза відомими точками.

```
коофіцієнти узяті з інтерполяційного поліному
y_pred = y[i] + h/24 * (55*f(x[i], y[i]) - 59*f(x[i-1], y[i-1]) + 
                        37*f(x[i-2], y[i-2]) - 9*f(x[i-3], y[i-3]))
                        
```

- який order методу адамса та методу rk4 та чому .
    4 похідні - 4й порядок 

- як обраховується аналітична похибка методу адамса та чому саме так (чому рунге правило)
    у рамках лабораторної роботи замість того, аби використовувати формули оцінки похибки методу адамса було використано метод рунге оскільки він простіший, який полягає у тому, щоб оцінити різницю між поточним кроком та у двічі меншим кроком. завеликий результат > кількох сотих означає різку змінну функції (k2 та k3) дві послідовні похідні використовуються у методі рунге 


що таке фазовий портрет та що він показує 
    - це графічне представлення траєкторій системи диференціальних рівнянь у фазовому просторі. Він показує якісну поведінку системи: стійкі/нестійкі точки, граничні цикли, атрактори тощо.


Що означає A-стійкість та L-стійкість методу? Який з представлених методів має кращу стійкість для жорстких систем?
        A-стійкість означає, що похибка не росте необмежено при будь-якому кроці для лінійного рівняння y' = λy при Re(λ) < 0. L-стійкість додатково вимагає загасання розв'язку при λh → -∞. З представлених методів найкращу стійкість для жорстких систем має неявний метод Ейлера.


Який математичний сенс має tau_upper та tau_lower в адаптивному методі? Як правильно підбирати ці параметри?
    au - це оцінка локальної похибки методу як відношення різниці послідовних наближень. tau_upper (≈0.01) - максимально допустима похибка, при перевищенні якої крок зменшується. tau_lower (≈0.0001) - мінімальна похибка, нижче якої крок можна збільшити для ефективності.

Як впливає нелінійність диференційного рівняння на вибір оптимального кроку в адаптивному методі?
    Чим сильніша нелінійність, тим менший крок потрібен для забезпечення точності та стійкості. В областях сильної нелінійності адаптивний метод автоматично зменшує крок, реагуючи на зростання локальної похибки.


Як можна оцінити оптимальний порядок методу залежно від потрібної точності та властивостей диференційного рівняння?
    Вибір порядку методу залежить від потрібної точності та гладкості розв'язку. Для точності 10⁻³ достатньо методів 2-3 порядку, для 10⁻⁶ потрібні методи 4-5 порядку. Вищий порядок вимагає більше обчислень на крок, але дозволяє робити більші кроки.