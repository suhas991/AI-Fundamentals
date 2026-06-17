# OOP Deep Dive: Class & Object — SDE 2 Interview Notes

> **Target:** SDE 2 roles at product-based companies (Google, Amazon, Microsoft, Flipkart, etc.)
> **Language coverage:** Java (primary), C++ (secondary), Python (where relevant)

---

## Table of Contents

1. [Class vs Object — Blueprint vs Instance](#1-class-vs-object)
2. [Constructor Types](#2-constructor-types)
3. [Destructor / Finalize](#3-destructor--finalize)
4. [The `this` Keyword](#4-the-this-keyword)
5. [Static vs Instance Members](#5-static-vs-instance-members)
6. [Object Creation in Memory — Stack vs Heap](#6-object-creation-in-memory)
7. [Tricky & Trap Questions](#7-tricky--trap-questions)

---

## 1. Class vs Object

### The Mental Model

A **class** is a compile-time blueprint — it defines *what* an entity looks like (fields) and *what it can do* (methods). It lives in the code segment of memory.

An **object** is a runtime instance — it consumes actual heap memory and has its own state. Multiple objects can share the same class but hold independent data.

```
Class → Template / Cookie Cutter
Object → Actual Cookie (with its own shape & flavor)
```

### Java Example

```java
// Class = blueprint
class BankAccount {
    String owner;       // instance field
    double balance;

    void deposit(double amount) {
        balance += amount;
    }
}

// Objects = instances
BankAccount acc1 = new BankAccount(); // own copy of owner & balance
BankAccount acc2 = new BankAccount(); // completely independent state

acc1.deposit(1000);  // only acc1.balance changes
```

### Key Distinction Table

| Aspect          | Class                        | Object                          |
|----------------|------------------------------|---------------------------------|
| Existence       | Compile-time concept         | Runtime entity                  |
| Memory          | Code/method area             | Heap memory                     |
| Count           | Defined once                 | Can have millions of instances  |
| State           | Defines fields (no state)    | Holds actual field values       |
| Created by      | Developer writes it          | `new` keyword (or factory)      |

### What actually gets loaded?

```
JVM Memory Layout
─────────────────────────────────────────────
Method Area     │ Class metadata, static fields,
(Metaspace)     │ bytecode (shared across objects)
─────────────────────────────────────────────
Heap            │ acc1 { owner=null, balance=0.0 }
                │ acc2 { owner=null, balance=0.0 }
─────────────────────────────────────────────
Stack           │ Reference variables: acc1, acc2
                │ (just 4/8 byte addresses to heap)
─────────────────────────────────────────────
```

---

## 2. Constructor Types

### What Is a Constructor?

A constructor is a **special method** invoked automatically when an object is created via `new`. It:
- Has the **same name as the class**
- Has **no return type** (not even void)
- Is **not inherited** (but can be chained)
- Runs **before** the object is usable

### 2.1 Default Constructor

A no-argument constructor. If you write no constructor, the compiler inserts one *silently*.

```java
class Dog {
    String name;
    int age;

    // Compiler inserts this if no constructor exists:
    // Dog() { super(); }  ← also calls Object()
}

Dog d = new Dog(); // valid
```

**Critical rule:** If you define ANY parameterized constructor, the compiler NO LONGER inserts the default constructor. You must write it explicitly if you need it.

```java
class Cat {
    String name;

    Cat(String name) {   // parameterized
        this.name = name;
    }
    // NO default Cat() exists anymore!
}

Cat c = new Cat();       // ❌ Compile error — no matching constructor
Cat c = new Cat("Mimi"); // ✅
```

### 2.2 Parameterized Constructor

Accepts arguments to initialize an object with specific state at creation time.

```java
class Rectangle {
    int width, height;

    Rectangle(int width, int height) {
        this.width = width;
        this.height = height;
    }

    int area() { return width * height; }
}

Rectangle r = new Rectangle(10, 5);
System.out.println(r.area()); // 50
```

**Constructor Overloading** — multiple constructors with different signatures:

```java
class Person {
    String name;
    int age;
    String email;

    Person(String name) {
        this(name, 0);            // chains to next constructor
    }

    Person(String name, int age) {
        this(name, age, "N/A");   // chains to full constructor
    }

    Person(String name, int age, String email) {
        this.name  = name;
        this.age   = age;
        this.email = email;
    }
}
```

> `this(...)` **must be the first statement** in a constructor. You cannot have both `this(...)` and `super(...)` as the first statement — only one is allowed.

### 2.3 Copy Constructor

Creates a new object as a **duplicate** of an existing one. Java has **no built-in copy constructor** — you write it manually. C++ does generate one, but it does a *shallow copy*.

#### Java — Manual Copy Constructor

```java
class Employee {
    String name;
    int    salary;
    int[]  bonusHistory;  // reference type — danger zone!

    // Copy constructor
    Employee(Employee other) {
        this.name         = other.name;           // String is immutable — safe
        this.salary       = other.salary;
        // Deep copy of array — must allocate new memory!
        this.bonusHistory = Arrays.copyOf(other.bonusHistory, other.bonusHistory.length);
    }
}

Employee e1 = new Employee("Alice", 100000, new int[]{5000, 6000});
Employee e2 = new Employee(e1);     // deep copy
e2.bonusHistory[0] = 9999;         // does NOT affect e1.bonusHistory
```

#### Shallow vs Deep Copy — The Classic Trap

```
Shallow Copy                Deep Copy
────────────────────────    ────────────────────────
e1 ──► [ name │ arr ──►[5000,6000] ]
e2 ──► [ name │ arr ──────────────┘  ← SAME array!

vs.

e1 ──► [ name │ arr ──►[5000,6000] ]
e2 ──► [ name │ arr ──►[5000,6000] ]  ← NEW array
```

#### C++ — Shallow vs Deep

```cpp
class Buffer {
public:
    int* data;
    int  size;

    Buffer(int sz) : size(sz) {
        data = new int[sz];
    }

    // Default copy constructor (SHALLOW — DANGEROUS):
    // Buffer(const Buffer& other) : data(other.data), size(other.size) {}
    // Both objects point to the SAME heap memory!

    // Correct DEEP copy constructor:
    Buffer(const Buffer& other) : size(other.size) {
        data = new int[size];
        std::copy(other.data, other.data + size, data);
    }

    ~Buffer() { delete[] data; }  // without deep copy, double-free crash!
};
```

### Constructor Execution Order

```java
class Animal {
    Animal() { System.out.println("Animal constructor"); }
}

class Dog extends Animal {
    Dog() {
        // super() called implicitly first
        System.out.println("Dog constructor");
    }
}

class GoldenRetriever extends Dog {
    GoldenRetriever() {
        System.out.println("GoldenRetriever constructor");
    }
}

new GoldenRetriever();
// Output:
// Animal constructor
// Dog constructor
// GoldenRetriever constructor
```

Order: **top-down** (parent → child), opposite of destructor order.

---

## 3. Destructor / Finalize

### C++ Destructor

Called automatically when an object goes out of scope or is `delete`d. Critical for releasing resources.

```cpp
class FileHandler {
    FILE* fp;
public:
    FileHandler(const char* path) {
        fp = fopen(path, "r");
    }

    ~FileHandler() {           // destructor
        if (fp) fclose(fp);   // always clean up!
        std::cout << "File closed\n";
    }
};

{
    FileHandler fh("data.txt");
    // ... use fh ...
}  // <-- ~FileHandler() called automatically here
```

**Rule of Three (C++):** If you define any of {destructor, copy constructor, copy assignment operator}, you likely need all three.

**Rule of Five (C++11):** Add move constructor + move assignment operator for efficiency.

### Java — `finalize()` (Deprecated & Unreliable)

Java has **no destructor**. The `finalize()` method in `Object` was the closest equivalent — called by GC before an object is collected.

```java
class OldCode {
    @Override
    protected void finalize() throws Throwable {
        try {
            // cleanup
        } finally {
            super.finalize();
        }
    }
}
```

**Why `finalize()` is problematic:**
- Called by GC at an **unpredictable time** (or never, if GC doesn't run)
- Can cause object **resurrection** (making it reachable again)
- Degrades GC performance
- **Deprecated in Java 9**, marked for removal

### Java Best Practice — `AutoCloseable` / try-with-resources

```java
class DatabaseConnection implements AutoCloseable {
    Connection conn;

    DatabaseConnection(String url) throws SQLException {
        conn = DriverManager.getConnection(url);
    }

    @Override
    public void close() throws SQLException {
        conn.close();   // guaranteed to run
        System.out.println("Connection closed");
    }
}

// try-with-resources guarantees close() is called
try (DatabaseConnection db = new DatabaseConnection("jdbc:...")) {
    // use db
}  // db.close() called here, even if exception thrown
```

### Python — `__del__`

```python
class Resource:
    def __del__(self):
        print("Cleanup!")  # called when ref count hits 0

# Better: use context managers
class Resource:
    def __enter__(self): return self
    def __exit__(self, *args): self.cleanup()

with Resource() as r:
    pass  # cleanup() guaranteed
```

---

## 4. The `this` Keyword

`this` is an implicit reference to the **current object** inside instance methods and constructors. It is resolved at **runtime** (based on the actual object), not compile-time.

### Use Case 1 — Disambiguate field vs parameter

```java
class Point {
    int x, y;

    Point(int x, int y) {
        this.x = x;  // this.x = field, x = parameter
        this.y = y;
    }
}
```

Without `this`, `x = x` would assign the parameter to itself — a silent bug.

### Use Case 2 — Constructor chaining

```java
class Config {
    String host;
    int port;
    boolean ssl;

    Config() {
        this("localhost", 8080);  // must be first statement
    }

    Config(String host, int port) {
        this(host, port, false);
    }

    Config(String host, int port, boolean ssl) {
        this.host = host;
        this.port = port;
        this.ssl  = ssl;
    }
}
```

### Use Case 3 — Return current object (Builder/Fluent pattern)

```java
class QueryBuilder {
    private String table;
    private String condition;

    QueryBuilder from(String table) {
        this.table = table;
        return this;              // enables chaining
    }

    QueryBuilder where(String condition) {
        this.condition = condition;
        return this;
    }

    String build() {
        return "SELECT * FROM " + table + " WHERE " + condition;
    }
}

String query = new QueryBuilder()
    .from("users")
    .where("age > 18")
    .build();
```

### Use Case 4 — Pass current object to another method

```java
class EventSource {
    void register(EventListener listener) {
        listener.onEvent(this);  // pass self as event source
    }
}
```

### `this` in Static Context — The Trap

```java
class Counter {
    int count = 0;
    static int total = 0;

    static void reset() {
        this.count = 0;  // ❌ Compile error!
        // 'this' doesn't exist in a static method
        // Static methods belong to the CLASS, not an instance
    }
}
```

### C++ — `this` is a Pointer

```cpp
class Node {
public:
    int val;
    Node* next;

    Node* getThis() { return this; }  // pointer, not reference

    bool isSelf(Node* other) {
        return this == other;  // pointer comparison
    }
};
```

---

## 5. Static vs Instance Members

### Conceptual Difference

| | Instance Member | Static Member |
|---|---|---|
| Belongs to | Each object (own copy) | The class itself (one copy) |
| Memory | Heap (per object) | Method area / static segment |
| Access | Via object reference | Via class name (preferred) |
| `this` inside | Available | NOT available |
| Lifecycle | Tied to object | Tied to class loading |

### Fields

```java
class Counter {
    int instanceCount;     // each Counter object has its own copy
    static int totalCount; // ONE copy shared by ALL Counter objects

    Counter() {
        instanceCount = 0;
        totalCount++;      // shared — tracks how many objects were created
    }
}

Counter a = new Counter(); // totalCount = 1, a.instanceCount = 0
Counter b = new Counter(); // totalCount = 2, b.instanceCount = 0

a.instanceCount = 5;
// b.instanceCount is still 0

Counter.totalCount = 99;   // modifies the single shared copy
```

Memory layout:

```
Heap:
  a → [ instanceCount = 5 ]
  b → [ instanceCount = 0 ]

Method Area (Metaspace):
  Counter.totalCount = 2   ← single shared field
```

### Methods

```java
class MathUtils {
    // Static — utility, no object state needed
    static double sqrt(double n) { return Math.sqrt(n); }

    // Instance — depends on object state
    double value;
    double doubled() { return value * 2; }
}

MathUtils.sqrt(16);        // ✅ call on class
new MathUtils().doubled(); // need object for instance method
```

**Can static methods call instance methods?**

```java
class Foo {
    int x = 10;

    void instanceMethod() { System.out.println(x); }

    static void staticMethod() {
        instanceMethod(); // ❌ No 'this' → no object → can't call instance method
        new Foo().instanceMethod(); // ✅ Create an object first
    }
}
```

### Static Initializer Block

Runs once when the class is first loaded. Used to initialize complex static state.

```java
class Config {
    static final Map<String, String> DEFAULTS;

    static {
        DEFAULTS = new HashMap<>();
        DEFAULTS.put("timeout", "30");
        DEFAULTS.put("retry",   "3");
        System.out.println("Config class loaded");
    }
}
```

Order: **Static block → Instance initializer → Constructor**

```java
class Demo {
    static { System.out.println("1: Static block"); }
    { System.out.println("2: Instance init block"); }
    Demo() { System.out.println("3: Constructor"); }
}

new Demo();
// 1: Static block      (only once ever)
// 2: Instance init block
// 3: Constructor

new Demo();
// 2: Instance init block   (static block NOT repeated)
// 3: Constructor
```

### Singleton Pattern — Classic Static Use Case

```java
class Singleton {
    private static Singleton instance;  // static — one per class

    private Singleton() {}              // private constructor

    public static Singleton getInstance() {
        if (instance == null) {
            instance = new Singleton();
        }
        return instance;
    }
}
```

**Thread-safe variant (double-checked locking):**

```java
class Singleton {
    private static volatile Singleton instance;

    public static Singleton getInstance() {
        if (instance == null) {
            synchronized (Singleton.class) {
                if (instance == null) {         // second check inside lock
                    instance = new Singleton();
                }
            }
        }
        return instance;
    }
}
```

---

## 6. Object Creation in Memory

### Stack vs Heap at a Glance

```
┌─────────────────────┐    ┌──────────────────────────────┐
│       STACK         │    │           HEAP               │
│─────────────────────│    │──────────────────────────────│
│ • Primitive locals  │    │ • All objects (new keyword)  │
│ • Object references │──► │ • Instance fields            │
│ • Method frames     │    │ • Arrays                     │
│ • Fast (LIFO)       │    │ • Slower (GC managed)        │
│ • Auto cleanup      │    │ • Manual (C++) or GC (Java)  │
│ • Thread-local      │    │ • Shared across threads      │
│ • Fixed size        │    │ • Large, expandable          │
└─────────────────────┘    └──────────────────────────────┘
```

### Java Memory Model

In Java, **ALL objects go to the heap**. The stack only holds primitives and references.

```java
void createObjects() {
    int x = 42;              // x → STACK (primitive)
    String s = "hello";      // s → STACK (reference), "hello" → Heap (String Pool)
    Person p = new Person(); // p → STACK (reference), Person object → HEAP
}
// When method returns: x, s, p are popped from stack
// Person object stays on heap until GC collects it (no more references)
```

Visual:

```
Stack (method frame for createObjects)     Heap
───────────────────────────────────────    ─────────────────────────────────
x  = 42                                    Person { name=null, age=0 } ◄───┐
s  = ──────────────────────────────────►   "hello" (String Pool)           │
p  = ─────────────────────────────────────────────────────────────────────┘
```

### C++ Stack vs Heap

```cpp
void demo() {
    int x = 10;         // STACK — auto-freed on scope exit
    int* y = new int;   // y on STACK, *y on HEAP — must delete manually
    *y = 20;

    Dog d1;             // STACK — destructor called automatically
    Dog* d2 = new Dog;  // d2 on STACK, *d2 on HEAP — must delete d2
    delete d2;          // manual cleanup
}   // d1 and x auto-freed here; y leaks if you forgot delete y
```

Stack-allocated objects:

```cpp
{
    Dog fido;           // constructor called
    fido.bark();
}                       // destructor called HERE automatically
```

Heap-allocated objects:

```cpp
Dog* fido = new Dog();   // constructor called
fido->bark();
delete fido;             // destructor called
fido = nullptr;          // good practice — prevent dangling pointer use
```

### Java Heap Generations (Important for GC Questions)

```
Java Heap
├── Young Generation
│   ├── Eden Space        ← new objects created here
│   ├── Survivor 0 (S0)   ← objects that survived 1 GC
│   └── Survivor 1 (S1)   ← objects that survived 2 GCs
└── Old Generation        ← long-lived objects promoted here
    └── (Metaspace)       ← class metadata (not on heap in Java 8+)
```

When `new Person()` is called:
1. Memory allocated in Eden
2. Minor GC runs → survivors moved to S0/S1
3. After N GCs, promoted to Old Gen
4. Major/Full GC eventually cleans Old Gen

### Object Size in Heap (Java)

Every Java object has a **header** (12-16 bytes) before any fields:

```
Java Object Header
├── Mark Word (8 bytes)  — hash code, lock state, GC age
└── Class Pointer (4-8 bytes) — pointer to Class metadata

Total minimum object size: 16 bytes (after padding to 8-byte alignment)
```

```java
// A class with zero fields still consumes ~16 bytes on heap!
class Empty {}

Empty e = new Empty(); // 16 bytes on heap, 4-8 bytes on stack (reference)
```

### Escape Analysis (JVM Optimization)

The JVM can allocate objects **on the stack** if it can prove the object doesn't "escape" the method (not returned, not stored in a field, not passed to unknown code). This is called **escape analysis**.

```java
void compute() {
    Point p = new Point(3, 4);  // may be stack-allocated by JIT!
    double dist = Math.sqrt(p.x * p.x + p.y * p.y);
    // p never escapes this method → JVM may skip heap allocation
}
```

---

## 7. Tricky & Trap Questions

---

### Q1 — What's the output?

```java
class A {
    A() { System.out.println("A"); }
}

class B extends A {
    B() { System.out.println("B"); }
}

class C extends B {
    C() { System.out.println("C"); }
}

public class Main {
    public static void main(String[] args) {
        new C();
    }
}
```

**Answer:**
```
A
B
C
```

**Why?** Each constructor implicitly calls `super()` first. So: `C()` → `B()` → `A()` executes first → prints A, then B unwinds, then C.

---

### Q2 — Shallow copy trap

```java
class MyList {
    int[] data;

    MyList(MyList other) {
        this.data = other.data;  // SHALLOW copy!
    }
}

int[] arr = {1, 2, 3};
MyList m1 = new MyList(arr);
MyList m2 = new MyList(m1);

m2.data[0] = 999;
System.out.println(m1.data[0]); // What prints?
```

**Answer:** `999`

**Why?** `this.data = other.data` copies the reference, not the array. Both objects point to the same array. Fix: `this.data = Arrays.copyOf(other.data, other.data.length);`

---

### Q3 — Static method overriding trap

```java
class Parent {
    static void greet() { System.out.println("Parent"); }
    void hello()        { System.out.println("Hello from Parent"); }
}

class Child extends Parent {
    static void greet() { System.out.println("Child"); }
    void hello()        { System.out.println("Hello from Child"); }
}

public class Main {
    public static void main(String[] args) {
        Parent p = new Child();
        p.greet();  // ?
        p.hello();  // ?
    }
}
```

**Answer:**
```
Parent   ← static method: METHOD HIDING, not overriding. Resolved at compile-time via type of reference.
Hello from Child  ← instance method: polymorphism. Resolved at runtime via actual object type.
```

**The trap:** Static methods cannot be overridden — they are *hidden*. `p.greet()` uses `Parent` because `p` is declared as `Parent`.

---

### Q4 — `this()` ordering trap

```java
class Box {
    int w, h;

    Box() {
        System.out.println("No-arg");
        this(10, 20);   // ❌ Compile error — this() must be FIRST statement
    }

    Box(int w, int h) {
        this.w = w;
        this.h = h;
    }
}
```

**Answer:** Compile error. `this(...)` and `super(...)` **must be the very first statement** in a constructor. No code can precede it.

---

### Q5 — Can you call an instance method from a constructor?

```java
class Demo {
    int value;

    Demo() {
        init();   // calling instance method from constructor
    }

    void init() {
        value = 42;
    }
}
```

**Answer:** Yes, it compiles and runs. But it's a **trap with inheritance**:

```java
class Parent {
    Parent() {
        init();   // calls overridden version in child!
    }
    void init() { System.out.println("Parent init"); }
}

class Child extends Parent {
    int x;
    Child() {
        super();      // implicit — calls Parent()
        x = 100;
    }
    @Override
    void init() { System.out.println("Child init, x = " + x); }
}

new Child();
// Prints: "Child init, x = 0"  ← x is 0 because Child() hasn't run yet!
```

**Why dangerous?** `Parent()` runs first. Inside it, `init()` resolves to `Child.init()` (polymorphism). But at that moment, `Child`'s constructor body hasn't executed, so `x` is still `0`. This is the **never call overridable methods from constructors** rule.

---

### Q6 — Stack Overflow vs OutOfMemoryError

```java
// Stack Overflow — deep/infinite recursion
void recurse() { recurse(); }  // ❌ StackOverflowError

// OutOfMemoryError — heap exhausted
void oom() {
    List<byte[]> list = new ArrayList<>();
    while (true) list.add(new byte[1024 * 1024]); // ❌ OutOfMemoryError
}
```

**Know the difference:** Stack is fixed-size per thread. Heap is large but finite and GC-managed.

---

### Q7 — String Pool trap

```java
String s1 = "hello";           // String pool
String s2 = "hello";           // same pool object
String s3 = new String("hello"); // NEW heap object, NOT from pool

System.out.println(s1 == s2);       // true  (same reference)
System.out.println(s1 == s3);       // false (different objects)
System.out.println(s1.equals(s3));  // true  (same content)
System.out.println(s3.intern() == s1); // true (intern() looks up pool)
```

**The trap:** `==` compares references. Always use `.equals()` for string content comparison.

---

### Q8 — Static field initialization order

```java
class Tricky {
    static int x = getX();
    static int y = 10;

    static int getX() {
        return y;  // y not initialized yet!
    }
}

System.out.println(Tricky.x); // What prints?
```

**Answer:** `0`

**Why?** Static fields are initialized in **declaration order**. When `x = getX()` runs, `y` hasn't been assigned `10` yet — it holds the default value `0`.

---

### Q9 — Can a constructor be private?

```java
class Singleton {
    private static final Singleton INSTANCE = new Singleton();

    private Singleton() {}   // ✅ private constructor — perfectly valid!

    public static Singleton getInstance() { return INSTANCE; }
}
```

**Answer:** Yes! Private constructors prevent direct instantiation from outside the class. Used in Singleton, Factory, and utility classes.

---

### Q10 — Destructor vs Finalizer execution guarantee

**Q:** "Is the destructor in C++ guaranteed to run?"

**Answer:**
- **C++ destructor on stack object:** ✅ **Guaranteed** — runs when scope exits, even if exception thrown (RAII)
- **C++ destructor on heap object (`delete` called):** ✅ **Guaranteed** — if you call `delete`; undefined behavior if you don't
- **Java `finalize()`:** ❌ **NOT guaranteed** — may never run if GC doesn't collect the object, or JVM exits

---

### Q11 — `this` vs `super` in constructors

```java
class Vehicle {
    String type;
    Vehicle(String type) { this.type = type; }
}

class Car extends Vehicle {
    int doors;

    Car(int doors) {
        super("car");      // ✅ calls Vehicle(String)
        this.doors = doors;
    }

    Car() {
        this(4);           // ✅ chains to Car(int)
        // NOTE: you cannot have BOTH super() and this() — only one first-statement call allowed
    }
}
```

---

### Q12 — Memory leak in Java (despite GC)

```java
class Cache {
    static List<Object> cache = new ArrayList<>();

    void addToCache(Object o) {
        cache.add(o);  // static list holds references → objects NEVER collected!
    }
}
```

**The trap:** Java has GC, but if a **static collection** holds references, objects are never eligible for collection. This is a common source of memory leaks in Java.

Fix: Use `WeakHashMap`, `SoftReference`, or explicitly remove entries.

---

## Quick Summary Cheat Sheet

```
CLASS          → blueprint, compile-time, lives in method area
OBJECT         → instance, runtime, lives on heap
CONSTRUCTOR    → same name as class, no return type, called on new
DEFAULT CTOR   → compiler adds only if NO ctor defined; lost if parameterized ctor added
COPY CTOR      → Java: manual; C++: auto but shallow; always consider deep copy
DESTRUCTOR     → C++: guaranteed; Java: no destructor, use AutoCloseable
finalize()     → deprecated, unreliable, never use
this           → reference to current object; unavailable in static context
this(...)      → constructor chaining; MUST be first statement
static field   → one shared copy per class; initialized in declaration order
instance field → one copy per object; holds individual state
stack          → primitives, references, method frames; auto-cleaned; thread-local
heap           → all objects; GC-managed (Java); manual (C++)
```

---

*These notes cover SDE 2 interview depth. For SDE 3+, also study: JVM internals (GC algorithms, generational GC), escape analysis, lock-free singleton, memory model (happens-before), and JMM visibility.*
