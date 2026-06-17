# OOP Four Pillars — SDE 2 Interview Notes (Java)

> **Target:** SDE 2 roles at product-based companies
> **Language:** Java (primary), with C++ notes where Java differs critically
> **Depth:** Internals + traps + design reasoning, not just syntax

---

## Table of Contents

1. [Encapsulation](#1-encapsulation)
2. [Abstraction](#2-abstraction)
3. [Polymorphism](#3-polymorphism)
4. [Inheritance](#4-inheritance)
5. [Master Trap Questions](#5-master-trap-questions)

---

# 1. Encapsulation

## The Core Idea

Encapsulation = **bundling data + behavior** that operates on that data into a single unit (class), and **controlling access** to that data.

It's not just "making fields private." It's about **invariant protection** — ensuring an object never enters an invalid state.

```
Without Encapsulation          With Encapsulation
──────────────────────         ─────────────────────────────
BankAccount.balance = -500;    account.withdraw(500) → throws if insufficient
// Invalid state — no guard    // Object controls its own state transitions
```

---

## 1.1 Access Modifiers

Java has four access levels. **No keyword = default (package-private)** — not `public`, not `private`.

| Modifier    | Same Class | Same Package | Subclass (diff pkg) | Any Class |
|-------------|:----------:|:------------:|:-------------------:|:---------:|
| `private`   | ✅          | ❌            | ❌                   | ❌         |
| `default`   | ✅          | ✅            | ❌                   | ❌         |
| `protected` | ✅          | ✅            | ✅                   | ❌         |
| `public`    | ✅          | ✅            | ✅                   | ✅         |

### Visualized

```
com.bank
  ├── Account.java          ← defines private balance
  ├── AccountUtils.java     ← default access: can see package-private members
  └── sub/
        └── SavingsAccount.java  ← subclass in different package
              protected: ✅ visible
              default:   ❌ NOT visible
              private:   ❌ NOT visible

com.ui
  └── Dashboard.java        ← unrelated class
        public:    ✅ visible
        protected: ❌
        default:   ❌
        private:   ❌
```

### Access Modifier on Class Itself

```java
public class PublicClass {}     // accessible from anywhere
class PackageClass {}           // only within the same package
// private class → NOT allowed at top level (only for nested/inner classes)
// protected class → NOT allowed at top level
```

### Concrete Example

```java
package com.bank;

public class Account {
    private double balance;               // hidden — only this class
    protected String accountNumber;      // subclasses in any package can access
    String branch;                       // package-private — AccountUtils can see
    public String ownerName;             // everyone can see (usually avoid for fields)

    private void validateAmount(double amt) {  // private helper
        if (amt <= 0) throw new IllegalArgumentException("Amount must be positive");
    }

    public void deposit(double amount) {   // public API
        validateAmount(amount);
        balance += amount;
    }

    public double getBalance() { return balance; }
}
```

---

## 1.2 Getters & Setters

Getters and setters are the **controlled entry points** into private state. Their power is in the logic they can contain, not just the get/set.

### Basic Pattern

```java
class Temperature {
    private double celsius;

    // Getter — can add formatting, conversion, logging
    public double getCelsius() { return celsius; }

    public double getFahrenheit() {  // derived getter — no separate field needed
        return celsius * 9.0 / 5.0 + 32;
    }

    // Setter — validates before accepting
    public void setCelsius(double celsius) {
        if (celsius < -273.15) {
            throw new IllegalArgumentException("Below absolute zero: " + celsius);
        }
        this.celsius = celsius;
    }
}
```

### Read-Only and Write-Once Fields

```java
class ImmutablePoint {
    private final int x;
    private final int y;

    ImmutablePoint(int x, int y) {
        this.x = x;
        this.y = y;
    }

    public int getX() { return x; }   // getter only — no setter
    public int getY() { return y; }   // field can never change after construction
}
```

### Lazy Initialization via Getter

```java
class ReportGenerator {
    private List<String> cachedData;   // null initially

    public List<String> getData() {
        if (cachedData == null) {      // compute only when needed
            cachedData = loadFromDatabase();
        }
        return cachedData;
    }
}
```

### The Over-Setter Anti-Pattern

```java
// ❌ BAD — exposes every field; object can be put in invalid state
class Rectangle {
    private int width;
    private int height;
    private int area;          // derived — should NOT have a setter

    public void setArea(int area) { this.area = area; } // caller can set area=999 while width=2, height=3!
}

// ✅ GOOD — area is always derived, never directly set
class Rectangle {
    private int width;
    private int height;

    public void setWidth(int w)  { if (w > 0) this.width = w;  recalculate(); }
    public void setHeight(int h) { if (h > 0) this.height = h; recalculate(); }
    public int  getArea()        { return width * height; }  // always consistent
}
```

---

## 1.3 Data Hiding vs Encapsulation

These are related but distinct — interviewers love testing this.

| Aspect       | Data Hiding                              | Encapsulation                                   |
|-------------|------------------------------------------|-------------------------------------------------|
| Definition  | Restricting direct access to fields      | Bundling data + methods + access control together |
| Mechanism   | `private` keyword                        | Class + access modifiers + methods              |
| Scope       | Narrower concept                         | Broader concept                                 |
| Goal        | Prevent unauthorized access              | Protect invariants, manage complexity           |
| Relationship| **Data hiding is a part of encapsulation** | Encapsulation includes data hiding             |

```java
// Data Hiding only (no real encapsulation — getter/setter expose everything raw)
class Bad {
    private int x;
    public int getX() { return x; }
    public void setX(int x) { this.x = x; }  // just a wrapper — zero protection
}

// True Encapsulation — hides AND protects invariant
class BankAccount {
    private double balance;      // data hiding

    public void transfer(BankAccount target, double amount) {
        if (amount > balance) throw new InsufficientFundsException();
        this.balance  -= amount;
        target.balance += amount;  // direct access OK — same class!
    }
    // The invariant "balance >= 0" is always maintained by the class itself
}
```

---

## 1.4 Tight vs Loose Encapsulation

### Tight Encapsulation (Preferred)

Every field is `private`. All access goes through methods. Object fully controls its state.

```java
class Order {
    private List<Item> items = new ArrayList<>();
    private OrderStatus status = OrderStatus.PENDING;
    private double totalPrice;

    public void addItem(Item item) {
        if (status != OrderStatus.PENDING) throw new IllegalStateException("Cannot modify placed order");
        items.add(item);
        totalPrice += item.getPrice();  // automatically kept in sync
    }

    public double getTotalPrice() { return totalPrice; }
    public List<Item> getItems()  { return Collections.unmodifiableList(items); } // defensive copy
    public OrderStatus getStatus(){ return status; }
}
```

### Loose Encapsulation (Risky)

Fields are `public` or `protected`, or getters return mutable references without defensive copying.

```java
// ❌ Loose — caller can mutate internal list directly
class LooseOrder {
    public List<Item> items = new ArrayList<>();  // public field — no control
    protected double total;                       // subclasses can corrupt it
}

// Caller can do: order.items.clear(); order.total = -1;  ← invariant broken!
```

### The Mutable Reference Leak Problem

```java
class Config {
    private Date lastUpdated = new Date();

    // ❌ Leaks mutable reference — caller can modify internal Date!
    public Date getLastUpdated() { return lastUpdated; }

    // ✅ Defensive copy — safe
    public Date getLastUpdated() { return new Date(lastUpdated.getTime()); }
}
```

---

# 2. Abstraction

## The Core Idea

Abstraction = **hiding the "how," exposing the "what."** The user interacts with a simplified model without knowing the internal complexity.

```
ATM Machine             Coffee Machine
What you see:           What you see:
  [Enter PIN]             [Press Espresso]
  [Withdraw Cash]
What's hidden:          What's hidden:
  Bank auth protocol      Grinder, boiler, pump pressure,
  Transaction ledger      temperature control, timer
```

In code: define **contracts** (what operations exist) and hide **implementations** (how they work).

---

## 2.1 Abstract Class

- Declared with `abstract` keyword
- **Cannot be instantiated** directly
- Can have **both abstract AND concrete methods**
- Can have **constructors, fields, static methods**
- A class can extend **only one** abstract class

```java
abstract class Shape {
    private String color;   // concrete field

    Shape(String color) {   // concrete constructor
        this.color = color;
    }

    abstract double area();        // MUST be implemented by subclass
    abstract double perimeter();   // MUST be implemented by subclass

    // Concrete method — shared behavior
    public void printInfo() {
        System.out.printf("Color: %s | Area: %.2f | Perimeter: %.2f%n",
                           color, area(), perimeter());
    }
}

class Circle extends Shape {
    private double radius;

    Circle(String color, double radius) {
        super(color);
        this.radius = radius;
    }

    @Override public double area()      { return Math.PI * radius * radius; }
    @Override public double perimeter() { return 2 * Math.PI * radius; }
}

class Square extends Shape {
    private double side;

    Square(String color, double side) {
        super(color);
        this.side = side;
    }

    @Override public double area()      { return side * side; }
    @Override public double perimeter() { return 4 * side; }
}

// Usage — program to abstraction, not implementation
List<Shape> shapes = List.of(new Circle("red", 5), new Square("blue", 4));
shapes.forEach(Shape::printInfo);
```

### When to use Abstract Class

Use when subclasses share a common base implementation and you want to avoid code duplication:

```java
abstract class DataProcessor {
    // Template Method Pattern
    public final void process() {  // final — workflow is fixed
        readData();
        validate();       // concrete — shared logic
        transform();      // abstract — subclass-specific
        writeOutput();    // concrete — shared logic
    }

    abstract void transform();    // must override

    private void readData()    { System.out.println("Reading..."); }
    private void validate()    { System.out.println("Validating..."); }
    private void writeOutput() { System.out.println("Writing..."); }
}

class CSVProcessor extends DataProcessor {
    @Override void transform() { System.out.println("Parsing CSV columns..."); }
}

class JSONProcessor extends DataProcessor {
    @Override void transform() { System.out.println("Deserializing JSON fields..."); }
}
```

---

## 2.2 Interface

- All methods are `public abstract` by default (pre-Java 8)
- Java 8+: can have `default` and `static` methods
- Java 9+: can have `private` methods
- All fields are `public static final` (constants)
- A class can implement **multiple interfaces**
- An interface can extend **multiple interfaces**

```java
interface Flyable {
    int MAX_ALTITUDE = 50000;    // implicitly public static final

    void fly();                  // implicitly public abstract
    void land();

    default void describe() {    // Java 8+ — default implementation
        System.out.println("I can fly up to " + MAX_ALTITUDE + " feet");
    }
}

interface Swimmable {
    void swim();

    default void describe() {    // name clash with Flyable.describe()!
        System.out.println("I can swim");
    }
}

class Duck implements Flyable, Swimmable {
    @Override public void fly()  { System.out.println("Duck flying"); }
    @Override public void land() { System.out.println("Duck landing"); }
    @Override public void swim() { System.out.println("Duck swimming"); }

    // MUST override to resolve ambiguity between two default methods
    @Override public void describe() {
        Flyable.super.describe();   // explicitly choose which default
        Swimmable.super.describe();
    }
}
```

### Interface vs Abstract Class Decision Table

| Question                                           | Use                  |
|----------------------------------------------------|----------------------|
| Multiple inheritance needed?                        | Interface            |
| Shared state (fields) needed?                       | Abstract Class       |
| Unrelated classes should implement same contract?  | Interface            |
| Strong "is-a" relationship with shared code?       | Abstract Class       |
| Evolving API where you control all implementations?| Abstract Class       |
| Library contract for unknown implementors?         | Interface            |
| Need constructors?                                  | Abstract Class       |

### Real-World Design Example

```java
// Interface — the contract (abstraction)
interface PaymentGateway {
    PaymentResult charge(double amount, String currency, CardDetails card);
    void refund(String transactionId);
}

// Abstract class — shared infrastructure
abstract class BasePaymentGateway implements PaymentGateway {
    protected Logger logger = Logger.getLogger(getClass().getName());
    private RateLimiter rateLimiter = new RateLimiter(100); // 100 req/sec

    @Override
    public PaymentResult charge(double amount, String currency, CardDetails card) {
        rateLimiter.acquire();
        logger.info("Charging " + amount + " " + currency);
        PaymentResult result = doCharge(amount, currency, card); // template
        logger.info("Result: " + result.getStatus());
        return result;
    }

    protected abstract PaymentResult doCharge(double amount, String currency, CardDetails card);
}

// Concrete — specific implementations hidden from callers
class StripeGateway extends BasePaymentGateway {
    @Override
    protected PaymentResult doCharge(double amount, String currency, CardDetails card) {
        // Stripe-specific HTTP calls, SDK usage — hidden from caller
        return new PaymentResult("stripe_txn_123", "SUCCESS");
    }
    @Override
    public void refund(String txnId) { /* Stripe refund API */ }
}

// Caller only knows the interface — can swap gateway without changing caller
PaymentGateway gateway = new StripeGateway();
gateway.charge(99.99, "USD", card);
```

---

# 3. Polymorphism

## The Core Idea

Polymorphism = **one interface, multiple behaviors.** The same method call produces different results depending on context (overloading) or the actual object type (overriding).

---

## 3.1 Compile-Time (Static) Polymorphism — Method Overloading

Resolved at **compile time** based on method signature (name + parameter types + count). Return type alone does NOT distinguish overloads.

```java
class Printer {
    void print(int x)      { System.out.println("int: " + x); }
    void print(double x)   { System.out.println("double: " + x); }
    void print(String x)   { System.out.println("String: " + x); }
    void print(int x, int y) { System.out.println("two ints: " + x + ", " + y); }

    // ❌ NOT valid overloading — same signature, only return type differs
    // int  getValue() { return 1; }
    // long getValue() { return 1L; }  // Compile error
}

Printer p = new Printer();
p.print(42);        // → int: 42
p.print(3.14);      // → double: 3.14
p.print("hello");   // → String: hello
```

### Widening in Overloading — The Trap

```java
class Overload {
    void method(long x)   { System.out.println("long"); }
    void method(double x) { System.out.println("double"); }
}

Overload o = new Overload();
o.method(10);     // 10 is int → widened to long → prints "long"
o.method(10L);    // explicit long → prints "long"
o.method(10.0);   // double → prints "double"
```

Widening priority: `byte → short → int → long → float → double`

### Autoboxing vs Widening Priority

```java
class BoxTrap {
    void method(Integer x) { System.out.println("Integer"); }
    void method(long x)    { System.out.println("long"); }
}

BoxTrap b = new BoxTrap();
b.method(10);  // widening (int→long) wins over autoboxing (int→Integer)
               // prints "long"
```

Compiler preference order: widening > autoboxing > varargs

---

## 3.2 Operator Overloading

Java does **not** support operator overloading for user-defined types. Only the JVM itself uses it (e.g., `+` for String concatenation).

```java
// Java — NOT possible to overload operators
// Vector v1 + v2;  ← illegal

// Must use methods instead:
class Vector {
    double x, y;
    Vector(double x, double y) { this.x = x; this.y = y; }

    Vector add(Vector other) { return new Vector(this.x + other.x, this.y + other.y); }
    @Override public String toString() { return "(" + x + ", " + y + ")"; }
}

Vector v3 = v1.add(v2);
```

C++ allows it; Python uses `__add__`, `__mul__` dunder methods.

---

## 3.3 Runtime (Dynamic) Polymorphism — Method Overriding

Resolved at **runtime** based on the **actual object type**, not the reference type. Enabled by the vtable (virtual dispatch) mechanism in the JVM.

### Rules for Overriding

```java
class Animal {
    public Animal create()         { return new Animal(); }  // return type
    public void sound()            { System.out.println("..."); }
    protected void breathe()       { System.out.println("breathe"); }

    // Cannot override:
    private void secret()          {}   // private — not visible to child
    final void lifespan()          {}   // final — locked
    static void taxonomy()         {}   // static — hidden, not overridden
}

class Dog extends Animal {
    @Override
    public Dog create()            { return new Dog(); }  // covariant return type ✅
    @Override
    public void sound()            { System.out.println("Woof"); } // ✅
    @Override
    public void breathe()          { System.out.println("pant"); } // ✅ can widen access (protected→public)

    // @Override private void secret() {}   // ❌ can't see parent's private
    // @Override void lifespan() {}         // ❌ can't override final
    // @Override static void taxonomy() {}  // ❌ this would be hiding, not overriding
}
```

### @Override Annotation — Always Use It

```java
class Parent {
    public void doWork() {}
}

class Child extends Parent {
    // Without @Override — silently creates a NEW method if you typo:
    public void dowork() {}    // typo — not an override! Bug silently introduced.

    // With @Override — compile error if signature doesn't match:
    @Override
    public void dowork() {}    // ❌ Compile error: "method does not override"
}
```

### The Virtual Dispatch Mechanism

```
JVM Runtime — vtable lookup
─────────────────────────────────────────────
Animal a = new Dog();

Reference type: Animal   (compiler checks this for method existence)
Actual type:    Dog      (JVM uses this at runtime for dispatch)

a.sound() → JVM looks up Dog's vtable → finds Dog.sound() → executes "Woof"
```

```java
Animal a = new Dog();
a.sound();   // "Woof" — runtime dispatch to Dog.sound()

// Compile-time:  "Does Animal have a sound() method?" → yes → compile OK
// Runtime:       "What is the actual type?" → Dog → call Dog.sound()
```

---

## 3.4 Early Binding vs Late Binding

| | Early Binding (Static)          | Late Binding (Dynamic)              |
|---|--------------------------------|-------------------------------------|
| When resolved | Compile time               | Runtime                             |
| Applies to  | Static, private, final methods | Non-final instance methods          |
| Mechanism   | Direct call in bytecode         | vtable / method dispatch table      |
| Performance | Faster                          | Tiny overhead (negligible)          |
| Enables     | Overloading                     | Overriding / Polymorphism           |

```java
class Parent { void hello() { System.out.println("Parent"); } }
class Child extends Parent { void hello() { System.out.println("Child"); } }

Parent p = new Child();
p.hello();
// Early binding: compiler sees Parent.hello() exists → OK
// Late binding: JVM at runtime sees actual object is Child → calls Child.hello()
// Output: "Child"
```

---

## 3.5 Covariant Return Types

A subclass can override a method and return a **more specific (narrower) type** than the parent's return type.

```java
class Animal {
    public Animal create() { return new Animal(); }
}

class Dog extends Animal {
    @Override
    public Dog create() { return new Dog(); }  // Dog is a subtype of Animal → ✅
}

Animal factory = new Dog();
Animal result = factory.create();  // returns a Dog at runtime
```

Why useful? **Builder pattern and clone()**:

```java
class Node implements Cloneable {
    int val;
    @Override
    public Node clone() {           // return type narrowed from Object to Node
        return new Node(this.val);  // no cast needed by caller
    }
}

Node original = new Node(5);
Node copy = original.clone();  // no (Node) cast needed!
```

---

# 4. Inheritance

## The Core Idea

Inheritance = a class (child/subclass) **acquires** the properties and behaviors of another class (parent/superclass). Models an **"is-a"** relationship.

**Favor composition over inheritance** (Effective Java Item 18) when the relationship is "has-a" or when the parent class isn't designed for extension.

---

## 4.1 Types of Inheritance

### Single Inheritance

```java
class Vehicle { void move() { System.out.println("Moving"); } }
class Car extends Vehicle { void honk() { System.out.println("Beep"); } }

Car c = new Car();
c.move();  // inherited
c.honk();  // own method
```

### Multilevel Inheritance

```java
class Animal   { void breathe() { System.out.println("breathe"); } }
class Mammal extends Animal  { void nurseYoung() { System.out.println("nurse"); } }
class Dog extends Mammal { void bark() { System.out.println("woof"); } }

Dog d = new Dog();
d.breathe();    // from Animal (2 levels up)
d.nurseYoung(); // from Mammal
d.bark();       // own
```

### Hierarchical Inheritance

```java
class Shape  { abstract double area(); }   // one parent
class Circle extends Shape { ... }         // multiple children
class Square extends Shape { ... }         //
class Triangle extends Shape { ... }       //
```

### Multiple Inheritance — NOT supported in Java for classes

Java does NOT allow `class C extends A, B`. Reason: Diamond Problem (see section 4.5).

**Java's solution:** Multiple inheritance of *type* via interfaces.

```java
interface Flyable { void fly(); }
interface Swimmable { void swim(); }
class Duck implements Flyable, Swimmable {  // ✅ multiple interface implementation
    public void fly()  { ... }
    public void swim() { ... }
}
```

### Hybrid Inheritance

A combination of multiple types. Only achievable in Java via interfaces.

```java
interface A { default void hello() { System.out.println("A"); } }
interface B extends A { }
interface C extends A { }
class D implements B, C {
    @Override
    public void hello() { B.super.hello(); }  // resolve diamond
}
```

---

## 4.2 The `super` Keyword

`super` refers to the **parent class**. Used to:
1. Access parent's constructor
2. Access parent's overridden method
3. Access parent's hidden field (rare)

```java
class Vehicle {
    String type;
    int speed;

    Vehicle(String type, int speed) {
        this.type  = type;
        this.speed = speed;
    }

    String describe() {
        return type + " going " + speed + " km/h";
    }
}

class ElectricCar extends Vehicle {
    int batteryKWh;

    ElectricCar(int speed, int battery) {
        super("Electric Car", speed);  // must be first statement
        this.batteryKWh = battery;
    }

    @Override
    String describe() {
        return super.describe() + " | Battery: " + batteryKWh + " kWh";
    }
}

ElectricCar tesla = new ElectricCar(200, 100);
System.out.println(tesla.describe());
// Electric Car going 200 km/h | Battery: 100 kWh
```

### super() Rules

- `super()` or `super(args)` must be the **first statement** in a constructor
- If omitted, compiler inserts `super()` (no-arg) automatically
- If parent has no no-arg constructor, you **must** explicitly call the correct `super(args)` or the code won't compile

```java
class Parent {
    Parent(int x) { }   // parameterized only — no default
}

class Child extends Parent {
    Child() {
        // compiler tries to insert super() → ❌ no-arg Parent() doesn't exist
        // Fix: super(42);
    }
}
```

---

## 4.3 Constructor Chaining Across Inheritance

```java
class A {
    A()    { System.out.println("A()"); }
    A(int x){ System.out.println("A(int): " + x); }
}

class B extends A {
    B()    { this(10); System.out.println("B()"); }  // chains to B(int)
    B(int x){ super(x); System.out.println("B(int): " + x); }  // chains to A(int)
}

class C extends B {
    C()    { super(5); System.out.println("C()"); }  // chains to B(int)
}

new C();
// A(int): 5    ← super chain reaches A first
// B(int): 5    ← B(int) runs
// C()          ← C() finally runs
```

Execution is always **outermost super-constructor first, then back down**.

---

## 4.4 Method Hiding vs Method Overriding

This is one of the **most tested distinctions** at SDE 2 level.

| | Method Overriding                     | Method Hiding                       |
|---|--------------------------------------|-------------------------------------|
| Applies to | Instance methods                  | Static methods                      |
| Binding | **Late (runtime)**                    | **Early (compile-time)**            |
| Dispatch | Based on actual object type           | Based on reference type             |
| `@Override` | Valid and recommended              | Can annotate but it's misleading    |

```java
class Parent {
    static void staticMethod() { System.out.println("Parent static"); }
    void instanceMethod()      { System.out.println("Parent instance"); }
}

class Child extends Parent {
    static void staticMethod() { System.out.println("Child static"); }    // HIDING
    @Override
    void instanceMethod()      { System.out.println("Child instance"); }  // OVERRIDING
}

Parent ref = new Child();  // reference is Parent, object is Child

ref.staticMethod();   // "Parent static"  ← compile-time, uses reference type (HIDING)
ref.instanceMethod(); // "Child instance" ← runtime, uses actual type (OVERRIDING)

Child cRef = new Child();
cRef.staticMethod();  // "Child static"   ← reference type is Child now
```

**Memory aid:** Static = Stuck to the class. Instance = Dynamic dispatch.

---

## 4.5 The Diamond Problem & Resolution

### The Problem

```
       A
      / \
     B   C
      \ /
       D
```

If `B` and `C` both override a method from `A`, and `D` inherits from both, which version does `D` get? This is the diamond problem.

Java prevents this for **classes** by not allowing multiple class inheritance.

```java
// C++ allows this (and it's messy):
class A { public: void hello() { cout << "A"; } };
class B : public A { public: void hello() { cout << "B"; } };
class C : public A { public: void hello() { cout << "C"; } };
class D : public B, public C {};  // ambiguous! D d; d.hello(); → compile error

// C++ fix: virtual inheritance
class B : virtual public A { ... };
class C : virtual public A { ... };
class D : public B, public C { ... };  // now resolved — only one A subobject
```

### Java's Diamond with Interfaces (Java 8+)

```java
interface A {
    default void hello() { System.out.println("A"); }
}

interface B extends A {
    default void hello() { System.out.println("B"); }
}

interface C extends A {
    default void hello() { System.out.println("C"); }
}

// Rule 1: Class method wins over interface default
// Rule 2: More specific interface wins (B wins over A if B extends A)
// Rule 3: If ambiguous, must override explicitly

class D implements B, C {
    @Override
    public void hello() {
        B.super.hello();  // explicitly choose B's version
        // or: C.super.hello();
        // or: System.out.println("D");
    }
}

// Rule 2 example — no ambiguity
class E implements B, A {
    // B is more specific (extends A) → B.hello() wins automatically
    // No override needed
}
```

### Resolution Rules (Java) — In Priority Order

```
1. Class wins  → If the class has its own method, use it (no matter what interfaces say)
2. Specificity → If interface X extends interface Y, X's method wins over Y's
3. Explicit    → If still ambiguous, you MUST override and explicitly call: InterfaceName.super.method()
```

---

# 5. Master Trap Questions

---

### T1 — Can an abstract class have a constructor?

```java
abstract class Shape {
    String color;
    Shape(String color) { this.color = color; }  // ✅ perfectly valid
}

// Shape s = new Shape("red"); // ❌ can't instantiate
// But when Circle extends Shape:
class Circle extends Shape {
    Circle(String color) { super(color); }  // constructor is called via super()
}
```

**Answer:** Yes. Abstract class constructors run when a subclass is instantiated via `super()`. They initialize the shared state.

---

### T2 — Output Prediction: Runtime Polymorphism with Fields

```java
class Parent {
    int x = 10;
    void print() { System.out.println("Parent: " + x); }
}

class Child extends Parent {
    int x = 20;
    @Override
    void print() { System.out.println("Child: " + x); }
}

public class Main {
    public static void main(String[] args) {
        Parent p = new Child();
        p.print();          // ?
        System.out.println(p.x);  // ?
    }
}
```

**Answer:**
```
Child: 20    ← method is dynamically dispatched → Child.print()
10           ← fields are NOT polymorphic! Resolved at compile time via reference type
```

**The trap:** Polymorphism applies to **methods**, not **fields**. Field access uses the reference type (Parent), not the object type.

---

### T3 — Can you override a private method?

```java
class Parent {
    private void secret() { System.out.println("Parent secret"); }

    void expose() { secret(); }  // calls its own private method
}

class Child extends Parent {
    private void secret() { System.out.println("Child secret"); }  // NEW method, not override!
}

new Child().expose();  // What prints?
```

**Answer:** `"Parent secret"`

`Child.secret()` is a completely new private method invisible to `Parent`. `expose()` is defined in `Parent` and calls `Parent.secret()` — there's no polymorphism for private methods. `@Override` on `Child.secret()` would cause a compile error.

---

### T4 — What is the output?

```java
class A {
    A() { show(); }
    void show() { System.out.println("A.show"); }
}

class B extends A {
    int val = 99;
    B() { super(); val = 100; }
    @Override void show() { System.out.println("B.show: val = " + val); }
}

new B();
```

**Answer:**
```
B.show: val = 0
```

**Execution order:**
1. `new B()` starts
2. `super()` → `A()` runs
3. Inside `A()`, `show()` is called — polymorphism kicks in → `B.show()` runs
4. BUT at this moment, `B`'s constructor body hasn't run yet → `val` is still `0` (default int)
5. `A()` finishes
6. `B`'s instance initializer runs: `val = 99`
7. `B()` body continues: `val = 100`

---

### T5 — Can an interface extend multiple interfaces?

```java
interface Readable  { String read(); }
interface Writable  { void write(String s); }
interface Closeable { void close(); }

// ✅ Interface CAN extend multiple interfaces
interface ReadWriteCloseable extends Readable, Writable, Closeable {}

class FileStream implements ReadWriteCloseable {
    public String read()        { return "data"; }
    public void write(String s) { System.out.println("writing: " + s); }
    public void close()         { System.out.println("closed"); }
}
```

**Answer:** Yes, interfaces can extend multiple interfaces. Only classes have single-inheritance restriction.

---

### T6 — Widening access in overriding

```java
class Parent {
    protected void method() { System.out.println("parent"); }
}

class Child extends Parent {
    @Override
    public void method() { System.out.println("child"); }  // ✅ widened protected → public

    // @Override
    // private void method() { }  // ❌ narrowed public → private — compile error
}
```

**Rule:** Overriding method can widen (protected → public) but **cannot narrow** access.

---

### T7 — Can you instantiate an anonymous abstract class?

```java
abstract class Greeting {
    abstract void greet();
}

// ✅ Anonymous class — creates a one-off implementation inline
Greeting g = new Greeting() {
    @Override
    public void greet() { System.out.println("Hello!"); }
};
g.greet();  // "Hello!"
```

**Answer:** You cannot instantiate the abstract class directly, but you CAN create an **anonymous subclass** that immediately provides the implementation. This is the basis for Java's lambda expressions and functional interfaces.

---

### T8 — Interface fields trap

```java
interface Constants {
    int VALUE = 100;  // implicitly: public static final int VALUE = 100;
}

class Impl implements Constants {
    void show() {
        // VALUE = 200;  // ❌ Compile error — it's final!
        System.out.println(VALUE);       // ✅ 100
        System.out.println(Constants.VALUE);  // ✅ 100 — preferred, explicit
    }
}
```

All interface fields are `public static final` — constants. You cannot reassign them.

---

### T9 — Method overloading with null

```java
class Overloaded {
    void method(String s)  { System.out.println("String"); }
    void method(Object o)  { System.out.println("Object"); }
    void method(Integer i) { System.out.println("Integer"); }
}

Overloaded o = new Overloaded();
o.method(null);  // Which one is called?
```

**Answer:** `"String"` — the compiler picks the **most specific type** that `null` is assignable to. `String` is more specific than `Object`. (`Integer` is also more specific than `Object`, but `String` and `Integer` are unrelated, so if both were present without `Object`, it would be a compile-time **ambiguity error**.)

```java
// Without the String method:
// o.method(null); → ambiguous between String and Integer → compile error!
```

---

### T10 — Overloading vs Overriding: return type

```java
class Parent {
    int compute(int x) { return x; }
}

class Child extends Parent {
    // Attempt 1: change only return type
    // double compute(int x) { return x; }  // ❌ Compile error — can't override by changing only return type

    // Attempt 2: covariant return (narrower subtype)
    // int compute(int x) { return x + 1; }  // ✅ valid override

    // Attempt 3: different param — this is overloading, not overriding
    double compute(double x) { return x; }  // ✅ new overloaded method, not an override
}
```

---

### T11 — `final` keyword interactions

```java
// final class — cannot be extended
final class ImmutablePoint {
    final int x, y;
    ImmutablePoint(int x, int y) { this.x = x; this.y = y; }
}
// class MutablePoint extends ImmutablePoint {}  // ❌ Compile error

// final method — cannot be overridden
class Base {
    final void lock() { System.out.println("locked"); }
}
class Derived extends Base {
    // @Override void lock() {}  // ❌ Compile error
}

// final variable — must be initialized once, cannot change
class Config {
    final int port;
    Config(int port) { this.port = port; }  // set in constructor
    // port = 9090;  // ❌ Cannot reassign
}
```

---

### T12 — The complete override checklist

For a method in a subclass to be a valid **override**:

```
✅ Same method name
✅ Same parameter types and count (exact match)
✅ Return type: same OR covariant (subtype)
✅ Access: same OR wider (not narrower)
✅ Exceptions: same OR fewer/narrower checked exceptions (can throw more unchecked)
✅ Method must not be: private, static, or final in parent
✅ Annotate with @Override to get compile-time verification
```

```java
class Parent {
    protected Number compute(int x) throws IOException { return x; }
}

class Child extends Parent {
    @Override
    public Integer compute(int x) throws FileNotFoundException {  // ✅ all rules satisfied
        return x * 2;
        // public > protected ✅
        // Integer is subtype of Number (covariant) ✅
        // FileNotFoundException is subtype of IOException ✅
    }
}
```

---

## Quick Summary Cheat Sheet

```
ENCAPSULATION
  private fields + public methods = control over state transitions
  Data Hiding ⊂ Encapsulation (hiding is a tool, encapsulation is the goal)
  Loose = public fields or leaking mutable references
  Tight = private fields + defensive copies + validated setters

ABSTRACTION
  Abstract class: shared state + partial implementation + single inheritance
  Interface: pure contract + multiple inheritance + no state (only constants)
  "Program to an interface, not an implementation" — GoF

POLYMORPHISM
  Overloading  → compile-time, same name, different params, early binding
  Overriding   → runtime, same signature, child replaces parent, late binding
  Hiding       → static methods, compile-time, reference type determines call
  Fields       → NOT polymorphic — always resolved by reference type
  Covariant    → narrower return type allowed in override

INHERITANCE
  is-a → use inheritance; has-a → use composition
  super() must be first statement in constructor
  Method hiding ≠ overriding (static vs instance, compile vs runtime)
  Diamond (classes): prevented by Java's single class inheritance
  Diamond (interfaces): resolved by specificity rule, then explicit override
  Overriding rules: widen access ✅, narrow access ❌, covariant return ✅
```

---

*Next topics to study at SDE 2+ depth: SOLID principles, Design Patterns (Builder, Factory, Strategy, Observer), Composition over Inheritance, Liskov Substitution Principle with examples, and how Java generics interact with inheritance (type erasure, wildcards).*
