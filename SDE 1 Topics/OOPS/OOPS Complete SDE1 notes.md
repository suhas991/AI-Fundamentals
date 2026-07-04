**You:**

a) Encapsulation
- Access modifiers — private, protected, public, default
- Getters & Setters
- Data hiding vs Encapsulation (difference)
- Tight vs Loose encapsulation

Java topics explain in depth wirh examples and charts necessary

* * *

**ChatGPT:**

# Encapsulation in Java (Complete Interview Guide)

Encapsulation is one of the **four pillars of Object-Oriented Programming (OOP)**.

The four pillars are:

1. Encapsulation
2. Inheritance
3. Polymorphism
4. Abstraction

* * *

# What is Encapsulation?

> **Encapsulation is the process of wrapping data (variables) and methods (functions) into a single unit (class) while restricting direct access to the data.**

Think of a **capsule medicine**.

A capsule contains different medicines inside, but you cannot directly touch them.

Similarly,

* Data is hidden
* Access is controlled through methods

```
          +-----------------------+
          |       Employee        |
          +-----------------------+
          | - id                 |
          | - salary             |
          | - password           |
          +-----------------------+
          | + getSalary()        |
          | + setSalary()        |
          | + login()            |
          +-----------------------+
```

Everything is packaged together inside one class.

* * *

# Why Encapsulation?

Without encapsulation:

```
person.age = -50;
person.salary = -10000;
person.password = "";
```

Anyone can modify anything.

With encapsulation:

```
person.setAge(25);
person.setSalary(50000);
```

The class validates data before storing it.

Advantages:

✔ Data Security

✔ Data Validation

✔ Better Maintainability

✔ Loose Coupling

✔ Easier Code Changes

* * *

# Real World Example

Think of an ATM.

```
Customer
     |
     |
     V
 +-----------+
 |  ATM      |
 +-----------+
     |
     |
     V
 Bank Database
```

You cannot directly modify your bank balance.

You must use

* Withdraw
* Deposit
* Check Balance

These methods control access.

That is Encapsulation.

* * *

# Java Example

Without Encapsulation

```Java
class Student {
    public String name;
    public int age;
}

public class Main {
    public static void main(String[] args) {

        Student s = new Student();

        s.name = "Rahul";
        s.age = -20; // Invalid

        System.out.println(s.age);
    }
}
```

Output

```
-20
```

Bad design.

* * *

With Encapsulation

```Java
class Student {

    private String name;
    private int age;

    public void setAge(int age) {

        if(age > 0)
            this.age = age;
    }

    public int getAge() {
        return age;
    }
}

public class Main {

    public static void main(String[] args) {

        Student s = new Student();

        s.setAge(-10);

        System.out.println(s.getAge());
    }
}
```

Output

```
0
```

Invalid value was rejected.

* * *

# Access Modifiers

Access modifiers control who can access variables, methods, and classes.

There are **4 access modifiers**.

| Modifier | Same Class | Same Package | Subclass | Other Package |
| --- | --- | --- | --- | --- |
| private | ✅ | ❌ | ❌ | ❌ |
| default | ✅ | ✅ | ❌ | ❌ |
| protected | ✅ | ✅ | ✅ | ❌ _(except through inheritance)_ |
| public | ✅ | ✅ | ✅ | ✅ |

* * *

## 1. Private

Accessible only inside the same class.

```Java
class Employee{

    private int salary = 50000;

    public void display() {
        System.out.println(salary);
    }
}
```

Cannot do

```Java
Employee e = new Employee();

System.out.println(e.salary);
```

Compilation Error

* * *

## 2. Default (Package-Private)

No keyword.

```Java
class Student{

    int age;
}
```

Accessible only inside the same package.

Package A

```
Student.java
Teacher.java
```

Teacher can access Student.age.

Package B

```
Main.java
```

Cannot access.

* * *

## 3. Protected

Accessible

* Same package
* Child classes

```Java
class Animal{

    protected void sound(){

    }
}
```

Subclass

```Java
class Dog extends Animal{

    public void bark(){

        sound();
    }

}
```

Works.

* * *

## 4. Public

Accessible from anywhere.

```Java
public class Employee{

    public void display(){

    }

}
```

Anyone can call

```Java
Employee e = new Employee();

e.display();
```

* * *

# Visual Chart

```
                 ACCESS LEVEL

          PUBLIC
   +------------------+

      PROTECTED
   +--------------+

       DEFAULT
   +----------+

      PRIVATE
   +------+
```

Private is the most restrictive.

Public is the least restrictive.

* * *

# Getters and Setters

Encapsulation is usually achieved using

* Private variables
* Public getters
* Public setters

Example

```Java
class Employee{

    private String name;

    public String getName(){

        return name;
    }

    public void setName(String name){

        this.name = name;
    }

}
```

Usage

```Java
Employee e = new Employee();

e.setName("Suhas");

System.out.println(e.getName());
```

Output

```
Suhas
```

* * *

## Getter

Returns value.

```Java
public int getAge(){

    return age;
}
```

* * *

## Setter

Sets value.

```Java
public void setAge(int age){

    this.age = age;
}
```

* * *

# Why not make variables public?

Bad

```Java
public int age;
```

Anyone can do

```Java
age = -100;
```

Good

```Java
private int age;

public void setAge(int age){

    if(age > 0)
        this.age = age;
}
```

Validation added.

* * *

# Data Hiding vs Encapsulation

Many interviews ask this.

## Data Hiding

Hiding implementation details.

Achieved using

```
private variables
```

Example

```Java
private int salary;
```

Nobody can access it.

Purpose

Security.

* * *

## Encapsulation

Wrapping

* Variables
* Methods

into one class.

Example

```Java
class Employee{

    private int salary;

    public void setSalary(){

    }

    public int getSalary(){

    }

}
```

Purpose

Data + behavior together.

* * *

## Difference Table

| Data Hiding | Encapsulation |
| --- | --- |
| Hides data | Wraps data and methods |
| Achieved using private keyword | Achieved using class |
| Focuses on security | Focuses on design |
| Part of encapsulation | Broader OOP concept |

* * *

Simple memory trick

```
Data Hiding
↓

Hide Data

Encapsulation
↓

Hide + Wrap + Control
```

* * *

# Tight Encapsulation

A class is tightly encapsulated if

Every variable is private.

Example

```Java
class Employee{

    private int id;

    private String name;

    private double salary;

}
```

All fields are private.

This is Tight Encapsulation.

* * *

Another example

```Java
class Student{

    private int age;

    private String name;

    private String college;

}
```

Perfect.

* * *

# Loose Encapsulation

One or more variables are

* protected
* default
* public

Example

```Java
class Employee{

    private int id;

    public String name;

}
```

Not tightly encapsulated.

* * *

Another example

```Java
class Student{

    protected int age;

    private String name;

}
```

Also loosely encapsulated.

* * *

# Chart

```
Tightly Encapsulated

class Student{

private int age;
private int id;
private String name;

}

      ✓ Correct
```

```
Loosely Encapsulated

class Student{

private int age;
public int id;

}

      ✗ Not Tight
```

* * *

# Can a tightly encapsulated class extend a loosely encapsulated class?

Example

```Java
class Parent{

    public int x;
}

class Child extends Parent{

    private int y;
}
```

Is Child tightly encapsulated?

**No.**

Reason:

`Child` inherits the public variable `x`, so the object still exposes public state.

* * *

# Best Practices

✔ Make instance variables `private`.

✔ Expose only necessary methods.

✔ Validate input in setters.

✔ Avoid public fields.

✔ Make objects immutable when possible (use `private final` fields and no setters where appropriate).

* * *

# Interview Questions

### 1. What is encapsulation?

Wrapping data and methods into a single unit while controlling access to the data using access modifiers.

* * *

### 2. How is encapsulation achieved in Java?

* Private variables
* Public getters/setters
* Access modifiers

* * *

### 3. Why use private variables?

To prevent unauthorized access and enforce validation.

* * *

### 4. Difference between encapsulation and abstraction?

| Encapsulation | Abstraction |
| --- | --- |
| Hides data | Hides implementation details |
| Achieved using classes and access modifiers | Achieved using interfaces and abstract classes |
| Focuses on data protection | Focuses on exposing only essential functionality |

* * *

### 5. Can encapsulation exist without getters and setters?

**Yes.** Getters and setters are common, but not mandatory. A class can expose behavior through other public methods (e.g., `deposit()`, `withdraw()`) while keeping its fields private.

* * *

### 6. Why are setters useful?

They allow validation, logging, transformation, or other business rules before updating an object's state.

* * *

# Summary Mind Map

```
                 ENCAPSULATION
                        |
        ----------------------------------
        |               |                |
    private         Getters        Setters
        |
        |
   Data Hiding
        |
        |
 Access Modifiers
        |
 ---------------------------------------------
 |          |           |                    |
Private   Default    Protected            Public
        |
        |
 Tight Encapsulation
        |
 All Variables Private
        |
 Better Security
        |
 Better Maintainability
```

This covers the topic in the depth typically expected for **SDE-1 Java interviews**, including concepts, diagrams, code examples, common interview questions, and best practices.

* * *

**You:**

b) Abstraction
- Hiding implementation, exposing behavior
- Abstract class vs Interface as abstraction tools
- Real-world design examples

* * *

**ChatGPT:**

# Abstraction in Java (Complete Interview Guide)

Abstraction is one of the **four pillars of Object-Oriented Programming (OOP)**.

```
OOP
├── Encapsulation
├── Inheritance
├── Polymorphism
└── Abstraction
```

* * *

# What is Abstraction?

> **Abstraction is the process of hiding implementation details and exposing only the essential behavior to the user.**

In simple words:

> **The user knows WHAT an object does, but not HOW it does it.**

* * *

# Real-World Example 1: Car

When you drive a car:

```
       Driver
          |
          |
      Press Start Button
          |
          V
     +--------------+
     |   Car Engine |
     +--------------+
          |
          |
   Fuel Injection
   Spark Plug
   Battery
   ECU
   Sensors
```

The driver only knows:

```
Start()
Stop()
Accelerate()
Brake()
```

The driver does **not** know:

* Fuel injection process
* Spark timing
* Engine combustion
* ECU calculations

These are hidden.

This is **Abstraction**.

* * *

# Real-World Example 2: ATM

```
      Customer
          |
      Withdraw()
          |
      +---------+
      |   ATM   |
      +---------+
          |
      Bank Server
          |
 Authentication
 Balance Check
 Cash Calculation
 Transaction Logging
```

Customer only knows:

```
Withdraw
Deposit
Check Balance
```

Implementation is hidden.

* * *

# Real-World Example 3: Mobile Phone

You click

```
Camera
```

Internally,

```
Open Lens
Focus
Adjust Brightness
Capture Frame
Apply AI Processing
Store Image
```

You don't know any of these.

You only know

```
Take Photo
```

* * *

# Definition

```
Abstraction

↓

Hide Internal Logic

↓

Expose Necessary Operations
```

* * *

# Why Do We Need Abstraction?

Suppose a TV has 500 internal operations.

Without abstraction:

```
Turn Power Supply On
Check Voltage
Enable Processor
Initialize Display
Load Drivers
Connect Speakers
...
```

Impossible for users.

Instead,

```
Power Button
```

One method.

* * *

Advantages

✔ Simplicity

✔ Security

✔ Easy Maintenance

✔ Loose Coupling

✔ Reusability

* * *

# Abstraction in Java

Java provides abstraction using

1. Abstract Class
2. Interface

```
          Abstraction
                |
      --------------------
      |                  |
Abstract Class      Interface
```

* * *

# Abstract Class

An abstract class is a class that **cannot be instantiated** and may contain both **implemented** and **abstract** methods.

Syntax

```Java
abstract class Animal {

    abstract void sound();

}
```

Cannot create object

```Java
Animal a = new Animal();
```

Compilation Error

* * *

# Example

```Java
abstract class Animal {

    abstract void sound();

    public void sleep() {
        System.out.println("Sleeping");
    }
}

class Dog extends Animal {

    @Override
    public void sound() {
        System.out.println("Bark");
    }
}

public class Main {

    public static void main(String[] args) {

        Dog d = new Dog();

        d.sound();
        d.sleep();
    }
}
```

Output

```
Bark
Sleeping
```

* * *

# Why Abstract Method?

```Java
abstract void sound();
```

Every animal has a sound.

But implementation differs.

Dog

```
Bark
```

Cat

```
Meow
```

Lion

```
Roar
```

The parent doesn't know implementation.

Children provide it.

* * *

# Diagram

```
           Animal
      ----------------
      sound()   (?)
      sleep()
           |
 -------------------------
 |            |           |
Dog          Cat        Lion
 |            |           |
Bark()      Meow()     Roar()
```

* * *

# Rules of Abstract Class

### Rule 1

Cannot create object.

```Java
Animal a = new Animal();
```

Not allowed.

* * *

### Rule 2

Can have constructors.

```Java
abstract class Animal{

    Animal(){

        System.out.println("Constructor");
    }

}
```

Called when subclass object is created.

* * *

### Rule 3

Can have variables.

```Java
abstract class Animal{

    int age;

}
```

* * *

### Rule 4

Can have implemented methods.

```Java
public void eat(){

}
```

* * *

### Rule 5

Can have abstract methods.

```Java
abstract void sound();
```

* * *

### Rule 6

Can have static methods.

```Java
static void test(){

}
```

* * *

### Rule 7

Can have final methods.

```Java
final void display(){

}
```

Cannot override.

* * *

# Interface

Interface is a contract.

It tells

> "These methods must be implemented."

```
Interface

↓

100% Contract

↓

Classes must implement it
```

Example

```Java
interface Vehicle {

    void start();

}
```

Implementation

```Java
class Car implements Vehicle {

    @Override
    public void start() {

        System.out.println("Car Started");
    }

}
```

* * *

# Complete Example

```Java
interface Payment {

    void pay(double amount);

}

class CreditCardPayment implements Payment {

    @Override
    public void pay(double amount) {

        System.out.println("Paid using Credit Card : " + amount);

    }
}

class UpiPayment implements Payment {

    @Override
    public void pay(double amount) {

        System.out.println("Paid using UPI : " + amount);

    }
}

public class Main {

    public static void main(String[] args) {

        Payment payment = new UpiPayment();

        payment.pay(5000);

    }
}
```

Output

```
Paid using UPI : 5000
```

Notice that the caller depends on the `Payment` interface, not a specific implementation.

* * *

# Interface Diagram

```
              Payment
          +--------------+
          | pay()        |
          +--------------+
            /          \
           /            \
 CreditCard          UPI
     |                 |
 pay()              pay()
```

* * *

# Abstract Class vs Interface

| Feature | Abstract Class | Interface |
| --- | --- | --- |
| Object creation | ❌ No | ❌ No |
| Constructors | ✅ Yes | ❌ No |
| Instance variables | ✅ Yes | ❌ Only `public static final` constants |
| Abstract methods | ✅ Yes | ✅ Yes |
| Concrete methods | ✅ Yes | ✅ Yes (`default` and `static` methods since Java 8) |
| Inheritance | One class can extend only one abstract class | A class can implement multiple interfaces |
| Keyword | `extends` | `implements` |
| Purpose | Share common state and partial implementation | Define a capability or contract |

* * *

# When to Use Abstract Class?

Use an abstract class when:

* Classes share common code.
* You want common fields.
* You want constructors.
* There is an **"is-a"** relationship.

Example

```
Animal

↓

Dog
Cat
Lion
```

All animals

* eat
* sleep
* breathe

Common code belongs in the abstract class.

* * *

# When to Use Interface?

Use an interface when different classes share a capability, regardless of their inheritance hierarchy.

Example

```
Flyable

↓

Bird
Airplane
Drone
```

These are not the same kind of object.

But all can fly.

* * *

# Real-World Design Example 1: Payment Gateway

```
                Payment
            +--------------+
            | pay()        |
            +--------------+
             /     |      \
            /      |       \
        UPI   CreditCard   NetBanking
```

Application code

```Java
Payment payment = new UpiPayment();

payment.pay(1000);
```

Tomorrow,

```
ApplePay

GooglePay

PayPal
```

can be added without changing existing client code.

* * *

# Real-World Design Example 2: Notification System

```
            Notification
        +------------------+
        | send(message)    |
        +------------------+
          /      |        \
         /       |         \
     Email      SMS      Push
```

Application

```Java
Notification notification = new EmailNotification();

notification.send("Welcome");
```

Easy to switch to SMS or Push notifications.

* * *

# Real-World Design Example 3: Database Layer

```
              Database
         +----------------+
         | connect()      |
         | save()         |
         +----------------+
            /        \
           /          \
     MySQL          PostgreSQL
```

Business logic

```Java
Database db = new MySQLDatabase();
```

Later

```Java
Database db = new PostgreSQLDatabase();
```

Only one line changes.

* * *

# Real-World Design Example 4: Shape Drawing

```Java
abstract class Shape {

    abstract void draw();

}
```

```
Shape
 |
 |-----------------------------
 |             |              |
Circle      Rectangle      Triangle
```

Each implements

```
draw()
```

differently.

* * *

# Abstraction + Polymorphism Together

```Java
interface Animal {

    void sound();
}

class Dog implements Animal {

    public void sound() {
        System.out.println("Bark");
    }
}

class Cat implements Animal {

    public void sound() {
        System.out.println("Meow");
    }
}
```

Client

```Java
Animal animal = new Dog();

animal.sound();
```

Output

```
Bark
```

If changed to

```Java
animal = new Cat();
```

Output

```
Meow
```

The client only knows the abstraction (`Animal`), while runtime polymorphism selects the correct implementation.

* * *

# Abstraction vs Encapsulation

| Abstraction | Encapsulation |
| --- | --- |
| Hides implementation | Hides data |
| Focuses on _what_ an object does | Focuses on _how data is protected_ |
| Achieved using abstract classes and interfaces | Achieved using classes, access modifiers, getters/setters |
| Simplifies usage | Secures object state |

**Memory Trick**

```
Abstraction
↓

"What can this object do?"

Encapsulation
↓

"How is this object's data protected?"
```

* * *

# Interview Questions

### 1. What is abstraction?

Hiding implementation details while exposing only the required functionality.

* * *

### 2. How is abstraction achieved in Java?

Using:

* Abstract classes
* Interfaces

* * *

### 3. Can an abstract class have constructors?

Yes. They are executed when a subclass object is created.

* * *

### 4. Can an interface have method implementations?

Yes. Since Java 8, interfaces can have `default` and `static` methods, and since Java 9 they can also have `private` helper methods.

* * *

### 5. Which should you prefer: abstract class or interface?

* Use an **abstract class** when related classes share common state and implementation.
* Use an **interface** to define capabilities or contracts that unrelated classes can implement.

* * *

### 6. Can a class extend multiple abstract classes?

No. Java supports single class inheritance.

* * *

### 7. Can a class implement multiple interfaces?

Yes.

```Java
class SmartPhone implements Camera, MusicPlayer, GPS {
    // Implement required methods
}
```

This is a common way to achieve multiple inheritance of behavior in Java.

* * *

# Summary Mind Map

```
                    ABSTRACTION
                          |
            -----------------------------
            |                           |
     Hide Implementation         Expose Behavior
            |
     -----------------------
     |                     |
 Abstract Class        Interface
     |                     |
 Shared State        Contract/Capability
 Constructors        Multiple Implementation
 Common Code         Loose Coupling
            |
      Runtime Polymorphism
            |
   Flexible & Maintainable Design
```

## SDE-1 Interview Tip

A common interview question is: **"Why do frameworks like Spring use interfaces so much?"**

The answer is that coding against interfaces enables loose coupling, easier testing (mock implementations), dependency injection, and the ability to swap implementations without changing client code. This is a practical application of abstraction in enterprise Java development.

* * *

**You:**

d) Inheritance
- Types — Single, Multilevel, Hierarchical, Multiple, Hybrid
- super / base keyword
- Constructor chaining
- Method hiding vs Method overriding
- Diamond problem & resolution

* * *

**ChatGPT:**

# Inheritance in Java (Complete Interview Guide)

Inheritance is one of the **four pillars of Object-Oriented Programming (OOP)**.

```
OOP
├── Encapsulation
├── Abstraction
├── Inheritance
└── Polymorphism
```

* * *

# What is Inheritance?

> **Inheritance is the mechanism by which one class acquires the properties (fields) and behaviors (methods) of another class.**

It allows **code reuse**, **extensibility**, and forms an **IS-A relationship**.

Example:

```
Dog IS-A Animal
Car IS-A Vehicle
SavingsAccount IS-A BankAccount
```

Java uses the **`extends`** keyword for class inheritance.

* * *

# Why Do We Need Inheritance?

Without inheritance:

```Java
class Dog {
    void eat() {}
    void sleep() {}
    void breathe() {}
}

class Cat {
    void eat() {}
    void sleep() {}
    void breathe() {}
}
```

Lots of duplicate code.

With inheritance:

```Java
class Animal {

    void eat() {
        System.out.println("Eating");
    }

    void sleep() {
        System.out.println("Sleeping");
    }
}

class Dog extends Animal {

}

class Cat extends Animal {

}
```

Now both classes automatically inherit `eat()` and `sleep()`.

* * *

# Real World Example

```
             Vehicle
          ----------------
          speed
          color
          start()
          stop()
                |
      -----------------------
      |                     |
     Car                 Bike
```

Car and Bike inherit common features from Vehicle.

* * *

# Syntax

```Java
class Parent {

}

class Child extends Parent {

}
```

Child automatically gets access to Parent's **non-private** members.

* * *

# Types of Inheritance

Java supports some inheritance types directly and others indirectly through interfaces.

```
               Inheritance
                     |
------------------------------------------------
|        |            |          |             |
Single Multilevel Hierarchical Multiple Hybrid
```

* * *

# 1. Single Inheritance

One parent → One child

```
Animal
   |
 Dog
```

Example

```Java
class Animal {

    void eat() {
        System.out.println("Eating");
    }
}

class Dog extends Animal {

    void bark() {
        System.out.println("Barking");
    }
}

public class Main {

    public static void main(String[] args) {

        Dog d = new Dog();

        d.eat();
        d.bark();
    }
}
```

Output

```
Eating
Barking
```

* * *

# 2. Multilevel Inheritance

A chain of inheritance.

```
Animal
   |
 Mammal
   |
 Dog
```

Example

```Java
class Animal {

    void eat() {
        System.out.println("Eating");
    }
}

class Mammal extends Animal {

    void walk() {
        System.out.println("Walking");
    }
}

class Dog extends Mammal {

    void bark() {
        System.out.println("Barking");
    }
}
```

Dog inherits:

* eat()
* walk()
* bark()

* * *

# 3. Hierarchical Inheritance

One parent → Multiple children

```
            Animal
          /    |    \
       Dog    Cat   Lion
```

Example

```Java
class Animal {

    void eat() {
        System.out.println("Eating");
    }
}

class Dog extends Animal {

}

class Cat extends Animal {

}

class Lion extends Animal {

}
```

All subclasses inherit `eat()`.

* * *

# 4. Multiple Inheritance

Multiple parents → One child

```
        A        B
         \      /
          \    /
           \  /
            C
```

### Does Java support this for classes?

**No.**

This is **not allowed**:

```Java
class A {

}

class B {

}

class C extends A, B {

}
```

Compilation Error.

* * *

# Why?

Because of the **Diamond Problem**, explained later.

* * *

# 5. Hybrid Inheritance

Combination of multiple inheritance types.

```
      A
     / \
    B   C
     \ /
      D
```

Java **does not support hybrid inheritance using classes** because it requires multiple inheritance of classes.

However, hybrid designs are possible using **interfaces**.

* * *

# Summary Table

| Type | Supported in Java Classes? |
| --- | --- |
| Single | ✅ Yes |
| Multilevel | ✅ Yes |
| Hierarchical | ✅ Yes |
| Multiple | ❌ No |
| Hybrid | ❌ No (with classes) / ✅ Yes (using interfaces) |

* * *

# The `super` Keyword

`super` refers to the **immediate parent class object**.

It is used to:

* Access parent variables
* Call parent methods
* Invoke parent constructors

* * *

## 1. Access Parent Variable

```Java
class Animal {

    String color = "White";
}

class Dog extends Animal {

    String color = "Black";

    void display() {

        System.out.println(super.color);
        System.out.println(this.color);
    }
}
```

Output

```
White
Black
```

* * *

## 2. Call Parent Method

```Java
class Animal {

    void sound() {
        System.out.println("Animal Sound");
    }
}

class Dog extends Animal {

    void sound() {
        System.out.println("Dog Bark");
    }

    void display() {

        super.sound();
        sound();
    }
}
```

Output

```
Animal Sound
Dog Bark
```

* * *

## 3. Call Parent Constructor

```Java
class Animal {

    Animal() {
        System.out.println("Animal Constructor");
    }
}

class Dog extends Animal {

    Dog() {

        super();

        System.out.println("Dog Constructor");
    }
}
```

Output

```
Animal Constructor
Dog Constructor
```

If `super()` is omitted, Java inserts it automatically as the first statement.

* * *

# `this` vs `super`

| `this` | `super` |
| --- | --- |
| Refers to current object | Refers to immediate parent |
| Calls current constructor using `this()` | Calls parent constructor using `super()` |
| Accesses current class members | Accesses parent class members |

* * *

# Constructor Chaining

Whenever an object is created, constructors execute **from parent to child**.

Example

```Java
class A {

    A() {
        System.out.println("A");
    }
}

class B extends A {

    B() {
        System.out.println("B");
    }
}

class C extends B {

    C() {
        System.out.println("C");
    }
}

public class Main {

    public static void main(String[] args) {

        new C();
    }
}
```

Execution flow

```
A
↓
B
↓
C
```

Output

```
A
B
C
```

* * *

# Constructor Chaining Diagram

```
new C()

↓

C()

↓

super()

↓

B()

↓

super()

↓

A()
```

* * *

# Rules of Constructor Chaining

* `super()` must be the **first statement** in a constructor.
* If not written explicitly, Java inserts it automatically.
* If the parent has **no default constructor**, the child must call the appropriate parent constructor explicitly.

Example

```Java
class Parent {

    Parent(int x) {

    }
}

class Child extends Parent {

    Child() {

        super(10);

    }
}
```

* * *

# Method Overriding

A child class provides a new implementation of a parent method using the **same method signature**.

Example

```Java
class Animal {

    void sound() {
        System.out.println("Animal Sound");
    }
}

class Dog extends Animal {

    @Override
    void sound() {
        System.out.println("Bark");
    }
}
```

Usage

```Java
Animal a = new Dog();

a.sound();
```

Output

```
Bark
```

This is **runtime polymorphism**.

* * *

# Method Hiding

Method hiding applies only to **static methods**.

Example

```Java
class Parent {

    static void display() {
        System.out.println("Parent");
    }
}

class Child extends Parent {

    static void display() {
        System.out.println("Child");
    }
}
```

Usage

```Java
Parent p = new Child();

p.display();
```

Output

```
Parent
```

Because static methods are resolved at **compile time** using the reference type.

* * *

# Method Hiding vs Method Overriding

| Method Hiding | Method Overriding |
| --- | --- |
| Applies to static methods | Applies to instance methods |
| Compile-time binding | Runtime binding |
| Based on reference type | Based on actual object |
| No runtime polymorphism | Supports runtime polymorphism |

* * *

# Variable Hiding

Fields are **hidden**, not overridden.

```Java
class Parent {

    int x = 10;
}

class Child extends Parent {

    int x = 20;
}

public class Main {

    public static void main(String[] args) {

        Parent p = new Child();

        System.out.println(p.x);
    }
}
```

Output

```
10
```

Field access depends on the **reference type**, unlike overridden methods.

* * *

# Diamond Problem

Suppose Java allowed multiple inheritance of classes.

```
        Animal
        /    \
       /      \
      Dog     Cat
        \    /
         \  /
       Hybrid
```

Both `Dog` and `Cat` override `sound()`.

Question:

```
Hybrid.sound()

↓

Should it call Dog.sound()?

OR

Cat.sound()?
```

This ambiguity is called the **Diamond Problem**.

* * *

# Why Java Doesn't Allow Multiple Class Inheritance

Imagine:

```Java
class A {

    void display() {
        System.out.println("A");
    }
}

class B extends A {

    void display() {
        System.out.println("B");
    }
}

class C extends A {

    void display() {
        System.out.println("C");
    }
}

// Hypothetical (not allowed)
class D extends B, C {

}
```

If `D.display()` is called, should it execute:

* `B.display()`
* `C.display()`

Java avoids this ambiguity by disallowing multiple inheritance of classes.

* * *

# Diamond Problem with Interfaces

Since Java 8, interfaces can have `default` methods.

Example

```Java
interface A {

    default void show() {
        System.out.println("A");
    }
}

interface B {

    default void show() {
        System.out.println("B");
    }
}

class C implements A, B {

    @Override
    public void show() {
        System.out.println("Resolved");
    }
}
```

The class **must override** `show()` to resolve the conflict.

You can also explicitly call one interface's default method:

```Java
A.super.show();
```

or

```Java
B.super.show();
```

This is Java's resolution to the interface version of the diamond problem.

* * *

# Real-World Example: Employee Hierarchy

```
               Employee
          -----------------
          id
          name
          salary
          work()
                |
     ------------------------
     |                      |
Developer              Tester
     |                      |
writeCode()          testSoftware()
```

Common functionality stays in `Employee`, while specialized behavior is added in subclasses.

* * *

# Best Practices

* Favor inheritance only when there is a true **IS-A** relationship.
* Keep parent classes focused on common behavior.
* Prefer **composition over inheritance** if the relationship is **HAS-A**.
* Use `@Override` when overriding methods.
* Avoid exposing unnecessary implementation details in parent classes.
* Do not override `final` methods.

* * *

# Interview Questions

### 1. What is inheritance?

Inheritance allows one class to acquire the properties and behaviors of another class, promoting code reuse and extensibility.

* * *

### 2. Which inheritance types are supported in Java?

* ✅ Single
* ✅ Multilevel
* ✅ Hierarchical
* ❌ Multiple (classes)
* ❌ Hybrid (classes)
* ✅ Multiple and hybrid behavior through interfaces

* * *

### 3. What is the `super` keyword?

It refers to the immediate parent class and is used to access parent members or invoke parent constructors.

* * *

### 4. Why doesn't Java support multiple inheritance of classes?

To avoid ambiguity caused by the **Diamond Problem**.

* * *

### 5. What is constructor chaining?

The automatic sequence in which parent constructors execute before child constructors during object creation.

* * *

### 6. Difference between method hiding and method overriding?

| Feature | Hiding | Overriding |
| --- | --- | --- |
| Method type | Static | Instance |
| Binding | Compile time | Runtime |
| Polymorphism | ❌ No | ✅ Yes |

* * *

### 7. What is the execution order of constructors?

```
Grandparent Constructor
        ↓
Parent Constructor
        ↓
Child Constructor
```

The parent constructor always executes before the child constructor.

* * *

# Summary Mind Map

```
                     INHERITANCE
                          |
          ----------------------------------
          |               |                |
       Code Reuse      IS-A Relation   Extensibility
                          |
     ---------------------------------------------
     |        |          |         |             |
  Single  Multilevel Hierarchical Multiple   Hybrid
     |                                     (Interfaces)
                          |
                       super
                          |
          -------------------------------
          |              |              |
     Variables        Methods     Constructors
                          |
                 Constructor Chaining
                          |
          --------------------------------
          |                              |
   Method Overriding             Method Hiding
   (Instance Methods)            (Static Methods)
                          |
                   Diamond Problem
                          |
        No Multiple Inheritance of Classes
        Interfaces resolve default-method conflicts by overriding
```

### SDE-1 Interview Tip

A frequent design question is: **"When should you use inheritance versus composition?"**

* Use **inheritance** when there is a genuine **IS-A** relationship (e.g., `Dog extends Animal`).
* Use **composition** when there is a **HAS-A** relationship (e.g., `Car has an Engine`).

Modern Java design (including Spring) generally favors **composition over inheritance** because it leads to more flexible, loosely coupled code.

* * *

**You:**

c) Polymorphism
- Compile-time (Static) — Method Overloading, Operator Overloading
- Runtime (Dynamic) — Method Overriding
- virtual keyword / @Override
- Early binding vs Late binding
- Covariant return types

* * *

**ChatGPT:**

# Polymorphism in Java (Complete Interview Guide)

Polymorphism is one of the **four pillars of Object-Oriented Programming (OOP).**

```
                 OOP
                  |
------------------------------------------------
|              |               |               |
Encapsulation  Abstraction   Inheritance   Polymorphism
```

The word **Polymorphism** comes from two Greek words:

* **Poly** = Many
* **Morph** = Forms

> **Polymorphism means "One interface, many implementations."**

The same method can behave differently depending on the object that invokes it.

* * *

# What is Polymorphism?

Suppose you have a method called:

```Java
draw();
```

Depending on the object,

```
Circle.draw()     → Draw Circle

Rectangle.draw()  → Draw Rectangle

Triangle.draw()   → Draw Triangle
```

Same method

Different behavior.

This is Polymorphism.

* * *

# Real World Example 1

### Human Behavior

```
Person

↓

Speak()
```

Different people speak differently.

```
English
Hindi
Kannada
French
```

Same action.

Different implementation.

* * *

# Real World Example 2

Payment System

```
             Payment
                |
      ----------------------
      |          |         |
     UPI      Card      NetBanking
```

Application

```Java
payment.pay(1000);
```

Internally

```
UPI → QR Payment

Card → Card Gateway

NetBanking → Bank Portal
```

User doesn't care.

Same method.

Different implementation.

* * *

# Types of Polymorphism

Java supports two types.

```
              Polymorphism
                    |
        ------------------------
        |                      |
 Compile Time            Runtime
   (Static)              (Dynamic)
        |                      |
 Method Overloading     Method Overriding
```

* * *

# Compile-Time Polymorphism (Static)

Also called

* Static Polymorphism
* Early Binding

Achieved using

* Method Overloading

(Java **does not support true operator overloading** for user-defined classes.)

* * *

# Method Overloading

Two or more methods having

* Same name
* Different parameters

Example

```Java
class Calculator {

    int add(int a, int b) {

        return a + b;
    }

    int add(int a, int b, int c) {

        return a + b + c;
    }

    double add(double a, double b) {

        return a + b;
    }
}
```

Usage

```Java
Calculator c = new Calculator();

System.out.println(c.add(10,20));

System.out.println(c.add(10,20,30));

System.out.println(c.add(10.5,20.5));
```

Output

```
30

60

31.0
```

Compiler decides which method to call.

Hence

Compile-Time Polymorphism.

* * *

# How Compiler Chooses?

Compiler checks

```
Method Name

+

Number of Parameters

+

Data Type

+

Order of Parameters
```

Example

```Java
display(int x)

display(String x)

display(int x, int y)
```

All are valid overloads.

* * *

# Invalid Overloading

Changing only return type

❌ Not allowed

```Java
int add(int a,int b){

}

double add(int a,int b){

}
```

Compilation Error.

Reason

Compiler cannot distinguish methods by return type alone.

* * *

# Valid Overloading

Different parameter list

```Java
void print()

void print(int x)

void print(double x)

void print(String x)
```

Perfectly valid.

* * *

# Rules of Method Overloading

Methods can differ by:

✔ Number of parameters

✔ Type of parameters

✔ Order of parameters

Example

```Java
void display(int,String)

void display(String,int)
```

Valid.

* * *

Cannot differ only by

```
Return Type
```

* * *

# Constructor Overloading

Constructors can also be overloaded.

```Java
class Student{

    Student(){

    }

    Student(String name){

    }

    Student(String name,int age){

    }

}
```

Different constructors

Different parameter list.

* * *

# Operator Overloading

Many languages like C++ support operator overloading.

Example

```C++
Complex c3 = c1 + c2;
```

Java **does not allow user-defined operator overloading.**

However,

Java internally overloads **`+`**.

Example

Numbers

```Java
System.out.println(10 + 20);
```

Output

```
30
```

Strings

```Java
System.out.println("Hello " + "Java");
```

Output

```
Hello Java
```

Mixed

```Java
System.out.println("Age : " + 25);
```

Output

```
Age : 25
```

Internally

```Java
StringBuilder.append()
```

is used by the compiler.

So,

Java provides **limited operator overloading** only for `+`.

* * *

# Runtime Polymorphism

Also called

* Dynamic Polymorphism
* Late Binding

Achieved using

```
Method Overriding
```

* * *

# Method Overriding

Child class provides its own implementation of parent method.

Example

```Java
class Animal{

    void sound(){

        System.out.println("Animal Sound");

    }

}

class Dog extends Animal{

    @Override
    void sound(){

        System.out.println("Bark");

    }

}
```

Usage

```Java
Animal a = new Dog();

a.sound();
```

Output

```
Bark
```

Runtime decides

Which method to execute.

* * *

# Runtime Diagram

```
Reference

Animal a

       |

       V

Actual Object

Dog

↓

Dog.sound()
```

Reference type

≠

Actual object.

Java uses

Actual object.

* * *

# Another Example

```Java
class Shape{

    void draw(){

        System.out.println("Shape");

    }

}

class Circle extends Shape{

    @Override
    void draw(){

        System.out.println("Circle");

    }

}

class Rectangle extends Shape{

    @Override
    void draw(){

        System.out.println("Rectangle");

    }

}
```

```Java
Shape s = new Circle();

s.draw();
```

Output

```
Circle
```

* * *

Later

```Java
s = new Rectangle();

s.draw();
```

Output

```
Rectangle
```

Same reference.

Different behavior.

* * *

# @Override Annotation

`@Override` tells the compiler

"This method is intended to override a superclass method."

Example

```Java
class Animal{

    void sound(){

    }

}

class Dog extends Animal{

    @Override
    void sound(){

    }

}
```

Benefits

✔ Better readability

✔ Compile-time checking

✔ Prevents spelling mistakes

* * *

Example

```Java
class Animal{

    void sound(){

    }

}

class Dog extends Animal{

    @Override
    void sounds(){

    }

}
```

Compiler Error

Because

Parent has

```
sound()
```

Not

```
sounds()
```

Without `@Override`, this mistake would silently create a new method instead of overriding.

* * *

# Virtual Keyword

Many languages like **C++** require the `virtual` keyword for runtime polymorphism.

Example (C++)

```C++
virtual void sound();
```

Java has **no `virtual` keyword**.

**Every non-static, non-final, non-private instance method is virtual by default.**

Therefore

```Java
void sound(){

}
```

is automatically virtual.

* * *

Methods that **cannot** be overridden

```
private

final

static
```

These are **not** dynamically dispatched.

* * *

# Early Binding vs Late Binding

Very common interview question.

* * *

## Early Binding

Binding happens

During compilation.

Example

```Java
class Test{

    void add(int a,int b){

    }

    void add(int a,int b,int c){

    }

}
```

Compiler already knows

Which method to call.

* * *

Also applies to

```
Static methods

Final methods

Private methods

Method Overloading
```

* * *

## Late Binding

Binding happens

At runtime.

Example

```Java
Animal a = new Dog();

a.sound();
```

Compiler knows only

```
Animal
```

Runtime sees

```
Dog
```

Calls

```
Dog.sound()
```

* * *

# Comparison

| Early Binding | Late Binding |
| --- | --- |
| Compile Time | Runtime |
| Faster | Slightly slower |
| Method Overloading | Method Overriding |
| Static methods | Instance methods |
| No runtime polymorphism | Runtime polymorphism |

* * *

# Method Overloading vs Overriding

| Overloading | Overriding |
| --- | --- |
| Same class (usually) | Parent & Child |
| Compile Time | Runtime |
| Different parameters | Same parameters |
| No inheritance required | Inheritance required |
| Static binding | Dynamic binding |

* * *

# Dynamic Method Dispatch

One of the most important concepts.

```Java
class Animal{

    void sound(){

        System.out.println("Animal");

    }

}

class Dog extends Animal{

    @Override
    void sound(){

        System.out.println("Dog");

    }

}

class Cat extends Animal{

    @Override
    void sound(){

        System.out.println("Cat");

    }

}
```

Main

```Java
Animal a;

a = new Dog();

a.sound();

a = new Cat();

a.sound();
```

Output

```
Dog

Cat
```

Java dynamically dispatches the method based on the actual object.

* * *

# Covariant Return Types

Before Java 5,

Overridden methods had to return the **exact same type**.

Since Java 5,

Child class may return a **subclass** of the parent's return type.

This is called

> **Covariant Return Type**

* * *

Example

```Java
class Animal{

}

class Dog extends Animal{

}
```

Parent

```Java
class Parent{

    Animal getAnimal(){

        return new Animal();

    }

}
```

Child

```Java
class Child extends Parent{

    @Override
    Dog getAnimal(){

        return new Dog();

    }

}
```

Perfectly valid.

Because

```
Dog IS-A Animal
```

* * *

# Another Example

```Java
class Vehicle{

}

class Car extends Vehicle{

}
```

Parent

```Java
Vehicle create(){

    return new Vehicle();

}
```

Child

```Java
Car create(){

    return new Car();

}
```

Valid.

* * *

# Invalid Covariant Return

```Java
class Parent{

    Animal get(){

        return new Animal();

    }

}

class Child extends Parent{

    @Override
    String get(){

        return "Hello";

    }

}
```

Compilation Error.

Because

```
String

IS NOT

Animal
```

* * *

# Real World Example

Factory Pattern

```
           Vehicle Factory
                 |
      ------------------------
      |                      |
    Car                  Bike
```

Parent

```Java
Vehicle createVehicle();
```

Child

```Java
Car createVehicle();
```

Perfect example of covariant return type.

* * *

# Complete Polymorphism Flow

```
                   Animal
                sound()
                    |
         ------------------------
         |          |           |
        Dog       Cat         Lion
         |          |           |
      Bark()     Meow()      Roar()

Animal a = new Dog();

↓

Runtime

↓

Dog.sound()

↓

Bark
```

* * *

# Best Practices

✔ Always use `@Override` when overriding methods.

✔ Program to interfaces or abstract classes, not concrete implementations.

✔ Use method overloading only when methods perform similar operations.

✔ Prefer runtime polymorphism for extensible designs.

✔ Do not overload methods with confusing parameter combinations.

* * *

# Interview Questions

### 1. What is polymorphism?

Polymorphism means **one interface, many implementations**, allowing the same method call to exhibit different behaviors depending on the object.

* * *

### 2. How many types of polymorphism exist in Java?

* Compile-time (Static)
* Runtime (Dynamic)

* * *

### 3. How is compile-time polymorphism achieved?

Using:

* Method Overloading
* Constructor Overloading

(Java does not support user-defined operator overloading.)

* * *

### 4. How is runtime polymorphism achieved?

Using **Method Overriding** and **Dynamic Method Dispatch**.

* * *

### 5. Does Java support operator overloading?

No, except for the built-in behavior of the `+` operator with strings and primitive numeric addition.

* * *

### 6. Why use `@Override`?

* Ensures the method actually overrides a parent method.
* Prevents accidental method overloading due to spelling or signature mistakes.
* Improves code readability.

* * *

### 7. Does Java have a `virtual` keyword?

No.

All **non-static, non-final, non-private instance methods are virtual by default**, enabling runtime polymorphism.

* * *

### 8. What is a covariant return type?

An overridden method can return a **subclass** of the original return type.

Example:

```Java
Parent: Animal get();

Child : Dog get();
```

Since `Dog` extends `Animal`, this is valid.

* * *

### 9. Why is runtime polymorphism important?

It allows code to be **extensible**, **loosely coupled**, and **open for new implementations without modifying existing client code**.

* * *

# Summary Mind Map

```
                      POLYMORPHISM
                             |
          ---------------------------------------
          |                                     |
   Compile-Time                          Runtime
 (Static Polymorphism)             (Dynamic Polymorphism)
          |                                     |
   Method Overloading                  Method Overriding
          |                                     |
     Early Binding                     Late Binding
          |                                     |
   Constructor Overloading          Dynamic Method Dispatch
          |                                     |
     Limited '+' Support             @Override Annotation
    (Operator Overloading)                  |
                                             |
                                  Covariant Return Types
                                             |
                                 One Interface, Many Forms
```

* * *

# SDE-1 Interview Tip

A very common interview question is:

> **"How do Abstraction, Inheritance, and Polymorphism work together?"**

Example:

```Java
interface Payment {
    void pay(double amount);
}

class UpiPayment implements Payment {
    @Override
    public void pay(double amount) {
        System.out.println("Paid using UPI");
    }
}

class CardPayment implements Payment {
    @Override
    public void pay(double amount) {
        System.out.println("Paid using Card");
    }
}

public class Main {
    public static void main(String[] args) {
        Payment payment = new UpiPayment(); // Abstraction
        payment.pay(1000);                  // Runtime Polymorphism
    }
}
```

* **Abstraction** → `Payment` interface defines the contract.
* **Inheritance** → `UpiPayment` and `CardPayment` implement the interface.
* **Polymorphism** → The same `pay()` call executes different implementations at runtime.

This design pattern is widely used in **Spring Boot**, **Hibernate**, and enterprise Java applications, making it an essential concept for **SDE-1 Java interviews**.

* * *

**ChatGPT:**

# Abstract Class in Java (Complete Interview Guide)

An **Abstract Class** is one of the two ways to achieve **Abstraction** in Java.

```
Abstraction
      |
-----------------------
|                     |
Abstract Class     Interface
```

Unlike a normal class, an **abstract class cannot be instantiated**. It acts as a **blueprint** for its child classes by providing:

* Common state (fields)
* Common behavior (concrete methods)
* Partial implementation (abstract methods)

* * *

# What is an Abstract Class?

> **An abstract class is a class declared with the `abstract` keyword. It cannot be instantiated and may contain both abstract and concrete methods.**

Syntax

```Java
abstract class Animal {

}
```

Here,

```
Animal
↓

Cannot create object
```

This is invalid:

```Java
Animal a = new Animal();
```

Output

```
Compilation Error
Animal is abstract; cannot be instantiated
```

* * *

# Why Do We Need an Abstract Class?

Suppose every animal can

* Eat
* Sleep
* Breathe

But every animal makes a different sound.

```
Dog  → Bark

Cat  → Meow

Lion → Roar
```

Should we implement `sound()` inside Animal?

No.

Because every animal behaves differently.

Instead,

```Java
abstract class Animal {

    abstract void sound();

    void eat() {
        System.out.println("Eating");
    }

}
```

Each child provides its own implementation.

* * *

# Real World Example

Think about an Employee.

Every employee has

```
ID

Name

Salary
```

Every employee works differently.

```
Developer

↓

Write Code

Tester

↓

Test Software

Manager

↓

Manage Team
```

So

```
Employee

↓

work()
```

should be abstract.

* * *

# Abstract Class Diagram

```
                 Animal
      -----------------------------
      + eat()
      + sleep()
      + breathe()
      + sound() (Abstract)
               |
     --------------------------
     |            |           |
    Dog          Cat        Lion
     |            |           |
  sound()      sound()     sound()
   Bark         Meow        Roar
```

* * *

# Cannot Be Instantiated

An abstract class represents an **incomplete object**.

Example

```Java
abstract class Animal {

}
```

Invalid

```Java
Animal a = new Animal();
```

Why?

Because Java doesn't know which type of animal to create.

Instead

```Java
Animal a = new Dog();
```

Valid.

Output depends on Dog implementation.

* * *

# Abstract Methods

An abstract method has

* No body
* Only declaration

Syntax

```Java
abstract void sound();
```

No implementation.

Child must implement it.

* * *

Example

```Java
abstract class Animal {

    abstract void sound();

}

class Dog extends Animal {

    @Override
    void sound() {

        System.out.println("Bark");

    }

}
```

Output

```
Bark
```

* * *

# Concrete Methods

Abstract classes may also contain implemented methods.

Example

```Java
abstract class Animal {

    void eat() {

        System.out.println("Eating");

    }

}
```

Child automatically inherits it.

```Java
class Dog extends Animal {

    @Override
    void sound() {

        System.out.println("Bark");

    }

}
```

Usage

```Java
Dog d = new Dog();

d.eat();
```

Output

```
Eating
```

* * *

# Abstract + Concrete Methods Together

Example

```Java
abstract class Animal {

    abstract void sound();

    void sleep() {

        System.out.println("Sleeping");

    }

}

class Dog extends Animal {

    @Override
    void sound() {

        System.out.println("Bark");

    }

}
```

Main

```Java
Dog d = new Dog();

d.sound();

d.sleep();
```

Output

```
Bark

Sleeping
```

* * *

# Can an Abstract Class Have Constructors?

Yes.

This is one of the biggest interview questions.

Many think

```
No object

↓

No constructor
```

Wrong.

Abstract classes **can have constructors**.

Example

```Java
abstract class Animal {

    Animal() {

        System.out.println("Animal Constructor");

    }

}

class Dog extends Animal {

    Dog() {

        System.out.println("Dog Constructor");

    }

}
```

Main

```Java
Dog d = new Dog();
```

Output

```
Animal Constructor

Dog Constructor
```

* * *

# Why Constructor?

Constructor initializes

* Parent fields
* Common resources
* Default values

before child object is created.

Execution Flow

```
new Dog()

↓

Dog()

↓

super()

↓

Animal()
```

* * *

# Constructor Chaining Diagram

```
Create Dog Object

↓

Dog Constructor

↓

super()

↓

Animal Constructor

↓

Return

↓

Dog Constructor Executes
```

* * *

# Can an Abstract Class Have Fields (State)?

Yes.

Unlike interfaces, abstract classes can store state.

Example

```Java
abstract class Employee {

    int id;

    String name;

}
```

Child

```Java
class Developer extends Employee {

}
```

Usage

```Java
Developer d = new Developer();

d.id = 101;

d.name = "Suhas";
```

Fields are inherited.

* * *

# Complete Example

```Java
abstract class Employee {

    int id;

    Employee(int id) {

        this.id = id;

    }

    abstract void work();

    void display() {

        System.out.println("ID : " + id);

    }

}

class Developer extends Employee {

    Developer(int id) {

        super(id);

    }

    @Override
    void work() {

        System.out.println("Writing Code");

    }

}

public class Main {

    public static void main(String[] args) {

        Developer d = new Developer(101);

        d.display();

        d.work();

    }

}
```

Output

```
ID : 101

Writing Code
```

* * *

# Rules of Abstract Classes

### Rule 1

Cannot instantiate.

```Java
new Animal();
```

Not allowed.

* * *

### Rule 2

Can contain abstract methods.

```Java
abstract void sound();
```

* * *

### Rule 3

Can contain concrete methods.

```Java
void eat(){

}
```

* * *

### Rule 4

Can contain constructors.

```Java
Animal(){

}
```

* * *

### Rule 5

Can contain variables.

```Java
int age;
```

* * *

### Rule 6

Can contain static methods.

```Java
static void test(){

}
```

* * *

### Rule 7

Can contain final methods.

```Java
final void display(){

}
```

Cannot be overridden.

* * *

### Rule 8

If a class contains an abstract method, the class itself **must** be declared abstract.

Example

```Java
class Animal {

    abstract void sound();

}
```

Compilation Error.

Correct

```Java
abstract class Animal {

    abstract void sound();

}
```

* * *

### Rule 9

If a child does not implement all inherited abstract methods, the child must also be declared abstract.

```Java
abstract class Animal {

    abstract void sound();

}

abstract class Dog extends Animal {

}
```

Valid.

* * *

# When to Use an Abstract Class?

Use an abstract class when:

### 1. Multiple classes share common code

Example

```
Vehicle

↓

Car

Bike

Bus
```

Common

```
speed

fuel

start()

stop()
```

Place these in an abstract class.

* * *

### 2. Some methods are common, others differ

Example

```
Animal

↓

eat()

sleep()

↓

sound()
```

Only `sound()` varies.

* * *

### 3. You need constructors

Interfaces don't have constructors.

Abstract classes do.

Useful for initialization.

* * *

### 4. You need instance variables (state)

Example

```Java
abstract class Employee {

    int id;

    String name;

}
```

* * *

### 5. There is a clear "IS-A" relationship

```
Dog IS-A Animal

Car IS-A Vehicle

SavingsAccount IS-A BankAccount
```

* * *

# Real-World Design Example

Banking System

```
                Account
          ------------------
          accountNumber
          balance
          deposit()
          withdraw()
          calculateInterest()
                 |
      --------------------------
      |                        |
 SavingsAccount         CurrentAccount
```

Common

```
deposit()

withdraw()
```

Different

```
calculateInterest()
```

Make it abstract.

* * *

# Abstract Class vs Concrete Class

| Concrete Class | Abstract Class |
| --- | --- |
| Can create objects | Cannot create objects |
| All methods implemented | Can contain abstract + concrete methods |
| Used directly | Used as a base class |
| Optional inheritance | Designed to be extended |

* * *

# Abstract Class vs Interface

| Abstract Class | Interface |
| --- | --- |
| Can have constructors | No constructors |
| Can have fields | Only `public static final` constants |
| Single inheritance (`extends`) | Multiple implementation (`implements`) |
| Partial implementation | Contract/capability |
| Best for common state and behavior | Best for unrelated capabilities |

* * *

# Best Practices

✔ Use abstract classes when related classes share common code.

✔ Keep only truly common logic in the abstract class.

✔ Declare only those methods abstract that subclasses must customize.

✔ Prefer interfaces when classes are unrelated but share a capability.

✔ Use `@Override` in every implementation.

* * *

# Interview Questions

### 1. Can we create an object of an abstract class?

No.

```Java
Animal a = new Animal();
```

Compilation Error.

* * *

### 2. Why can't we instantiate an abstract class?

Because it represents an incomplete implementation. Java doesn't know how to create an object when required abstract methods have no implementation.

* * *

### 3. Can an abstract class have constructors?

Yes.

They are executed when a subclass object is created.

* * *

### 4. Can an abstract class have static methods?

Yes.

```Java
abstract class Test {

    static void display() {

        System.out.println("Static Method");

    }

}
```

* * *

### 5. Can an abstract class have final methods?

Yes.

```Java
final void show() {

}
```

Cannot be overridden.

* * *

### 6. Can an abstract class have private methods?

Yes.

Private methods are useful as helper methods inside the abstract class but are not inherited by subclasses.

* * *

### 7. Can an abstract class have all concrete methods?

Yes.

Example

```Java
abstract class Animal {

    void eat() {

    }

    void sleep() {

    }

}
```

Even without abstract methods, marking the class `abstract` prevents direct instantiation.

* * *

### 8. Can an abstract class have all abstract methods?

Yes.

Example

```Java
abstract class Animal {

    abstract void eat();

    abstract void sleep();

}
```

* * *

### 9. When should you use an abstract class instead of an interface?

Use an abstract class when:

* Related classes share common fields.
* You need constructors.
* You want partial implementation.
* There is an **IS-A** relationship.

* * *

# Summary Mind Map

```
                   ABSTRACT CLASS
                         |
     ------------------------------------------
     |          |          |                  |
 Cannot Create  Fields   Constructors    Concrete Methods
    Objects        |          |                |
                   |          |                |
            Abstract Methods  Shared Initialization
                   |
        Child Must Implement
                   |
         Common State + Common Behavior
                   |
          Best for IS-A Relationships
```

## SDE-1 Interview Tip

A common interview question is:

> **"If an abstract class can contain only concrete methods, why declare it abstract?"**

Answer:

You may want to **prevent direct object creation** while still providing common implementation for subclasses.

Example:

```Java
abstract class Vehicle {

    void start() {
        System.out.println("Vehicle Started");
    }
}
```

Although `Vehicle` has no abstract methods, marking it `abstract` ensures that only specific vehicles like `Car` or `Bike` can be instantiated, preserving the intended design.

* * *

**You:**

4. Interface
- All methods implicitly abstract (pre Java 8)
- Sealed interfaces (Java 17+)
- Multiple interface implementation
- Marker interfaces — Serializable, Cloneable

* * *

**ChatGPT:**

# Interface in Java (Complete Interview Guide)

An **Interface** is one of the primary mechanisms used to achieve **Abstraction** in Java.

It defines a **contract** that implementing classes must follow.

> **Interface tells _WHAT_ should be done, not _HOW_ it should be done.**

* * *

# What is an Interface?

An interface is a blueprint of a class that contains method declarations (contracts) which implementing classes must provide.

Syntax

```Java
interface Animal {

    void sound();

}
```

Implementation

```Java
class Dog implements Animal {

    @Override
    public void sound() {

        System.out.println("Bark");

    }

}
```

Usage

```Java
Animal animal = new Dog();

animal.sound();
```

Output

```
Bark
```

* * *

# Why Do We Need Interfaces?

Suppose you are building a payment application.

Different payment methods:

```
UPI

Credit Card

PayPal

Net Banking
```

All support

```
pay()
```

But implementation differs.

Instead of writing separate code everywhere,

Create one interface.

```
           Payment
      ----------------
          pay()
      ----------------
       |      |      |
      UPI   Card   PayPal
```

Application

```Java
Payment payment = new UpiPayment();

payment.pay(1000);
```

Tomorrow,

Google Pay or Apple Pay can be added without changing client code.

* * *

# Interface Characteristics

An interface:

✔ Cannot be instantiated

✔ Supports abstraction

✔ Supports multiple inheritance

✔ Defines a contract

✔ Enables loose coupling

* * *

# Pre Java 8 Interface

Before Java 8,

Interfaces could contain only

* Abstract methods
* Constants

Example

```Java
interface Animal {

    void sound();

    void eat();

}
```

Compiler internally converts to

```Java
public abstract void sound();

public abstract void eat();
```

Hence,

All methods were **implicitly**

```
public abstract
```

* * *

## Fields in Interface

Example

```Java
interface Test {

    int MAX = 100;

}
```

Compiler converts it to

```Java
public static final int MAX = 100;
```

All interface variables are automatically

```
public

static

final
```

Meaning they are constants.

* * *

# Complete Example

```Java
interface Animal {

    void sound();

}

class Dog implements Animal {

    @Override
    public void sound() {

        System.out.println("Bark");

    }

}

public class Main {

    public static void main(String[] args) {

        Animal a = new Dog();

        a.sound();

    }

}
```

Output

```
Bark
```

* * *

# Java 8 Interface Enhancements

From Java 8 onwards,

Interfaces can contain

* Abstract methods
* Default methods
* Static methods

* * *

## Default Method

Example

```Java
interface Vehicle {

    void start();

    default void stop() {

        System.out.println("Vehicle Stopped");

    }

}
```

Implementation

```Java
class Car implements Vehicle {

    @Override
    public void start() {

        System.out.println("Car Started");

    }

}
```

Usage

```Java
Car car = new Car();

car.stop();
```

Output

```
Vehicle Stopped
```

* * *

Why default methods?

Suppose Java adds a new method.

Without default methods,

Every implementation would break.

Default methods solve backward compatibility.

* * *

## Static Method

Example

```Java
interface Calculator {

    static int add(int a,int b){

        return a+b;

    }

}
```

Usage

```Java
System.out.println(Calculator.add(10,20));
```

Output

```
30
```

* * *

# Java 9 Enhancement

Java 9 introduced

```
Private methods

Private static methods
```

Used internally by default methods.

Example

```Java
interface Logger {

    default void logInfo() {

        print("INFO");

    }

    default void logError() {

        print("ERROR");

    }

    private void print(String level) {

        System.out.println(level);

    }

}
```

* * *

# Java 17 - Sealed Interfaces

One of the biggest additions.

* * *

## What is Sealed Interface?

A sealed interface restricts

**which classes or interfaces can implement it.**

Syntax

```Java
public sealed interface Shape
    permits Circle, Rectangle {

}
```

Only

```
Circle

Rectangle
```

can implement Shape.

* * *

Example

```Java
sealed interface Payment
    permits UpiPayment, CardPayment {

    void pay();

}
```

Allowed

```Java
final class UpiPayment implements Payment {

    @Override
    public void pay() {

        System.out.println("UPI Payment");

    }

}
```

Allowed

```Java
final class CardPayment implements Payment {

    @Override
    public void pay() {

        System.out.println("Card Payment");

    }

}
```

Not Allowed

```Java
class CryptoPayment implements Payment {

}
```

Compilation Error.

* * *

# Why Sealed Interfaces?

Imagine

```
Payment

↓

Hundreds of Unknown Classes
```

Hard to control.

Instead

```
Payment

↓

UPI

Card

Wallet
```

Only known implementations.

Advantages

✔ Better Security

✔ Controlled Hierarchy

✔ Better Pattern Matching

✔ Easier Maintenance

* * *

# Diagram

```
          sealed Payment
      ------------------------
           pay()
      ------------------------
         |               |
   UpiPayment      CardPayment

❌ CryptoPayment
```

* * *

# Multiple Interface Implementation

Java doesn't support

Multiple inheritance of classes.

But

It supports

Multiple inheritance using interfaces.

Example

```Java
interface Camera {

    void click();

}

interface MusicPlayer {

    void playMusic();

}

class Smartphone implements Camera, MusicPlayer {

    @Override
    public void click() {

        System.out.println("Photo Clicked");

    }

    @Override
    public void playMusic() {

        System.out.println("Music Playing");

    }

}
```

Usage

```Java
Smartphone phone = new Smartphone();

phone.click();

phone.playMusic();
```

Output

```
Photo Clicked

Music Playing
```

* * *

# Diagram

```
         Camera

             \

              \

           Smartphone

              /

             /

      MusicPlayer
```

Multiple interface inheritance.

* * *

# Why Java Allows Multiple Interfaces?

Interfaces contain

Contracts

Not implementation (historically).

Hence,

No ambiguity.

Since Java 8, if multiple interfaces provide the same `default` method, the implementing class **must override** it to resolve the conflict.

Example

```Java
interface A {

    default void show() {

        System.out.println("A");

    }

}

interface B {

    default void show() {

        System.out.println("B");

    }

}

class Test implements A,B {

    @Override
    public void show() {

        System.out.println("Resolved");

    }

}
```

* * *

# Marker Interfaces

Very important interview topic.

* * *

## What is Marker Interface?

A marker interface

Contains **no methods and no fields**.

Purpose

To provide metadata to JVM or frameworks.

Example

```Java
interface Test{

}
```

Nothing inside.

* * *

# Serializable

```Java
class Employee implements Serializable{

}
```

No methods.

Purpose

Marks object as serializable.

Serialization means

Converting object into byte stream.

Used for

* File storage
* Network communication
* Caching

* * *

Example

```Java
class Student implements Serializable {

    private String name;

}
```

Now

```Java
Student

↓

Byte Stream

↓

File
```

* * *

# Cloneable

```Java
class Employee implements Cloneable{

}
```

Purpose

Allows object cloning.

Example

```Java
Employee e2 = (Employee)e1.clone();
```

Without `Cloneable`

Runtime Exception

```
CloneNotSupportedException
```

* * *

# Other Marker Interfaces

| Interface | Purpose |
| --- | --- |
| Serializable | Object Serialization |
| Cloneable | Object Cloning |
| Remote | Used in RMI (Remote Method Invocation) |
| RandomAccess | Indicates efficient random access for collections like `ArrayList` |

* * *

# Interface vs Abstract Class

| Interface | Abstract Class |
| --- | --- |
| Contract | Partial implementation |
| No constructors | Constructors allowed |
| Multiple implementation | Single inheritance |
| Constants only | Instance variables allowed |
| Best for capabilities | Best for common state & behavior |

* * *

# Real World Design Example

Notification System

```
            Notification
        --------------------
          send(message)
        --------------------
        |       |         |
      Email    SMS      Push
```

Application

```Java
Notification notification = new EmailNotification();

notification.send("Welcome");
```

Later

```
Slack

WhatsApp

Teams
```

can be added without changing existing client code.

* * *

# Best Practices

✔ Use interfaces to define capabilities (e.g., `Flyable`, `Runnable`, `Payable`).

✔ Program to interfaces rather than concrete classes.

✔ Keep interfaces focused on a single responsibility.

✔ Use sealed interfaces when you want a fixed set of implementations.

✔ Prefer composition with interfaces for loose coupling and testability.

* * *

# Interview Questions

### 1. What is an interface?

An interface is a contract that specifies methods which implementing classes must define.

* * *

### 2. Can we create an object of an interface?

No.

```Java
Animal a = new Animal();
```

Compilation Error.

* * *

### 3. What were interfaces like before Java 8?

They contained only:

* Abstract methods
* `public static final` constants

* * *

### 4. What changed in Java 8?

Interfaces gained:

* Default methods
* Static methods

* * *

### 5. What changed in Java 9?

Interfaces gained:

* Private methods
* Private static methods

* * *

### 6. What are sealed interfaces?

Introduced in Java 17, they restrict which classes or interfaces are allowed to implement them.

* * *

### 7. Why doesn't Java support multiple inheritance of classes but supports multiple interfaces?

Multiple class inheritance can create ambiguity (Diamond Problem).

Interfaces primarily define contracts, and any default method conflicts must be explicitly resolved by the implementing class.

* * *

### 8. What is a marker interface?

An empty interface used to convey metadata or capabilities.

Examples:

* `Serializable`
* `Cloneable`

* * *

### 9. What is the difference between `extends` and `implements`?

| Keyword | Used For |
| --- | --- |
| `extends` | Class inherits another class or interface extends another interface |
| `implements` | Class implements one or more interfaces |

Example

```Java
class Dog extends Animal implements Pet {
}
```

* * *

# Summary Mind Map

```
                         INTERFACE
                              |
        ------------------------------------------------
        |                  |                 |
     Contract       Multiple Inheritance   Loose Coupling
        |
   -------------------------------
   |              |              |
 Abstract     Default       Static
 Methods      Methods       Methods
        |
    Java 9
        |
 Private Methods
        |
    Java 17
        |
 Sealed Interfaces
        |
 Restrict Implementations
        |
 Marker Interfaces
        |
 -----------------------------
 |            |              |
Serializable Cloneable   RandomAccess
```

* * *

# SDE-1 Interview Tip

A common Spring Boot interview question is:

> **"Why are interfaces used extensively in Spring?"**

Example:

```Java
public interface UserService {
    User getUser(Long id);
}

@Service
public class UserServiceImpl implements UserService {
    @Override
    public User getUser(Long id) {
        return repository.findById(id).orElseThrow();
    }
}
```

Spring injects the dependency using the **interface**, not the implementation:

```Java
@Autowired
private UserService userService;
```

**Benefits:**

* ✅ Loose coupling
* ✅ Easy to replace implementations
* ✅ Easier unit testing using mocks
* ✅ Supports Dependency Injection (DI)
* ✅ Makes applications easier to maintain and extend

This is one of the most common real-world uses of interfaces in enterprise Java and is frequently asked in **SDE-1 interviews**.

* * *

**You:**

5.Relationships Between Classes
- IS-A → Inheritance
- HAS-A → Composition & Aggregation
- Association → loose relationship
- Composition vs Aggregation vs Association (with lifecycle)

6. Advanced OOP Concepts
- Coupling — tight vs loose (aim for loose)
- Cohesion — high cohesion is good
- Delegation over inheritance
- Favor Composition over Inheritance principle
- Object cloning — shallow vs deep copy
- Immutable classes — how to design one

* * *

**ChatGPT:**

# 5. Relationships Between Classes (Complete Interview Guide)

One of the most important topics in Java OOP is understanding **how classes relate to each other**.

There are mainly **three types of relationships**:

```
                    Relationships
                          |
        --------------------------------------
        |                |                  |
      IS-A             HAS-A          ASSOCIATION
        |                |
   Inheritance    Composition / Aggregation
```

* * *

# 1. IS-A Relationship (Inheritance)

An **IS-A** relationship means one class **is a specialized version of another class**.

Achieved using:

```Java
extends
```

or

```Java
implements
```

* * *

## Real World Example

```
Animal
   |
   |
  Dog
```

Dog **IS-A** Animal.

Similarly,

```
Vehicle
   |
  Car
```

Car **IS-A** Vehicle.

* * *

## Example

```Java
class Animal {

    void eat() {
        System.out.println("Eating");
    }

}

class Dog extends Animal {

    void bark() {
        System.out.println("Barking");
    }

}
```

Usage

```Java
Dog d = new Dog();

d.eat();

d.bark();
```

Output

```
Eating

Barking
```

Dog inherited Animal's behavior.

* * *

# Characteristics

✔ Code Reuse

✔ Polymorphism

✔ Method Overriding

✔ Runtime Binding

* * *

# Diagram

```
        Animal
      ----------
      eat()
      sleep()
          |
     extends
          |
        Dog
      ----------
      bark()
```

* * *

# When to Use?

Whenever

```
A IS-A B
```

Example

```
Student IS-A Person

Circle IS-A Shape

SavingsAccount IS-A BankAccount
```

* * *

# 2. HAS-A Relationship

HAS-A means

One class contains another class.

Achieved using

```
Object Reference
```

Two types

```
HAS-A
   |
---------------------
|                   |
Composition     Aggregation
```

* * *

# Composition

Composition is the **strongest HAS-A relationship**.

> Child cannot exist without Parent.

* * *

## Real World Example

```
House

↓

Rooms
```

If House is destroyed,

Rooms are destroyed.

* * *

Another example

```
Human

↓

Heart
```

Heart cannot exist independently.

* * *

## Java Example

```Java
class Engine {

    void start() {
        System.out.println("Engine Started");
    }

}

class Car {

    private Engine engine = new Engine();

    void drive() {

        engine.start();

        System.out.println("Car Moving");
    }

}
```

Usage

```Java
Car car = new Car();

car.drive();
```

Output

```
Engine Started

Car Moving
```

Notice

Engine object is created **inside Car**.

* * *

## Diagram

```
        Car
     ------------
      Engine
```

Destroy Car

↓

Engine also destroyed.

* * *

# Characteristics

✔ Strong relationship

✔ Same lifecycle

✔ Ownership

✔ Better encapsulation

* * *

# Aggregation

Aggregation is a **weak HAS-A relationship**.

Objects can exist independently.

* * *

## Real World Example

```
College

↓

Students
```

If College closes,

Students still exist.

* * *

Another example

```
Department

↓

Teacher
```

Teacher exists without Department.

* * *

## Java Example

```Java
class Student {

    String name;

    Student(String name) {
        this.name = name;
    }

}

class College {

    Student student;

    College(Student student) {

        this.student = student;
    }

}
```

Usage

```Java
Student s = new Student("Suhas");

College c = new College(s);
```

Student created separately.

College just uses it.

* * *

# Diagram

```
Student

↓

College
```

Destroy College

↓

Student still exists.

* * *

# Composition vs Aggregation

| Composition | Aggregation |
| --- | --- |
| Strong HAS-A | Weak HAS-A |
| Same lifecycle | Independent lifecycle |
| Parent owns child | Parent references child |
| Child cannot exist alone | Child can exist alone |
| Example: Car-Engine | College-Student |

* * *

# Association

Association is a **general relationship** between two independent objects.

Neither owns the other.

* * *

## Real World Example

```
Doctor

↓

Patient
```

Doctor treats Patient.

Neither owns the other.

* * *

Another example

```
Customer

↓

Bank
```

Customer uses Bank.

* * *

## Java Example

```Java
class Teacher {

    void teach() {

        System.out.println("Teaching");

    }

}

class Student {

    void learn(Teacher teacher) {

        teacher.teach();

    }

}
```

Usage

```Java
Teacher t = new Teacher();

Student s = new Student();

s.learn(t);
```

Output

```
Teaching
```

Teacher and Student are independent.

* * *

# Types of Association

## One-to-One

```
Person

↓

Passport
```

* * *

## One-to-Many

```
Teacher

↓

Students
```

* * *

## Many-to-Many

```
Students

↓

Courses
```

One student takes many courses.

One course has many students.

* * *

# Lifecycle Comparison

```
               Parent Destroyed

Inheritance

Parent
↓

Child still exists if instantiated

----------------------------

Composition

Car
↓

Engine destroyed

----------------------------

Aggregation

College
↓

Student still exists

----------------------------

Association

Teacher

Student

Independent
```

* * *

# Complete Comparison

| Feature | Inheritance | Composition | Aggregation | Association |
| --- | --- | --- | --- | --- |
| Relationship | IS-A | Strong HAS-A | Weak HAS-A | Uses-A / Knows-A |
| Ownership | Parent | Strong | Weak | None |
| Lifecycle | Independent | Same | Independent | Independent |
| Reuse | Yes | Yes | Yes | Limited |
| Example | Dog-Animal | Car-Engine | College-Student | Teacher-Student |

* * *

# Interview Questions

### Difference between Composition and Aggregation?

Composition

```
Car

↓

Engine
```

Engine dies with Car.

Aggregation

```
College

↓

Student
```

Student lives independently.

* * *

### Difference between IS-A and HAS-A?

IS-A

```
Dog IS-A Animal
```

HAS-A

```
Car HAS-A Engine
```

* * *

# 6. Advanced OOP Concepts

* * *

# Coupling

Coupling measures

> **How much one class depends on another class.**

Lower dependency

↓

Better design.

* * *

# Tight Coupling

Classes depend heavily on each other.

Example

```Java
class Engine {

    void start() {

        System.out.println("Engine Started");

    }

}

class Car {

    Engine engine = new Engine();

    void drive() {

        engine.start();

    }

}
```

Problem

Cannot replace Engine easily.

* * *

# Loose Coupling

Depend on abstraction.

```Java
interface Engine {

    void start();

}

class PetrolEngine implements Engine {

    public void start() {

        System.out.println("Petrol Engine");

    }

}

class ElectricEngine implements Engine {

    public void start() {

        System.out.println("Electric Engine");

    }

}

class Car {

    private Engine engine;

    Car(Engine engine) {

        this.engine = engine;

    }

    void drive() {

        engine.start();

    }

}
```

Usage

```Java
Car car = new Car(new ElectricEngine());
```

Tomorrow

```Java
new PetrolEngine()
```

Works without changing Car.

* * *

# Tight vs Loose Coupling

| Tight | Loose |
| --- | --- |
| Depends on implementation | Depends on abstraction |
| Hard to modify | Easy to modify |
| Hard to test | Easy to mock/test |
| Less flexible | Highly flexible |

* * *

# Cohesion

Cohesion measures

> **How closely related the responsibilities inside a class are.**

* * *

## High Cohesion

One class

One responsibility.

Example

```
InvoiceService

↓

Generate Invoice
```

Good.

* * *

## Low Cohesion

```
Employee

↓

Send Email

↓

Generate Report

↓

Database Access

↓

Payment
```

Too many responsibilities.

* * *

# High vs Low Cohesion

| High | Low |
| --- | --- |
| Single Responsibility | Many responsibilities |
| Easy maintenance | Difficult maintenance |
| Reusable | Less reusable |

* * *

# Delegation over Inheritance

Instead of inheriting,

Delegate work.

* * *

Inheritance

```
Car

↓

Engine
```

Better

```
Car

↓

Uses Engine
```

Example

```Java
class Printer {

    void print() {

        System.out.println("Printing");

    }

}

class Employee {

    private Printer printer = new Printer();

    void printReport() {

        printer.print();

    }

}
```

Employee delegates printing.

* * *

# Favor Composition over Inheritance

One of the most famous design principles.

Instead of

```
Car extends Engine
```

Use

```
Car HAS-A Engine
```

Why?

Composition is

✔ Flexible

✔ Reusable

✔ Easy Testing

✔ Runtime Replaceable

Spring Boot heavily follows this principle.

* * *

# Object Cloning

Cloning means

Creating an exact copy of an object.

* * *

# Shallow Copy

Only top-level object copied.

Nested objects are shared.

Example

```Java
Employee e2 = e1.clone();
```

```
Employee1

↓

Address

Employee2

↓

Same Address
```

Modify Address

↓

Both objects change.

* * *

# Deep Copy

Entire object graph copied.

```
Employee1

↓

Address1

Employee2

↓

Address2
```

Modify one.

Other remains unchanged.

* * *

# Comparison

| Shallow Copy | Deep Copy |
| --- | --- |
| References shared | Independent objects |
| Faster | Slower |
| Less memory | More memory |
| Nested changes affect both | No shared state |

* * *

# Immutable Class

An immutable object

Cannot change after creation.

Example

```Java
String s = "Java";
```

New value

```Java
s = s.concat("8");
```

Creates new object.

Old object unchanged.

* * *

# How to Design Immutable Class

### Rule 1

Make class final.

```Java
final class Employee {

}
```

* * *

### Rule 2

Private final fields.

```Java
private final int id;
```

* * *

### Rule 3

Initialize through constructor.

```Java
Employee(int id){

    this.id=id;

}
```

* * *

### Rule 4

No setters.

Only getters.

* * *

### Rule 5

Defensive copy for mutable fields.

Example

```Java
private final Date joiningDate;

public Employee(Date joiningDate) {
    this.joiningDate = new Date(joiningDate.getTime());
}

public Date getJoiningDate() {
    return new Date(joiningDate.getTime());
}
```

This prevents external code from modifying the internal `Date`.

* * *

# Complete Immutable Example

```Java
final class Employee {

    private final int id;
    private final String name;

    Employee(int id, String name) {

        this.id = id;
        this.name = name;
    }

    public int getId() {

        return id;
    }

    public String getName() {

        return name;
    }
}
```

No setters.

Cannot modify object.

* * *

# Benefits of Immutable Objects

✔ Thread Safe

✔ Easy Caching

✔ Predictable

✔ Safe for Collections

✔ No Synchronization Needed

Example

```
String

LocalDate

LocalDateTime

BigInteger

BigDecimal
```

are immutable classes.

* * *

# Advanced OOP Mind Map

```
                    ADVANCED OOP
                         |
      --------------------------------------------
      |              |            |              |
  Coupling       Cohesion     Relationships   Cloning
      |              |            |              |
 Tight/Loose     High/Low      IS-A/HAS-A   Shallow/Deep
      |
 Delegation
      |
Composition over Inheritance
      |
 Immutable Objects
      |
 final class
 private final fields
 no setters
 defensive copies
```

* * *

# SDE-1 Interview Tips

### Q1. Which is better: Inheritance or Composition?

**Answer:**

Prefer **Composition over Inheritance** because it provides:

* Loose coupling
* Better flexibility
* Runtime behavior changes
* Easier testing
* Better maintainability

Use **Inheritance** only when there is a true **IS-A** relationship.

* * *

### Q2. Why is Spring Boot loosely coupled?

Spring Boot uses:

* Interfaces
* Dependency Injection (DI)
* Constructor Injection

Example:

```Java
@Service
class UserService {

    private final UserRepository repository;

    UserService(UserRepository repository) {
        this.repository = repository;
    }
}
```

`UserService` depends on the `UserRepository` abstraction instead of creating the implementation itself, resulting in **loose coupling** and easier testing.

* * *

### Q3. Which OOP concepts are most commonly asked in SDE-1 interviews?

* ✅ IS-A vs HAS-A
* ✅ Composition vs Aggregation vs Association
* ✅ Tight vs Loose Coupling
* ✅ High vs Low Cohesion
* ✅ Composition over Inheritance
* ✅ Shallow vs Deep Copy
* ✅ Immutable Class Design
* ✅ Dependency Injection and Delegation