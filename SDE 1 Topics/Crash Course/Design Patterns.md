# Java Design Patterns — Interview Notes

> **Covers:** Creational · Structural · Behavioral  
> **Language:** Java | **Level:** Interview-Ready

---

## What is a Design Pattern?

A **design pattern** is a reusable, proven solution to a commonly occurring problem in software design. They are not finished code — they are **templates or blueprints** you adapt to your situation.

Introduced by the **"Gang of Four" (GoF)** — Erich Gamma, Richard Helm, Ralph Johnson, John Vlissides — in the book *Design Patterns: Elements of Reusable Object-Oriented Software* (1994).

### The Three Categories

| Category | Purpose | Patterns Covered |
|---|---|---|
| **Creational** | How objects are created | Singleton, Factory, Abstract Factory, Builder |
| **Structural** | How objects are composed/structured | Adapter, Facade, Composite |
| **Behavioral** | How objects communicate and behave | Strategy, Observer, Iterator, State |

---

# PART 1 — CREATIONAL PATTERNS

> *"Control how objects are instantiated."*

---

## 1. Singleton Pattern

### Intent
Ensure a class has **only one instance** and provide a **global access point** to it.

### When to Use
- Database connection pool
- Logger
- Configuration manager
- Thread pool
- Cache

### Implementation (Thread-Safe — Double-Checked Locking)

```java
public class Singleton {

    // volatile ensures visibility across threads
    private static volatile Singleton instance;

    // Private constructor prevents direct instantiation
    private Singleton() {
        // initialization
    }

    public static Singleton getInstance() {
        if (instance == null) {                    // First check (no lock)
            synchronized (Singleton.class) {
                if (instance == null) {            // Second check (with lock)
                    instance = new Singleton();
                }
            }
        }
        return instance;
    }

    public void doSomething() {
        System.out.println("Singleton in action");
    }
}

// Usage
Singleton s1 = Singleton.getInstance();
Singleton s2 = Singleton.getInstance();
System.out.println(s1 == s2); // true
```

### Bill Pugh Singleton (Best Practice — Lazy + Thread-Safe)

```java
public class Singleton {

    private Singleton() {}

    // Inner static class is loaded only when getInstance() is called
    private static class SingletonHelper {
        private static final Singleton INSTANCE = new Singleton();
    }

    public static Singleton getInstance() {
        return SingletonHelper.INSTANCE;
    }
}
```

### Enum Singleton (Simplest + Serialization-Safe)

```java
public enum Singleton {
    INSTANCE;

    public void doSomething() {
        System.out.println("Enum Singleton");
    }
}

// Usage
Singleton.INSTANCE.doSomething();
```

### Key Points for Interview

| Aspect | Detail |
|---|---|
| `volatile` keyword | Prevents instruction reordering by JVM/CPU |
| Why double-check? | Avoids `synchronized` overhead after first initialization |
| Enum Singleton | Handles serialization, reflection attacks automatically |
| Pitfall | Singleton breaks with multiple ClassLoaders or serialization without `readResolve()` |

---

## 2. Factory Method Pattern

### Intent
Define an interface for creating an object, but let **subclasses decide which class to instantiate**. The factory method defers instantiation to subclasses.

### When to Use
- You don't know the exact class to create until runtime
- Subclasses should control the type of objects created
- `java.util.Calendar.getInstance()`, `NumberFormat.getInstance()`

### Example — Shape Factory

```java
// Step 1: Product interface
public interface Shape {
    void draw();
}

// Step 2: Concrete Products
public class Circle implements Shape {
    @Override
    public void draw() {
        System.out.println("Drawing Circle");
    }
}

public class Rectangle implements Shape {
    @Override
    public void draw() {
        System.out.println("Drawing Rectangle");
    }
}

public class Triangle implements Shape {
    @Override
    public void draw() {
        System.out.println("Drawing Triangle");
    }
}

// Step 3: Factory class
public class ShapeFactory {

    public static Shape getShape(String shapeType) {
        if (shapeType == null) return null;

        switch (shapeType.toLowerCase()) {
            case "circle":    return new Circle();
            case "rectangle": return new Rectangle();
            case "triangle":  return new Triangle();
            default: throw new IllegalArgumentException("Unknown shape: " + shapeType);
        }
    }
}

// Step 4: Client
public class Main {
    public static void main(String[] args) {
        Shape shape1 = ShapeFactory.getShape("circle");
        shape1.draw(); // Drawing Circle

        Shape shape2 = ShapeFactory.getShape("rectangle");
        shape2.draw(); // Drawing Rectangle
    }
}
```

### Key Points for Interview

- **Decouples** object creation from usage — client never calls `new` directly
- Follows **Open/Closed Principle** — add new shapes without changing the factory (use a registry or reflection)
- Real-world: `ConnectionFactory` in JDBC, `LoggerFactory` in SLF4J

---

## 3. Abstract Factory Pattern

### Intent
Provide an interface for creating **families of related objects** without specifying their concrete classes.

> Think of it as a **factory of factories**.

### When to Use
- UI toolkit (Windows vs Mac buttons, checkboxes)
- Cross-platform applications
- When product families must be used together

### Example — Cross-Platform UI

```java
// Step 1: Abstract product interfaces
public interface Button {
    void click();
}

public interface Checkbox {
    void check();
}

// Step 2: Windows family
public class WindowsButton implements Button {
    @Override
    public void click() {
        System.out.println("Windows Button clicked");
    }
}

public class WindowsCheckbox implements Checkbox {
    @Override
    public void check() {
        System.out.println("Windows Checkbox checked");
    }
}

// Step 3: Mac family
public class MacButton implements Button {
    @Override
    public void click() {
        System.out.println("Mac Button clicked");
    }
}

public class MacCheckbox implements Checkbox {
    @Override
    public void check() {
        System.out.println("Mac Checkbox checked");
    }
}

// Step 4: Abstract Factory interface
public interface GUIFactory {
    Button createButton();
    Checkbox createCheckbox();
}

// Step 5: Concrete factories
public class WindowsFactory implements GUIFactory {
    @Override
    public Button createButton() {
        return new WindowsButton();
    }

    @Override
    public Checkbox createCheckbox() {
        return new WindowsCheckbox();
    }
}

public class MacFactory implements GUIFactory {
    @Override
    public Button createButton() {
        return new MacButton();
    }

    @Override
    public Checkbox createCheckbox() {
        return new MacCheckbox();
    }
}

// Step 6: Client
public class Application {
    private Button button;
    private Checkbox checkbox;

    public Application(GUIFactory factory) {
        button   = factory.createButton();
        checkbox = factory.createCheckbox();
    }

    public void render() {
        button.click();
        checkbox.check();
    }
}

// Step 7: Main
public class Main {
    public static void main(String[] args) {
        GUIFactory factory;

        String os = System.getProperty("os.name").toLowerCase();
        if (os.contains("win")) {
            factory = new WindowsFactory();
        } else {
            factory = new MacFactory();
        }

        Application app = new Application(factory);
        app.render();
    }
}
```

### Factory vs Abstract Factory

| | Factory Method | Abstract Factory |
|---|---|---|
| Creates | One product | Family of related products |
| How | Subclass overrides a method | Separate factory object per family |
| Complexity | Simpler | More complex |
| Use case | One type needed | Multiple related types needed together |

---

## 4. Builder Pattern

### Intent
Separate the **construction of a complex object** from its representation so the same process can create different representations.

### When to Use
- Object has many optional parameters (avoids telescoping constructors)
- Construction requires multiple steps
- `StringBuilder`, `Stream.Builder`, Lombok's `@Builder`

### Example — Building a Computer

```java
// Product
public class Computer {
    // Required
    private final String cpu;
    private final String ram;
    // Optional
    private final String storage;
    private final String gpu;
    private final boolean hasBluetooth;
    private final boolean hasWifi;

    // Private constructor — only Builder can call it
    private Computer(Builder builder) {
        this.cpu          = builder.cpu;
        this.ram          = builder.ram;
        this.storage      = builder.storage;
        this.gpu          = builder.gpu;
        this.hasBluetooth = builder.hasBluetooth;
        this.hasWifi      = builder.hasWifi;
    }

    @Override
    public String toString() {
        return "Computer{cpu='" + cpu + "', ram='" + ram +
               "', storage='" + storage + "', gpu='" + gpu +
               "', bluetooth=" + hasBluetooth + ", wifi=" + hasWifi + "}";
    }

    // Static inner Builder class
    public static class Builder {
        // Required fields
        private final String cpu;
        private final String ram;

        // Optional fields (defaults)
        private String storage      = "256GB SSD";
        private String gpu          = "Integrated";
        private boolean hasBluetooth = false;
        private boolean hasWifi      = false;

        public Builder(String cpu, String ram) {
            this.cpu = cpu;
            this.ram = ram;
        }

        public Builder storage(String storage) {
            this.storage = storage;
            return this;  // Return 'this' for method chaining
        }

        public Builder gpu(String gpu) {
            this.gpu = gpu;
            return this;
        }

        public Builder bluetooth(boolean val) {
            this.hasBluetooth = val;
            return this;
        }

        public Builder wifi(boolean val) {
            this.hasWifi = val;
            return this;
        }

        public Computer build() {
            // Validation can go here
            if (cpu == null || cpu.isEmpty()) {
                throw new IllegalStateException("CPU is required");
            }
            return new Computer(this);
        }
    }
}

// Usage — fluent API
public class Main {
    public static void main(String[] args) {
        Computer gamingPC = new Computer.Builder("Intel i9", "32GB")
                .storage("2TB NVMe SSD")
                .gpu("RTX 4090")
                .bluetooth(true)
                .wifi(true)
                .build();

        Computer officePC = new Computer.Builder("Intel i5", "8GB")
                .storage("512GB SSD")
                .build();

        System.out.println(gamingPC);
        System.out.println(officePC);
    }
}
```

### Key Points for Interview

- Solves the **telescoping constructor anti-pattern**
- The object is **immutable** once built (final fields)
- `build()` is the right place for **validation logic**
- **Fluent interface** — each setter returns `this`

---

# PART 2 — STRUCTURAL PATTERNS

> *"How classes and objects are composed to form larger structures."*

---

## 5. Adapter Pattern

### Intent
Convert the interface of a class into **another interface that clients expect**. Lets incompatible interfaces work together.

> Like a power socket adapter — same electricity, different plug shapes.

### When to Use
- Integrating legacy code with new systems
- Using third-party libraries with incompatible interfaces
- `java.io.InputStreamReader` adapts `InputStream` to `Reader`

### Example — Media Player

```java
// Step 1: Target interface (what the client expects)
public interface MediaPlayer {
    void play(String audioType, String fileName);
}

// Step 2: Adaptee (incompatible, existing interface)
public interface AdvancedMediaPlayer {
    void playVlc(String fileName);
    void playMp4(String fileName);
}

public class VlcPlayer implements AdvancedMediaPlayer {
    @Override
    public void playVlc(String fileName) {
        System.out.println("Playing VLC file: " + fileName);
    }

    @Override
    public void playMp4(String fileName) {
        // do nothing
    }
}

public class Mp4Player implements AdvancedMediaPlayer {
    @Override
    public void playVlc(String fileName) {
        // do nothing
    }

    @Override
    public void playMp4(String fileName) {
        System.out.println("Playing MP4 file: " + fileName);
    }
}

// Step 3: Adapter — bridges MediaPlayer ↔ AdvancedMediaPlayer
public class MediaAdapter implements MediaPlayer {

    private AdvancedMediaPlayer advancedPlayer;

    public MediaAdapter(String audioType) {
        if (audioType.equalsIgnoreCase("vlc")) {
            advancedPlayer = new VlcPlayer();
        } else if (audioType.equalsIgnoreCase("mp4")) {
            advancedPlayer = new Mp4Player();
        }
    }

    @Override
    public void play(String audioType, String fileName) {
        if (audioType.equalsIgnoreCase("vlc")) {
            advancedPlayer.playVlc(fileName);
        } else if (audioType.equalsIgnoreCase("mp4")) {
            advancedPlayer.playMp4(fileName);
        }
    }
}

// Step 4: Concrete implementation of target interface
public class AudioPlayer implements MediaPlayer {

    @Override
    public void play(String audioType, String fileName) {
        if (audioType.equalsIgnoreCase("mp3")) {
            System.out.println("Playing MP3 file: " + fileName);
        } else if (audioType.equalsIgnoreCase("vlc")
                || audioType.equalsIgnoreCase("mp4")) {
            // Delegate to adapter
            MediaAdapter adapter = new MediaAdapter(audioType);
            adapter.play(audioType, fileName);
        } else {
            System.out.println("Format not supported: " + audioType);
        }
    }
}

// Step 5: Client
public class Main {
    public static void main(String[] args) {
        AudioPlayer player = new AudioPlayer();
        player.play("mp3", "song.mp3");   // Playing MP3 file: song.mp3
        player.play("mp4", "video.mp4");  // Playing MP4 file: video.mp4
        player.play("vlc", "movie.vlc");  // Playing VLC file: movie.vlc
        player.play("avi", "clip.avi");   // Format not supported: avi
    }
}
```

### Key Points for Interview

| Aspect | Detail |
|---|---|
| Also called | **Wrapper** pattern |
| Two types | **Class Adapter** (uses inheritance) vs **Object Adapter** (uses composition — preferred) |
| Real Java | `Arrays.asList()`, `Collections.list()`, `InputStreamReader` |

---

## 6. Facade Pattern

### Intent
Provide a **simplified interface** to a complex subsystem. The facade hides the complexity and provides a unified, easy-to-use API.

> Like a **hotel concierge** — you ask them for dinner reservations, a taxi, and a wake-up call. You don't manage each service yourself.

### When to Use
- Simplify usage of complex libraries or frameworks
- Layered architecture (Service layer is a facade over Repositories, external APIs)
- `javax.faces.context.FacesContext` in JSF

### Example — Home Theater System

```java
// Complex subsystem classes
public class DVDPlayer {
    public void on()  { System.out.println("DVD Player ON"); }
    public void off() { System.out.println("DVD Player OFF"); }
    public void play(String movie) { System.out.println("Playing: " + movie); }
}

public class Projector {
    public void on()         { System.out.println("Projector ON"); }
    public void off()        { System.out.println("Projector OFF"); }
    public void widescreen() { System.out.println("Projector in widescreen mode"); }
}

public class SoundSystem {
    public void on()             { System.out.println("Sound System ON"); }
    public void off()            { System.out.println("Sound System OFF"); }
    public void setVolume(int v) { System.out.println("Volume set to " + v); }
    public void setSurroundSound() { System.out.println("Surround sound ON"); }
}

public class Lights {
    public void dim(int level) { System.out.println("Lights dimmed to " + level + "%"); }
    public void on()           { System.out.println("Lights ON"); }
}

// FACADE — simple interface over all subsystems
public class HomeTheaterFacade {

    private DVDPlayer    dvd;
    private Projector    projector;
    private SoundSystem  sound;
    private Lights       lights;

    public HomeTheaterFacade(DVDPlayer dvd, Projector projector,
                             SoundSystem sound, Lights lights) {
        this.dvd       = dvd;
        this.projector = projector;
        this.sound     = sound;
        this.lights    = lights;
    }

    // Simplified "watch movie" operation
    public void watchMovie(String movie) {
        System.out.println("--- Get ready to watch a movie ---");
        lights.dim(20);
        projector.on();
        projector.widescreen();
        sound.on();
        sound.setSurroundSound();
        sound.setVolume(8);
        dvd.on();
        dvd.play(movie);
    }

    // Simplified "end movie" operation
    public void endMovie() {
        System.out.println("--- Shutting down movie theater ---");
        dvd.off();
        sound.off();
        projector.off();
        lights.on();
    }
}

// Client
public class Main {
    public static void main(String[] args) {
        HomeTheaterFacade theater = new HomeTheaterFacade(
            new DVDPlayer(), new Projector(),
            new SoundSystem(), new Lights()
        );

        theater.watchMovie("Inception");  // One call replaces 8 steps
        System.out.println();
        theater.endMovie();
    }
}
```

### Key Points for Interview

- Does **not** prevent clients from using the subsystem directly if needed
- Promotes **loose coupling** between client and subsystem
- Common in **Spring** — `JdbcTemplate` is a Facade over JDBC
- Difference from Adapter: Adapter makes incompatible interfaces work; Facade simplifies a complex interface

---

## 7. Composite Pattern

### Intent
Compose objects into **tree structures** to represent part-whole hierarchies. Lets clients treat **individual objects and compositions uniformly**.

> Like a **file system** — a folder contains files and other folders. Both respond to the same `display()` or `getSize()` call.

### When to Use
- Tree structures: file system, org charts, UI component trees
- Clients should ignore differences between individual objects and groups
- `java.awt.Container`, XML/HTML DOM trees

### Example — File System

```java
import java.util.ArrayList;
import java.util.List;

// Component — common interface for files and folders
public interface FileSystemComponent {
    void display(String indent);
    long getSize();
}

// Leaf — a file (no children)
public class File implements FileSystemComponent {

    private String name;
    private long   size;

    public File(String name, long size) {
        this.name = name;
        this.size = size;
    }

    @Override
    public void display(String indent) {
        System.out.println(indent + "📄 " + name + " (" + size + " KB)");
    }

    @Override
    public long getSize() {
        return size;
    }
}

// Composite — a folder (can contain files AND other folders)
public class Folder implements FileSystemComponent {

    private String                   name;
    private List<FileSystemComponent> children = new ArrayList<>();

    public Folder(String name) {
        this.name = name;
    }

    public void add(FileSystemComponent component) {
        children.add(component);
    }

    public void remove(FileSystemComponent component) {
        children.remove(component);
    }

    @Override
    public void display(String indent) {
        System.out.println(indent + "📁 " + name + " (" + getSize() + " KB)");
        for (FileSystemComponent child : children) {
            child.display(indent + "   ");   // Recurse into children
        }
    }

    @Override
    public long getSize() {
        // Sum of all children's sizes
        return children.stream()
                       .mapToLong(FileSystemComponent::getSize)
                       .sum();
    }
}

// Client
public class Main {
    public static void main(String[] args) {
        // Create files
        File file1 = new File("resume.pdf",   200);
        File file2 = new File("photo.jpg",    500);
        File file3 = new File("notes.txt",     10);
        File file4 = new File("project.zip", 1500);

        // Create folders
        Folder documents = new Folder("Documents");
        Folder pictures  = new Folder("Pictures");
        Folder root      = new Folder("Root");

        // Build tree
        documents.add(file1);
        documents.add(file3);
        pictures.add(file2);
        root.add(documents);
        root.add(pictures);
        root.add(file4);

        // Client treats Folder and File the same way
        root.display("");
        System.out.println("Total size: " + root.getSize() + " KB");
    }
}
```

**Output:**
```
📁 Root (2210 KB)
   📁 Documents (210 KB)
      📄 resume.pdf (200 KB)
      📄 notes.txt (10 KB)
   📁 Pictures (500 KB)
      📄 photo.jpg (500 KB)
   📄 project.zip (1500 KB)
Total size: 2210 KB
```

### Key Points for Interview

| Role | Class | Description |
|---|---|---|
| **Component** | `FileSystemComponent` | Common interface |
| **Leaf** | `File` | No children, base behavior |
| **Composite** | `Folder` | Has children, delegates operations |

- Enables **recursive tree operations** via polymorphism
- Client doesn't need to know if it's dealing with a leaf or composite

---

# PART 3 — BEHAVIORAL PATTERNS

> *"How objects interact and distribute responsibility."*

---

## 8. Strategy Pattern

### Intent
Define a **family of algorithms**, encapsulate each one, and make them **interchangeable**. Strategy lets the algorithm vary independently from clients that use it.

> Like choosing a **payment method** at checkout — credit card, PayPal, or UPI. The checkout process stays the same; only the payment strategy changes.

### When to Use
- Multiple variations of an algorithm
- Avoid large if-else or switch chains
- `java.util.Comparator`, `Collections.sort()`

### Example — Payment System

```java
// Step 1: Strategy interface
public interface PaymentStrategy {
    boolean pay(double amount);
}

// Step 2: Concrete strategies
public class CreditCardPayment implements PaymentStrategy {
    private String cardNumber;
    private String cvv;

    public CreditCardPayment(String cardNumber, String cvv) {
        this.cardNumber = cardNumber;
        this.cvv        = cvv;
    }

    @Override
    public boolean pay(double amount) {
        System.out.printf("Paid ₹%.2f using Credit Card ending in %s%n",
                          amount, cardNumber.substring(cardNumber.length() - 4));
        return true;
    }
}

public class PayPalPayment implements PaymentStrategy {
    private String email;

    public PayPalPayment(String email) {
        this.email = email;
    }

    @Override
    public boolean pay(double amount) {
        System.out.printf("Paid ₹%.2f via PayPal account: %s%n", amount, email);
        return true;
    }
}

public class UPIPayment implements PaymentStrategy {
    private String upiId;

    public UPIPayment(String upiId) {
        this.upiId = upiId;
    }

    @Override
    public boolean pay(double amount) {
        System.out.printf("Paid ₹%.2f via UPI ID: %s%n", amount, upiId);
        return true;
    }
}

// Step 3: Context — uses a strategy
public class ShoppingCart {
    private PaymentStrategy paymentStrategy;
    private double          total;

    public ShoppingCart(double total) {
        this.total = total;
    }

    // Strategy can be set/changed at runtime
    public void setPaymentStrategy(PaymentStrategy strategy) {
        this.paymentStrategy = strategy;
    }

    public void checkout() {
        if (paymentStrategy == null) {
            throw new IllegalStateException("No payment strategy set");
        }
        boolean success = paymentStrategy.pay(total);
        System.out.println("Payment " + (success ? "successful" : "failed"));
    }
}

// Client
public class Main {
    public static void main(String[] args) {
        ShoppingCart cart = new ShoppingCart(1499.99);

        // Use Credit Card
        cart.setPaymentStrategy(new CreditCardPayment("1234567812345678", "123"));
        cart.checkout();

        // Switch to UPI at runtime
        cart.setPaymentStrategy(new UPIPayment("user@okbank"));
        cart.checkout();
    }
}
```

### Key Points for Interview

- Follows **Open/Closed Principle** — add new strategies without changing context
- Eliminates conditional logic in the context class
- **Difference from State:** Strategy is about interchangeable algorithms; State is about behavior changing based on internal state

---

## 9. Observer Pattern

### Intent
Define a **one-to-many dependency** so that when one object (subject) changes state, all its dependents (observers) are **notified and updated automatically**.

> Like a **YouTube subscription** — when a channel uploads a video, all subscribers get notified.

### When to Use
- Event-driven systems
- GUI event handling
- MVC: Model notifies Views
- `java.util.EventListener`, Spring's `ApplicationEventPublisher`

### Example — Stock Price Alert System

```java
import java.util.ArrayList;
import java.util.List;

// Step 1: Observer interface
public interface Observer {
    void update(String stockSymbol, double price);
}

// Step 2: Subject (Observable) interface
public interface Subject {
    void registerObserver(Observer o);
    void removeObserver(Observer o);
    void notifyObservers();
}

// Step 3: Concrete Subject — StockMarket
public class StockMarket implements Subject {

    private List<Observer> observers = new ArrayList<>();
    private String stockSymbol;
    private double stockPrice;

    public StockMarket(String symbol) {
        this.stockSymbol = symbol;
    }

    @Override
    public void registerObserver(Observer o) {
        observers.add(o);
    }

    @Override
    public void removeObserver(Observer o) {
        observers.remove(o);
    }

    @Override
    public void notifyObservers() {
        for (Observer o : observers) {
            o.update(stockSymbol, stockPrice);
        }
    }

    // State change triggers notification
    public void setStockPrice(double price) {
        System.out.println("\n[Market Update] " + stockSymbol + " → ₹" + price);
        this.stockPrice = price;
        notifyObservers();
    }
}

// Step 4: Concrete Observers
public class MobileApp implements Observer {
    private String userName;

    public MobileApp(String userName) {
        this.userName = userName;
    }

    @Override
    public void update(String symbol, double price) {
        System.out.println("  📱 [" + userName + "'s Mobile] " +
                           symbol + " is now ₹" + price);
    }
}

public class EmailAlert implements Observer {
    private String email;

    public EmailAlert(String email) {
        this.email = email;
    }

    @Override
    public void update(String symbol, double price) {
        System.out.println("  📧 [Email to " + email + "] " +
                           symbol + " price changed to ₹" + price);
    }
}

// Client
public class Main {
    public static void main(String[] args) {
        StockMarket tcs = new StockMarket("TCS");

        Observer alice  = new MobileApp("Alice");
        Observer bob    = new MobileApp("Bob");
        Observer alert  = new EmailAlert("trader@example.com");

        tcs.registerObserver(alice);
        tcs.registerObserver(bob);
        tcs.registerObserver(alert);

        tcs.setStockPrice(3500.00);

        // Bob unsubscribes
        tcs.removeObserver(bob);
        tcs.setStockPrice(3620.50);  // Bob won't be notified
    }
}
```

**Output:**
```
[Market Update] TCS → ₹3500.0
  📱 [Alice's Mobile] TCS is now ₹3500.0
  📱 [Bob's Mobile] TCS is now ₹3500.0
  📧 [Email to trader@example.com] TCS price changed to ₹3500.0

[Market Update] TCS → ₹3620.5
  📱 [Alice's Mobile] TCS is now ₹3620.5
  📧 [Email to trader@example.com] TCS price changed to ₹3620.5
```

### Key Points for Interview

| Aspect | Detail |
|---|---|
| Also known as | **Publish-Subscribe**, **Event-Listener** |
| Push vs Pull | Push: subject sends data; Pull: observer fetches data |
| Java built-in | `java.util.Observable` (deprecated in Java 9), `PropertyChangeListener` |
| Modern Java | Use `EventBus` (Guava), Spring Events, or reactive streams (RxJava) |

---

## 10. Iterator Pattern

### Intent
Provide a way to **sequentially access elements** of a collection without exposing its underlying structure.

> You iterate over an `ArrayList`, `LinkedList`, or `TreeSet` using the same `for-each` syntax — that's Iterator in action.

### When to Use
- Traverse different collection types with a unified interface
- `java.util.Iterator`, `java.lang.Iterable`, enhanced for-each

### Example — Custom Book Collection Iterator

```java
import java.util.Iterator;
import java.util.NoSuchElementException;

// Element
public class Book {
    private String title;
    private String author;

    public Book(String title, String author) {
        this.title  = title;
        this.author = author;
    }

    @Override
    public String toString() {
        return "\"" + title + "\" by " + author;
    }
}

// Aggregate — the collection
public class BookShelf implements Iterable<Book> {

    private Book[] books;
    private int    count = 0;

    public BookShelf(int capacity) {
        books = new Book[capacity];
    }

    public void addBook(Book book) {
        if (count < books.length) {
            books[count++] = book;
        }
    }

    // Returns an Iterator for this collection
    @Override
    public Iterator<Book> iterator() {
        return new BookIterator();
    }

    // Inner Iterator class — knows the internals of BookShelf
    private class BookIterator implements Iterator<Book> {
        private int index = 0;

        @Override
        public boolean hasNext() {
            return index < count;
        }

        @Override
        public Book next() {
            if (!hasNext()) {
                throw new NoSuchElementException();
            }
            return books[index++];
        }
    }
}

// Client
public class Main {
    public static void main(String[] args) {
        BookShelf shelf = new BookShelf(5);
        shelf.addBook(new Book("Clean Code", "Robert Martin"));
        shelf.addBook(new Book("Effective Java", "Joshua Bloch"));
        shelf.addBook(new Book("Design Patterns", "Gang of Four"));

        // Standard for-each works because BookShelf implements Iterable
        for (Book book : shelf) {
            System.out.println(book);
        }

        // Or use iterator explicitly
        Iterator<Book> it = shelf.iterator();
        while (it.hasNext()) {
            System.out.println(it.next());
        }
    }
}
```

### Key Points for Interview

- Java's **enhanced for-each** works on any class implementing `Iterable<T>`
- The iterator **encapsulates traversal logic** — the client doesn't care how the data is stored
- **Fail-fast iterators** (like in `ArrayList`) throw `ConcurrentModificationException` if modified during iteration

---

## 11. State Pattern

### Intent
Allow an object to **alter its behavior when its internal state changes**. The object will appear to change its class.

> Like a **traffic light** — its behavior (what it does next) depends entirely on its current state (Red → Green → Yellow → Red).

### When to Use
- Object behavior changes based on state (vending machine, ATM, traffic light, order lifecycle)
- Replace large `if-else` or `switch` based on state flags
- State transitions need to be explicit

### Example — Vending Machine

```java
// Step 1: State interface
public interface VendingMachineState {
    void insertCoin(VendingMachine machine);
    void selectItem(VendingMachine machine);
    void dispenseItem(VendingMachine machine);
}

// Step 2: Concrete states
public class IdleState implements VendingMachineState {
    @Override
    public void insertCoin(VendingMachine machine) {
        System.out.println("Coin inserted. Please select an item.");
        machine.setState(new HasCoinState());
    }

    @Override
    public void selectItem(VendingMachine machine) {
        System.out.println("Please insert a coin first.");
    }

    @Override
    public void dispenseItem(VendingMachine machine) {
        System.out.println("Please insert a coin and select an item.");
    }
}

public class HasCoinState implements VendingMachineState {
    @Override
    public void insertCoin(VendingMachine machine) {
        System.out.println("Coin already inserted. Please select an item.");
    }

    @Override
    public void selectItem(VendingMachine machine) {
        System.out.println("Item selected. Dispensing...");
        machine.setState(new DispensingState());
    }

    @Override
    public void dispenseItem(VendingMachine machine) {
        System.out.println("Please select an item first.");
    }
}

public class DispensingState implements VendingMachineState {
    @Override
    public void insertCoin(VendingMachine machine) {
        System.out.println("Please wait, dispensing in progress.");
    }

    @Override
    public void selectItem(VendingMachine machine) {
        System.out.println("Already dispensing. Please wait.");
    }

    @Override
    public void dispenseItem(VendingMachine machine) {
        System.out.println("Item dispensed! Enjoy your snack.");
        if (machine.getItemCount() > 0) {
            machine.setItemCount(machine.getItemCount() - 1);
            machine.setState(new IdleState());  // Transition back to idle
        } else {
            System.out.println("Out of stock!");
            machine.setState(new OutOfStockState());
        }
    }
}

public class OutOfStockState implements VendingMachineState {
    @Override
    public void insertCoin(VendingMachine machine) {
        System.out.println("Machine is out of stock. Coin returned.");
    }

    @Override
    public void selectItem(VendingMachine machine) {
        System.out.println("Out of stock. Cannot select item.");
    }

    @Override
    public void dispenseItem(VendingMachine machine) {
        System.out.println("Out of stock.");
    }
}

// Step 3: Context — the Vending Machine
public class VendingMachine {
    private VendingMachineState currentState;
    private int                 itemCount;

    public VendingMachine(int itemCount) {
        this.itemCount = itemCount;
        this.currentState = (itemCount > 0) ? new IdleState() : new OutOfStockState();
    }

    public void insertCoin()   { currentState.insertCoin(this); }
    public void selectItem()   { currentState.selectItem(this); }
    public void dispenseItem() { currentState.dispenseItem(this); }

    public void setState(VendingMachineState state) {
        this.currentState = state;
    }

    public int getItemCount()             { return itemCount; }
    public void setItemCount(int count)   { this.itemCount = count; }
}

// Client
public class Main {
    public static void main(String[] args) {
        VendingMachine vm = new VendingMachine(2);

        vm.insertCoin();    // Coin inserted
        vm.selectItem();    // Item selected
        vm.dispenseItem();  // Dispensed, item count: 1

        vm.selectItem();    // No coin yet
        vm.insertCoin();
        vm.insertCoin();    // Already has coin
        vm.selectItem();
        vm.dispenseItem();  // Dispensed, item count: 0 → Out of stock

        vm.insertCoin();    // Out of stock, coin returned
    }
}
```

### Strategy vs State

| Aspect | Strategy | State |
|---|---|---|
| Intent | Choose an algorithm | Manage state-dependent behavior |
| Who changes? | Client sets the strategy | Context/State transitions itself |
| Awareness | Strategies usually don't know each other | States often know other states |
| Analogy | Payment method | Vending machine state |

---

# PART 4 — COMPARISON & INTERVIEW Q&A

---

## Quick Pattern Comparison Table

| Pattern | Category | Solves | Key Class Relationship |
|---|---|---|---|
| Singleton | Creational | One instance only | Single class |
| Factory Method | Creational | Decouple creation from use | Creator + Product |
| Abstract Factory | Creational | Families of related objects | Factory + Product families |
| Builder | Creational | Complex object construction | Builder + Product |
| Adapter | Structural | Incompatible interfaces | Wrapper around Adaptee |
| Facade | Structural | Simplify complex subsystem | Facade + Subsystems |
| Composite | Structural | Part-whole tree hierarchies | Component, Leaf, Composite |
| Strategy | Behavioral | Interchangeable algorithms | Context + Strategy |
| Observer | Behavioral | Event notification | Subject + Observers |
| Iterator | Behavioral | Sequential collection access | Iterator + Iterable |
| State | Behavioral | State-dependent behavior | Context + States |

---

## SOLID Principles in Design Patterns

| Principle | Patterns That Apply |
|---|---|
| **S** — Single Responsibility | Builder (separates construction), Facade (hides complexity) |
| **O** — Open/Closed | Strategy, Factory, Observer (extend without modifying) |
| **L** — Liskov Substitution | All patterns using interface/abstract class |
| **I** — Interface Segregation | Iterator, Observer (small, focused interfaces) |
| **D** — Dependency Inversion | Factory, Abstract Factory, Strategy (depend on abstractions) |

---

## Common Interview Questions & Answers

**Q: What is the difference between Factory Method and Abstract Factory?**
Factory Method creates **one product** via a method that subclasses override. Abstract Factory creates **families of related products** through a factory object. Use Abstract Factory when products must be used together (e.g., Windows Button + Windows Checkbox).

---

**Q: How do you make a Singleton thread-safe?**
Three options: (1) **Synchronized `getInstance()`** — safe but slow. (2) **Double-checked locking with `volatile`** — fast and safe. (3) **Bill Pugh (static inner class)** — lazy, thread-safe, no synchronization needed. (4) **Enum** — safest; handles serialization and reflection.

---

**Q: What is the difference between Strategy and State patterns?**
Both use polymorphism and look structurally similar, but their **intent** differs. Strategy selects an algorithm at runtime (the context doesn't manage transitions). State manages object behavior based on its internal state (states often transition to other states). In State, the behavior change is driven from within; in Strategy, it's driven by the client.

---

**Q: When would you use Builder over a constructor?**
When a class has **4+ parameters**, many of which are optional. Builders prevent telescoping constructors, make code more readable, allow validation before object creation, and support immutability by using `final` fields.

---

**Q: What is the difference between Adapter and Facade?**
Adapter makes **two incompatible interfaces work together** — it wraps one interface to look like another. Facade **simplifies access to a complex subsystem** — it doesn't adapt an interface, it provides a new, friendlier one on top. Adapter is about compatibility; Facade is about simplicity.

---

**Q: Can you give a real-world Java API example of the Observer pattern?**
`java.util.EventListener` in Swing, `PropertyChangeListener`, and `java.util.Observer` (deprecated). Modern examples include Spring's `ApplicationEvent`/`ApplicationListener` and reactive streams in RxJava and Project Reactor.

---

**Q: How does the Composite pattern differ from a regular collection?**
A regular collection only holds items of the same type. Composite creates a **tree structure** where both leaves and containers implement the same interface, so operations like `getSize()` or `display()` work **recursively and uniformly** on both.

---

**Q: What problem does the Iterator pattern solve?**
It decouples the **traversal logic** from the **collection structure**. Clients can iterate over an array-backed structure, a linked list, or a tree using the exact same code. It also enables multiple simultaneous iterators on the same collection.

---

## Anti-Patterns to Know

| Anti-Pattern | Description | Solution Pattern |
|---|---|---|
| **Singleton overuse** | Everything becomes global state, hard to test | Dependency Injection |
| **God Object** | One class knows and does too much | Facade, SRP |
| **Spaghetti state** | Huge if-else chains for state | State pattern |
| **Telescoping constructor** | `new Obj(a, b, null, null, true, false)` | Builder pattern |
| **Hardcoded creation** | `new ConcreteClass()` scattered everywhere | Factory / DI |

---

## Quick Cheat Sheet

```
CREATIONAL — Object Creation
├── Singleton   → One instance, global access, thread-safe with volatile/enum
├── Factory     → Decouple client from concrete type, return via interface
├── Abs Factory → Family of related objects, switch families easily
└── Builder     → Fluent API, optional params, immutable object

STRUCTURAL — Object Composition
├── Adapter     → Wrap incompatible class, expose target interface
├── Facade      → Unified simple API over complex subsystems
└── Composite   → Tree structure, Leaf + Composite share same interface

BEHAVIORAL — Object Communication
├── Strategy    → Swap algorithms at runtime via interface
├── Observer    → Subject notifies all registered observers on change
├── Iterator    → Uniform traversal, implement Iterable<T>
└── State       → Behavior changes with internal state, avoid if-else chains
```

---

*Master the **intent** of each pattern first, then the structure. Interviewers care most about knowing **when to apply** a pattern, not just reciting its code.*
