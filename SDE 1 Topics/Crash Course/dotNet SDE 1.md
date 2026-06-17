# 🔷 .NET / C# Backend — Complete Interview Notes (SDE-1)

---

## Table of Contents
1. [C# Language Fundamentals](#1-c-language-fundamentals)
2. [Object-Oriented Programming in C#](#2-object-oriented-programming-in-c)
3. [Collections & Generics](#3-collections--generics)
4. [LINQ](#4-linq)
5. [Async / Await & Multithreading](#5-async--await--multithreading)
6. [Exception Handling](#6-exception-handling)
7. [.NET Core & ASP.NET Core Basics](#7-net-core--aspnet-core-basics)
8. [Building REST APIs with ASP.NET Core](#8-building-rest-apis-with-aspnet-core)
9. [Entity Framework Core](#9-entity-framework-core)
10. [Dependency Injection in .NET](#10-dependency-injection-in-net)
11. [Middleware & Filters](#11-middleware--filters)
12. [Configuration & Environments](#12-configuration--environments)
13. [Authentication & Authorization](#13-authentication--authorization)
14. [Testing in .NET](#14-testing-in-net)
15. [Common Interview Q&A](#15-common-interview-qa)

---

## 1. C# Language Fundamentals

### Value Types vs Reference Types
This is one of the **most asked** C# fundamentals.

| | Value Types | Reference Types |
|---|---|---|
| Stored in | **Stack** | **Heap** |
| Contains | Actual value | Pointer to value |
| Examples | `int`, `double`, `bool`, `struct`, `enum` | `class`, `string`, `array`, `interface` |
| Default value | `0`, `false`, etc. | `null` |
| Copy behavior | Copies the value | Copies the reference |

```csharp
// Value type — independent copies
int a = 10;
int b = a;
b = 20;
Console.WriteLine(a); // 10 — unchanged

// Reference type — shared reference
int[] arr1 = { 1, 2, 3 };
int[] arr2 = arr1;   // both point to same array
arr2[0] = 99;
Console.WriteLine(arr1[0]); // 99 — changed!

// Exception: string is reference type but IMMUTABLE
string s1 = "hello";
string s2 = s1;
s2 = "world";        // creates a new string object
Console.WriteLine(s1); // "hello" — unchanged
```

### Nullable Types
```csharp
int  age = null;   // ❌ compile error — value type can't be null
int? age = null;   // ✅ nullable value type

// Null coalescing operator (??)
int result = age ?? 0;  // if age is null, use 0

// Null conditional operator (?.)
string name = user?.Name;        // null if user is null
int? len = user?.Name?.Length;   // safe chained access

// Null coalescing assignment (??=)
user.Name ??= "Anonymous";       // assign only if null
```

### `var` Keyword
```csharp
var name = "Ravi";       // compiler infers string
var count = 42;          // compiler infers int
var list = new List<int>(); // compiler infers List<int>

// ❌ NOT the same as JavaScript's var — still strongly typed
var x = 10;
x = "hello"; // ❌ compile error
```

### `const` vs `readonly`
```csharp
// const — compile-time constant, must assign at declaration
public const double PI = 3.14159;

// readonly — runtime constant, can assign in constructor
public readonly DateTime CreatedAt;
public MyClass() {
    CreatedAt = DateTime.Now; // ✅ allowed in constructor
}
```

### String Handling
```csharp
// String interpolation (preferred)
string name = "Ravi";
string msg = $"Hello, {name}! You are {25 + 1} years old.";

// Verbatim string (no escape sequences)
string path = @"C:\Users\Ravi\Documents";

// String methods
"hello".ToUpper()                    // "HELLO"
"  hello  ".Trim()                   // "hello"
"hello world".Contains("world")      // true
"hello world".Replace("world", "C#") // "hello C#"
"a,b,c".Split(',')                   // ["a", "b", "c"]
string.Join("-", new[]{"a","b","c"}) // "a-b-c"
string.IsNullOrEmpty(str)            // null or ""
string.IsNullOrWhiteSpace(str)       // null, "" or "   "

// StringBuilder — for heavy string concatenation
var sb = new StringBuilder();
for (int i = 0; i < 1000; i++) {
    sb.Append(i).Append(", ");
}
string result = sb.ToString();
```

### Tuples
```csharp
// Return multiple values without creating a class
(string Name, int Age) GetUser() {
    return ("Ravi", 25);
}

var user = GetUser();
Console.WriteLine(user.Name); // Ravi
Console.WriteLine(user.Age);  // 25

// Deconstruction
var (name, age) = GetUser();
```

### Pattern Matching
```csharp
object obj = "Hello";

// Type pattern
if (obj is string s) {
    Console.WriteLine(s.ToUpper());
}

// Switch expression (C# 8+)
string Describe(object o) => o switch {
    int n when n > 0  => "positive number",
    int n when n < 0  => "negative number",
    string s          => $"string of length {s.Length}",
    null              => "null",
    _                 => "something else"    // default
};
```

---

## 2. Object-Oriented Programming in C#

### The 4 Pillars

#### 1. Encapsulation
Bundling data and methods together, hiding internal state.
```csharp
public class BankAccount {
    // Private field — hidden from outside
    private decimal _balance;

    // Public property — controlled access
    public decimal Balance {
        get => _balance;
        private set {
            if (value < 0) throw new InvalidOperationException("Balance cannot be negative");
            _balance = value;
        }
    }

    public void Deposit(decimal amount) {
        if (amount <= 0) throw new ArgumentException("Amount must be positive");
        Balance += amount;
    }

    public void Withdraw(decimal amount) {
        if (amount > Balance) throw new InvalidOperationException("Insufficient funds");
        Balance -= amount;
    }
}
```

#### 2. Inheritance
A class inheriting members from a parent class.
```csharp
public class Animal {
    public string Name { get; set; }

    public Animal(string name) { Name = name; }

    public virtual void Speak() {
        Console.WriteLine($"{Name} makes a sound");
    }
}

public class Dog : Animal {
    public Dog(string name) : base(name) { }

    // Override parent method
    public override void Speak() {
        Console.WriteLine($"{Name} says: Woof!");
    }
}

public class Cat : Animal {
    public Cat(string name) : base(name) { }

    public override void Speak() {
        Console.WriteLine($"{Name} says: Meow!");
    }
}

// Polymorphism in action
Animal[] animals = { new Dog("Rex"), new Cat("Whiskers") };
foreach (var animal in animals) {
    animal.Speak(); // calls the correct override
}
// Rex says: Woof!
// Whiskers says: Meow!
```

#### 3. Abstraction — Abstract Classes vs Interfaces

**Abstract Class**
```csharp
public abstract class Shape {
    public string Color { get; set; }

    // Abstract — subclass MUST implement
    public abstract double Area();

    // Concrete — subclass inherits as-is (or overrides)
    public void Display() {
        Console.WriteLine($"Shape: {GetType().Name}, Area: {Area():F2}");
    }
}

public class Circle : Shape {
    public double Radius { get; set; }
    public override double Area() => Math.PI * Radius * Radius;
}

public class Rectangle : Shape {
    public double Width { get; set; }
    public double Height { get; set; }
    public override double Area() => Width * Height;
}
```

**Interface**
```csharp
public interface IPayable {
    decimal CalculatePayment();
    bool ProcessPayment(decimal amount);
}

public interface INotifiable {
    void SendNotification(string message);
}

// A class can implement multiple interfaces (C# has no multiple inheritance for classes)
public class Order : IPayable, INotifiable {
    public decimal CalculatePayment() => TotalAmount + DeliveryFee;
    public bool ProcessPayment(decimal amount) { /* ... */ return true; }
    public void SendNotification(string msg) { /* send email/SMS */ }
}
```

**Abstract Class vs Interface**
| | Abstract Class | Interface |
|---|---|---|
| Multiple inheritance | ❌ Only one | ✅ Many |
| Fields / state | ✅ Can have | ❌ Only properties |
| Constructor | ✅ Can have | ❌ Cannot |
| Access modifiers | ✅ Any | Public only (default) |
| Default impl | ✅ Yes | ✅ Since C# 8 (default methods) |
| Use when | "IS-A" relationship, shared code | "CAN-DO" behavior contract |

#### 4. Polymorphism
```csharp
// Compile-time (Method Overloading)
public class Calculator {
    public int Add(int a, int b) => a + b;
    public double Add(double a, double b) => a + b;
    public int Add(int a, int b, int c) => a + b + c;
}

// Runtime (Method Overriding — as shown above with virtual/override)
```

### Properties
```csharp
public class Product {
    // Auto property
    public string Name { get; set; }

    // Read-only auto property (init in constructor)
    public Guid Id { get; } = Guid.NewGuid();

    // Init-only property (C# 9) — set only during object initialization
    public string Category { get; init; }

    // Computed property
    public bool IsExpensive => Price > 1000;

    // Full property with backing field + validation
    private decimal _price;
    public decimal Price {
        get => _price;
        set {
            if (value < 0) throw new ArgumentException("Price cannot be negative");
            _price = value;
        }
    }
}

// Object initializer
var product = new Product {
    Name = "Laptop",
    Category = "Electronics",  // init-only
    Price = 999.99m
};
```

### Records (C# 9+) — Immutable Data Classes
```csharp
// Record — immutable by default, value equality, deconstruction
public record Person(string FirstName, string LastName, int Age);

var p1 = new Person("Ravi", "Kumar", 25);
var p2 = new Person("Ravi", "Kumar", 25);
Console.WriteLine(p1 == p2); // true — value equality (unlike classes)

// With expression — create modified copy
var p3 = p1 with { Age = 26 };

// Great for DTOs, API responses
public record ProductResponseDto(int Id, string Name, decimal Price);
```

### Structs
```csharp
// Struct — value type, stack allocated, no inheritance
public struct Point {
    public int X { get; set; }
    public int Y { get; set; }

    public double DistanceTo(Point other) {
        int dx = X - other.X, dy = Y - other.Y;
        return Math.Sqrt(dx * dx + dy * dy);
    }
}

// Use structs for: small, frequently created, value-semantic data
// (coordinates, colors, money amounts)
```

### `sealed` Class
```csharp
// Cannot be inherited
public sealed class Singleton {
    private static Singleton _instance;
    private Singleton() { }
    public static Singleton Instance => _instance ??= new Singleton();
}
```

---

## 3. Collections & Generics

### Common Collections
```csharp
// List<T> — dynamic array, order preserved, duplicates allowed
var numbers = new List<int> { 3, 1, 4, 1, 5 };
numbers.Add(9);
numbers.Remove(1);           // removes first occurrence
numbers.RemoveAt(0);         // removes at index
numbers.Contains(4);         // true
numbers.Count;               // length
numbers.Sort();
numbers.Find(x => x > 4);   // 5

// Dictionary<TKey, TValue> — key-value pairs, O(1) lookup
var scores = new Dictionary<string, int> {
    ["Alice"] = 95,
    ["Bob"] = 87
};
scores["Charlie"] = 92;
scores.ContainsKey("Alice");         // true
scores.TryGetValue("Dave", out int val); // safe get — no exception
foreach (var (key, value) in scores) {
    Console.WriteLine($"{key}: {value}");
}

// HashSet<T> — unique elements, O(1) lookup
var unique = new HashSet<int> { 1, 2, 3 };
unique.Add(2);             // no-op, already exists
unique.Contains(1);        // true

// Queue<T> — FIFO
var queue = new Queue<string>();
queue.Enqueue("first");
queue.Enqueue("second");
queue.Dequeue();           // "first"
queue.Peek();              // "second" — look without removing

// Stack<T> — LIFO
var stack = new Stack<int>();
stack.Push(1); stack.Push(2); stack.Push(3);
stack.Pop();               // 3
stack.Peek();              // 2
```

### Generics
```csharp
// Generic class — reusable for any type
public class Repository<T> {
    private readonly List<T> _items = new();

    public void Add(T item) => _items.Add(item);
    public T GetById(int index) => _items[index];
    public IEnumerable<T> GetAll() => _items;
}

// Usage
var userRepo = new Repository<User>();
var productRepo = new Repository<Product>();

// Generic method
public T Max<T>(T a, T b) where T : IComparable<T> {
    return a.CompareTo(b) >= 0 ? a : b;
}

// Generic constraints
public class Service<T> where T : class, IEntity, new() {
    // T must be: a class, implement IEntity, have parameterless constructor
}
```

### IEnumerable vs ICollection vs IList
```
IEnumerable<T>       → read-only, forward-only iteration (foreach)
    ↓ extends
ICollection<T>       → adds Count, Add, Remove, Contains
    ↓ extends
IList<T>             → adds index access (list[0]), Insert, RemoveAt
    ↓ implements
List<T>              → concrete implementation
```

---

## 4. LINQ

### What is LINQ?
**L**anguage **IN**tegrated **Q**uery — query collections (lists, arrays, DBs) using C# syntax instead of loops.

```csharp
var products = new List<Product> {
    new Product { Id=1, Name="Laptop",  Price=999, Category="Electronics" },
    new Product { Id=2, Name="Phone",   Price=499, Category="Electronics" },
    new Product { Id=3, Name="Chair",   Price=199, Category="Furniture"   },
    new Product { Id=4, Name="Desk",    Price=299, Category="Furniture"   },
    new Product { Id=5, Name="Monitor", Price=399, Category="Electronics" },
};
```

### Query Syntax vs Method Syntax
```csharp
// QUERY SYNTAX (SQL-like)
var electronics = from p in products
                  where p.Category == "Electronics"
                  orderby p.Price descending
                  select p;

// METHOD SYNTAX (Lambda — more common in practice)
var electronics = products
    .Where(p => p.Category == "Electronics")
    .OrderByDescending(p => p.Price);
```

### Most Important LINQ Methods
```csharp
// Filtering
products.Where(p => p.Price > 300)

// Projection — transform each element
products.Select(p => new { p.Name, p.Price })          // anonymous type
products.Select(p => p.Name)                           // IEnumerable<string>

// Ordering
products.OrderBy(p => p.Price)
products.OrderByDescending(p => p.Price)
products.ThenBy(p => p.Name)                          // secondary sort

// Aggregation
products.Count()                                       // 5
products.Count(p => p.Price > 300)                    // 3
products.Sum(p => p.Price)                            // 2395
products.Average(p => p.Price)                        // 479
products.Min(p => p.Price)                            // 199
products.Max(p => p.Price)                            // 999

// Element
products.First(p => p.Category == "Furniture")        // throws if not found
products.FirstOrDefault(p => p.Price > 5000)          // null if not found ✅
products.Single(p => p.Id == 1)                       // throws if 0 or >1
products.SingleOrDefault(p => p.Id == 99)             // null if not found

// Existence
products.Any(p => p.Price > 900)                      // true
products.All(p => p.Price > 0)                        // true
products.Contains(someProduct)                         // true/false

// Grouping
var grouped = products.GroupBy(p => p.Category);
foreach (var group in grouped) {
    Console.WriteLine($"{group.Key}: {group.Count()} items");
    foreach (var item in group) Console.WriteLine($"  {item.Name}");
}

// Pagination
products.Skip(10).Take(10)  // page 2, 10 items per page

// Flattening
var tags = products.SelectMany(p => p.Tags); // flatten list of lists

// Joining
var result = products.Join(
    categories,
    p => p.CategoryId,
    c => c.Id,
    (p, c) => new { p.Name, CategoryName = c.Name }
);

// Distinct / Distinct by
products.Select(p => p.Category).Distinct()   // unique categories

// ToList / ToArray / ToDictionary
products.Where(p => p.Price > 300).ToList()
products.ToDictionary(p => p.Id, p => p.Name) // { 1: "Laptop", 2: "Phone" }
```

### LINQ Deferred Execution
```csharp
// Query is NOT executed here — just defined
var query = products.Where(p => p.Price > 300);

// Executed only when iterated (ToList, foreach, Count, etc.)
var result = query.ToList(); // NOW it executes

// Importance: if underlying data changes between definition and execution,
// you get the updated data. Always use .ToList() to materialize when needed.
```

---

## 5. Async / Await & Multithreading

### Why Async?
```
Synchronous (blocking):
  Thread waits idle while DB/API responds → wastes thread → fewer concurrent users

Asynchronous (non-blocking):
  Thread is released during wait → handles other requests → more concurrent users
```

### async / await Basics
```csharp
// Synchronous — blocks the thread
public string GetDataSync() {
    Thread.Sleep(2000);       // thread blocked for 2 seconds
    return "data";
}

// Asynchronous — thread released during wait
public async Task<string> GetDataAsync() {
    await Task.Delay(2000);   // thread released, returns when done
    return "data";
}

// Calling async method
public async Task RunAsync() {
    string data = await GetDataAsync(); // await = wait here, but don't block thread
    Console.WriteLine(data);
}
```

### Task — The Core Type
```csharp
Task        // async operation with no return value (like void)
Task<T>     // async operation returning T

// Running parallel tasks
var task1 = FetchUsersAsync();
var task2 = FetchProductsAsync();
await Task.WhenAll(task1, task2);  // wait for BOTH to complete
var users = await task1;
var products = await task2;

// Wait for FIRST to complete
var completed = await Task.WhenAny(task1, task2);

// Task.Run — offload CPU-bound work to thread pool
var result = await Task.Run(() => HeavyCpuComputation());
```

### Real Example — Async API Call
```csharp
public class ProductService {
    private readonly HttpClient _httpClient;
    private readonly IProductRepository _repo;

    // ✅ Always async all the way down — don't mix sync/async
    public async Task<List<Product>> GetProductsAsync(CancellationToken cancellationToken = default) {
        return await _repo.GetAllAsync(cancellationToken);
    }

    public async Task<ProductDto> GetByIdAsync(int id) {
        var product = await _repo.GetByIdAsync(id)
            ?? throw new NotFoundException($"Product {id} not found");
        return MapToDto(product);
    }

    // Async with external HTTP call
    public async Task<string> FetchExternalDataAsync() {
        var response = await _httpClient.GetAsync("https://api.example.com/data");
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadAsStringAsync();
    }
}
```

### CancellationToken — Let clients cancel requests
```csharp
[HttpGet]
public async Task<IActionResult> GetProducts(CancellationToken cancellationToken) {
    // If client disconnects, cancellationToken is triggered
    var products = await _service.GetAllAsync(cancellationToken);
    return Ok(products);
}
```

### Common Async Mistakes
```csharp
// ❌ async void — can't be awaited, exceptions are unhandled
public async void DoWork() { await Task.Delay(1000); }

// ✅ async Task instead
public async Task DoWorkAsync() { await Task.Delay(1000); }

// ❌ .Result / .Wait() — causes deadlock in ASP.NET Core
var data = GetDataAsync().Result;       // DEADLOCK ❌

// ✅ await instead
var data = await GetDataAsync();        // ✅

// ❌ Forgetting await — fires and forgets (silent errors)
SaveDataAsync(data);     // no await — not waited, errors ignored

// ✅
await SaveDataAsync(data);
```

---

## 6. Exception Handling

### try / catch / finally
```csharp
public Product GetProduct(int id) {
    try {
        var product = _repo.GetById(id);
        if (product == null)
            throw new NotFoundException($"Product {id} not found");
        return product;
    }
    catch (NotFoundException ex) {
        _logger.LogWarning("Product not found: {Id}", id);
        throw;  // re-throw — preserve original stack trace
    }
    catch (SqlException ex) {
        _logger.LogError(ex, "DB error fetching product {Id}", id);
        throw new ServiceException("Database error occurred", ex);  // wrap
    }
    catch (Exception ex) when (ex is not OperationCanceledException) {
        // Exception filter — only catch if condition true
        _logger.LogError(ex, "Unexpected error");
        throw;
    }
    finally {
        // Always runs — for cleanup (close connections, dispose, etc.)
        _logger.LogDebug("GetProduct completed for id: {Id}", id);
    }
}
```

### Custom Exceptions
```csharp
// Custom exception hierarchy
public class AppException : Exception {
    public int StatusCode { get; }
    public AppException(string message, int statusCode = 500)
        : base(message) {
        StatusCode = statusCode;
    }
}

public class NotFoundException : AppException {
    public NotFoundException(string message) : base(message, 404) { }
}

public class ValidationException : AppException {
    public List<string> Errors { get; }
    public ValidationException(List<string> errors)
        : base("Validation failed", 400) {
        Errors = errors;
    }
}

public class UnauthorizedException : AppException {
    public UnauthorizedException() : base("Unauthorized", 401) { }
}
```

### Global Exception Handling — Middleware
```csharp
public class GlobalExceptionMiddleware {
    private readonly RequestDelegate _next;
    private readonly ILogger<GlobalExceptionMiddleware> _logger;

    public GlobalExceptionMiddleware(RequestDelegate next, ILogger<GlobalExceptionMiddleware> logger) {
        _next = next;
        _logger = logger;
    }

    public async Task InvokeAsync(HttpContext context) {
        try {
            await _next(context);
        }
        catch (NotFoundException ex) {
            await WriteErrorResponse(context, 404, ex.Message);
        }
        catch (ValidationException ex) {
            await WriteErrorResponse(context, 400, ex.Message, ex.Errors);
        }
        catch (Exception ex) {
            _logger.LogError(ex, "Unhandled exception");
            await WriteErrorResponse(context, 500, "An internal server error occurred");
        }
    }

    private static async Task WriteErrorResponse(HttpContext context, int statusCode,
        string message, List<string> errors = null) {
        context.Response.ContentType = "application/json";
        context.Response.StatusCode = statusCode;
        var response = new { status = statusCode, message, errors, timestamp = DateTime.UtcNow };
        await context.Response.WriteAsJsonAsync(response);
    }
}

// Register in Program.cs
app.UseMiddleware<GlobalExceptionMiddleware>();
```

---

## 7. .NET Core & ASP.NET Core Basics

### What is .NET Core?
- **Cross-platform** — Windows, Linux, macOS
- **High-performance** — one of the fastest web frameworks (TechEmpower benchmarks)
- **Open-source** — github.com/dotnet
- **Modular** — include only what you need (NuGet packages)

### .NET Version History (Key Points)
| Version | Notes |
|---|---|
| .NET Framework | Windows-only, legacy |
| .NET Core 1-3 | Cross-platform, new start |
| .NET 5 | Unified platform (merged Core + Framework) |
| .NET 6 / 7 / 8 | Current LTS versions, minimal APIs introduced |

### Program.cs — Modern .NET 6+ Structure
```csharp
// No Startup.cs in .NET 6+, everything in Program.cs
var builder = WebApplication.CreateBuilder(args);

// 1. REGISTER SERVICES (Dependency Injection)
builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();
builder.Services.AddDbContext<AppDbContext>(options =>
    options.UseSqlServer(builder.Configuration.GetConnectionString("DefaultConnection")));
builder.Services.AddScoped<IProductService, ProductService>();
builder.Services.AddScoped<IProductRepository, ProductRepository>();

var app = builder.Build();

// 2. CONFIGURE MIDDLEWARE PIPELINE
if (app.Environment.IsDevelopment()) {
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();
app.UseAuthentication();
app.UseAuthorization();
app.UseMiddleware<GlobalExceptionMiddleware>();
app.MapControllers();

app.Run();
```

### The Request Pipeline — How It Works
```
Incoming HTTP Request
        ↓
[Middleware 1] Exception Handler
        ↓
[Middleware 2] HTTPS Redirection
        ↓
[Middleware 3] Authentication
        ↓
[Middleware 4] Authorization
        ↓
[Middleware 5] Routing → matches URL to Controller
        ↓
[Filter 1] Action Filter (before)
        ↓
Controller Action Method executes
        ↓
[Filter 2] Action Filter (after)
        ↓
Response flows back through middleware in reverse
```

---

## 8. Building REST APIs with ASP.NET Core

### Full CRUD Controller
```csharp
[ApiController]                    // enables model validation, problem details
[Route("api/v1/[controller]")]     // → api/v1/products
public class ProductsController : ControllerBase {

    private readonly IProductService _service;
    private readonly ILogger<ProductsController> _logger;

    public ProductsController(IProductService service, ILogger<ProductsController> logger) {
        _service = service;
        _logger = logger;
    }

    // GET api/v1/products?page=1&size=10&category=electronics
    [HttpGet]
    [ProducesResponseType(typeof(PagedResult<ProductDto>), StatusCodes.Status200OK)]
    public async Task<IActionResult> GetAll(
        [FromQuery] int page = 1,
        [FromQuery] int size = 10,
        [FromQuery] string? category = null,
        CancellationToken cancellationToken = default) {
        var result = await _service.GetAllAsync(page, size, category, cancellationToken);
        return Ok(result);
    }

    // GET api/v1/products/5
    [HttpGet("{id:int}")]
    [ProducesResponseType(typeof(ProductDto), StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    public async Task<IActionResult> GetById(int id) {
        var product = await _service.GetByIdAsync(id);
        return Ok(product);  // NotFoundException handled globally
    }

    // POST api/v1/products
    [HttpPost]
    [ProducesResponseType(typeof(ProductDto), StatusCodes.Status201Created)]
    [ProducesResponseType(StatusCodes.Status400BadRequest)]
    public async Task<IActionResult> Create([FromBody] CreateProductDto dto) {
        var created = await _service.CreateAsync(dto);
        return CreatedAtAction(nameof(GetById), new { id = created.Id }, created);
    }

    // PUT api/v1/products/5
    [HttpPut("{id:int}")]
    [ProducesResponseType(typeof(ProductDto), StatusCodes.Status200OK)]
    public async Task<IActionResult> Update(int id, [FromBody] UpdateProductDto dto) {
        var updated = await _service.UpdateAsync(id, dto);
        return Ok(updated);
    }

    // PATCH api/v1/products/5/status
    [HttpPatch("{id:int}/status")]
    public async Task<IActionResult> UpdateStatus(int id, [FromBody] UpdateStatusDto dto) {
        await _service.UpdateStatusAsync(id, dto.IsActive);
        return NoContent();
    }

    // DELETE api/v1/products/5
    [HttpDelete("{id:int}")]
    [ProducesResponseType(StatusCodes.Status204NoContent)]
    public async Task<IActionResult> Delete(int id) {
        await _service.DeleteAsync(id);
        return NoContent();
    }
}
```

### DTOs (Data Transfer Objects)
```csharp
// Request DTOs
public record CreateProductDto(
    [Required][StringLength(200)] string Name,
    [Required][StringLength(100)] string Category,
    [Range(0.01, 1000000)] decimal Price,
    string? Description
);

public record UpdateProductDto(
    [Required][StringLength(200)] string Name,
    [Range(0.01, 1000000)] decimal Price,
    string? Description
);

// Response DTO
public record ProductDto(
    int Id,
    string Name,
    string Category,
    decimal Price,
    string? Description,
    bool IsActive,
    DateTime CreatedAt
);

// Pagination wrapper
public class PagedResult<T> {
    public List<T> Items { get; set; } = new();
    public int Page { get; set; }
    public int PageSize { get; set; }
    public int TotalCount { get; set; }
    public int TotalPages => (int)Math.Ceiling((double)TotalCount / PageSize);
    public bool HasNext => Page < TotalPages;
    public bool HasPrevious => Page > 1;
}
```

### Route Constraints
```csharp
[HttpGet("{id:int}")]           // id must be integer
[HttpGet("{name:alpha}")]       // name must be letters only
[HttpGet("{slug:minlength(3)}")] // minimum length 3
[HttpGet("{id:int:min(1)}")]    // int, minimum value 1
[HttpGet("{date:datetime}")]    // must be valid datetime
```

---

## 9. Entity Framework Core

### What is EF Core?
EF Core is an **ORM (Object-Relational Mapper)** — maps C# classes to database tables, letting you query databases using C# instead of raw SQL.

### DbContext Setup
```csharp
public class AppDbContext : DbContext {

    public AppDbContext(DbContextOptions<AppDbContext> options) : base(options) { }

    // Tables
    public DbSet<Product> Products { get; set; }
    public DbSet<Category> Categories { get; set; }
    public DbSet<Order> Orders { get; set; }
    public DbSet<OrderItem> OrderItems { get; set; }

    protected override void OnModelCreating(ModelBuilder modelBuilder) {
        // Configure Product entity
        modelBuilder.Entity<Product>(entity => {
            entity.HasKey(p => p.Id);
            entity.Property(p => p.Name).IsRequired().HasMaxLength(200);
            entity.Property(p => p.Price).HasPrecision(18, 2);
            entity.HasIndex(p => p.Sku).IsUnique();

            // Relationship
            entity.HasOne(p => p.Category)
                  .WithMany(c => c.Products)
                  .HasForeignKey(p => p.CategoryId)
                  .OnDelete(DeleteBehavior.Restrict);
        });

        // Seed data
        modelBuilder.Entity<Category>().HasData(
            new Category { Id = 1, Name = "Electronics" },
            new Category { Id = 2, Name = "Furniture" }
        );
    }
}
```

### Entity / Model
```csharp
public class Product {
    public int Id { get; set; }
    public string Name { get; set; } = string.Empty;
    public string Sku { get; set; } = string.Empty;
    public decimal Price { get; set; }
    public bool IsActive { get; set; } = true;
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    // Navigation properties (EF Core relationships)
    public int CategoryId { get; set; }
    public Category Category { get; set; } = null!;  // null! = not null in runtime
    public ICollection<OrderItem> OrderItems { get; set; } = new List<OrderItem>();
}
```

### Repository Pattern with EF Core
```csharp
public interface IProductRepository {
    Task<List<Product>> GetAllAsync(CancellationToken ct = default);
    Task<Product?> GetByIdAsync(int id, CancellationToken ct = default);
    Task<Product> AddAsync(Product product);
    Task UpdateAsync(Product product);
    Task DeleteAsync(int id);
    Task<bool> ExistsAsync(int id);
}

public class ProductRepository : IProductRepository {
    private readonly AppDbContext _context;

    public ProductRepository(AppDbContext context) {
        _context = context;
    }

    public async Task<List<Product>> GetAllAsync(CancellationToken ct = default) {
        return await _context.Products
            .Include(p => p.Category)        // eager load related data
            .Where(p => p.IsActive)
            .OrderBy(p => p.Name)
            .AsNoTracking()                  // read-only → better performance
            .ToListAsync(ct);
    }

    public async Task<Product?> GetByIdAsync(int id, CancellationToken ct = default) {
        return await _context.Products
            .Include(p => p.Category)
            .FirstOrDefaultAsync(p => p.Id == id, ct);
    }

    public async Task<Product> AddAsync(Product product) {
        _context.Products.Add(product);
        await _context.SaveChangesAsync();
        return product;
    }

    public async Task UpdateAsync(Product product) {
        _context.Products.Update(product);
        await _context.SaveChangesAsync();
    }

    public async Task DeleteAsync(int id) {
        var product = await _context.Products.FindAsync(id)
            ?? throw new NotFoundException($"Product {id} not found");
        _context.Products.Remove(product);
        await _context.SaveChangesAsync();
    }

    public async Task<bool> ExistsAsync(int id) {
        return await _context.Products.AnyAsync(p => p.Id == id);
    }
}
```

### Migrations — Code First
```bash
# Add a new migration (after changing your model)
dotnet ef migrations add AddProductSkuColumn

# Apply migrations to database
dotnet ef database update

# Rollback to specific migration
dotnet ef database update PreviousMigrationName

# Remove last migration (if not yet applied)
dotnet ef migrations remove

# Generate SQL script (for production)
dotnet ef migrations script
```

### LINQ Queries with EF Core
```csharp
// Filtering
var products = await _context.Products
    .Where(p => p.Price >= minPrice && p.Price <= maxPrice)
    .ToListAsync();

// Include related entities (avoid N+1)
var orders = await _context.Orders
    .Include(o => o.Customer)
    .Include(o => o.Items)
        .ThenInclude(i => i.Product)  // nested include
    .ToListAsync();

// Projection — select only needed columns (more efficient)
var names = await _context.Products
    .Where(p => p.IsActive)
    .Select(p => new { p.Id, p.Name, p.Price })
    .ToListAsync();

// Raw SQL (when LINQ isn't enough)
var products = await _context.Products
    .FromSqlRaw("SELECT * FROM Products WHERE Price > {0}", minPrice)
    .ToListAsync();

// Pagination
var paged = await _context.Products
    .OrderBy(p => p.Name)
    .Skip((page - 1) * size)
    .Take(size)
    .ToListAsync();
```

### Tracking vs No-Tracking
```csharp
// Tracking (default) — EF watches for changes, needed for UPDATE/DELETE
var product = await _context.Products.FindAsync(id);
product.Price = 999;
await _context.SaveChangesAsync(); // EF detects change, runs UPDATE

// No-Tracking — for read-only queries, faster (no change detection overhead)
var products = await _context.Products
    .AsNoTracking()
    .ToListAsync();
```

---

## 10. Dependency Injection in .NET

### Service Lifetimes — Critical to Know
| Lifetime | Registered via | Created | Destroyed | Use for |
|---|---|---|---|---|
| **Singleton** | `AddSingleton` | Once, app start | App shutdown | Config, cache, logging, HttpClient |
| **Scoped** | `AddScoped` | Once per HTTP request | End of request | DbContext, repositories, services ✅ |
| **Transient** | `AddTransient` | Every time requested | After use | Lightweight, stateless services |

```csharp
// Registration in Program.cs
builder.Services.AddSingleton<IConfiguration>(builder.Configuration);
builder.Services.AddSingleton<ICacheService, RedisCacheService>();

builder.Services.AddScoped<AppDbContext>();          // ✅ DbContext must be Scoped
builder.Services.AddScoped<IProductRepository, ProductRepository>();
builder.Services.AddScoped<IProductService, ProductService>();

builder.Services.AddTransient<IEmailService, SmtpEmailService>();
builder.Services.AddTransient<IPasswordHasher, BCryptPasswordHasher>();

// Register HttpClient with DI (managed lifetime)
builder.Services.AddHttpClient<IPaymentService, PaymentService>(client => {
    client.BaseAddress = new Uri("https://payment-api.example.com");
    client.Timeout = TimeSpan.FromSeconds(30);
});
```

### Constructor Injection (Standard Pattern)
```csharp
public class ProductService : IProductService {
    private readonly IProductRepository _repo;
    private readonly ILogger<ProductService> _logger;
    private readonly ICacheService _cache;

    // Constructor injection — all dependencies declared explicitly
    public ProductService(
        IProductRepository repo,
        ILogger<ProductService> logger,
        ICacheService cache) {
        _repo = repo;
        _logger = logger;
        _cache = cache;
    }
}
```

### ⚠️ Captive Dependency — Common Mistake
```csharp
// ❌ WRONG — Singleton capturing a Scoped service
// Scoped service (DbContext) lives longer than intended → bugs!
builder.Services.AddSingleton<IMyService, MyService>(); // Singleton
// MyService depends on IProductRepository (Scoped)
// → Scoped repo lives as long as Singleton → stale DbContext

// ✅ FIX — Make MyService Scoped too, or use IServiceScopeFactory
public class MySingletonService {
    private readonly IServiceScopeFactory _scopeFactory;
    public MySingletonService(IServiceScopeFactory scopeFactory) {
        _scopeFactory = scopeFactory;
    }
    public void DoWork() {
        using var scope = _scopeFactory.CreateScope();
        var repo = scope.ServiceProvider.GetRequiredService<IProductRepository>();
        // use repo
    }
}
```

---

## 11. Middleware & Filters

### Custom Middleware
```csharp
// Middleware — processes EVERY request in the pipeline
public class RequestLoggingMiddleware {
    private readonly RequestDelegate _next;
    private readonly ILogger<RequestLoggingMiddleware> _logger;

    public RequestLoggingMiddleware(RequestDelegate next, ILogger<RequestLoggingMiddleware> logger) {
        _next = next;
        _logger = logger;
    }

    public async Task InvokeAsync(HttpContext context) {
        var stopwatch = Stopwatch.StartNew();

        _logger.LogInformation("→ {Method} {Path}",
            context.Request.Method, context.Request.Path);

        await _next(context); // call next middleware

        stopwatch.Stop();
        _logger.LogInformation("← {StatusCode} in {Ms}ms",
            context.Response.StatusCode, stopwatch.ElapsedMilliseconds);
    }
}

// Register
app.UseMiddleware<RequestLoggingMiddleware>();
```

### Action Filters — Scoped to Controllers/Actions
```csharp
// Filter — runs before/after controller action methods
public class ValidateModelFilter : IActionFilter {

    public void OnActionExecuting(ActionExecutingContext context) {
        // Runs BEFORE the action
        if (!context.ModelState.IsValid) {
            var errors = context.ModelState
                .Where(m => m.Value!.Errors.Count > 0)
                .ToDictionary(
                    m => m.Key,
                    m => m.Value!.Errors.Select(e => e.ErrorMessage).ToArray()
                );
            context.Result = new BadRequestObjectResult(new { errors });
        }
    }

    public void OnActionExecuted(ActionExecutedContext context) {
        // Runs AFTER the action
    }
}

// Apply globally
builder.Services.AddControllers(options => {
    options.Filters.Add<ValidateModelFilter>();
});

// Apply to specific controller or action
[ServiceFilter(typeof(ValidateModelFilter))]
public class ProductsController : ControllerBase { }
```

### Middleware vs Filter
| | Middleware | Filter |
|---|---|---|
| Scope | Entire pipeline | MVC layer only |
| Order | Before routing | After routing |
| Access to MVC context | ❌ No | ✅ Yes (ModelState, ActionDescriptor) |
| Use for | Auth, logging, error handling, CORS | Validation, logging per-action, caching |

---

## 12. Configuration & Environments

### appsettings.json
```json
{
  "ConnectionStrings": {
    "DefaultConnection": "Server=localhost;Database=MyApp;Trusted_Connection=true;"
  },
  "Jwt": {
    "SecretKey": "your-secret-key-here",
    "Issuer": "MyApp",
    "Audience": "MyAppUsers",
    "ExpiryMinutes": 60
  },
  "App": {
    "Name": "My Application",
    "MaxPageSize": 100,
    "AllowedOrigins": ["https://myfrontend.com"]
  },
  "Logging": {
    "LogLevel": {
      "Default": "Information",
      "Microsoft.AspNetCore": "Warning"
    }
  }
}
```

### Environment-Specific Config
```
appsettings.json            → base config (always loaded)
appsettings.Development.json → overrides for local dev
appsettings.Staging.json    → overrides for staging
appsettings.Production.json → overrides for production
```

### Options Pattern — Strongly Typed Config
```csharp
// Config class
public class JwtSettings {
    public string SecretKey { get; set; } = string.Empty;
    public string Issuer { get; set; } = string.Empty;
    public string Audience { get; set; } = string.Empty;
    public int ExpiryMinutes { get; set; } = 60;
}

// Register
builder.Services.Configure<JwtSettings>(
    builder.Configuration.GetSection("Jwt"));

// Inject and use
public class TokenService {
    private readonly JwtSettings _jwtSettings;

    public TokenService(IOptions<JwtSettings> options) {
        _jwtSettings = options.Value;
    }

    public string GenerateToken(User user) {
        var key = new SymmetricSecurityKey(
            Encoding.UTF8.GetBytes(_jwtSettings.SecretKey));
        // ...
    }
}
```

### Reading Config Values
```csharp
// Direct read (less preferred)
var connStr = builder.Configuration.GetConnectionString("DefaultConnection");
var appName = builder.Configuration["App:Name"];

// Strongly typed (preferred — Options Pattern above)
```

---

## 13. Authentication & Authorization

### JWT Authentication Setup
```csharp
// Program.cs
builder.Services.AddAuthentication(JwtBearerDefaults.AuthenticationScheme)
    .AddJwtBearer(options => {
        var jwtSettings = builder.Configuration.GetSection("Jwt").Get<JwtSettings>()!;
        options.TokenValidationParameters = new TokenValidationParameters {
            ValidateIssuer = true,
            ValidateAudience = true,
            ValidateLifetime = true,
            ValidateIssuerSigningKey = true,
            ValidIssuer = jwtSettings.Issuer,
            ValidAudience = jwtSettings.Audience,
            IssuerSigningKey = new SymmetricSecurityKey(
                Encoding.UTF8.GetBytes(jwtSettings.SecretKey))
        };
    });

builder.Services.AddAuthorization();

// Middleware (order matters!)
app.UseAuthentication();  // first
app.UseAuthorization();   // second
```

### Generate JWT Token
```csharp
public class TokenService {
    private readonly JwtSettings _settings;

    public string GenerateToken(User user) {
        var claims = new[] {
            new Claim(ClaimTypes.NameIdentifier, user.Id.ToString()),
            new Claim(ClaimTypes.Email, user.Email),
            new Claim(ClaimTypes.Role, user.Role),
            new Claim("fullName", user.FullName)
        };

        var key = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(_settings.SecretKey));
        var creds = new SigningCredentials(key, SecurityAlgorithms.HmacSha256);

        var token = new JwtSecurityToken(
            issuer: _settings.Issuer,
            audience: _settings.Audience,
            claims: claims,
            expires: DateTime.UtcNow.AddMinutes(_settings.ExpiryMinutes),
            signingCredentials: creds
        );

        return new JwtSecurityTokenHandler().WriteToken(token);
    }
}
```

### Authorization Attributes
```csharp
[Authorize]                          // any authenticated user
[Authorize(Roles = "Admin")]         // only Admin role
[Authorize(Roles = "Admin,Manager")] // Admin OR Manager
[AllowAnonymous]                     // no auth required (overrides [Authorize])

[ApiController]
[Route("api/[controller]")]
[Authorize]                          // all endpoints in controller require auth
public class ProductsController : ControllerBase {

    [HttpGet]
    public async Task<IActionResult> GetAll() { ... }  // requires auth

    [HttpGet("public")]
    [AllowAnonymous]                 // this one doesn't
    public async Task<IActionResult> GetPublic() { ... }

    [HttpDelete("{id}")]
    [Authorize(Roles = "Admin")]     // only admin
    public async Task<IActionResult> Delete(int id) { ... }
}
```

### Policy-Based Authorization
```csharp
// Register policy
builder.Services.AddAuthorization(options => {
    options.AddPolicy("MinimumAge", policy =>
        policy.RequireClaim("age", "18"));
    options.AddPolicy("PremiumUser", policy =>
        policy.RequireRole("Premium")
              .RequireClaim("subscriptionActive", "true"));
});

// Use
[Authorize(Policy = "PremiumUser")]
public IActionResult GetPremiumContent() { ... }
```

---

## 14. Testing in .NET

### Unit Testing with xUnit + Moq
```csharp
// Test project: install xUnit, Moq, FluentAssertions

public class ProductServiceTests {
    private readonly Mock<IProductRepository> _mockRepo;
    private readonly Mock<ILogger<ProductService>> _mockLogger;
    private readonly ProductService _service;

    public ProductServiceTests() {
        _mockRepo = new Mock<IProductRepository>();
        _mockLogger = new Mock<ILogger<ProductService>>();
        _service = new ProductService(_mockRepo.Object, _mockLogger.Object);
    }

    [Fact]
    public async Task GetByIdAsync_WhenProductExists_ReturnsProduct() {
        // Arrange
        var product = new Product { Id = 1, Name = "Laptop", Price = 999 };
        _mockRepo.Setup(r => r.GetByIdAsync(1, default))
                 .ReturnsAsync(product);

        // Act
        var result = await _service.GetByIdAsync(1);

        // Assert
        result.Should().NotBeNull();           // FluentAssertions
        result.Id.Should().Be(1);
        result.Name.Should().Be("Laptop");
        _mockRepo.Verify(r => r.GetByIdAsync(1, default), Times.Once);
    }

    [Fact]
    public async Task GetByIdAsync_WhenNotFound_ThrowsNotFoundException() {
        // Arrange
        _mockRepo.Setup(r => r.GetByIdAsync(99, default))
                 .ReturnsAsync((Product?)null);

        // Act & Assert
        await Assert.ThrowsAsync<NotFoundException>(
            () => _service.GetByIdAsync(99));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    [InlineData(-100)]
    public async Task CreateAsync_WithInvalidPrice_ThrowsValidationException(decimal price) {
        var dto = new CreateProductDto("Test", "Category", price, null);

        await Assert.ThrowsAsync<ValidationException>(
            () => _service.CreateAsync(dto));
    }
}
```

### Integration Testing with WebApplicationFactory
```csharp
public class ProductsControllerTests : IClassFixture<WebApplicationFactory<Program>> {
    private readonly HttpClient _client;

    public ProductsControllerTests(WebApplicationFactory<Program> factory) {
        _client = factory.WithWebHostBuilder(builder => {
            builder.ConfigureServices(services => {
                // Replace real DB with in-memory DB for tests
                services.RemoveAll<DbContextOptions<AppDbContext>>();
                services.AddDbContext<AppDbContext>(options =>
                    options.UseInMemoryDatabase("TestDb"));
            });
        }).CreateClient();
    }

    [Fact]
    public async Task GetAll_ReturnsOkWithProducts() {
        var response = await _client.GetAsync("/api/v1/products");

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var products = await response.Content.ReadFromJsonAsync<List<ProductDto>>();
        products.Should().NotBeNull();
    }

    [Fact]
    public async Task Create_ValidProduct_Returns201() {
        var dto = new CreateProductDto("Test Laptop", "Electronics", 999.99m, null);

        var response = await _client.PostAsJsonAsync("/api/v1/products", dto);

        response.StatusCode.Should().Be(HttpStatusCode.Created);
        var created = await response.Content.ReadFromJsonAsync<ProductDto>();
        created!.Name.Should().Be("Test Laptop");
    }
}
```

---

## 15. Common Interview Q&A

**Q: What is the difference between `IEnumerable` and `IQueryable`?**
> `IEnumerable` executes the query **in memory** — data is loaded from DB first, then filtered in C#. `IQueryable` translates the query to **SQL and executes it in the database** — only matching rows are fetched. Always use `IQueryable` (EF Core default) for DB queries; use `IEnumerable` after `.ToList()` when working in-memory.
```csharp
// IQueryable — SQL: SELECT * FROM Products WHERE IsActive = 1
IQueryable<Product> query = _context.Products.Where(p => p.IsActive);

// IEnumerable — fetches ALL products, filters in C# memory ❌ inefficient
IEnumerable<Product> all = _context.Products.ToList().Where(p => p.IsActive);
```

---

**Q: What is the difference between `==` and `.Equals()` in C#?**
> For **value types**, both compare values. For **reference types**, `==` compares **references** (same object in memory) by default, while `.Equals()` can be overridden to compare **values**. `string` overrides `==` to compare content. Records also override `==` for value equality.
```csharp
string a = new string("hello");
string b = new string("hello");
Console.WriteLine(a == b);       // true  — string overrides ==
Console.WriteLine(a.Equals(b));  // true

object x = new object();
object y = new object();
Console.WriteLine(x == y);       // false — different references
Console.WriteLine(x.Equals(y));  // false
```

---

**Q: What is boxing and unboxing?**
> **Boxing** = converting a value type to `object` (heap allocation). **Unboxing** = converting back. Both are expensive — avoid in hot paths.
```csharp
int num = 42;
object boxed = num;      // Boxing — heap allocation
int unboxed = (int)boxed; // Unboxing — must cast explicitly
```

---

**Q: What is the difference between `abstract class` and `interface`?**
> Use **abstract class** when classes share code and have an "is-a" relationship with common state. Use **interface** when you want to define a capability/contract that unrelated classes can implement. A class can implement **multiple interfaces** but inherit only **one abstract class**.

---

**Q: Explain `async`/`await`. What happens under the hood?**
> When you `await` an async operation, the compiler transforms the method into a **state machine**. The current thread is released back to the thread pool while waiting. When the operation completes, a thread (possibly the same) picks up execution from where it left off. This allows one thread to handle thousands of concurrent requests without blocking.

---

**Q: What is the difference between `Scoped`, `Singleton`, and `Transient`?**
> `Singleton` — created once, shared by all. `Scoped` — created once per HTTP request, shared within that request. `Transient` — new instance every injection. DbContext should be Scoped (one per request). Stateless utilities can be Transient. Config/cache should be Singleton.

---

**Q: What is middleware in ASP.NET Core?**
> Middleware are components forming the **request processing pipeline**. Each component receives the request, does work, and calls `next()` to pass control to the next component. The response then flows back through the same middleware in reverse. Order matters — Authentication must come before Authorization.

---

**Q: What is the Repository Pattern and why use it?**
> Repository Pattern abstracts the data access logic behind an interface. Benefits: decouples business logic from EF Core, makes unit testing easy (mock the interface), and allows swapping DB implementation without changing service code.

---

**Q: What does `AsNoTracking()` do in EF Core?**
> Tells EF Core not to track the returned entities for changes. This improves performance for read-only queries since EF skips change detection. Always use it when you don't intend to update the returned entities.

---

**Q: What is the difference between `Task.WhenAll` and `Task.WhenAny`?**
> `Task.WhenAll` waits for **all** tasks to complete (parallel execution). `Task.WhenAny` waits for the **first** task to complete. Use `WhenAll` when all results are needed; use `WhenAny` for racing or timeout patterns.

---

## Quick Reference Cheatsheet

```
C# TYPE SYSTEM
──────────────────────────────────────────────────
Value types (stack): int, double, bool, struct, enum
Reference types (heap): class, string, array, interface
Nullable value: int? age = null;
Safe access: user?.Name ?? "Unknown"

OOP KEYWORDS
──────────────────────────────────────────────────
virtual   → method can be overridden in child
override  → overrides parent virtual method
abstract  → must be overridden (no implementation)
sealed    → cannot be inherited or overridden
new       → hides parent method (not override)
base      → calls parent constructor/method

LINQ ESSENTIALS
──────────────────────────────────────────────────
.Where()          → filter
.Select()         → project/transform
.OrderBy()        → sort ascending
.GroupBy()        → group elements
.First/OrDefault() → single element
.Any() / .All()   → existence check
.Count() / .Sum() / .Average() → aggregation
.Skip().Take()    → pagination
.ToList()         → materialize / execute

DI LIFETIMES
──────────────────────────────────────────────────
Singleton  → once per app (config, cache)
Scoped     → once per request (DbContext, services) ✅
Transient  → every injection (lightweight utils)

ASYNC RULES
──────────────────────────────────────────────────
✅ async Task (not async void)
✅ await all the way down
✅ Use CancellationToken in long-running ops
❌ Never .Result or .Wait() → deadlock
❌ Never fire-and-forget without handling exceptions

HTTP STATUS CODES
──────────────────────────────────────────────────
200 OK          → GET, PUT, PATCH success
201 Created     → POST success
204 No Content  → DELETE success
400 Bad Request → validation error
401 Unauthorized → not authenticated
403 Forbidden   → not authorized
404 Not Found   → resource missing
500 Server Error → unexpected crash

EF CORE CHEATSHEET
──────────────────────────────────────────────────
.Include()       → eager load navigation property
.ThenInclude()   → nested eager load
.AsNoTracking()  → read-only, faster
.FirstOrDefault() → null if not found (safe)
.AnyAsync()      → check existence
SaveChangesAsync() → commit to DB
```
