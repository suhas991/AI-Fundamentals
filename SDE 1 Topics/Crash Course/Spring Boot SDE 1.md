# 🍃 Spring Boot — Complete Interview Notes (SDE-1)

---

## Table of Contents
1. [Spring Core Concepts](#1-spring-core-concepts)
2. [Spring Boot Basics](#2-spring-boot-basics)
3. [Dependency Injection & IoC](#3-dependency-injection--ioc)
4. [Spring Annotations](#4-spring-annotations)
5. [Spring MVC & REST APIs](#5-spring-mvc--rest-apis)
6. [Spring Data JPA](#6-spring-data-jpa)
7. [Spring Boot Configuration](#7-spring-boot-configuration)
8. [Exception Handling](#8-exception-handling)
9. [Validation](#9-validation)
10. [Spring Security (Basics)](#10-spring-security-basics)
11. [Lombok](#11-lombok)
12. [Spring Boot Actuator](#12-spring-boot-actuator)
13. [Testing in Spring Boot](#13-testing-in-spring-boot)
14. [Common Interview Q&A](#14-common-interview-qa)

---

## 1. Spring Core Concepts

### What is Spring?
Spring is a **lightweight, open-source Java framework** for building enterprise applications. It provides infrastructure support so developers focus on business logic.

### Core Modules
| Module | Purpose |
|---|---|
| **Spring Core** | IoC container, Dependency Injection |
| **Spring MVC** | Web layer, REST APIs |
| **Spring Data** | Database access, JPA, repositories |
| **Spring Security** | Authentication & Authorization |
| **Spring Boot** | Auto-configuration, embedded server |
| **Spring AOP** | Aspect-Oriented Programming (cross-cutting concerns) |

### Spring Container
The Spring IoC container is responsible for **creating, configuring, and managing beans** (Java objects). Two main types:
- `BeanFactory` — basic container (lazy loading)
- `ApplicationContext` — advanced container (eager loading, events, AOP support) ✅ preferred

---

## 2. Spring Boot Basics

### What is Spring Boot?
Spring Boot is an **opinionated extension of Spring** that eliminates boilerplate configuration through:
- **Auto-configuration** — automatically configures Spring based on classpath
- **Embedded server** — Tomcat/Jetty/Undertow built-in (no WAR deployment needed)
- **Starter dependencies** — curated dependency bundles
- **Production-ready features** — Actuator, metrics, health checks

### Spring vs Spring Boot
| Feature | Spring | Spring Boot |
|---|---|---|
| Configuration | Manual XML/Java | Auto-configured |
| Server | External (Tomcat WAR) | Embedded |
| Setup time | High | Minimal |
| Dependency management | Manual versions | Managed via BOM |

### Creating a Spring Boot App
```java
@SpringBootApplication  // = @Configuration + @EnableAutoConfiguration + @ComponentScan
public class MyApp {
    public static void main(String[] args) {
        SpringApplication.run(MyApp.class, args);
    }
}
```

### `@SpringBootApplication` — what it does
```
@SpringBootApplication
    ├── @Configuration        → marks class as bean config source
    ├── @EnableAutoConfiguration → enables auto-config based on classpath
    └── @ComponentScan        → scans for components in current package and sub-packages
```

### Spring Boot Starters
Pre-packaged dependency sets — add one, get all needed libraries:
```xml
<!-- Web (includes Spring MVC + Tomcat) -->
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-web</artifactId>
</dependency>

<!-- JPA (includes Hibernate + Spring Data) -->
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-data-jpa</artifactId>
</dependency>

<!-- Security -->
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-security</artifactId>
</dependency>

<!-- Test -->
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-test</artifactId>
    <scope>test</scope>
</dependency>
```

---

## 3. Dependency Injection & IoC

### Inversion of Control (IoC)
IoC is a design principle where the **control of object creation is transferred from the developer to the Spring container**. Instead of `new MyService()`, Spring creates and injects it.

### Dependency Injection (DI)
DI is how IoC is implemented — the container **injects dependencies** into a class.

### 3 Ways to Inject Dependencies

#### 1. Constructor Injection ✅ Recommended
```java
@Service
public class OrderService {
    private final ProductRepository productRepo; // final — immutable

    // @Autowired optional when there's only one constructor (Spring 4.3+)
    public OrderService(ProductRepository productRepo) {
        this.productRepo = productRepo;
    }
}
```
**Why preferred?** Immutability, easier testing, avoids circular dependency issues.

#### 2. Setter Injection
```java
@Service
public class OrderService {
    private ProductRepository productRepo;

    @Autowired
    public void setProductRepo(ProductRepository productRepo) {
        this.productRepo = productRepo;
    }
}
```
**When to use?** Optional dependencies.

#### 3. Field Injection ❌ Not Recommended
```java
@Service
public class OrderService {
    @Autowired
    private ProductRepository productRepo; // Hard to test, hides dependencies
}
```
**Why avoid?** Cannot make final, hides dependencies, harder to unit test.

### Bean Scopes
| Scope | Description | Default? |
|---|---|---|
| **singleton** | One instance per Spring container | ✅ Yes |
| **prototype** | New instance every time requested | No |
| **request** | One instance per HTTP request (web) | No |
| **session** | One instance per HTTP session (web) | No |

```java
@Component
@Scope("prototype") // new instance every time
public class ReportGenerator { }
```

---

## 4. Spring Annotations

### Stereotype Annotations (Component Scanning)
| Annotation | Purpose |
|---|---|
| `@Component` | Generic Spring-managed bean |
| `@Service` | Business logic layer |
| `@Repository` | Data access layer — also translates SQL exceptions |
| `@Controller` | Spring MVC web controller (returns views) |
| `@RestController` | `@Controller` + `@ResponseBody` — returns JSON/XML |

```java
@Service
public class UserService {
    // business logic
}

@Repository
public interface UserRepository extends JpaRepository<User, Long> {
    // data access
}
```

### Configuration Annotations
```java
@Configuration  // marks class as source of bean definitions
public class AppConfig {

    @Bean  // declares a bean managed by Spring
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();
    }
}
```

### `@Autowired`
Tells Spring to inject a dependency automatically.
```java
@Autowired  // can inject by constructor, setter, or field
private EmailService emailService;
```

### `@Qualifier`
Used when **multiple beans of the same type** exist:
```java
@Component("smtpEmailService")
public class SmtpEmailService implements EmailService { }

@Component("mockEmailService")
public class MockEmailService implements EmailService { }

// Inject specific one
@Autowired
@Qualifier("smtpEmailService")
private EmailService emailService;
```

### `@Primary`
Marks one bean as the **default** when multiple beans of the same type exist:
```java
@Component
@Primary
public class SmtpEmailService implements EmailService { }
```

### `@Value`
Inject values from properties:
```java
@Value("${app.name}")
private String appName;

@Value("${app.max-connections:10}") // with default value
private int maxConnections;
```

### `@Lazy`
Bean is created **on first use**, not at startup:
```java
@Component
@Lazy
public class HeavyService { }
```

---

## 5. Spring MVC & REST APIs

### MVC Architecture
```
Client Request
    ↓
DispatcherServlet (Front Controller)
    ↓
HandlerMapping → finds Controller method
    ↓
Controller → processes request, calls Service
    ↓
Service → business logic
    ↓
Repository → database
    ↓
Response (JSON via HttpMessageConverter)
```

### Building a REST API

#### Entity
```java
@Entity
@Table(name = "products")
public class Product {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    private String name;
    private Double price;
    // getters, setters
}
```

#### Repository
```java
@Repository
public interface ProductRepository extends JpaRepository<Product, Long> {
    List<Product> findByName(String name);
}
```

#### Service
```java
@Service
public class ProductService {
    private final ProductRepository repo;

    public ProductService(ProductRepository repo) {
        this.repo = repo;
    }

    public List<Product> getAllProducts() {
        return repo.findAll();
    }

    public Product getById(Long id) {
        return repo.findById(id)
            .orElseThrow(() -> new ResourceNotFoundException("Product not found: " + id));
    }

    public Product save(Product product) {
        return repo.save(product);
    }

    public void delete(Long id) {
        repo.deleteById(id);
    }
}
```

#### Controller
```java
@RestController
@RequestMapping("/api/products")
public class ProductController {

    private final ProductService productService;

    public ProductController(ProductService productService) {
        this.productService = productService;
    }

    // GET all
    @GetMapping
    public ResponseEntity<List<Product>> getAll() {
        return ResponseEntity.ok(productService.getAllProducts());
    }

    // GET by ID
    @GetMapping("/{id}")
    public ResponseEntity<Product> getById(@PathVariable Long id) {
        return ResponseEntity.ok(productService.getById(id));
    }

    // POST - create
    @PostMapping
    public ResponseEntity<Product> create(@RequestBody @Valid Product product) {
        Product saved = productService.save(product);
        return ResponseEntity.status(HttpStatus.CREATED).body(saved);
    }

    // PUT - update
    @PutMapping("/{id}")
    public ResponseEntity<Product> update(@PathVariable Long id,
                                           @RequestBody Product product) {
        product.setId(id);
        return ResponseEntity.ok(productService.save(product));
    }

    // DELETE
    @DeleteMapping("/{id}")
    public ResponseEntity<Void> delete(@PathVariable Long id) {
        productService.delete(id);
        return ResponseEntity.noContent().build();
    }
}
```

### HTTP Method Mapping Annotations
| Annotation | HTTP Method | Use Case |
|---|---|---|
| `@GetMapping` | GET | Retrieve resource(s) |
| `@PostMapping` | POST | Create resource |
| `@PutMapping` | PUT | Update entire resource |
| `@PatchMapping` | PATCH | Update partial resource |
| `@DeleteMapping` | DELETE | Delete resource |

### Parameter Annotations
| Annotation | Source | Example |
|---|---|---|
| `@PathVariable` | URL path | `/products/{id}` |
| `@RequestParam` | Query string | `/products?name=phone` |
| `@RequestBody` | Request body (JSON) | POST/PUT body |
| `@RequestHeader` | HTTP header | `Authorization` header |

```java
@GetMapping("/search")
public List<Product> search(
    @RequestParam String name,
    @RequestParam(defaultValue = "0") int page,
    @RequestParam(defaultValue = "10") int size) {
    // ...
}
```

### ResponseEntity
Allows full control over HTTP response (status, headers, body):
```java
return ResponseEntity.ok(data);                          // 200
return ResponseEntity.created(location).body(data);     // 201
return ResponseEntity.noContent().build();              // 204
return ResponseEntity.notFound().build();               // 404
return ResponseEntity.status(HttpStatus.CONFLICT).body(msg); // 409
```

---

## 6. Spring Data JPA

### What is JPA?
JPA (Java Persistence API) is a specification for **ORM (Object-Relational Mapping)** — mapping Java objects to database tables. **Hibernate** is the most common JPA implementation.

### Entity Annotations
```java
@Entity                            // marks as JPA entity
@Table(name = "users")             // maps to table name
public class User {

    @Id                            // primary key
    @GeneratedValue(strategy = GenerationType.IDENTITY)  // auto-increment
    private Long id;

    @Column(name = "email", nullable = false, unique = true, length = 100)
    private String email;

    @Column(name = "created_at")
    private LocalDateTime createdAt;

    @Transient                     // NOT persisted to DB
    private String tempToken;
}
```

### Repository Interfaces
```java
// CrudRepository — basic CRUD
// PagingAndSortingRepository — adds pagination
// JpaRepository — extends both + JPA-specific methods ✅ use this
public interface UserRepository extends JpaRepository<User, Long> {

    // Derived query methods (Spring auto-generates SQL from method name)
    Optional<User> findByEmail(String email);
    List<User> findByAgeGreaterThan(int age);
    List<User> findByNameContainingIgnoreCase(String name);
    boolean existsByEmail(String email);
    long countByAge(int age);
    void deleteByEmail(String email);
}
```

### Custom Queries with `@Query`
```java
// JPQL (Java Persistence Query Language — uses entity names, not table names)
@Query("SELECT u FROM User u WHERE u.email = :email AND u.active = true")
Optional<User> findActiveByEmail(@Param("email") String email);

// Native SQL
@Query(value = "SELECT * FROM users WHERE age > :age", nativeQuery = true)
List<User> findByAgeNative(@Param("age") int age);

// Modifying query
@Modifying
@Transactional
@Query("UPDATE User u SET u.active = false WHERE u.lastLogin < :date")
int deactivateOldUsers(@Param("date") LocalDateTime date);
```

### Relationships
#### One-to-Many / Many-to-One
```java
@Entity
public class Order {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id")
    private User user;

    @OneToMany(mappedBy = "order", cascade = CascadeType.ALL, orphanRemoval = true)
    private List<OrderItem> items = new ArrayList<>();
}
```

#### Many-to-Many
```java
@Entity
public class Student {
    @ManyToMany
    @JoinTable(
        name = "student_course",
        joinColumns = @JoinColumn(name = "student_id"),
        inverseJoinColumns = @JoinColumn(name = "course_id")
    )
    private Set<Course> courses = new HashSet<>();
}
```

### Fetch Types
| Type | Behavior | Default for |
|---|---|---|
| `LAZY` | Load related data only when accessed ✅ | `@OneToMany`, `@ManyToMany` |
| `EAGER` | Load related data immediately | `@ManyToOne`, `@OneToOne` |

> **Interview Tip:** Always prefer `LAZY` for collections to avoid N+1 problem. Use `@EntityGraph` or JOIN FETCH for explicit loading.

### Pagination
```java
// In repository
Page<User> findByActive(boolean active, Pageable pageable);

// In service/controller
Pageable pageable = PageRequest.of(0, 10, Sort.by("name").ascending());
Page<User> users = userRepository.findByActive(true, pageable);

// Page contains: content, totalPages, totalElements, number, size
```

### `@Transactional`
```java
@Service
public class TransferService {

    @Transactional  // all or nothing — if any step fails, everything rolls back
    public void transfer(Long fromId, Long toId, Double amount) {
        Account from = accountRepo.findById(fromId).orElseThrow();
        Account to = accountRepo.findById(toId).orElseThrow();
        from.setBalance(from.getBalance() - amount);
        to.setBalance(to.getBalance() + amount);
        accountRepo.save(from);
        accountRepo.save(to);
    }
}
```

---

## 7. Spring Boot Configuration

### application.properties vs application.yml
```properties
# application.properties
server.port=8080
spring.datasource.url=jdbc:mysql://localhost:3306/mydb
spring.datasource.username=root
spring.datasource.password=secret
spring.jpa.hibernate.ddl-auto=update
spring.jpa.show-sql=true
```

```yaml
# application.yml (equivalent — more readable for nested config)
server:
  port: 8080

spring:
  datasource:
    url: jdbc:mysql://localhost:3306/mydb
    username: root
    password: secret
  jpa:
    hibernate:
      ddl-auto: update
    show-sql: true
```

### `spring.jpa.hibernate.ddl-auto` Options
| Value | Behavior |
|---|---|
| `none` | No schema changes |
| `validate` | Validates schema, no changes |
| `update` | Updates schema without dropping data |
| `create` | Creates schema on startup (drops existing) |
| `create-drop` | Creates on startup, drops on shutdown |

> **Production:** Use `none` or `validate`. Use Flyway/Liquibase for migrations.

### Profiles
Profiles let you have **environment-specific configuration**:

```yaml
# application.yml (base config)
app:
  name: MyApp

---
# application-dev.yml
spring:
  config:
    activate:
      on-profile: dev
  datasource:
    url: jdbc:h2:mem:testdb

---
# application-prod.yml
spring:
  config:
    activate:
      on-profile: prod
  datasource:
    url: jdbc:mysql://prod-server:3306/mydb
```

Activate a profile:
```bash
# application.properties
spring.profiles.active=dev

# Or via command line
java -jar app.jar --spring.profiles.active=prod

# Or via environment variable
SPRING_PROFILES_ACTIVE=prod
```

In code:
```java
@Profile("dev")
@Bean
public DataSource devDataSource() { /* H2 */ }

@Profile("prod")
@Bean
public DataSource prodDataSource() { /* MySQL */ }
```

### Custom Configuration Properties
```java
@ConfigurationProperties(prefix = "app")
@Component
public class AppProperties {
    private String name;
    private int maxConnections = 10;
    private List<String> allowedOrigins;
    // getters and setters
}
```
```yaml
app:
  name: MyApp
  max-connections: 20
  allowed-origins:
    - https://frontend.com
    - https://admin.com
```

---

## 8. Exception Handling

### `@ExceptionHandler` — Local (per controller)
```java
@RestController
public class ProductController {

    @ExceptionHandler(ResourceNotFoundException.class)
    public ResponseEntity<String> handleNotFound(ResourceNotFoundException ex) {
        return ResponseEntity.status(HttpStatus.NOT_FOUND).body(ex.getMessage());
    }
}
```

### `@ControllerAdvice` — Global Exception Handler ✅ Best Practice
```java
// Custom exception
public class ResourceNotFoundException extends RuntimeException {
    public ResourceNotFoundException(String message) {
        super(message);
    }
}

// Error response DTO
public class ErrorResponse {
    private int status;
    private String message;
    private LocalDateTime timestamp;
    // constructors, getters
}

// Global handler
@RestControllerAdvice  // = @ControllerAdvice + @ResponseBody
public class GlobalExceptionHandler {

    @ExceptionHandler(ResourceNotFoundException.class)
    public ResponseEntity<ErrorResponse> handleNotFound(ResourceNotFoundException ex) {
        ErrorResponse error = new ErrorResponse(404, ex.getMessage(), LocalDateTime.now());
        return ResponseEntity.status(HttpStatus.NOT_FOUND).body(error);
    }

    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ResponseEntity<ErrorResponse> handleValidation(MethodArgumentNotValidException ex) {
        String message = ex.getBindingResult().getFieldErrors().stream()
            .map(e -> e.getField() + ": " + e.getDefaultMessage())
            .collect(Collectors.joining(", "));
        return ResponseEntity.badRequest()
            .body(new ErrorResponse(400, message, LocalDateTime.now()));
    }

    @ExceptionHandler(Exception.class)  // catch-all
    public ResponseEntity<ErrorResponse> handleGeneral(Exception ex) {
        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
            .body(new ErrorResponse(500, "Internal Server Error", LocalDateTime.now()));
    }
}
```

---

## 9. Validation

### Adding Dependency
```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-validation</artifactId>
</dependency>
```

### Bean Validation Annotations
```java
public class UserDTO {

    @NotNull(message = "Name cannot be null")
    @NotBlank(message = "Name cannot be blank")
    @Size(min = 2, max = 50, message = "Name must be between 2 and 50 characters")
    private String name;

    @Email(message = "Invalid email format")
    @NotBlank
    private String email;

    @Min(value = 18, message = "Age must be at least 18")
    @Max(value = 120, message = "Age must be at most 120")
    private int age;

    @NotNull
    @Pattern(regexp = "^\\d{10}$", message = "Phone must be 10 digits")
    private String phone;

    @Past(message = "Date of birth must be in the past")
    private LocalDate dateOfBirth;

    @Positive(message = "Price must be positive")
    private Double price;
}
```

### Using `@Valid` in Controller
```java
@PostMapping("/users")
public ResponseEntity<User> createUser(@RequestBody @Valid UserDTO dto) {
    // if validation fails, MethodArgumentNotValidException is thrown
    return ResponseEntity.status(HttpStatus.CREATED).body(userService.create(dto));
}
```

### Common Validation Annotations
| Annotation | Description |
|---|---|
| `@NotNull` | Value must not be null |
| `@NotBlank` | String must not be null or whitespace |
| `@NotEmpty` | Collection/String must not be empty |
| `@Size` | String/Collection size constraints |
| `@Min` / `@Max` | Numeric range |
| `@Email` | Valid email format |
| `@Pattern` | Regex pattern match |
| `@Past` / `@Future` | Date constraints |
| `@Positive` / `@Negative` | Number sign |

---

## 10. Spring Security (Basics)

### What it does
- **Authentication** — who are you? (login)
- **Authorization** — what can you do? (roles/permissions)

### Basic Setup
```java
@Configuration
@EnableWebSecurity
public class SecurityConfig {

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
            .csrf().disable()
            .authorizeHttpRequests(auth -> auth
                .requestMatchers("/api/public/**").permitAll()
                .requestMatchers("/api/admin/**").hasRole("ADMIN")
                .anyRequest().authenticated()
            )
            .httpBasic(); // or .formLogin() or JWT filter
        return http.build();
    }

    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();
    }
}
```

### UserDetailsService
```java
@Service
public class CustomUserDetailsService implements UserDetailsService {

    private final UserRepository userRepository;

    @Override
    public UserDetails loadUserByUsername(String email) throws UsernameNotFoundException {
        User user = userRepository.findByEmail(email)
            .orElseThrow(() -> new UsernameNotFoundException("User not found: " + email));

        return org.springframework.security.core.userdetails.User
            .withUsername(user.getEmail())
            .password(user.getPassword()) // must be BCrypt encoded
            .roles(user.getRole())
            .build();
    }
}
```

### JWT Flow (High Level)
```
1. User POST /login with credentials
2. Server validates → generates JWT token
3. Client stores JWT (localStorage/cookie)
4. Client sends JWT in Authorization: Bearer <token> header
5. Server validates JWT on every request via filter
6. Access granted/denied based on claims
```

### `@PreAuthorize`
```java
@RestController
public class AdminController {

    @GetMapping("/admin/users")
    @PreAuthorize("hasRole('ADMIN')")
    public List<User> getAllUsers() { ... }

    @DeleteMapping("/admin/users/{id}")
    @PreAuthorize("hasRole('ADMIN') or #id == authentication.principal.id")
    public void deleteUser(@PathVariable Long id) { ... }
}
```

---

## 11. Lombok

### What is Lombok?
A Java library that reduces boilerplate code via **annotations processed at compile time**.

```xml
<dependency>
    <groupId>org.projectlombok</groupId>
    <artifactId>lombok</artifactId>
    <optional>true</optional>
</dependency>
```

### Common Annotations
```java
@Data                  // @Getter + @Setter + @ToString + @EqualsAndHashCode + @RequiredArgsConstructor
@NoArgsConstructor     // generates no-arg constructor
@AllArgsConstructor    // generates constructor with all fields
@RequiredArgsConstructor // constructor for final/non-null fields
@Builder               // builder pattern
@Getter                // generates getters
@Setter                // generates setters
@ToString              // generates toString()
@EqualsAndHashCode     // generates equals() and hashCode()
@Slf4j                 // injects logger: private static final Logger log = LoggerFactory.getLogger(...)
```

### Example
```java
@Entity
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class Product {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String name;
    private Double price;
    private String category;
}

// Using @Builder
Product p = Product.builder()
    .name("Laptop")
    .price(999.99)
    .category("Electronics")
    .build();
```

```java
// Using @Slf4j
@Service
@Slf4j
public class ProductService {
    public void process() {
        log.info("Processing product...");
        log.error("Error occurred: {}", message);
    }
}
```

### `@Data` Warning with JPA
Avoid `@Data` on JPA entities — `@EqualsAndHashCode` and `@ToString` can cause issues with lazy-loaded collections. Instead use:
```java
@Entity
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
public class User { ... }
```

---

## 12. Spring Boot Actuator

### What is it?
Provides **production-ready monitoring endpoints** out of the box.

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-actuator</artifactId>
</dependency>
```

### Key Endpoints
| Endpoint | Description |
|---|---|
| `/actuator/health` | App health status |
| `/actuator/info` | App info |
| `/actuator/metrics` | JVM, HTTP metrics |
| `/actuator/env` | Environment variables |
| `/actuator/beans` | All Spring beans |
| `/actuator/mappings` | All request mappings |
| `/actuator/loggers` | Logger levels |
| `/actuator/threaddump` | Thread dump |

### Configuration
```yaml
management:
  endpoints:
    web:
      exposure:
        include: health,info,metrics  # expose specific endpoints
        # include: "*"               # expose all
  endpoint:
    health:
      show-details: always           # show full health details
```

### Custom Health Indicator
```java
@Component
public class DatabaseHealthIndicator implements HealthIndicator {

    @Override
    public Health health() {
        // check connectivity
        boolean isUp = checkDatabase();
        if (isUp) {
            return Health.up().withDetail("db", "Responding").build();
        }
        return Health.down().withDetail("db", "Not responding").build();
    }
}
```

---

## 13. Testing in Spring Boot

### Types of Tests
| Type | Annotation | What it tests |
|---|---|---|
| Unit Test | `@ExtendWith(MockitoExtension.class)` | Single class with mocked deps |
| Slice Test | `@WebMvcTest`, `@DataJpaTest` | Specific layer only |
| Integration Test | `@SpringBootTest` | Full application context |

### Unit Test with Mockito
```java
@ExtendWith(MockitoExtension.class)
class ProductServiceTest {

    @Mock
    private ProductRepository productRepository;

    @InjectMocks
    private ProductService productService;

    @Test
    void getById_WhenProductExists_ReturnsProduct() {
        // Arrange
        Product product = new Product(1L, "Laptop", 999.99);
        when(productRepository.findById(1L)).thenReturn(Optional.of(product));

        // Act
        Product result = productService.getById(1L);

        // Assert
        assertNotNull(result);
        assertEquals("Laptop", result.getName());
        verify(productRepository, times(1)).findById(1L);
    }

    @Test
    void getById_WhenNotFound_ThrowsException() {
        when(productRepository.findById(99L)).thenReturn(Optional.empty());

        assertThrows(ResourceNotFoundException.class,
            () -> productService.getById(99L));
    }
}
```

### Controller Slice Test with `@WebMvcTest`
```java
@WebMvcTest(ProductController.class)  // loads only web layer
class ProductControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @MockBean  // mocks the service in Spring context
    private ProductService productService;

    @Autowired
    private ObjectMapper objectMapper;

    @Test
    void getAll_ReturnsProductList() throws Exception {
        List<Product> products = List.of(
            new Product(1L, "Laptop", 999.99),
            new Product(2L, "Phone", 499.99)
        );
        when(productService.getAllProducts()).thenReturn(products);

        mockMvc.perform(get("/api/products")
                .contentType(MediaType.APPLICATION_JSON))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.length()").value(2))
            .andExpect(jsonPath("$[0].name").value("Laptop"));
    }

    @Test
    void create_ValidProduct_Returns201() throws Exception {
        Product product = new Product(null, "Tablet", 299.99);
        Product saved = new Product(1L, "Tablet", 299.99);

        when(productService.save(any(Product.class))).thenReturn(saved);

        mockMvc.perform(post("/api/products")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(product)))
            .andExpect(status().isCreated())
            .andExpect(jsonPath("$.id").value(1));
    }
}
```

### Repository Slice Test with `@DataJpaTest`
```java
@DataJpaTest  // in-memory H2 db, loads only JPA components
class UserRepositoryTest {

    @Autowired
    private UserRepository userRepository;

    @Test
    void findByEmail_WhenExists_ReturnsUser() {
        User user = new User();
        user.setEmail("test@test.com");
        user.setName("Test User");
        userRepository.save(user);

        Optional<User> found = userRepository.findByEmail("test@test.com");
        assertTrue(found.isPresent());
        assertEquals("Test User", found.get().getName());
    }
}
```

### Integration Test with `@SpringBootTest`
```java
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
class ProductIntegrationTest {

    @Autowired
    private TestRestTemplate restTemplate;

    @Test
    void createAndRetrieveProduct() {
        Product product = new Product(null, "Laptop", 999.99);

        ResponseEntity<Product> created = restTemplate
            .postForEntity("/api/products", product, Product.class);
        assertEquals(HttpStatus.CREATED, created.getStatusCode());

        Long id = created.getBody().getId();
        ResponseEntity<Product> fetched = restTemplate
            .getForEntity("/api/products/" + id, Product.class);
        assertEquals(HttpStatus.OK, fetched.getStatusCode());
        assertEquals("Laptop", fetched.getBody().getName());
    }
}
```

---

## 14. Common Interview Q&A

**Q: What is auto-configuration in Spring Boot?**
> Spring Boot automatically configures beans based on the libraries on the classpath. For example, if `spring-boot-starter-data-jpa` is on the classpath, it automatically configures a DataSource, EntityManagerFactory, and TransactionManager — you don't have to define them manually. You can exclude specific auto-configurations using `@SpringBootApplication(exclude = {DataSourceAutoConfiguration.class})`.

---

**Q: What is the difference between `@Component`, `@Service`, `@Repository`, and `@Controller`?**
> All four are specializations of `@Component` and enable component scanning. Functionally they're the same, but semantically: `@Service` marks business logic, `@Repository` marks the data layer (also converts JPA exceptions to Spring's `DataAccessException`), and `@Controller` marks the web layer. Using the right annotation makes the code more readable and enables layer-specific features.

---

**Q: What is the difference between `@Controller` and `@RestController`?**
> `@RestController` is a convenience annotation combining `@Controller` and `@ResponseBody`. Every method in a `@RestController` automatically serializes the return value to JSON/XML and writes it to the HTTP response body. With plain `@Controller`, you'd need `@ResponseBody` on each method, or it would look for a view template to render.

---

**Q: What is the N+1 problem in JPA?**
> The N+1 problem occurs when fetching a list of N entities triggers N additional queries to load their associations. For example, loading 100 Orders triggers 100 additional queries to load each Order's User. Solution: Use `JOIN FETCH` in JPQL, `@EntityGraph`, or set batch size with `@BatchSize`.
```java
@Query("SELECT o FROM Order o JOIN FETCH o.user")
List<Order> findAllWithUser();
```

---

**Q: What is the difference between `@Bean` and `@Component`?**
> `@Component` (and its specializations) is used with **class-level annotation** and Spring automatically detects it via component scanning. `@Bean` is used inside a `@Configuration` class on a **method** — useful when you don't own the class (e.g., third-party library) or need complex construction logic.

---

**Q: What is `@Transactional` and when would you use it?**
> `@Transactional` wraps a method in a database transaction — if the method throws an unchecked exception, the transaction is rolled back. Use it on service methods that perform multiple database operations that must succeed or fail together (atomicity). By default it rolls back on `RuntimeException` and `Error`, not checked exceptions.

---

**Q: Constructor injection vs field injection — which is better and why?**
> Constructor injection is preferred because: (1) dependencies are explicit and required, (2) fields can be `final` (immutability), (3) easier to unit test — just `new MyService(mockRepo)`, (4) Spring detects circular dependencies early. Field injection hides dependencies and makes testing harder.

---

**Q: What are Spring Profiles and when would you use them?**
> Profiles allow different configurations for different environments (dev, test, prod). Annotate beans or config files with `@Profile("dev")` and activate via `spring.profiles.active=dev`. Common use: different datasources, mock vs real email services, feature flags.

---

**Q: Explain the lifecycle of a Spring Bean.**
```
1. Bean instantiated (constructor called)
2. Dependencies injected (@Autowired)
3. @PostConstruct method called (initialization)
4. Bean is ready for use
5. @PreDestroy method called (on shutdown)
6. Bean destroyed
```
```java
@Component
public class MyBean {
    @PostConstruct
    public void init() { System.out.println("Bean initialized"); }

    @PreDestroy
    public void destroy() { System.out.println("Bean destroyed"); }
}
```

---

**Q: What is the difference between `findById()` and `getOne()`/`getReferenceById()`?**
> `findById()` eagerly loads the entity and returns `Optional<T>` — hits the DB immediately. `getReferenceById()` returns a **proxy** — only hits the DB when you access a property (lazy). Use `getReferenceById()` when you only need the entity reference (e.g., for setting a foreign key relationship), not its data.

---

**Q: How do you handle circular dependencies in Spring?**
> Circular dependency: A depends on B, B depends on A. Solutions:
> - **Refactor** the design (best approach)
> - Use `@Lazy` on one of the dependencies
> - Use setter injection instead of constructor injection (Spring can inject after creation)
```java
@Autowired
public MyService(@Lazy DependencyB b) { this.b = b; }
```

---

## Quick Reference Cheatsheet

```
Spring Boot App Layers:
──────────────────────────────────────────────
Controller (@RestController)   ← HTTP layer
    ↕
Service (@Service)             ← Business logic
    ↕
Repository (@Repository)       ← Data access
    ↕
Database (JPA/SQL)

Key Annotations Summary:
──────────────────────────────────────────────
@SpringBootApplication    → App entry point
@RestController           → REST API controller
@Service                  → Business logic
@Repository               → Data access layer
@Entity                   → JPA entity/table
@Autowired                → Inject dependency
@Value                    → Inject property value
@ConfigurationProperties  → Bind config block
@Transactional            → Database transaction
@Valid / @Validated       → Trigger validation
@ExceptionHandler         → Handle exceptions
@ControllerAdvice         → Global exception handling
@GetMapping / @PostMapping / @PutMapping / @DeleteMapping

HTTP Status Codes:
──────────────────────────────────────────────
200 OK                    → GET success
201 Created               → POST success
204 No Content            → DELETE success
400 Bad Request           → Validation error
401 Unauthorized          → Not authenticated
403 Forbidden             → Not authorized
404 Not Found             → Resource missing
409 Conflict              → Duplicate resource
500 Internal Server Error → Unexpected error
```
