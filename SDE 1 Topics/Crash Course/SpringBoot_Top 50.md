# Spring Boot – Interview Preparation Notes

---

## 1. 2 Main Advantages of Spring Boot over Spring

1. **Auto-Configuration**: Spring Boot automatically configures your application based on the dependencies present in the classpath. You don't need to write verbose XML or Java config files.
2. **Embedded Server**: Spring Boot comes with an embedded Tomcat/Jetty/Undertow server, so you don't need to deploy a WAR to an external server — just run the JAR.

> **Bonus**: It also provides a starter dependency system (`spring-boot-starter-*`) that bundles related dependencies together.

---

## 2. Advantages of `@SpringBootApplication`

`@SpringBootApplication` is a convenience annotation that combines three annotations:

- `@Configuration` – Marks the class as a source of bean definitions.
- `@EnableAutoConfiguration` – Tells Spring Boot to auto-configure the application context.
- `@ComponentScan` – Scans the current package and sub-packages for Spring components.

> **Interview tip**: It reduces boilerplate and bootstraps the entire Spring context in one annotation.

---

## 3. Difference between JAR and WAR

| Feature | JAR | WAR |
|---|---|---|
| Full Form | Java ARchive | Web Application ARchive |
| Contains | Classes, resources, embedded server | Classes, JSPs, web.xml, no embedded server |
| Deployment | Run standalone (`java -jar`) | Deploy to external server (Tomcat, JBoss) |
| Spring Boot default | ✅ Yes | Optional |

> Spring Boot prefers JAR packaging with an embedded server.

---

## 4. `@RestController`

- A combination of `@Controller` + `@ResponseBody`.
- Every method in a `@RestController` automatically serializes the return value to JSON/XML and writes it to the HTTP response body.
- Used for building RESTful APIs.

```java
@RestController
public class UserController {
    @GetMapping("/users")
    public List<User> getUsers() {
        return userService.findAll(); // serialized to JSON
    }
}
```

---

## 5. What is Maven and How Does it Help?

**Maven** is a build automation and project management tool.

**How it helps:**
- **Dependency Management**: Automatically downloads libraries from Maven Central via `pom.xml`.
- **Build Lifecycle**: Provides standard phases like `compile`, `test`, `package`, `install`, `deploy`.
- **Plugins**: Extends functionality (e.g., Spring Boot Maven Plugin for running/packaging).
- **Standardization**: Enforces a consistent project structure.

---

## 6. `@GetMapping`, `@PostMapping`, `@PutMapping`, `@DeleteMapping`

These are shorthand annotations for HTTP method-specific request mappings (specializations of `@RequestMapping`):

| Annotation | HTTP Method | Use Case |
|---|---|---|
| `@GetMapping` | GET | Fetch/read data |
| `@PostMapping` | POST | Create new resource |
| `@PutMapping` | PUT | Update entire resource |
| `@DeleteMapping` | DELETE | Delete a resource |

```java
@GetMapping("/user/{id}")       // Read
@PostMapping("/user")           // Create
@PutMapping("/user/{id}")       // Update
@DeleteMapping("/user/{id}")    // Delete
```

---

## 7 & 8. Maven Build Lifecycle

Maven has **3 built-in lifecycles**. The most important is the **default lifecycle**:

| Phase | Description |
|---|---|
| `validate` | Validates the project structure |
| `compile` | Compiles source code |
| `test` | Runs unit tests |
| `package` | Packages code into JAR/WAR |
| `verify` | Runs integration checks |
| `install` | Installs package to local `.m2` repo |
| `deploy` | Copies to remote repository |

Maven is called a **build automation tool** because it automates this entire process. Running `mvn package` automatically runs all prior phases (validate → compile → test → package).

---

## 9. `pom.xml` Full Form

**POM = Project Object Model**

`pom.xml` is the core configuration file in a Maven project. It contains:
- Project metadata (groupId, artifactId, version)
- Dependencies
- Plugins
- Build configurations
- Profiles

---

## 10. FatJAR / Uber JAR – Why "Self-Contained"?

A **FatJAR** (also called Uber JAR) is a JAR that contains:
- Your application's compiled classes
- **All its dependencies** bundled inside
- An embedded web server (e.g., Tomcat)

It is called **Self-Contained** because it has everything needed to run — no external server or classpath setup required. You just do:
```bash
java -jar myapp.jar
```

---

## 11. Why is it called IoC?

**IoC = Inversion of Control**

Normally (without IoC), your code controls the creation and lifecycle of objects:
```java
UserService service = new UserService(); // you control it
```

With IoC, the **Spring Container controls** the creation and injection of objects. You give up control (invert it) to the framework:
```java
@Autowired
UserService service; // Spring controls it
```

> Control is *inverted* from the developer to the framework — hence "Inversion of Control."

---

## 12. IoC Container / Application Context

The **IoC Container** is the core of Spring. It:
- Creates and manages **beans** (Spring-managed objects)
- Injects dependencies (Dependency Injection)
- Manages the lifecycle of beans

The main implementation used in Spring Boot is `ApplicationContext` (specifically `AnnotationConfigApplicationContext` or embedded in `SpringApplication`).

---

## 13. `@Component`

Marks a class as a Spring-managed bean. Spring will automatically detect and register it during component scanning.

```java
@Component
public class EmailService {
    public void sendEmail() { ... }
}
```

> Sub-specializations: `@Service` (business logic), `@Repository` (data layer), `@Controller` (web layer).

---

## 14. REST API

**REST = Representational State Transfer**

Key principles:
- **Stateless**: Each request is independent; server stores no session state.
- **Client-Server**: Clear separation of UI and data.
- **Uniform Interface**: Uses HTTP verbs (GET, POST, PUT, DELETE) and URIs.
- **Resource-Based**: Everything is a resource identified by a URL.

Data is typically exchanged in **JSON** format.

---

## 15. Difference between `@RestController` and `@Controller`

| Feature | `@Controller` | `@RestController` |
|---|---|---|
| Returns | View name (HTML page) | Data (JSON/XML) |
| `@ResponseBody` needed | Yes, on each method | No, implicit |
| Used for | MVC / Thymeleaf apps | REST APIs |

```java
// @Controller - returns a view
@Controller
public class PageController {
    @GetMapping("/home")
    public String home() { return "home"; } // resolves to home.html
}

// @RestController - returns data
@RestController
public class ApiController {
    @GetMapping("/data")
    public String data() { return "Hello"; } // returns "Hello" as response body
}
```

---

## 16. 2 Ways a Controller Can Respond

1. **Return a View Name** (used with `@Controller`): The return string resolves to an HTML template (e.g., Thymeleaf).
2. **Return Data / ResponseEntity** (used with `@RestController`): Returns JSON/XML data directly, optionally wrapped in `ResponseEntity` to control status codes and headers.

---

## 17. When do we get HTTP ERROR 405?

**405 Method Not Allowed** occurs when:
- The URL exists, but the **HTTP method used is not supported** by that endpoint.

Example: A `@GetMapping("/user")` exists but you send a `POST` request to `/user` → **405 Error**.

---

## 18. `@RequestMapping`

A general-purpose annotation to map HTTP requests to handler methods. Can be placed at class or method level.

```java
@RequestMapping("/api")
public class UserController {

    @RequestMapping(value = "/users", method = RequestMethod.GET)
    public List<User> getUsers() { ... }
}
```

> `@GetMapping`, `@PostMapping`, etc. are shortcuts for `@RequestMapping(method = RequestMethod.GET/POST/...)`.

---

## 19. Annotations included in `@SpringBootApplication`

```
@SpringBootApplication
    ├── @Configuration
    ├── @EnableAutoConfiguration
    └── @ComponentScan
```

---

## 20. `@PathVariable`

Extracts values from the **URI path**.

```java
@GetMapping("/user/{id}")
public User getUser(@PathVariable int id) {
    return userService.findById(id);
}
// GET /user/5  →  id = 5
```

---

## 21. `@RequestParam`

Extracts values from **query parameters** in the URL.

```java
@GetMapping("/search")
public List<User> search(@RequestParam String name) {
    return userService.findByName(name);
}
// GET /search?name=John  →  name = "John"
```

---

## 22. ORM (Object Relational Mapping)

**ORM** maps Java objects to database tables, eliminating the need for manual SQL.

- A Java class → Database Table
- A class field → Table Column
- An object instance → Table Row

**Spring Data JPA** uses **Hibernate** as the ORM provider by default.

> Advantage: Write Java code instead of SQL; changes to DB schema are reflected in the entity class.

---

## 23. Spring Data MongoDB

A Spring module that provides easy integration with **MongoDB** (a NoSQL, document-oriented database).

- Eliminates boilerplate MongoDB driver code.
- Provides `MongoRepository` for CRUD operations out of the box.
- Uses annotations like `@Document`, `@Id`, `@Field`.

Add dependency: `spring-boot-starter-data-mongodb`

---

## 24. `MongoRepository`

An interface extending `CrudRepository` that provides built-in CRUD and query methods for MongoDB.

```java
public interface UserRepository extends MongoRepository<User, String> {
    List<User> findByName(String name); // derived query method
}
```

> Spring auto-generates the implementation at runtime — no need to write query code.

---

## 25. `@Document`

Marks a Java class as a MongoDB document (equivalent to a row/record in a collection).

```java
@Document(collection = "users")
public class User {
    @Id
    private String id;
    private String name;
}
```

---

## 26. `@Id`

Marks the field as the **primary identifier** of the document/entity.

- In MongoDB: maps to the `_id` field.
- In JPA/SQL: marks the primary key column.

---

## 27. `@Autowired`

Tells Spring to **automatically inject** a bean dependency. Spring resolves the dependency from the IoC container.

```java
@RestController
public class UserController {
    @Autowired
    private UserService userService; // Spring injects the UserService bean
}
```

> Best practice: Use **constructor injection** over field injection for testability.

---

## 28. HTTP Status Codes

| Code | Meaning |
|---|---|
| 200 | OK – Request succeeded |
| 201 | Created – Resource created successfully |
| 204 | No Content – Success but no body |
| 400 | Bad Request – Invalid input |
| 401 | Unauthorized – Authentication required |
| 403 | Forbidden – Authenticated but no permission |
| 404 | Not Found – Resource doesn't exist |
| 405 | Method Not Allowed |
| 409 | Conflict – Duplicate resource |
| 500 | Internal Server Error |

---

## 29. `ResponseEntity`

A wrapper that gives full control over the **HTTP response**: status code, headers, and body.

```java
@GetMapping("/user/{id}")
public ResponseEntity<User> getUser(@PathVariable String id) {
    User user = userService.findById(id);
    if (user == null) {
        return ResponseEntity.status(HttpStatus.NOT_FOUND).body(null);
    }
    return ResponseEntity.ok(user); // 200 OK with body
}
```

---

## 30. Lombok

A Java library that reduces boilerplate code by auto-generating common methods at compile time via annotations.

**Common Lombok annotations:**
- `@Getter` / `@Setter` – Generates getters and setters
- `@Data` – Combines `@Getter`, `@Setter`, `@ToString`, `@EqualsAndHashCode`, `@RequiredArgsConstructor`
- `@NoArgsConstructor` / `@AllArgsConstructor`
- `@Builder` – Implements Builder design pattern

---

## 31. `@Data`

A Lombok shortcut annotation equivalent to applying:
- `@Getter` – all fields
- `@Setter` – all non-final fields
- `@ToString`
- `@EqualsAndHashCode`
- `@RequiredArgsConstructor`

```java
@Data
@Document
public class User {
    @Id
    private String id;
    private String name;
    private String email;
    // getters, setters, toString, etc. are auto-generated
}
```

---

## 32. `@Indexed(unique = true)`

A MongoDB annotation that creates a **unique index** on the annotated field, preventing duplicate values.

```java
@Indexed(unique = true)
private String email;
```

> Requires enabling indexing: `spring.data.mongodb.auto-index-creation=true` in `application.properties`.

---

## 33. `@NonNull` (Lombok)

Generates a null-check at runtime for the annotated field or parameter. Throws `NullPointerException` with a descriptive message if null is passed.

```java
public User(@NonNull String name) {
    this.name = name; // throws NPE if name is null
}
```

---

## 34. Error when Inserting Duplicate on `@Indexed(unique = true)`

You get a **`DuplicateKeyException`** (from Spring Data) which wraps a MongoDB `MongoWriteException` with error code **11000**.

> Handle it with a try-catch or a `@ControllerAdvice` global exception handler and return a `409 Conflict` response.

---

## 35. `@DBRef`

Used in MongoDB to store a **reference** (foreign key-like relationship) to another document instead of embedding it.

```java
@DBRef
private Department department;
```

> The `department` is stored as a reference (`$ref` and `$id`) in MongoDB. When fetched, Spring Data will resolve the reference by performing a separate query.

---

## 36. `@Transactional` and Required Companion Annotation

`@Transactional` ensures that a method (or all methods in a class) run within a database transaction — either all succeed or all roll back.

**Important**: When using `@Transactional` with **MongoDB**, you must annotate the **main class** with:

```java
@EnableTransactionManagement // required on main/config class
@SpringBootApplication
public class MyApp { ... }
```

And configure a `MongoTransactionManager` bean. MongoDB transactions require a **replica set** (not standalone).

---

## 37. Spring Security

Spring Security provides **authentication** (who are you?) and **authorization** (what can you do?).

**By default**, Spring Security uses **HTTP Basic Authentication**.

**To add Spring Security**: Add `spring-boot-starter-security` dependency — it auto-secures all endpoints.

Default behavior:
- All endpoints require login.
- A default user `user` is created with a generated password printed in the console.

---

## 38. Customized Security in Spring Security

Create a class extending `SecurityFilterChain` configuration:

```java
@Configuration
@EnableWebSecurity
public class SecurityConfig {

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
            .authorizeHttpRequests(auth -> auth
                .requestMatchers("/public/**").permitAll()
                .anyRequest().authenticated()
            )
            .formLogin()
            .and()
            .httpBasic();
        return http.build();
    }
}
```

> Override `UserDetailsService` bean to load users from a database.

---

## 39. `@Component` vs `@Bean` vs `@Autowired`

| Annotation | Purpose | Where Used |
|---|---|---|
| `@Component` | Declares a class as a Spring bean (auto-detected via scanning) | On class |
| `@Bean` | Declares a method's return value as a Spring bean (manual declaration) | On method inside `@Configuration` class |
| `@Autowired` | Injects an existing Spring bean into a field/constructor/method | On field, constructor, or method |

```java
// @Component
@Component
public class EmailService { }

// @Bean (inside @Configuration)
@Bean
public PasswordEncoder passwordEncoder() {
    return new BCryptPasswordEncoder();
}

// @Autowired
@Autowired
private EmailService emailService;
```

---

## 40. Where Spring Security Credentials are Stored / How to Access

By default, authenticated user credentials are stored in the **`SecurityContextHolder`** (an in-memory, thread-local storage).

**To access the authenticated user:**

```java
Authentication auth = SecurityContextHolder.getContext().getAuthentication();
String username = auth.getName();
Object principal = auth.getPrincipal(); // UserDetails object
```

For persistent storage (e.g., from a DB), you implement `UserDetailsService` and load users from a repository.

---

## 41. JUnit and Where it is Available

**JUnit** is a Java unit testing framework used to write and run repeatable tests.

**JUnit 5** (Jupiter) is the current version, and it is available in Spring Boot via:
- `spring-boot-starter-test` dependency (included by default in Spring Boot projects).

It pulls in JUnit 5, Mockito, AssertJ, and more.

---

## 42. `@SpringBootTest`

Loads the **full application context** for integration testing.

```java
@SpringBootTest
class UserServiceTest {
    @Autowired
    private UserService userService;

    @Test
    void testFindUser() { ... }
}
```

> Use when you need Spring beans, database, etc. For pure unit tests, avoid this (use Mockito instead).

---

## 43. `@Test`

Marks a method as a JUnit test case.

```java
@Test
void shouldReturnTrueWhenUserExists() {
    boolean result = userService.exists("john");
    assertTrue(result);
}
```

---

## 44. Methods in Test Class (JUnit 5)

| Annotation | When it Runs |
|---|---|
| `@BeforeEach` | Before **each** test method |
| `@AfterEach` | After **each** test method |
| `@BeforeAll` | Once before **all** tests (method must be `static`) |
| `@AfterAll` | Once after **all** tests (method must be `static`) |
| `@Test` | The actual test |
| `@Disabled` | Skips the test |

---

## 45. `@ParameterizedTest`

Allows running the same test with **multiple inputs**.

```java
@ParameterizedTest
@ValueSource(ints = {1, 2, 3})
void testWithMultipleValues(int number) {
    assertTrue(number > 0);
}
```

---

## 46. `@CsvSource`

Provides **comma-separated values** as arguments to a parameterized test.

```java
@ParameterizedTest
@CsvSource({"1, John", "2, Jane", "3, Bob"})
void testWithCsv(int id, String name) {
    assertNotNull(name);
}
```

---

## 47. `@Disabled`

Skips the annotated test or test class. Used to temporarily disable tests.

```java
@Test
@Disabled("Not yet implemented")
void futureTest() { }
```

---

## 48. `@CsvFileSource`

Loads test data from an **external CSV file** (placed in `src/test/resources`).

```java
@ParameterizedTest
@CsvFileSource(resources = "/test-data.csv", numLinesToSkip = 1)
void testFromFile(int id, String name) { ... }
```

---

## 49. `@ValueSource`

Provides a single array of values of a primitive/String type for parameterized tests.

```java
@ParameterizedTest
@ValueSource(strings = {"Alice", "Bob", "Charlie"})
void testNames(String name) {
    assertNotNull(name);
}
```

---

## 50. Plugin Required to Run Tests Using `mvn`

The **Maven Surefire Plugin** is required to run JUnit tests via `mvn test`.

For JUnit 5 specifically, you may need:
```xml
<plugin>
    <groupId>org.apache.maven.plugins</groupId>
    <artifactId>maven-surefire-plugin</artifactId>
    <version>3.x.x</version>
</plugin>
```

> Spring Boot's parent POM usually includes this automatically.

---

## 51. `@BeforeEach`, `@BeforeAll`, `@AfterEach`, `@AfterAll`

See **Question 44** for full table.

```java
@BeforeAll
static void initAll() { /* runs once before all tests */ }

@BeforeEach
void init() { /* runs before each test */ }

@AfterEach
void tearDown() { /* runs after each test */ }

@AfterAll
static void tearDownAll() { /* runs once after all tests */ }
```

---

## 52. `@ActiveProfiles`

Activates specific **Spring profiles** for a test class. Useful for loading a test-specific configuration.

```java
@SpringBootTest
@ActiveProfiles("test")
class UserServiceTest { ... }
```

> Loads `application-test.properties` or beans annotated with `@Profile("test")`.

---

## 53. Common Logging Levels (in order)

From lowest to highest severity:

```
TRACE → DEBUG → INFO → WARN → ERROR → FATAL (OFF)
```

- **TRACE**: Very fine-grained details.
- **DEBUG**: Diagnostic info for debugging.
- **INFO**: General application flow (default in Spring Boot).
- **WARN**: Potential issues.
- **ERROR**: Errors that need attention.

Set in `application.properties`:
```properties
logging.level.root=INFO
logging.level.com.myapp=DEBUG
```

---

## 54. Serialization and Deserialization

| Term | Meaning |
|---|---|
| **Serialization** | Converting a Java object → JSON/XML (for sending over network/storing) |
| **Deserialization** | Converting JSON/XML → Java object (for reading received data) |

Spring Boot uses **Jackson** library (`ObjectMapper`) to handle this automatically.

```java
// Serialization: User object → JSON
// {"id": 1, "name": "John"}

// Deserialization: JSON → User object
@PostMapping("/user")
public void create(@RequestBody User user) { ... } // @RequestBody triggers deserialization
```

---

## 55. `RestTemplate` and the Key Method

`RestTemplate` is a synchronous HTTP client provided by Spring for **consuming REST APIs** from within a Spring application (making outgoing HTTP calls).

**Key methods:**

| Method | Use |
|---|---|
| `getForObject(url, Class)` | GET and return response body |
| `getForEntity(url, Class)` | GET and return `ResponseEntity` |
| `postForObject(url, body, Class)` | POST and return response body |
| `exchange(url, method, entity, Class)` | Full control over request/response |
| `delete(url)` | DELETE request |

```java
RestTemplate restTemplate = new RestTemplate();
User user = restTemplate.getForObject("http://api.example.com/user/1", User.class);
```

> **Note**: `RestTemplate` is in maintenance mode. Prefer `WebClient` (from Spring WebFlux) for new projects.

---

*End of Spring Boot Interview Notes*
