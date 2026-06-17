# 🏗️ System Design — Interview Notes (SDE-1)

> Focus: Concepts clarity + API design thinking + simple design questions.
> Not LLD/HLD depth of SDE-2. Know the *why* behind every concept.

---

## Table of Contents
1. [Client-Server Architecture](#1-client-server-architecture)
2. [Load Balancer Basics](#2-load-balancer-basics)
3. [Caching Basics](#3-caching-basics)
4. [Database Scaling Concepts](#4-database-scaling-concepts)
5. [REST Principles](#5-rest-principles)
6. [API Design](#6-api-design)
7. [Design: URL Shortener](#7-design-url-shortener)
8. [Design: Parking Lot](#8-design-parking-lot)
9. [Design: Library Management System](#9-design-library-management-system)
10. [Design: Food Delivery APIs](#10-design-food-delivery-apis)
11. [How to Approach Any Design Question](#11-how-to-approach-any-design-question)

---

## 1. Client-Server Architecture

### What is it?
A model where two parties communicate over a network:
- **Client** — makes requests (browser, mobile app, Postman)
- **Server** — processes requests and sends back responses

```
Client                         Server
──────                         ──────
Browser ──── HTTP Request ───► Spring Boot App
        ◄─── HTTP Response ─── (processes, queries DB, returns JSON)
```

### How a Request Flows (Full Picture)
```
User types URL in browser
        ↓
DNS Resolution → converts domain to IP address
        ↓
TCP Handshake → connection established
        ↓
HTTP Request sent to Server
        ↓
Load Balancer → routes to one of many servers
        ↓
Application Server → business logic
        ↓
Database → fetch/store data
        ↓
Response flows back to client
```

### Key Terms
| Term | Meaning |
|---|---|
| **HTTP** | Protocol for communication (text-based, stateless) |
| **HTTPS** | HTTP + TLS encryption |
| **Stateless** | Server doesn't remember previous requests |
| **Session** | Workaround for statelessness — stores state on server or client (JWT) |
| **API** | Contract between client and server |
| **Latency** | Time for a request to travel client → server → back |
| **Throughput** | Requests handled per second |

### Why Stateless Matters
Since HTTP is stateless, every request must carry all necessary info:
```
❌ Bad: Server remembers who you are between requests
✅ Good: Client sends JWT token on every request — server verifies it
```

---

## 2. Load Balancer Basics

### What is a Load Balancer?
A Load Balancer sits **between the client and your servers**, distributing incoming traffic across multiple servers so no single server is overwhelmed.

```
                    ┌─── Server 1 (handles 33%)
Client ──► LB ──────├─── Server 2 (handles 33%)
                    └─── Server 3 (handles 33%)
```

### Why Do We Need It?
- **High Availability** — if Server 1 crashes, traffic goes to Server 2 & 3
- **Scalability** — add more servers without changing client code
- **Performance** — spread load, reduce response time

### Load Balancing Algorithms
| Algorithm | How it works | Best for |
|---|---|---|
| **Round Robin** | Requests go to servers in rotation (1→2→3→1→2→3) | Servers with equal capacity |
| **Least Connections** | Route to server with fewest active connections | Varying request durations |
| **IP Hash** | Same client IP always goes to same server | Session-sticky apps |
| **Weighted Round Robin** | Powerful servers get more requests | Heterogeneous servers |

### Health Checks
Load balancers continuously **ping servers** (e.g., `GET /health`). If a server doesn't respond, it's removed from the pool automatically.

```
LB pings every 10 seconds:
  Server 1 → 200 OK ✅ (keep routing)
  Server 2 → No response ❌ (remove from pool, alert team)
  Server 3 → 200 OK ✅ (keep routing)
```

### Sticky Sessions
When a user must always hit the same server (e.g., in-memory session):
```
User A → always → Server 1
User B → always → Server 2
```
> **Interview Tip:** Sticky sessions are a sign of poor design. Better to make servers stateless and use a shared session store (Redis).

---

## 3. Caching Basics

### What is Caching?
Storing frequently accessed data in **fast memory (RAM)** so you don't hit the database every time.

```
Without Cache:            With Cache:
──────────────            ─────────────
Client → Server → DB      Client → Server → Cache HIT → return instantly
(every request, slow)                       Cache MISS → DB → store in Cache
```

### Cache Hit vs Cache Miss
- **Cache Hit** — data found in cache → fast response
- **Cache Miss** — data NOT in cache → fetch from DB, store in cache, return

### Where to Cache
| Level | Example | What's Cached |
|---|---|---|
| **Client-side** | Browser cache | Static assets (JS, CSS, images) |
| **CDN** | Cloudflare | Static files geographically close to user |
| **Server-side** | Redis, Memcached | DB query results, session data, computed data |
| **Database** | Query cache | Repeated SQL query results |

### Redis — Most Common Cache
```
// Conceptual flow in Spring Boot
public Product getProduct(Long id) {
    // 1. Check Redis cache
    Product cached = redis.get("product:" + id);
    if (cached != null) return cached;          // Cache HIT

    // 2. Cache MISS — go to DB
    Product product = productRepository.findById(id);

    // 3. Store in Redis with TTL (expiry)
    redis.set("product:" + id, product, TTL = 10 minutes);

    return product;
}
```

### Cache Eviction Policies
When cache is full, which data gets removed?
| Policy | Meaning |
|---|---|
| **LRU** (Least Recently Used) | Remove data not accessed for longest time ✅ Most common |
| **LFU** (Least Frequently Used) | Remove data accessed fewest times |
| **TTL** (Time To Live) | Auto-expire data after set duration |
| **FIFO** | Remove oldest inserted data |

### Cache Invalidation — The Hard Problem
When does cached data become stale?
```
Strategies:
1. TTL (Time-To-Live) — expire after N seconds (simple, may serve stale data briefly)
2. Write-Through — update cache whenever DB is updated (always fresh, more writes)
3. Write-Back (Lazy) — update cache first, sync DB later (fast writes, risk of data loss)
4. Cache-Aside — app manages cache manually (most flexible) ← most common
```

### When NOT to Cache
- Data that changes very frequently (live stock prices)
- User-specific sensitive data (without proper isolation)
- Large datasets where cache storage cost > benefit

---

## 4. Database Scaling Concepts

### The Problem
As users grow: DB becomes the bottleneck.
```
1 user  → 1 query/sec  → fine
1M users → 10,000 queries/sec → DB crashes
```

### Vertical Scaling (Scale Up)
Buy a bigger database server — more CPU, RAM, faster disks.
```
DB Server: 4 CPU, 16GB RAM  →  32 CPU, 256GB RAM
```
- ✅ Simple, no code changes
- ❌ Expensive, has hardware limits, single point of failure

### Horizontal Scaling (Scale Out)
Add more database servers.

#### Read Replicas
```
                    ┌── Read Replica 1 (handles SELECT queries)
Primary DB (Writes) ├── Read Replica 2 (handles SELECT queries)
                    └── Read Replica 3 (handles SELECT queries)
```
- All **writes** (INSERT, UPDATE, DELETE) go to Primary
- All **reads** (SELECT) go to Replicas
- **Replication lag** — replicas may be slightly behind primary (eventual consistency)
- ✅ Great for read-heavy apps (most apps are 80% reads)

#### Database Sharding
Split data across multiple databases based on a **shard key**:
```
Shard by user_id:
  Shard 1: users 1–1,000,000
  Shard 2: users 1,000,001–2,000,000
  Shard 3: users 2,000,001–3,000,000
```
- ✅ Massive horizontal scale
- ❌ Complex queries spanning shards, difficult to re-shard, joins are hard

### Indexing — Most Impactful Optimization
An index is a **lookup table** that speeds up queries:
```sql
-- Without index: full table scan (reads every row)
SELECT * FROM users WHERE email = 'ram@gmail.com';  -- O(n)

-- With index on email: instant lookup
CREATE INDEX idx_users_email ON users(email);        -- O(log n)
```
- ✅ Dramatically faster reads
- ❌ Slower writes (index must be updated), uses extra storage
- **Rule:** Add indexes on columns used in WHERE, JOIN, ORDER BY clauses

### SQL vs NoSQL — When to Use What
| | SQL (MySQL, PostgreSQL) | NoSQL (MongoDB, DynamoDB) |
|---|---|---|
| **Data** | Structured, relational | Flexible, unstructured |
| **Schema** | Fixed | Dynamic |
| **Scaling** | Vertical (mostly) | Horizontal (native) |
| **Transactions** | Strong ACID | Eventual consistency |
| **Use case** | Orders, users, finance | Catalogs, logs, social feeds |

> **SDE-1 Rule of Thumb:** Default to SQL (PostgreSQL/MySQL). Use NoSQL only when you have a clear reason (massive scale, flexible schema, geographically distributed).

### Connection Pooling
Opening a DB connection is expensive. A **connection pool** reuses existing connections:
```
App Server                  DB
──────────                  ──
Thread 1  ─┐
Thread 2  ─┤── Pool of 10 connections ──► MySQL
Thread 3  ─┘
(1000 threads share 10 connections — requests wait if pool is busy)
```
In Spring Boot: **HikariCP** (default) manages this automatically.

---

## 5. REST Principles

### What is REST?
**RE**presentational **S**tate **T**ransfer — an architectural style for designing networked applications using HTTP.

### 6 REST Constraints
| Constraint | Meaning |
|---|---|
| **Stateless** | Each request contains all info needed. Server stores no client state. |
| **Client-Server** | Separation of concerns — UI separate from data storage |
| **Cacheable** | Responses should indicate if they can be cached |
| **Uniform Interface** | Consistent URL structure and HTTP methods |
| **Layered System** | Client doesn't know if it's talking to server directly or via proxy |
| **Code on Demand** | (Optional) Server can send executable code (JS) to client |

### REST Resource Naming — Rules
```
✅ Use nouns, not verbs (the HTTP method is the verb)
✅ Use plural nouns for collections
✅ Use lowercase and hyphens
✅ Nest resources to show relationships

❌ /getUsers          → ✅ GET /users
❌ /createProduct     → ✅ POST /products
❌ /deleteOrder?id=5  → ✅ DELETE /orders/5
❌ /user_orders       → ✅ GET /users/{userId}/orders
```

### HTTP Methods — Correct Usage
| Method | Action | Idempotent? | Safe? |
|---|---|---|---|
| GET | Retrieve resource | ✅ Yes | ✅ Yes |
| POST | Create resource | ❌ No | ❌ No |
| PUT | Replace entire resource | ✅ Yes | ❌ No |
| PATCH | Partial update | ❌ No | ❌ No |
| DELETE | Delete resource | ✅ Yes | ❌ No |

> **Idempotent** = calling it multiple times gives the same result.
> **Safe** = no side effects, doesn't modify data.

### HTTP Status Codes You Must Know
```
2xx — Success
  200 OK              → GET, PUT, PATCH success
  201 Created         → POST success (include Location header)
  204 No Content      → DELETE success

4xx — Client Error
  400 Bad Request     → Invalid input, validation failed
  401 Unauthorized    → Not authenticated (no/invalid token)
  403 Forbidden       → Authenticated but not authorized
  404 Not Found       → Resource doesn't exist
  409 Conflict        → Duplicate resource (email already exists)
  422 Unprocessable   → Semantic validation error

5xx — Server Error
  500 Internal Server Error → Unexpected server crash
  503 Service Unavailable   → Server down/overloaded
```

---

## 6. API Design

### What Makes a Good API?
- **Intuitive** — developer can guess the URL without reading docs
- **Consistent** — same patterns everywhere
- **Versioned** — changes don't break existing clients
- **Documented** — Swagger/OpenAPI

### API Versioning
```
Option 1: URL versioning ✅ Most common
GET /api/v1/users
GET /api/v2/users

Option 2: Header versioning
GET /api/users
Header: Accept: application/vnd.myapp.v2+json

Option 3: Query param
GET /api/users?version=2
```

### Pagination — Never Return All Data
```
GET /api/products?page=0&size=10&sort=name,asc

Response:
{
  "content": [...],
  "page": 0,
  "size": 10,
  "totalElements": 250,
  "totalPages": 25,
  "last": false
}
```

### Filtering & Searching
```
GET /api/products?category=electronics&minPrice=100&maxPrice=500
GET /api/users?search=john&status=active
GET /api/orders?from=2024-01-01&to=2024-12-31&status=DELIVERED
```

### Consistent Error Response Format
```json
{
  "status": 404,
  "error": "Not Found",
  "message": "Product with id 42 not found",
  "path": "/api/products/42",
  "timestamp": "2024-06-13T10:30:00Z"
}
```

### API Request/Response Design Rules
```
Request:
  ✅ Use JSON body for POST/PUT
  ✅ Use path params for resource ID (/users/{id})
  ✅ Use query params for filters/pagination
  ✅ Validate all inputs, return 400 with clear messages

Response:
  ✅ Always return consistent structure
  ✅ Don't expose internal IDs or sensitive fields (passwords!)
  ✅ Use DTOs — don't return raw DB entities
  ✅ Include pagination metadata for lists
```

### DTOs (Data Transfer Objects)
Never expose your DB entity directly. Use DTOs:
```java
// Entity (DB layer) — has ALL fields
@Entity
public class User {
    private Long id;
    private String name;
    private String email;
    private String password;   // ← NEVER send this to client
    private String internalRef;
}

// Response DTO (what client sees)
public class UserResponseDTO {
    private Long id;
    private String name;
    private String email;
    // no password, no internalRef
}

// Request DTO (what client sends to create/update)
public class CreateUserDTO {
    @NotBlank private String name;
    @Email private String email;
    @Size(min=8) private String password;
}
```

---

## 7. Design: URL Shortener

> **Like bit.ly** — convert `https://very-long-url.com/some/path?with=params` → `https://short.ly/abc123`

### Requirements Clarification (ask these in interview)
- **Functional:** Create short URL, redirect to original, (optional: expiry, analytics)
- **Non-functional:** High availability, low latency reads, scale (millions of URLs)
- **Scale estimate:** 100M URLs created/day, 10:1 read:write ratio

---

### Core Entities
```
URL
─────────────────────────────
id            BIGINT PK
original_url  VARCHAR(2048)
short_code    VARCHAR(8)  UNIQUE INDEX
user_id       BIGINT FK (nullable, if auth supported)
expires_at    DATETIME (nullable)
created_at    DATETIME
click_count   BIGINT DEFAULT 0
```

### Short Code Generation — Key Design Decision

#### Option 1: Random String
```java
// Generate 6-character alphanumeric code
// 62 chars (a-z, A-Z, 0-9) ^ 6 = 56 billion combinations
String generateCode() {
    String chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    StringBuilder sb = new StringBuilder();
    Random random = new Random();
    for (int i = 0; i < 6; i++) {
        sb.append(chars.charAt(random.nextInt(chars.length())));
    }
    return sb.toString(); // e.g., "aB3kR9"
}
// Then check DB for collision — retry if exists
```

#### Option 2: Base62 of Auto-Increment ID ✅ Simpler
```java
// DB auto-increments ID: 1, 2, 3, ...
// Convert ID to Base62
// ID = 12345 → Base62 → "dnh"
// Guaranteed unique, no collision check needed

String toBase62(long id) {
    String chars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    StringBuilder sb = new StringBuilder();
    while (id > 0) {
        sb.append(chars.charAt((int)(id % 62)));
        id /= 62;
    }
    return sb.reverse().toString();
}
```

### API Design
```
POST   /api/urls
Body:  { "originalUrl": "https://very-long-url.com", "expiresIn": 7 }
Response 201:
{
  "shortCode": "abc123",
  "shortUrl": "https://short.ly/abc123",
  "originalUrl": "https://very-long-url.com",
  "expiresAt": "2024-06-20T00:00:00Z"
}

GET    /api/urls/{shortCode}       → 200 with URL details
DELETE /api/urls/{shortCode}       → 204
GET    /api/urls/{shortCode}/stats → { "clicks": 4200 }

// Redirect endpoint (non-API)
GET /{shortCode}                   → 301/302 Redirect to originalUrl
```

### Redirect: 301 vs 302
| | 301 Permanent | 302 Temporary |
|---|---|---|
| Browser caches? | ✅ Yes | ❌ No |
| Server gets future requests? | ❌ No (browser handles) | ✅ Yes |
| Analytics possible? | ❌ No | ✅ Yes |
> Use **302** if you want to track clicks. Use **301** for efficiency if analytics don't matter.

### Service Layer (Spring Boot)
```java
@Service
public class UrlShortenerService {

    private final UrlRepository urlRepo;

    public UrlResponseDTO createShortUrl(CreateUrlDTO dto) {
        // 1. Check if original URL already shortened (optional dedup)
        Optional<Url> existing = urlRepo.findByOriginalUrl(dto.getOriginalUrl());
        if (existing.isPresent()) return mapToDTO(existing.get());

        // 2. Save to DB (get auto-generated ID)
        Url url = new Url();
        url.setOriginalUrl(dto.getOriginalUrl());
        url.setCreatedAt(LocalDateTime.now());
        if (dto.getExpiresIn() != null) {
            url.setExpiresAt(LocalDateTime.now().plusDays(dto.getExpiresIn()));
        }
        Url saved = urlRepo.save(url);

        // 3. Generate short code from ID
        String shortCode = Base62Util.encode(saved.getId());
        saved.setShortCode(shortCode);
        urlRepo.save(saved);

        return mapToDTO(saved);
    }

    public String getOriginalUrl(String shortCode) {
        Url url = urlRepo.findByShortCode(shortCode)
            .orElseThrow(() -> new ResourceNotFoundException("Short URL not found"));

        if (url.getExpiresAt() != null && url.getExpiresAt().isBefore(LocalDateTime.now())) {
            throw new UrlExpiredException("This short URL has expired");
        }

        // Increment click count (can be async for performance)
        url.setClickCount(url.getClickCount() + 1);
        urlRepo.save(url);

        return url.getOriginalUrl();
    }
}
```

### Caching Layer
```
Read path (most traffic):
  GET /{shortCode}
    → Check Redis cache (key: shortCode, value: originalUrl)
    → Cache HIT: redirect instantly (sub-millisecond)
    → Cache MISS: query DB, store in Redis with TTL, redirect

Write path (less traffic):
  POST /api/urls
    → Generate shortCode, save to DB
    → Optionally pre-warm cache
```

### Architecture Diagram
```
Client
  │
  ▼
Load Balancer
  │
  ├─── App Server 1 ──┐
  ├─── App Server 2 ──┼──► Redis Cache ──► MySQL DB
  └─── App Server 3 ──┘       (reads)      (source of truth)
```

---

## 8. Design: Parking Lot

> Design a system to manage a parking lot — track spaces, issue tickets, calculate fees.

### Requirements
- **Functional:** Park vehicle, unpark vehicle, check availability, calculate fee
- **Types:** Multiple vehicle types (bike, car, truck), multiple floor levels
- **Fee:** Based on duration parked

---

### Core Entities & Their Relationships
```
ParkingLot
  └── has many Floors
        └── has many ParkingSpots
              └── has one VehicleType (BIKE, CAR, TRUCK)
              └── has one SpotStatus (AVAILABLE, OCCUPIED)

Vehicle
  ├── licensePlate
  └── vehicleType

ParkingTicket
  ├── ticketId
  ├── vehicle
  ├── spot
  ├── entryTime
  ├── exitTime
  └── amountCharged
```

### Database Schema
```sql
-- parking_spots table
CREATE TABLE parking_spots (
    id          BIGINT PRIMARY KEY AUTO_INCREMENT,
    floor_number INT NOT NULL,
    spot_number  INT NOT NULL,
    spot_type    ENUM('BIKE','CAR','TRUCK') NOT NULL,
    status       ENUM('AVAILABLE','OCCUPIED') DEFAULT 'AVAILABLE',
    UNIQUE (floor_number, spot_number)
);

-- vehicles table
CREATE TABLE vehicles (
    id            BIGINT PRIMARY KEY AUTO_INCREMENT,
    license_plate VARCHAR(20) UNIQUE NOT NULL,
    vehicle_type  ENUM('BIKE','CAR','TRUCK') NOT NULL
);

-- parking_tickets table
CREATE TABLE parking_tickets (
    id          BIGINT PRIMARY KEY AUTO_INCREMENT,
    vehicle_id  BIGINT NOT NULL REFERENCES vehicles(id),
    spot_id     BIGINT NOT NULL REFERENCES parking_spots(id),
    entry_time  DATETIME NOT NULL,
    exit_time   DATETIME,
    amount      DECIMAL(10,2),
    status      ENUM('ACTIVE','CLOSED') DEFAULT 'ACTIVE'
);
```

### API Design
```
// Check availability
GET  /api/parking/availability?type=CAR
Response: { "available": 42, "total": 100 }

// Park vehicle (entry)
POST /api/parking/entry
Body: { "licensePlate": "KA01AB1234", "vehicleType": "CAR" }
Response 201:
{
  "ticketId": 1001,
  "spotId": 45,
  "floor": 2,
  "spotNumber": 15,
  "vehicleType": "CAR",
  "licensePlate": "KA01AB1234",
  "entryTime": "2024-06-13T09:00:00"
}

// Unpark vehicle (exit)
POST /api/parking/exit/{ticketId}
Response 200:
{
  "ticketId": 1001,
  "entryTime": "2024-06-13T09:00:00",
  "exitTime": "2024-06-13T11:30:00",
  "duration": "2 hours 30 minutes",
  "amountCharged": 75.00
}

// Get ticket details
GET /api/parking/tickets/{ticketId}

// Admin: get all active tickets
GET /api/parking/tickets?status=ACTIVE&page=0&size=20
```

### Fee Calculation Strategy
```java
// Use Strategy Pattern for different fee structures
public interface FeeStrategy {
    double calculate(long durationMinutes);
}

@Component("carFeeStrategy")
public class CarFeeStrategy implements FeeStrategy {
    // First hour: ₹30, subsequent hours: ₹20/hour
    public double calculate(long durationMinutes) {
        if (durationMinutes <= 60) return 30.0;
        long extraHours = (long) Math.ceil((durationMinutes - 60) / 60.0);
        return 30.0 + (extraHours * 20.0);
    }
}

@Component("bikeFeeStrategy")
public class BikeFeeStrategy implements FeeStrategy {
    public double calculate(long durationMinutes) {
        return Math.ceil(durationMinutes / 60.0) * 10.0; // ₹10/hour
    }
}
```

### Service Layer
```java
@Service
public class ParkingService {

    private final ParkingSpotRepository spotRepo;
    private final ParkingTicketRepository ticketRepo;
    private final VehicleRepository vehicleRepo;
    private final Map<String, FeeStrategy> feeStrategies;

    @Transactional
    public TicketDTO parkVehicle(EntryRequestDTO request) {
        // 1. Find an available spot for vehicle type
        ParkingSpot spot = spotRepo
            .findFirstBySpotTypeAndStatus(request.getVehicleType(), SpotStatus.AVAILABLE)
            .orElseThrow(() -> new NoSpotAvailableException(
                "No available spots for " + request.getVehicleType()));

        // 2. Mark spot as occupied
        spot.setStatus(SpotStatus.OCCUPIED);
        spotRepo.save(spot);

        // 3. Find or create vehicle record
        Vehicle vehicle = vehicleRepo.findByLicensePlate(request.getLicensePlate())
            .orElseGet(() -> vehicleRepo.save(
                new Vehicle(request.getLicensePlate(), request.getVehicleType())));

        // 4. Create ticket
        ParkingTicket ticket = new ParkingTicket();
        ticket.setVehicle(vehicle);
        ticket.setSpot(spot);
        ticket.setEntryTime(LocalDateTime.now());
        ticket.setStatus(TicketStatus.ACTIVE);

        return mapToDTO(ticketRepo.save(ticket));
    }

    @Transactional
    public ExitResponseDTO unparkVehicle(Long ticketId) {
        ParkingTicket ticket = ticketRepo.findById(ticketId)
            .orElseThrow(() -> new ResourceNotFoundException("Ticket not found"));

        if (ticket.getStatus() == TicketStatus.CLOSED) {
            throw new IllegalStateException("Ticket already closed");
        }

        // 1. Set exit time
        LocalDateTime exitTime = LocalDateTime.now();
        ticket.setExitTime(exitTime);

        // 2. Calculate fee
        long minutes = ChronoUnit.MINUTES.between(ticket.getEntryTime(), exitTime);
        String strategyKey = ticket.getVehicle().getVehicleType().name().toLowerCase() + "FeeStrategy";
        double amount = feeStrategies.get(strategyKey).calculate(minutes);
        ticket.setAmount(amount);
        ticket.setStatus(TicketStatus.CLOSED);

        // 3. Free up the spot
        ParkingSpot spot = ticket.getSpot();
        spot.setStatus(SpotStatus.AVAILABLE);
        spotRepo.save(spot);
        ticketRepo.save(ticket);

        return new ExitResponseDTO(ticket, minutes, amount);
    }
}
```

---

## 9. Design: Library Management System

> Manage books, members, borrowing, returns, and fines.

### Requirements
- **Functional:** Add/search books, register members, borrow/return books, calculate fines
- **Constraints:** A book can have multiple copies; a member can borrow max 3 books

---

### Core Entities
```
Book (title, author, ISBN, genre)
  └── has many BookCopies (each is a physical copy with its own status)

Member (name, email, phone, membershipExpiry)
  └── can have many BorrowRecords

BorrowRecord
  ├── bookCopy
  ├── member
  ├── borrowDate
  ├── dueDate (borrowDate + 14 days)
  ├── returnDate
  └── fine (if returned late)
```

### Database Schema
```sql
CREATE TABLE books (
    id       BIGINT PRIMARY KEY AUTO_INCREMENT,
    title    VARCHAR(255) NOT NULL,
    author   VARCHAR(255) NOT NULL,
    isbn     VARCHAR(13) UNIQUE NOT NULL,
    genre    VARCHAR(100),
    publisher VARCHAR(255)
);

CREATE TABLE book_copies (
    id      BIGINT PRIMARY KEY AUTO_INCREMENT,
    book_id BIGINT NOT NULL REFERENCES books(id),
    status  ENUM('AVAILABLE','BORROWED','LOST','DAMAGED') DEFAULT 'AVAILABLE'
);

CREATE TABLE members (
    id                BIGINT PRIMARY KEY AUTO_INCREMENT,
    name              VARCHAR(255) NOT NULL,
    email             VARCHAR(255) UNIQUE NOT NULL,
    phone             VARCHAR(15),
    membership_expiry DATE NOT NULL,
    active            BOOLEAN DEFAULT TRUE
);

CREATE TABLE borrow_records (
    id           BIGINT PRIMARY KEY AUTO_INCREMENT,
    copy_id      BIGINT NOT NULL REFERENCES book_copies(id),
    member_id    BIGINT NOT NULL REFERENCES members(id),
    borrow_date  DATE NOT NULL,
    due_date     DATE NOT NULL,
    return_date  DATE,
    fine_amount  DECIMAL(10,2) DEFAULT 0,
    status       ENUM('ACTIVE','RETURNED','OVERDUE') DEFAULT 'ACTIVE'
);
```

### API Design
```
── BOOKS ──────────────────────────────────────────────────

GET    /api/books                    → list all books (paginated)
GET    /api/books/{id}               → get book details + available copies
GET    /api/books/search?q=harry&genre=fantasy  → search
POST   /api/books                    → add new book (admin)
PUT    /api/books/{id}               → update book (admin)
DELETE /api/books/{id}               → remove book (admin)

POST   /api/books/{id}/copies        → add a new physical copy (admin)

── MEMBERS ────────────────────────────────────────────────

POST   /api/members                  → register new member
GET    /api/members/{id}             → get member profile
GET    /api/members/{id}/borrows     → get member's borrow history
PUT    /api/members/{id}             → update profile

── BORROWING ──────────────────────────────────────────────

POST   /api/borrows
Body:  { "memberId": 5, "bookId": 12 }
Response 201:
{
  "borrowId": 301,
  "book": { "title": "Harry Potter", "isbn": "..." },
  "copyId": 45,
  "borrowDate": "2024-06-13",
  "dueDate": "2024-06-27",
  "member": { "name": "Ravi Kumar" }
}

POST   /api/borrows/{borrowId}/return
Response 200:
{
  "borrowId": 301,
  "returnDate": "2024-06-30",
  "daysOverdue": 3,
  "fineAmount": 15.00,    ← ₹5/day overdue
  "message": "Book returned with fine. Please pay at counter."
}

GET    /api/borrows?status=OVERDUE&page=0&size=20   → admin: overdue records
GET    /api/borrows/{borrowId}                       → get borrow details
```

### Service Layer — Borrow Logic
```java
@Service
public class BorrowService {

    private static final int MAX_BORROW_LIMIT = 3;
    private static final int LOAN_PERIOD_DAYS = 14;
    private static final double FINE_PER_DAY = 5.0;

    @Transactional
    public BorrowResponseDTO borrowBook(BorrowRequestDTO request) {
        Member member = memberRepo.findById(request.getMemberId())
            .orElseThrow(() -> new ResourceNotFoundException("Member not found"));

        // Validate membership
        if (!member.isActive() || member.getMembershipExpiry().isBefore(LocalDate.now())) {
            throw new InvalidMembershipException("Membership expired or inactive");
        }

        // Check borrow limit
        long activeCount = borrowRepo.countByMemberIdAndStatus(member.getId(), BorrowStatus.ACTIVE);
        if (activeCount >= MAX_BORROW_LIMIT) {
            throw new BorrowLimitExceededException("Member has already borrowed " + MAX_BORROW_LIMIT + " books");
        }

        // Find available copy
        BookCopy copy = bookCopyRepo
            .findFirstByBookIdAndStatus(request.getBookId(), CopyStatus.AVAILABLE)
            .orElseThrow(() -> new NoCopyAvailableException("No copies available for this book"));

        // Mark copy as borrowed
        copy.setStatus(CopyStatus.BORROWED);
        bookCopyRepo.save(copy);

        // Create borrow record
        BorrowRecord record = new BorrowRecord();
        record.setCopy(copy);
        record.setMember(member);
        record.setBorrowDate(LocalDate.now());
        record.setDueDate(LocalDate.now().plusDays(LOAN_PERIOD_DAYS));
        record.setStatus(BorrowStatus.ACTIVE);

        return mapToDTO(borrowRepo.save(record));
    }

    @Transactional
    public ReturnResponseDTO returnBook(Long borrowId) {
        BorrowRecord record = borrowRepo.findById(borrowId)
            .orElseThrow(() -> new ResourceNotFoundException("Borrow record not found"));

        if (record.getStatus() == BorrowStatus.RETURNED) {
            throw new IllegalStateException("Book already returned");
        }

        LocalDate returnDate = LocalDate.now();
        record.setReturnDate(returnDate);

        // Calculate fine
        double fine = 0.0;
        long daysOverdue = ChronoUnit.DAYS.between(record.getDueDate(), returnDate);
        if (daysOverdue > 0) {
            fine = daysOverdue * FINE_PER_DAY;
        }
        record.setFineAmount(fine);
        record.setStatus(BorrowStatus.RETURNED);

        // Free up copy
        record.getCopy().setStatus(CopyStatus.AVAILABLE);
        bookCopyRepo.save(record.getCopy());
        borrowRepo.save(record);

        return new ReturnResponseDTO(record, daysOverdue, fine);
    }
}
```

---

## 10. Design: Food Delivery APIs

> Like Zomato/Swiggy — users order food from restaurants, delivery partners deliver it.

### Requirements
- **Functional:** Browse restaurants/menus, place orders, track order status, manage cart
- **Actors:** Customer, Restaurant, Delivery Partner, Admin

---

### Core Entities
```
User (customer)
Restaurant → has many MenuItems
Order → has many OrderItems (each linked to a MenuItem)
  └── OrderStatus: PLACED → CONFIRMED → PREPARING → READY → PICKED_UP → DELIVERED
DeliveryPartner
Cart → has many CartItems (temporary, pre-order)
Address (saved addresses for user)
```

### Database Schema
```sql
CREATE TABLE restaurants (
    id          BIGINT PRIMARY KEY AUTO_INCREMENT,
    name        VARCHAR(255) NOT NULL,
    cuisine     VARCHAR(100),
    address     VARCHAR(500),
    rating      DECIMAL(2,1),
    is_open     BOOLEAN DEFAULT TRUE,
    owner_id    BIGINT REFERENCES users(id)
);

CREATE TABLE menu_items (
    id            BIGINT PRIMARY KEY AUTO_INCREMENT,
    restaurant_id BIGINT REFERENCES restaurants(id),
    name          VARCHAR(255) NOT NULL,
    description   TEXT,
    price         DECIMAL(10,2) NOT NULL,
    category      VARCHAR(100),
    is_available  BOOLEAN DEFAULT TRUE,
    is_veg        BOOLEAN DEFAULT TRUE
);

CREATE TABLE orders (
    id               BIGINT PRIMARY KEY AUTO_INCREMENT,
    customer_id      BIGINT REFERENCES users(id),
    restaurant_id    BIGINT REFERENCES restaurants(id),
    delivery_partner_id BIGINT REFERENCES delivery_partners(id),
    status           ENUM('PLACED','CONFIRMED','PREPARING','READY','PICKED_UP','DELIVERED','CANCELLED'),
    total_amount     DECIMAL(10,2),
    delivery_address TEXT,
    placed_at        DATETIME,
    delivered_at     DATETIME,
    delivery_fee     DECIMAL(10,2),
    special_notes    TEXT
);

CREATE TABLE order_items (
    id          BIGINT PRIMARY KEY AUTO_INCREMENT,
    order_id    BIGINT REFERENCES orders(id),
    menu_item_id BIGINT REFERENCES menu_items(id),
    quantity    INT NOT NULL,
    unit_price  DECIMAL(10,2) NOT NULL,  -- snapshot of price at order time
    subtotal    DECIMAL(10,2)
);
```

### Full API Design
```
── RESTAURANTS ─────────────────────────────────────────────

GET  /api/restaurants                         → list all open restaurants
GET  /api/restaurants?cuisine=indian&rating=4  → filter restaurants
GET  /api/restaurants/{id}                    → restaurant details
GET  /api/restaurants/{id}/menu               → full menu
GET  /api/restaurants/{id}/menu?category=pizza → filtered menu

POST /api/restaurants                         → register (restaurant owner)
PUT  /api/restaurants/{id}                    → update details (owner)
PATCH /api/restaurants/{id}/status            → open/close restaurant

── MENU ────────────────────────────────────────────────────

POST   /api/restaurants/{id}/menu             → add menu item
PUT    /api/restaurants/{id}/menu/{itemId}    → update item
PATCH  /api/restaurants/{id}/menu/{itemId}/availability → toggle availability
DELETE /api/restaurants/{id}/menu/{itemId}   → remove item

── CART ────────────────────────────────────────────────────

GET    /api/cart                              → view cart
POST   /api/cart/items
Body:  { "menuItemId": 5, "quantity": 2 }    → add item
PUT    /api/cart/items/{itemId}
Body:  { "quantity": 3 }                     → update quantity
DELETE /api/cart/items/{itemId}              → remove item
DELETE /api/cart                             → clear cart

── ORDERS ──────────────────────────────────────────────────

POST   /api/orders
Body:
{
  "restaurantId": 10,
  "deliveryAddressId": 3,
  "items": [
    { "menuItemId": 5, "quantity": 2 },
    { "menuItemId": 8, "quantity": 1 }
  ],
  "specialNotes": "Extra spicy please",
  "paymentMethod": "UPI"
}
Response 201:
{
  "orderId": 5001,
  "status": "PLACED",
  "restaurant": { "name": "Spice Garden" },
  "items": [...],
  "subtotal": 450.00,
  "deliveryFee": 30.00,
  "totalAmount": 480.00,
  "estimatedDelivery": "30-45 mins",
  "placedAt": "2024-06-13T12:00:00"
}

GET    /api/orders                            → customer order history
GET    /api/orders/{orderId}                  → order details + status
GET    /api/orders/{orderId}/track            → live tracking
POST   /api/orders/{orderId}/cancel           → cancel (if still PLACED)

── ORDER STATUS (Restaurant side) ──────────────────────────

PATCH  /api/orders/{orderId}/status
Body:  { "status": "CONFIRMED" }              → restaurant confirms
Body:  { "status": "PREPARING" }             → cooking started
Body:  { "status": "READY" }                 → ready for pickup

── ORDER STATUS (Delivery Partner side) ────────────────────

PATCH  /api/orders/{orderId}/status
Body:  { "status": "PICKED_UP" }             → partner picked up
Body:  { "status": "DELIVERED" }             → delivered to customer

── REVIEWS ─────────────────────────────────────────────────

POST   /api/orders/{orderId}/review
Body:  { "restaurantRating": 4, "deliveryRating": 5, "comment": "Great food!" }

GET    /api/restaurants/{id}/reviews?page=0&size=10
```

### Order Placement Service
```java
@Service
public class OrderService {

    @Transactional
    public OrderResponseDTO placeOrder(PlaceOrderDTO dto, Long customerId) {
        // 1. Validate restaurant is open
        Restaurant restaurant = restaurantRepo.findById(dto.getRestaurantId())
            .orElseThrow(() -> new ResourceNotFoundException("Restaurant not found"));
        if (!restaurant.isOpen()) {
            throw new RestaurantClosedException("Restaurant is currently closed");
        }

        // 2. Validate all menu items and compute total
        List<OrderItem> orderItems = new ArrayList<>();
        double subtotal = 0;

        for (OrderItemDTO itemDTO : dto.getItems()) {
            MenuItem menuItem = menuItemRepo.findById(itemDTO.getMenuItemId())
                .orElseThrow(() -> new ResourceNotFoundException("Menu item not found"));
            if (!menuItem.isAvailable()) {
                throw new ItemUnavailableException(menuItem.getName() + " is not available");
            }
            double itemTotal = menuItem.getPrice() * itemDTO.getQuantity();
            subtotal += itemTotal;
            orderItems.add(new OrderItem(menuItem, itemDTO.getQuantity(), menuItem.getPrice(), itemTotal));
        }

        // 3. Calculate delivery fee
        double deliveryFee = calculateDeliveryFee(restaurant, dto.getDeliveryAddressId());

        // 4. Create order
        Order order = new Order();
        order.setCustomerId(customerId);
        order.setRestaurant(restaurant);
        order.setStatus(OrderStatus.PLACED);
        order.setTotalAmount(subtotal + deliveryFee);
        order.setDeliveryFee(deliveryFee);
        order.setSpecialNotes(dto.getSpecialNotes());
        order.setPlacedAt(LocalDateTime.now());
        Order saved = orderRepo.save(order);

        // 5. Save order items
        orderItems.forEach(item -> { item.setOrder(saved); orderItemRepo.save(item); });

        // 6. Notify restaurant (async - via event/message queue)
        eventPublisher.publishEvent(new OrderPlacedEvent(saved.getId()));

        return mapToDTO(saved);
    }

    @Transactional
    public void updateOrderStatus(Long orderId, OrderStatus newStatus, Long actorId) {
        Order order = orderRepo.findById(orderId)
            .orElseThrow(() -> new ResourceNotFoundException("Order not found"));

        validateStatusTransition(order.getStatus(), newStatus);
        order.setStatus(newStatus);

        if (newStatus == OrderStatus.DELIVERED) {
            order.setDeliveredAt(LocalDateTime.now());
        }

        orderRepo.save(order);
        // Notify customer via push notification / SMS (async)
        notificationService.notifyCustomer(order.getCustomerId(), newStatus);
    }
}
```

### Order Status State Machine
```
PLACED
  ↓ (restaurant confirms)
CONFIRMED
  ↓ (restaurant starts cooking)
PREPARING
  ↓ (food ready, waiting for pickup)
READY
  ↓ (delivery partner picks up)
PICKED_UP
  ↓ (delivered to customer)
DELIVERED

CANCELLED ← can happen from PLACED or CONFIRMED only
```

---

## 11. How to Approach Any Design Question

### The 5-Step Framework (Use in Every Interview)

```
Step 1: CLARIFY Requirements (2-3 mins)
─────────────────────────────────────
Ask functional requirements: "What are the core features?"
Ask constraints: "How many users? Read-heavy or write-heavy?"
Ask assumptions: "Should I include authentication?"
Confirm scope: "Should I focus on the API design or the DB schema?"

Step 2: DEFINE Entities & Relationships (3-4 mins)
───────────────────────────────────────────────────
Identify the main objects (nouns in the requirements)
Define their attributes and relationships
Decide on primary keys, foreign keys

Step 3: DESIGN the API (5-6 mins)
──────────────────────────────────
List all endpoints grouped by resource
Use proper HTTP methods and status codes
Define request and response structure
Think about pagination, filtering

Step 4: WRITE the Schema (3-4 mins)
────────────────────────────────────
Create tables with columns and types
Add indexes on frequently queried columns
Show relationships via foreign keys

Step 5: DISCUSS Trade-offs (2-3 mins)
──────────────────────────────────────
"For caching, I'd use Redis to cache frequently accessed data"
"For scale, I'd add read replicas to the DB"
"One challenge here is... I'd handle it by..."
```

### What Interviewers Are Actually Evaluating
```
✅ Can you break down a vague problem into clear requirements?
✅ Do you know how to design clean, RESTful APIs?
✅ Do you understand basic DB design (normalization, indexes)?
✅ Do you know why caching/load balancing exists?
✅ Can you communicate your thought process clearly?

❌ They don't expect: Kafka, microservices, consistent hashing
❌ They don't expect: exact capacity calculations
❌ They don't expect: a perfect system with zero flaws
```

### Phrases to Use in the Interview
```
"Before I start, let me clarify a few things..."
"I'll assume X for now — let me know if that's wrong."
"The main entities I see are..."
"For this endpoint, I'm choosing POST because..."
"A potential issue here is... I'd handle it by..."
"For scale, we could add... but for now let's keep it simple."
```

---

## Quick Cheatsheet

```
CLIENT-SERVER
─────────────
Client → DNS → Load Balancer → App Server → Cache → DB → Response

LOAD BALANCER
─────────────
Round Robin → equal servers
Least Connections → variable duration requests
IP Hash → sticky sessions

CACHING
───────
Cache-Aside: app manages cache manually (most common)
TTL: auto-expire stale data
Redis: key-value, in-memory, sub-millisecond

DATABASE SCALING
────────────────
Read Replicas → scale reads (80% of traffic)
Indexing → fastest win for query performance
Sharding → scale writes (complex, last resort)
Connection Pooling → HikariCP (Spring Boot default)

REST
────
Nouns in URLs, verbs via HTTP methods
GET=read, POST=create, PUT=replace, PATCH=partial, DELETE=remove
Stateless: every request self-contained

API DESIGN CHECKLIST
────────────────────
✅ Versioning (/api/v1/)
✅ Pagination for lists
✅ DTOs (don't expose entities)
✅ Consistent error format
✅ Proper HTTP status codes
✅ Input validation
```
