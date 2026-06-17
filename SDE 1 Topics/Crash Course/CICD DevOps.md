# CI/CD, DevOps & SDE-1 Interview Notes
### Tailored for: CSG — Software Development Engineer I

> **Role focus:** Azure DevOps · CI/CD Pipelines · REST APIs · SQL · OAuth/Auth · JSON/XML · Agile · ETL · Logging & Monitoring  
> **Stack:** C#/.NET / Java · Git · Azure · Jira · Postman

---

## 📋 JD Snapshot — What CSG Expects

| Area | Key Skills from JD |
|---|---|
| **CI/CD** | Azure Pipelines, build/test/deploy, automated deployments |
| **Integration** | REST APIs, Jira↔ADO↔Helix, Jitterbit, iPaaS |
| **Data** | SQL, ETL (extract/transform/validate/load), data migration |
| **Auth** | OAuth, API keys, Basic Auth |
| **Data Formats** | JSON, XML, API-based integrations |
| **Tooling** | Azure DevOps, Jira, Postman, Git, Visual Studio |
| **Practices** | Agile, code reviews, runbooks, logging/monitoring/alerting |

---

# PART 1 — CI/CD & DevOps

---

## 1. What is CI/CD?

**CI/CD** stands for **Continuous Integration / Continuous Delivery (or Deployment)**. It is a set of practices and automated pipelines that allow teams to deliver software faster, with fewer bugs and less manual effort.

```
Developer pushes code
        ↓
  [CI] Build + Test         ← Triggered automatically on every push/PR
        ↓
  [CD] Deploy to Dev
        ↓
  [CD] Deploy to Test/QA
        ↓
  [CD] Deploy to Production ← Manual gate or fully automated
```

### CI vs CD vs CD

| Term | Stands For | Meaning |
|---|---|---|
| **CI** | Continuous Integration | Automatically build and test every code change |
| **CD** | Continuous Delivery | Automatically deploy to staging; prod needs manual approval |
| **CD** | Continuous Deployment | Automatically deploy all the way to production |

### Why it matters
- Catch bugs early (at commit time, not release time)
- Reduce manual deployment errors
- Faster feedback cycles
- Repeatable, auditable deployments

---

## 2. Azure DevOps (ADO) — Core Concepts

Azure DevOps is Microsoft's end-to-end DevOps platform. It bundles five services:

| Service | Purpose |
|---|---|
| **Azure Repos** | Git-based source control |
| **Azure Pipelines** | CI/CD automation |
| **Azure Boards** | Agile planning (Epics, Stories, Tasks, Bugs) |
| **Azure Test Plans** | Manual and automated testing |
| **Azure Artifacts** | Package management (NuGet, npm, Maven) |

### Key ADO Terms

| Term | Definition |
|---|---|
| **Organization** | Top-level container (company level) |
| **Project** | A product/team workspace within an org |
| **Repo** | A Git repository inside a project |
| **Pipeline** | Automated workflow definition |
| **Agent** | A machine (hosted or self-hosted) that runs pipeline jobs |
| **Stage** | A logical phase in a pipeline (Build, Test, Deploy) |
| **Job** | A set of steps that run on one agent |
| **Step / Task** | Individual action (compile, run tests, publish artifact) |
| **Artifact** | Build output (compiled binary, Docker image, zip) passed between stages |
| **Service Connection** | Credentials to connect ADO to external services (Azure, AWS, GitHub) |

---

## 3. Azure Pipelines — Build → Test → Deploy

### YAML Pipeline Structure

```yaml
# azure-pipelines.yml

trigger:
  branches:
    include:
      - main
      - develop

pool:
  vmImage: 'ubuntu-latest'       # Microsoft-hosted agent

variables:
  buildConfiguration: 'Release'

stages:

  # ─── STAGE 1: BUILD ───────────────────────────────
  - stage: Build
    displayName: 'Build Application'
    jobs:
      - job: BuildJob
        steps:
          - task: UseDotNet@2
            inputs:
              packageType: 'sdk'
              version: '8.x'

          - script: dotnet restore
            displayName: 'Restore NuGet Packages'

          - script: dotnet build --configuration $(buildConfiguration)
            displayName: 'Build Solution'

          - task: PublishBuildArtifacts@1
            inputs:
              pathToPublish: '$(Build.ArtifactStagingDirectory)'
              artifactName: 'drop'

  # ─── STAGE 2: TEST ────────────────────────────────
  - stage: Test
    displayName: 'Run Tests'
    dependsOn: Build
    jobs:
      - job: TestJob
        steps:
          - script: dotnet test --configuration $(buildConfiguration) --logger trx
            displayName: 'Run Unit Tests'

          - task: PublishTestResults@2
            inputs:
              testResultsFormat: 'VSTest'
              testResultsFiles: '**/*.trx'

  # ─── STAGE 3: DEPLOY TO DEV ───────────────────────
  - stage: DeployDev
    displayName: 'Deploy to Dev'
    dependsOn: Test
    condition: succeeded()
    jobs:
      - deployment: DeployToDev
        environment: 'dev'
        strategy:
          runOnce:
            deploy:
              steps:
                - script: echo "Deploying to Dev environment"
                - task: AzureWebApp@1
                  inputs:
                    azureSubscription: 'my-service-connection'
                    appName: 'myapp-dev'
                    package: '$(Pipeline.Workspace)/drop/**/*.zip'

  # ─── STAGE 4: DEPLOY TO PROD (with approval gate) ─
  - stage: DeployProd
    displayName: 'Deploy to Production'
    dependsOn: DeployDev
    jobs:
      - deployment: DeployToProd
        environment: 'prod'          # Approval gate configured in ADO UI
        strategy:
          runOnce:
            deploy:
              steps:
                - task: AzureWebApp@1
                  inputs:
                    azureSubscription: 'my-service-connection'
                    appName: 'myapp-prod'
                    package: '$(Pipeline.Workspace)/drop/**/*.zip'
```

### Key Pipeline Concepts

```yaml
condition: succeeded()         # Only run if previous stage passed
condition: always()            # Run regardless of prior result
dependsOn: [Build, Test]       # Explicit dependency ordering
timeoutInMinutes: 30           # Max runtime for a job
```

### Build vs Release Pipelines

| | Classic Build Pipeline | Classic Release Pipeline | YAML (Modern) |
|---|---|---|---|
| Purpose | Build + test | Deploy to environments | Both in one file |
| Config | GUI | GUI | Code (YAML) |
| Source control | No | No | Yes — lives in repo |
| Recommended? | Legacy | Legacy | ✅ Yes |

---

## 4. Environment Separation — Dev / Test / Prod

Separating environments prevents untested code from reaching customers and mirrors real-world conditions at each stage.

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│     DEV     │ →  │     QA      │ →  │   STAGING   │ →  │    PROD     │
│  (develop)  │    │  (test)     │    │ (pre-prod)  │    │   (main)    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
 Dev freedom        Automated tests     Load/perf tests    Real users
 Local/shared DB    Test data           Prod-like config   Live data
 No approval        Auto deploy         Manual approval    Manual gate
```

### Environment-Specific Configuration

```json
// appsettings.Development.json
{
  "ConnectionStrings": {
    "DefaultConnection": "Server=dev-db;Database=MyAppDev;"
  },
  "Logging": { "LogLevel": { "Default": "Debug" } },
  "FeatureFlags": { "NewUI": true }
}

// appsettings.Production.json
{
  "ConnectionStrings": {
    "DefaultConnection": "Server=prod-db;Database=MyApp;"
  },
  "Logging": { "LogLevel": { "Default": "Warning" } },
  "FeatureFlags": { "NewUI": false }
}
```

### ADO Environments & Approval Gates

In Azure DevOps, **Environments** are objects you create per stage (dev, test, prod). You can add:
- **Approval checks** — a named person must click "Approve" before deployment proceeds
- **Branch policies** — only deploy from `main`
- **Business hours gates** — only deploy Mon–Fri 9am–5pm

---

## 5. Deployment Types

### 5.1 Blue-Green Deployment

Two identical production environments run simultaneously. Traffic is switched from **Blue** (live) to **Green** (new version) all at once.

```
         ┌──────────────────┐
Users ──▶│   Load Balancer  │
         └────────┬─────────┘
          Switch traffic ↕
    ┌─────────────┐   ┌─────────────┐
    │    BLUE     │   │    GREEN    │
    │  (v1.0 -    │   │  (v2.0 -   │
    │   current)  │   │   new)     │
    └─────────────┘   └─────────────┘
```

**Flow:**
1. Green is deployed with new version while Blue serves production
2. Green is tested in isolation
3. Load balancer flips traffic from Blue → Green
4. Blue remains idle as instant rollback option

| Pros | Cons |
|---|---|
| Zero downtime | Double infrastructure cost |
| Instant rollback (flip back) | State/session management during switch |
| Full testing in prod-like env | DB schema changes need care |

---

### 5.2 Rolling Deployment

Replace instances of the old version **one by one** (or in small batches) without taking everything down at once.

```
Start:   [v1] [v1] [v1] [v1] [v1]  ← All running v1

Step 1:  [v2] [v1] [v1] [v1] [v1]  ← 1 instance updated

Step 2:  [v2] [v2] [v1] [v1] [v1]

Step 3:  [v2] [v2] [v2] [v1] [v1]

Done:    [v2] [v2] [v2] [v2] [v2]  ← All running v2
```

| Pros | Cons |
|---|---|
| No extra infrastructure cost | Both versions run simultaneously during rollout |
| Gradual — catch issues early | Rollback is slower (re-roll all instances) |
| Good for stateless services | Not ideal for breaking API changes |

---

### 5.3 Canary Deployment

Route a **small percentage of traffic** to the new version, observe, then gradually increase.

```
         ┌──────────────────┐
Users ──▶│   Load Balancer  │
         └───────┬──────────┘
          95%  ↙   ↘  5%
        [v1]        [v2 - Canary]
      (stable)      (new release)
```

**Typical progression:** 1% → 5% → 25% → 50% → 100%

| Pros | Cons |
|---|---|
| Real user testing at low risk | Needs good monitoring to detect issues |
| Easy rollback (route 0% to canary) | Complexity in routing/config |
| Data-driven rollout decisions | Both versions live simultaneously |

### Deployment Strategy Comparison

| Strategy | Downtime | Rollback Speed | Cost | Best For |
|---|---|---|---|---|
| **Blue-Green** | None | Instant | High (2x infra) | Critical services, databases |
| **Rolling** | None | Slow | Low | Stateless web apps |
| **Canary** | None | Fast | Medium | Risky features, large user bases |
| **Recreate** | Yes | Fast | Low | Dev/Test environments only |

---

## 6. Key DevOps Concepts for Interview

### 6.1 Infrastructure as Code (IaC)
Define infrastructure (servers, networks, databases) in code/config files — versioned, repeatable, reviewable.

```yaml
# Azure ARM Template / Bicep snippet
resource webApp 'Microsoft.Web/sites@2022-03-01' = {
  name: 'myapp-prod'
  location: 'eastus'
  properties: {
    serverFarmId: appServicePlan.id
  }
}
```

Tools: **Azure ARM / Bicep**, Terraform, Pulumi

### 6.2 Artifact Management
Build output stored in a package registry — pulled by deployment stages.

```
Build Stage  →  Publish artifact to Azure Artifacts (NuGet/npm)
Deploy Stage →  Download artifact → Deploy to environment
```

### 6.3 Pipeline Triggers

```yaml
trigger:                    # CI trigger on push
  branches:
    include: [main, develop]
  paths:
    include: [src/**]

pr:                         # PR trigger (run on pull requests)
  branches:
    include: [main]

schedules:                  # Nightly builds
  - cron: "0 2 * * *"
    branches:
      include: [main]
```

### 6.4 Pipeline Variables & Secrets

```yaml
variables:
  appName: 'myapp'            # Plain variable

# Secrets stored in Azure Key Vault or ADO Variable Groups
# Referenced as: $(MY_SECRET) — never echoed in logs
```

---

# PART 2 — REST APIs & Web Services

---

## 7. REST API Fundamentals

**REST** (Representational State Transfer) is an architectural style for building web services over HTTP.

### Core Principles
- **Stateless** — each request contains all information needed; server stores no session
- **Client-Server** — decoupled; client and server evolve independently
- **Uniform Interface** — resources accessed via consistent URLs and HTTP verbs
- **Resource-based** — everything is a resource (a user, an order, a ticket)

### HTTP Methods

| Method | CRUD | Description | Idempotent? |
|---|---|---|---|
| `GET` | Read | Retrieve a resource | ✅ Yes |
| `POST` | Create | Create a new resource | ❌ No |
| `PUT` | Update | Replace entire resource | ✅ Yes |
| `PATCH` | Update | Partially update a resource | ✅ Yes |
| `DELETE` | Delete | Remove a resource | ✅ Yes |

### HTTP Status Codes

| Range | Meaning | Examples |
|---|---|---|
| **2xx** | Success | 200 OK, 201 Created, 204 No Content |
| **3xx** | Redirect | 301 Moved Permanently, 304 Not Modified |
| **4xx** | Client Error | 400 Bad Request, 401 Unauthorized, 403 Forbidden, 404 Not Found, 409 Conflict, 422 Unprocessable Entity |
| **5xx** | Server Error | 500 Internal Server Error, 503 Service Unavailable |

### REST URL Design Best Practices

```
GET    /api/users             → List all users
GET    /api/users/42          → Get user 42
POST   /api/users             → Create a new user
PUT    /api/users/42          → Replace user 42 entirely
PATCH  /api/users/42          → Update specific fields of user 42
DELETE /api/users/42          → Delete user 42

GET    /api/users/42/orders   → Get all orders for user 42
GET    /api/users/42/orders/7 → Get order 7 of user 42

# Filtering, sorting, pagination
GET    /api/users?status=active&sort=name&page=2&limit=20
```

**Rules:**
- Use **nouns**, not verbs in URLs (`/users`, not `/getUsers`)
- Use **plural** resource names (`/users`, `/orders`)
- Use **lowercase** with hyphens (`/work-items`, not `/workItems`)
- Version your API: `/api/v1/users`

### Request/Response Example

```http
POST /api/v1/tickets HTTP/1.1
Host: api.csgi.com
Content-Type: application/json
Authorization: Bearer eyJhbGci...

{
  "title": "Login page crash",
  "priority": "high",
  "assignee": "john.doe@csgi.com"
}
```

```http
HTTP/1.1 201 Created
Content-Type: application/json
Location: /api/v1/tickets/5821

{
  "id": 5821,
  "title": "Login page crash",
  "status": "open",
  "createdAt": "2025-06-01T10:30:00Z"
}
```

---

## 8. JSON & XML

### JSON (JavaScript Object Notation)

```json
{
  "workItem": {
    "id": 1042,
    "title": "Implement OAuth login",
    "type": "UserStory",
    "state": "Active",
    "assignedTo": {
      "name": "Alice",
      "email": "alice@csgi.com"
    },
    "tags": ["auth", "security"],
    "storyPoints": 5,
    "isBlocked": false
  }
}
```

**Key rules:** Keys must be strings (double quotes), supports `string`, `number`, `boolean`, `null`, `array`, `object`. No comments allowed.

### XML (eXtensible Markup Language)

```xml
<?xml version="1.0" encoding="UTF-8"?>
<WorkItem id="1042">
  <Title>Implement OAuth login</Title>
  <Type>UserStory</Type>
  <State>Active</State>
  <AssignedTo>
    <Name>Alice</Name>
    <Email>alice@csgi.com</Email>
  </AssignedTo>
  <Tags>
    <Tag>auth</Tag>
    <Tag>security</Tag>
  </Tags>
  <StoryPoints>5</StoryPoints>
  <IsBlocked>false</IsBlocked>
</WorkItem>
```

### JSON vs XML Comparison

| Feature | JSON | XML |
|---|---|---|
| Readability | More readable | Verbose |
| Size | Compact | Larger (tags add overhead) |
| Data types | Native (number, bool, null) | Everything is text |
| Comments | Not supported | Supported |
| Parsing | `JSON.parse()` — fast | DOM/SAX parsers — heavier |
| Schema validation | JSON Schema | XSD (XML Schema Definition) |
| Use today | REST APIs, configs | SOAP, legacy enterprise, configs |

---

# PART 3 — AUTHENTICATION & SECURITY

---

## 9. Authentication Concepts

### 9.1 Basic Authentication

Credentials encoded in Base64 and sent in the `Authorization` header on every request.

```http
Authorization: Basic dXNlcjpwYXNzd29yZA==
# Decoded: user:password
```

- Simple but insecure unless used over HTTPS
- No token expiry or revocation
- Used in: internal tools, legacy APIs, testing

### 9.2 API Keys

A unique secret key issued to a client. Sent via header or query string.

```http
# Via header (preferred)
Authorization: ApiKey abc123xyz

# Via query param (avoid in production — logged in URLs)
GET /api/data?api_key=abc123xyz
```

- No identity (just authorization for a client/app)
- No built-in expiry
- Used in: third-party integrations, public APIs (OpenWeather, Stripe)

### 9.3 OAuth 2.0 — The Modern Standard

OAuth 2.0 is an **authorization framework** that lets an application access resources on behalf of a user **without sharing their password**.

**Key Roles:**
| Role | Description |
|---|---|
| **Resource Owner** | The user who owns the data |
| **Client** | The application requesting access |
| **Authorization Server** | Issues access tokens (e.g., Azure AD, Google) |
| **Resource Server** | The API that accepts and validates tokens |

**Authorization Code Flow (most common):**

```
User clicks "Login with Microsoft"
        ↓
App redirects user to Azure AD login page
        ↓
User authenticates + consents
        ↓
Azure AD returns an Authorization Code to the app
        ↓
App exchanges code for Access Token + Refresh Token
        ↓
App calls API with: Authorization: Bearer <access_token>
        ↓
API validates token and returns data
```

**Token Types:**

| Token | Purpose | Lifetime |
|---|---|---|
| **Access Token** | Bearer token for API calls | Short (15min – 1hr) |
| **Refresh Token** | Get a new access token without re-login | Long (days/weeks) |
| **ID Token** | Contains user identity info (OpenID Connect) | Short |

### OAuth vs API Key vs Basic Auth

| | Basic Auth | API Key | OAuth 2.0 |
|---|---|---|---|
| Security | Low (no expiry) | Medium | High |
| Identity | User | App/Client | User + App |
| Expiry | No | No (manual rotation) | Yes (short-lived) |
| Best for | Internal/dev | Service-to-service | User-delegated access |

### 9.4 JWT (JSON Web Token)

JWTs are the common format for OAuth access tokens. Self-contained — the API doesn't need to call the auth server to validate.

```
Header.Payload.Signature
eyJhbGci...  .  eyJzdWIi...  .  SflKxwRJSMeKKF2...
```

**Payload (decoded):**
```json
{
  "sub": "alice@csgi.com",
  "name": "Alice",
  "roles": ["developer"],
  "iss": "https://login.microsoftonline.com/...",
  "exp": 1735689600,
  "iat": 1735686000
}
```

---

# PART 4 — SQL & DATABASES

---

## 10. SQL Fundamentals

### Core DDL vs DML vs DCL

| Category | Commands | Description |
|---|---|---|
| **DDL** | CREATE, ALTER, DROP, TRUNCATE | Define/change schema |
| **DML** | SELECT, INSERT, UPDATE, DELETE | Manipulate data |
| **DCL** | GRANT, REVOKE | Control access |
| **TCL** | COMMIT, ROLLBACK, SAVEPOINT | Manage transactions |

### Essential Queries

```sql
-- Basic SELECT with filtering and ordering
SELECT id, name, email, created_at
FROM   users
WHERE  status = 'active'
  AND  created_at >= '2024-01-01'
ORDER BY name ASC
LIMIT 20 OFFSET 40;   -- Pagination: page 3, 20 per page

-- Aggregations
SELECT department, COUNT(*) AS headcount, AVG(salary) AS avg_salary
FROM   employees
GROUP BY department
HAVING COUNT(*) > 5
ORDER BY avg_salary DESC;

-- JOINs
-- INNER JOIN — only matching rows in both tables
SELECT o.id, o.total, u.name, u.email
FROM   orders o
INNER JOIN users u ON o.user_id = u.id;

-- LEFT JOIN — all rows from left table, NULLs for unmatched right rows
SELECT u.name, COUNT(o.id) AS order_count
FROM   users u
LEFT JOIN orders o ON u.id = o.user_id
GROUP BY u.name;

-- Subquery
SELECT name FROM users
WHERE id IN (
    SELECT DISTINCT user_id FROM orders WHERE total > 1000
);

-- Common Table Expression (CTE)
WITH TopCustomers AS (
    SELECT user_id, SUM(total) AS lifetime_value
    FROM   orders
    GROUP BY user_id
    HAVING SUM(total) > 5000
)
SELECT u.name, tc.lifetime_value
FROM   TopCustomers tc
JOIN   users u ON tc.user_id = u.id
ORDER BY tc.lifetime_value DESC;
```

### Indexes

```sql
-- Create index to speed up queries on email
CREATE INDEX idx_users_email ON users(email);

-- Composite index
CREATE INDEX idx_orders_user_date ON orders(user_id, created_at);

-- Unique index (also enforces uniqueness)
CREATE UNIQUE INDEX idx_users_email_unique ON users(email);

-- Drop index
DROP INDEX idx_users_email;
```

**When to index:** Columns used in `WHERE`, `JOIN ON`, `ORDER BY`. Do NOT index columns that are updated very frequently.

### Transactions

```sql
BEGIN TRANSACTION;

UPDATE accounts SET balance = balance - 500 WHERE id = 1;
UPDATE accounts SET balance = balance + 500 WHERE id = 2;

-- If anything fails:
ROLLBACK;

-- If all good:
COMMIT;
```

**ACID Properties:**

| Property | Meaning |
|---|---|
| **Atomicity** | All steps succeed or all fail — no partial updates |
| **Consistency** | Data always moves from one valid state to another |
| **Isolation** | Concurrent transactions don't interfere |
| **Durability** | Committed data survives crashes |

---

# PART 5 — ETL & DATA MIGRATION

---

## 11. ETL — Extract, Transform, Load

ETL is the process of moving data from one system to another — a core responsibility in this CSG role (Jira → ADO migration).

```
SOURCE SYSTEM          TRANSFORM LAYER          TARGET SYSTEM
(Jira / Helix)   →   (Validate, Map, Clean)  →  (Azure DevOps)
   Extract                Transform                  Load
```

### Stage 1: Extract
Pull data from the source system via API, SQL query, file export, or event stream.

```csharp
// Example: Extract tickets from Jira REST API
var response = await httpClient.GetAsync(
    "/rest/api/3/search?jql=project=MYPROJ&maxResults=100"
);
var jiraIssues = await response.Content.ReadFromJsonAsync<JiraSearchResult>();
```

### Stage 2: Transform
Clean, map, enrich, and validate data to match the target schema.

```csharp
// Map Jira issue to ADO WorkItem format
var workItem = new AdoWorkItem {
    Title       = jiraIssue.Summary,
    Description = HtmlToMarkdown(jiraIssue.Description), // Convert format
    State       = MapState(jiraIssue.Status.Name),       // "In Progress" → "Active"
    Priority    = MapPriority(jiraIssue.Priority.Name),  // "High" → 1
    Tags        = string.Join(";", jiraIssue.Labels)
};

// Validation
if (string.IsNullOrEmpty(workItem.Title))
    throw new ValidationException($"Issue {jiraIssue.Key} has no title");
```

### Stage 3: Load
Write the transformed data to the target system.

```csharp
// Create WorkItem in Azure DevOps
var patchDocument = new JsonPatchDocument {
    new JsonPatchOperation {
        Operation = Operation.Add,
        Path      = "/fields/System.Title",
        Value     = workItem.Title
    },
    new JsonPatchOperation {
        Operation = Operation.Add,
        Path      = "/fields/System.Description",
        Value     = workItem.Description
    }
};
await witClient.CreateWorkItemAsync(patchDocument, project, "User Story");
```

### Validation Strategies

```
1. Row counts       — source count == target count?
2. Checksums        — hash key fields to detect corruption
3. Spot checks      — randomly sample and compare records
4. Referential integrity — all FK references exist in target
5. Business rules   — no nulls in required fields, valid enum values
```

---

# PART 6 — LOGGING, MONITORING & ALERTING

---

## 12. Logging Best Practices

Good logging is critical for debugging production issues — key for the CSG role.

### Log Levels

| Level | When to Use |
|---|---|
| `TRACE` | Very detailed — method entry/exit, loop iterations |
| `DEBUG` | Diagnostic info useful in development |
| `INFO` | Normal application events (startup, requests received) |
| `WARN` | Unexpected but recoverable (slow query, deprecated API) |
| `ERROR` | Failures that need attention (exception, API call failed) |
| `CRITICAL` | System is down or data loss risk |

### Structured Logging (Preferred over plain text)

```csharp
// Bad — hard to query/filter
logger.LogInformation("Processing order for user " + userId);

// Good — structured, queryable, filterable
logger.LogInformation(
    "Processing order {OrderId} for user {UserId} with total {Total}",
    orderId, userId, total
);
```

### What to Log

```csharp
// API integration events
logger.LogInformation("Jira→ADO sync started. Fetching {Count} issues", count);
logger.LogWarning("Issue {JiraKey} has no assignee; defaulting to unassigned", key);
logger.LogError(ex, "Failed to create ADO work item for Jira issue {JiraKey}", key);

// Include: timestamp, correlation ID, user/system identity, action, result
```

### Azure Monitor + Application Insights

```
Application → SDK → Application Insights → Azure Monitor
                                                ↓
                              Dashboards, Alerts, Log queries (KQL)
```

```kql
// Kusto Query Language (KQL) — Azure's log query language
// Find all errors in the last hour
traces
| where timestamp > ago(1h)
| where severityLevel == 3   // Error
| project timestamp, message, customDimensions
| order by timestamp desc
```

---

## 13. Monitoring & Alerting

### Key Metrics to Monitor

| Metric | Description |
|---|---|
| **Availability** | Is the service responding? (Health endpoint `/health`) |
| **Latency** | How long do API calls take? (p50, p95, p99) |
| **Error Rate** | % of requests returning 5xx |
| **Throughput** | Requests per second |
| **Resource Usage** | CPU, memory, disk I/O |

### Alerting Strategy

```
Metric crosses threshold
        ↓
Alert fires in Azure Monitor / Application Insights
        ↓
Notification → Email / Teams / PagerDuty / Slack
        ↓
On-call engineer investigates (runbook guides resolution)
```

### Health Check Endpoint

```csharp
// ASP.NET Core — register health checks
builder.Services.AddHealthChecks()
    .AddSqlServer(connectionString)
    .AddUrlGroup(new Uri("https://jira.example.com/status"), "Jira");

app.MapHealthChecks("/health");
```

```json
// Response
{
  "status": "Healthy",
  "checks": [
    { "name": "sqlserver", "status": "Healthy", "duration": "00:00:00.012" },
    { "name": "Jira",      "status": "Healthy", "duration": "00:00:00.087" }
  ]
}
```

---

# PART 7 — AGILE METHODOLOGY

---

## 14. Agile & Scrum Fundamentals

### Agile Manifesto (4 Values)
1. **Individuals and interactions** over processes and tools
2. **Working software** over comprehensive documentation
3. **Customer collaboration** over contract negotiation
4. **Responding to change** over following a plan

### Scrum Framework

```
Product Backlog  →  Sprint Planning  →  Sprint (1–4 weeks)
                                              ↓
                                       Daily Standup
                                              ↓
                                       Sprint Review
                                              ↓
                                       Sprint Retrospective
                                              ↓
                                       Potentially Shippable Product
```

### Scrum Artifacts

| Artifact | Description |
|---|---|
| **Product Backlog** | Prioritized list of all features (Epics → Stories → Tasks) |
| **Sprint Backlog** | Items committed for the current sprint |
| **Increment** | Working software delivered at end of sprint |
| **Definition of Done (DoD)** | Criteria a story must meet to be "done" |

### Work Item Hierarchy in Azure Boards

```
Epic           → "Build Integration Platform"
  └── Feature  → "Jira ↔ ADO Sync"
        └── User Story → "As a PM, I want Jira tickets to appear in ADO"
              └── Task  → "Build Jira REST API client"
              └── Task  → "Write ADO work item creation logic"
              └── Bug   → "Sync fails for tickets with special characters"
```

### User Story Format

```
As a [persona]
I want to [action/goal]
So that [benefit/value]

Acceptance Criteria:
- [ ] Given X, when Y, then Z
- [ ] Given X, when Y, then Z

Story Points: 3
Priority: High
```

### Agile Ceremonies

| Ceremony | Frequency | Purpose | Duration |
|---|---|---|---|
| **Sprint Planning** | Start of sprint | Choose backlog items, define sprint goal | 2–4 hrs |
| **Daily Standup** | Every day | What did I do? What will I do? Any blockers? | 15 min |
| **Sprint Review** | End of sprint | Demo working software to stakeholders | 1–2 hrs |
| **Retrospective** | End of sprint | What went well? Improve? Experiments? | 1–2 hrs |
| **Backlog Refinement** | Mid-sprint | Estimate and clarify upcoming stories | 1–2 hrs |

---

# PART 8 — TOOLING

---

## 15. Postman & API Testing

### Core Postman Concepts

```
Collection  →  Folder  →  Request
                              ├── Method + URL
                              ├── Headers (Auth, Content-Type)
                              ├── Body (JSON / Form)
                              ├── Pre-request Script
                              └── Tests (assertions)
```

### Writing Postman Tests

```javascript
// Test: status code is 201
pm.test("Status code is 201 Created", function () {
    pm.response.to.have.status(201);
});

// Test: response contains id
pm.test("Response has work item id", function () {
    var json = pm.response.json();
    pm.expect(json).to.have.property("id");
    pm.expect(json.id).to.be.a("number");
});

// Save value to environment variable for next request
var token = pm.response.json().access_token;
pm.environment.set("ACCESS_TOKEN", token);
```

### Environments in Postman

```json
// Dev environment
{ "BASE_URL": "https://api-dev.csgi.com", "API_KEY": "dev-key-123" }

// Prod environment
{ "BASE_URL": "https://api.csgi.com", "API_KEY": "prod-key-xyz" }
```

Use `{{BASE_URL}}/api/v1/users` — Postman substitutes the variable based on active environment.

---

## 16. Jira vs Azure DevOps Boards

| Feature | Jira | Azure Boards |
|---|---|---|
| Work items | Issue (Story, Bug, Epic) | Work Item (Story, Bug, Epic, Task, Feature) |
| Board views | Scrum board, Kanban | Scrum board, Kanban, Backlogs |
| Query language | JQL (Jira Query Language) | WIQL (Work Item Query Language) |
| Integration | REST API at `/rest/api/3/` | REST API at `/_apis/wit/` |
| Sprints | Versions / Sprints | Iterations |

```
# JQL example — find high-priority open bugs assigned to me
project = MYPROJ AND issuetype = Bug AND status != Done 
AND priority = High AND assignee = currentUser()

# WIQL equivalent
SELECT [System.Id], [System.Title]
FROM   WorkItems
WHERE  [System.TeamProject] = 'MyProject'
  AND  [System.WorkItemType] = 'Bug'
  AND  [System.State] <> 'Done'
  AND  [Microsoft.VSTS.Common.Priority] = 1
  AND  [System.AssignedTo] = @Me
```

---

# PART 9 — INTERVIEW Q&A

---

## Common Interview Questions & Answers

**Q: What is the difference between CI and CD?**
CI (Continuous Integration) is the practice of automatically building and testing code on every push. CD can mean Continuous Delivery — automatically deploying to staging with a manual gate to production — or Continuous Deployment — fully automated all the way to production. CI catches integration bugs early; CD ensures software is always in a deployable state.

---

**Q: What is the difference between Blue-Green and Canary deployments?**
Blue-Green switches 100% of traffic from old to new at once, with an instant rollback by switching back. Canary gradually shifts a small percentage of traffic (e.g. 5%) to the new version, monitors for errors, then increases. Blue-Green has zero downtime and instant rollback but needs double infrastructure. Canary is lower risk but requires good monitoring.

---

**Q: What is OAuth 2.0 and how does it work?**
OAuth 2.0 is an authorization framework where a user grants an application limited access to their resources on another service without sharing their password. The app redirects the user to the authorization server, the user logs in and consents, the server returns a short-lived access token, and the app uses that token (as a Bearer token) to call APIs. The token expires, limiting the damage if it's stolen.

---

**Q: What is the difference between authentication and authorization?**
Authentication verifies *who you are* (login, identity). Authorization determines *what you're allowed to do* (permissions, roles). OAuth handles authorization; OpenID Connect (OIDC) adds authentication on top of OAuth.

---

**Q: Explain ACID properties.**
Atomicity ensures all operations in a transaction succeed or all fail. Consistency ensures the database moves from one valid state to another. Isolation ensures concurrent transactions don't interfere with each other. Durability ensures committed data survives failures. These properties are what make relational databases reliable for financial and critical data.

---

**Q: What's the difference between PUT and PATCH?**
`PUT` replaces the entire resource — you send the complete object. `PATCH` partially updates a resource — you send only the fields that changed. `PUT` is idempotent and replaces everything; `PATCH` is more efficient for partial updates.

---

**Q: What is a rolling deployment?**
Rolling deployment gradually replaces instances of the old version with the new version one-by-one or in batches. At any point during the rollout, both versions run simultaneously. It requires no extra infrastructure but rollback is slower since you need to re-deploy all instances.

---

**Q: What is structured logging and why does it matter?**
Structured logging records log data as key-value pairs (not plain strings), enabling filtering, querying, and alerting by specific fields. For example, logging `{OrderId: 42, UserId: 7, Duration: 120ms}` instead of `"Order 42 for user 7 took 120ms"` lets you query all slow orders across millions of log entries efficiently in tools like Application Insights or Splunk.

---

**Q: What is a service connection in Azure DevOps?**
A service connection stores credentials (service principal, API key, PAT) that allow Azure Pipelines to authenticate to external services — like deploying to an Azure subscription, pushing to a Docker registry, or calling a REST API — without embedding secrets in the pipeline YAML.

---

**Q: What is ETL? Can you describe each step?**
ETL stands for Extract, Transform, Load. Extract pulls data from source systems (via API, SQL, or file export). Transform cleans, maps, validates, and reshapes data to fit the target schema (field name mapping, type conversion, business rule validation). Load writes the transformed data into the target system. ETL is the backbone of data migration projects, like moving work items from Jira to Azure DevOps.

---

## Key Runbook Concepts (CSG-specific)

A **runbook** is a documented set of steps to handle a known operational scenario — like restarting a service, handling a failed sync, or rotating credentials.

```markdown
## Runbook: Jira → ADO Sync Failure

### Symptoms
- ADO work items not updated after Jira changes
- Alert: "Sync job failed" in Azure Monitor

### Steps
1. Check Azure Pipeline run logs for the sync job
2. Look for HTTP 429 (rate limit) or 401 (auth expired) errors
3. If 401: rotate Jira API key in Key Vault → re-run pipeline
4. If 429: check sync frequency config → reduce batch size
5. Validate record counts: source vs target
6. Re-run failed batch using the replay script

### Escalation
If unresolved in 30 min → escalate to @senior-engineer
```

---

## Topic Coverage Matrix — CSG SDE-1

| JD Requirement | Notes Section | Coverage |
|---|---|---|
| Azure DevOps / ADO | Parts 1–2 | ✅ Full |
| CI/CD Pipelines | Part 1 | ✅ Full |
| Environment separation | Section 4 | ✅ Full |
| Deployment types | Section 5 | ✅ Full |
| REST APIs | Section 7 | ✅ Full |
| JSON / XML | Section 8 | ✅ Full |
| OAuth / API Keys / Basic Auth | Section 9 | ✅ Full |
| SQL & relational databases | Section 10 | ✅ Full |
| ETL & data migration | Section 11 | ✅ Full |
| Logging / monitoring / alerting | Sections 12–13 | ✅ Full |
| Agile / Jira / Scrum | Section 14 | ✅ Full |
| Postman / API testing | Section 15 | ✅ Full |
| Runbooks / operational support | Section 16 + Q&A | ✅ Full |

---

*Study tip: For this role, interviewers will care most about **REST APIs + Azure DevOps + SQL + Auth concepts**. Be ready to walk through a CI/CD pipeline end-to-end, explain an OAuth flow, and write a basic SQL JOIN query.*
