# ☁️ Cloud Fundamentals & Azure — Interview Notes

---

## 1. Cloud Computing

### What is Cloud Computing?
Cloud computing is the **on-demand delivery of IT resources** (compute, storage, databases, networking, software) over the internet with **pay-as-you-go pricing**, eliminating the need to own/manage physical data centers.

### Key Characteristics (NIST Model)
| Characteristic | Description |
|---|---|
| On-demand self-service | Provision resources without human interaction |
| Broad network access | Accessible over the network via standard devices |
| Resource pooling | Multi-tenant model, resources shared across customers |
| Rapid elasticity | Scale up/down quickly based on demand |
| Measured service | Usage is monitored and billed accordingly |

### Benefits
- **Cost savings** — No CapEx; pay only for what you use
- **Scalability** — Scale globally in minutes
- **Reliability** — Built-in redundancy and disaster recovery
- **Security** — Enterprise-grade security managed by provider
- **Speed** — Provision resources in seconds

---

## 2. Types of Cloud

| Type | Description | Example |
|---|---|---|
| **Public Cloud** | Resources owned and operated by a third-party provider, shared across customers | Azure, AWS, GCP |
| **Private Cloud** | Cloud infrastructure dedicated to a single organization, hosted on-prem or by a provider | Azure Stack, VMware |
| **Hybrid Cloud** | Combination of public and private clouds, connected via secure links | Azure Arc, Azure VPN |
| **Multi-Cloud** | Using two or more public cloud providers simultaneously | Azure + AWS |

### Interview Tip
> "Hybrid cloud is preferred in regulated industries (finance, healthcare) where sensitive data must stay on-premises while leveraging public cloud for burst workloads."

---

## 3. Service Models — IaaS, PaaS, SaaS

```
User Manages ◄────────────────────────────► Provider Manages

On-Premises   IaaS          PaaS           SaaS
─────────     ─────────     ─────────      ─────────
App           App           App ✓          App ✓
Data          Data          Data ✓         Data ✓
Runtime       Runtime       Runtime ✓      Runtime ✓
OS            OS            OS ✓           OS ✓
VM            VM ✓          VM ✓           VM ✓
Storage       Storage ✓     Storage ✓      Storage ✓
Network       Network ✓     Network ✓      Network ✓
```

### IaaS — Infrastructure as a Service
- Provider manages: Networking, Storage, Servers, Virtualization
- You manage: OS, Runtime, App, Data
- **Azure Examples:** Azure Virtual Machines, Azure Blob Storage, Azure VNet
- **Use case:** Lift-and-shift migrations, custom OS environments, dev/test labs

### PaaS — Platform as a Service
- Provider manages everything up to the runtime
- You manage: App and Data only
- **Azure Examples:** Azure App Service, Azure SQL Database, Azure Functions
- **Use case:** Web app development, APIs, microservices — focus on code, not infra

### SaaS — Software as a Service
- Provider manages everything; you just use the software
- **Azure Examples:** Microsoft 365, Dynamics 365, Azure DevOps
- **Use case:** Email, CRM, collaboration tools

### Quick Comparison
| | IaaS | PaaS | SaaS |
|---|---|---|---|
| Control | High | Medium | Low |
| Flexibility | High | Medium | Low |
| Management | High | Low | Minimal |
| Example | Azure VM | Azure App Service | Office 365 |

---

## 4. Azure App Service

### What is it?
A fully managed **PaaS** platform for building, deploying, and scaling web apps, REST APIs, and mobile backends.

### Key Features
- Supports **multiple languages**: .NET, Java, Node.js, Python, PHP, Ruby
- Built-in **CI/CD** integration (GitHub, Azure DevOps, Bitbucket)
- **Auto-scaling** — scale out based on HTTP queue, CPU, or schedule
- **Custom domains** and **SSL certificates**
- **Deployment slots** — staging/production swap with zero downtime
- **App Service Plan** defines compute resources (CPU, RAM)

### App Service Plans (Tiers)
| Tier | Use Case |
|---|---|
| Free/Shared | Dev/test only, shared infrastructure |
| Basic | Dev/test with manual scale |
| Standard | Production workloads, auto-scale, custom domains |
| Premium | High performance, VNet integration |
| Isolated | Fully isolated, dedicated environment (App Service Environment) |

### Interview Questions
**Q: What is a deployment slot?**
> A deployment slot is a live app with its own hostname. You can deploy to a staging slot, warm it up, then **swap** it with production — enabling zero-downtime deployments and easy rollback.

**Q: Difference between App Service and Azure Functions?**
> App Service runs **long-running web apps/APIs continuously**. Azure Functions is **event-driven and serverless** — ideal for short-duration tasks triggered by events.

---

## 5. Azure Virtual Machines (VMs)

### What is it?
Azure VMs are **IaaS** offerings — virtualized computing resources in Azure's data centers. You control the OS, runtime, and installed software.

### Key Concepts
- **VM Size** — defines CPU, RAM, disk (e.g., Standard_D2s_v3 = 2 vCPUs, 8GB RAM)
- **VM Image** — OS image (Windows Server, Ubuntu, RHEL, etc.)
- **Availability Set** — protects against rack-level failures (Fault Domains + Update Domains)
- **Availability Zone** — protects against data center-level failures (separate physical buildings)
- **VM Scale Set (VMSS)** — group of identical VMs for auto-scaling
- **Managed Disks** — Azure-managed storage for VM OS and data disks

### VM Disk Types
| Disk | Use Case |
|---|---|
| OS Disk | Operating system |
| Data Disk | Application data, databases |
| Temp Disk | Ephemeral, not persistent (lost on reboot) |

### SLA
- Single VM with Premium SSD: **99.9%**
- VMs in Availability Set: **99.95%**
- VMs in Availability Zones: **99.99%**

### Interview Tip
> "Use **VMSS** for stateless workloads that need to scale. Use **Availability Zones** for high-availability production workloads. Use **Availability Sets** when zones aren't available."

---

## 6. Azure Blob Storage

### What is it?
Azure Blob Storage is Microsoft Azure's **object storage** solution for unstructured data — images, videos, logs, backups, static website content.

### Storage Account Structure
```
Storage Account
└── Container (like a folder/bucket)
    └── Blob (the actual file/object)
```

### Blob Types
| Type | Description | Use Case |
|---|---|---|
| **Block Blob** | Most common, stores text/binary data | Files, images, videos, backups |
| **Append Blob** | Optimized for append operations | Log files |
| **Page Blob** | Random read/write, 512-byte pages | Azure VM OS/Data disks (VHDs) |

### Access Tiers
| Tier | Cost | Access Frequency |
|---|---|---|
| **Hot** | High storage cost, low access cost | Frequently accessed data |
| **Cool** | Lower storage cost, higher access cost | Infrequently accessed (≥30 days) |
| **Cold** | Even lower storage cost | Rarely accessed (≥90 days) |
| **Archive** | Lowest storage cost, highest retrieval cost | Long-term backup (≥180 days) |

### Key Features
- **Lifecycle Management** — automatically move blobs between tiers
- **Soft Delete** — recover deleted blobs within retention period
- **Versioning** — keep previous versions of blobs
- **Immutability** — WORM (Write Once, Read Many) policies for compliance
- **Shared Access Signature (SAS)** — time-limited, permission-scoped URLs

### Redundancy Options
| Option | Description |
|---|---|
| LRS | Locally Redundant Storage — 3 copies in 1 datacenter |
| ZRS | Zone Redundant Storage — 3 copies across 3 zones |
| GRS | Geo Redundant Storage — LRS + async replication to secondary region |
| GZRS | Geo + Zone Redundant Storage |

---

## 7. Azure SQL & Cosmos DB

### Azure SQL Database
A fully managed **relational database** (PaaS) based on SQL Server engine.

**Key Features:**
- Automatic patching, backups, and high availability
- **Elastic pools** — share resources across multiple databases (cost-efficient)
- **DTU model** vs **vCore model** — pricing/scaling options
- Built-in **intelligence** for performance tuning
- Supports **geo-replication** for global read replicas
- **Active Geo-Replication** — up to 4 readable secondary replicas

**Service Tiers:**
| Tier | Use Case |
|---|---|
| Basic/Standard/Premium | DTU-based, predictable workloads |
| General Purpose | vCore, balanced compute/storage |
| Business Critical | vCore, high IOPS, in-memory OLTP |
| Hyperscale | vCore, massive scale (up to 100TB) |

---

### Azure Cosmos DB
A globally distributed, **multi-model NoSQL** database designed for low latency and high availability at any scale.

**Key Features:**
- **Global distribution** — replicate data to any Azure region with one click
- **Multi-master writes** — write to any region simultaneously
- **Guaranteed SLAs** — 99.999% availability, <10ms reads, <15ms writes
- **5 consistency levels** (from strong to eventual)
- **Multiple APIs** — Core (SQL), MongoDB, Cassandra, Gremlin (Graph), Table

**Consistency Levels (Strongest → Weakest):**
1. **Strong** — Always reads the latest committed write
2. **Bounded Staleness** — Reads lag behind by defined operations/time
3. **Session** — Consistent within a session (most popular default)
4. **Consistent Prefix** — Reads never see out-of-order writes
5. **Eventual** — Highest availability, lowest consistency

**Partition Key** — critical design decision; determines data distribution and scalability. Choose a key with high cardinality and uniform distribution.

### SQL vs Cosmos DB
| | Azure SQL | Cosmos DB |
|---|---|---|
| Data model | Relational (tables) | NoSQL (JSON, Graph, etc.) |
| Schema | Fixed | Flexible/Schema-less |
| Scaling | Vertical (mostly) | Horizontal (native) |
| Latency | Milliseconds | Single-digit milliseconds |
| Use case | Transactional apps, reporting | Global apps, IoT, real-time |

---

## 8. Scaling — Vertical vs Horizontal

### Vertical Scaling (Scale Up / Scale Down)
Increasing or decreasing the **size** of an existing resource.

- Add more CPU, RAM, or storage to the same VM/service
- **Azure Example:** Upgrading a VM from Standard_D2s_v3 → Standard_D4s_v3
- **Pros:** Simple, no code changes needed
- **Cons:** Has a limit (max VM size), causes downtime during resize, single point of failure

### Horizontal Scaling (Scale Out / Scale In)
Adding or removing **instances** of a resource.

- **Azure Example:** VM Scale Sets, App Service auto-scale, AKS node pools
- **Pros:** Near-unlimited scale, high availability, no downtime
- **Cons:** App must be stateless (or use external session/state management)

### Auto-Scaling in Azure
- **Metric-based:** Scale when CPU > 70% for 5 minutes
- **Schedule-based:** Scale out at 9 AM, scale in at 6 PM
- **Cooldown period:** Wait time between scale events (prevents thrashing)

```
Scale Out  ──►  More instances added (demand increases)
Scale In   ──►  Instances removed (demand decreases)
Scale Up   ──►  Bigger instance (more CPU/RAM)
Scale Down ──►  Smaller instance
```

---

## 9. Azure Monitor & Alerts

### Azure Monitor
Azure's **centralized monitoring platform** for collecting, analyzing, and acting on telemetry from cloud and on-premises environments.

### Data Types Collected
| Type | Description | Example |
|---|---|---|
| **Metrics** | Numerical time-series data | CPU %, memory, request count |
| **Logs** | Event and diagnostic data | App logs, activity logs |
| **Traces** | Distributed tracing | Application Insights |

### Key Components
- **Log Analytics Workspace** — centralized store for log data; queried with **KQL (Kusto Query Language)**
- **Application Insights** — APM tool for web apps; tracks requests, exceptions, dependencies, user flows
- **Metrics Explorer** — visualize and analyze metric data
- **Diagnostic Settings** — route resource logs to Log Analytics, Storage, or Event Hub
- **Azure Activity Log** — records subscription-level events (who did what, when)

### Alerts
Alerts notify you or trigger automated actions when conditions are met.

**Alert Components:**
1. **Scope** — which resource(s) to monitor
2. **Condition** — metric threshold or log query (e.g., CPU > 80%)
3. **Action Group** — what to do (email, SMS, webhook, Azure Function, Logic App, runbook)
4. **Alert Rule** — combines scope + condition + action group

**Alert Types:**
| Type | Based On |
|---|---|
| Metric Alert | Numeric thresholds (CPU, memory, etc.) |
| Log Alert | KQL query results from Log Analytics |
| Activity Log Alert | Azure resource operations |
| Smart Detection | AI-based anomaly detection (App Insights) |

**Severity Levels:** Sev 0 (Critical) → Sev 4 (Verbose)

### Sample KQL Query
```kql
// Find all errors in the last 1 hour
AppExceptions
| where TimeGenerated > ago(1h)
| summarize count() by type
| order by count_ desc
```

---

## 10. Terraform

### What is Terraform?
An open-source **Infrastructure as Code (IaC)** tool by HashiCorp that lets you define and provision infrastructure using a declarative configuration language (**HCL — HashiCorp Configuration Language**).

### Core Workflow Commands

#### `terraform init`
- Initializes the working directory
- Downloads required **provider plugins** (e.g., `hashicorp/azurerm`)
- Sets up the **backend** (where state is stored)
- Must be run first before any other command
```bash
terraform init
```

#### `terraform validate`
- Checks configuration files for **syntax errors and internal consistency**
- Does NOT check against the actual cloud provider (no API calls)
- Use before plan to catch typos/mistakes early
```bash
terraform validate
# Output: Success! The configuration is valid.
```

#### `terraform plan`
- Creates an **execution plan** — shows what actions Terraform will take
- Compares current state vs desired configuration
- **No changes are made** — read-only preview
- Outputs: `+` (create), `-` (destroy), `~` (update in-place), `-/+` (replace)
```bash
terraform plan
terraform plan -out=tfplan  # save plan to file
```

#### `terraform apply`
- **Executes the plan** — creates, updates, or destroys infrastructure
- Prompts for confirmation (`yes`) unless `-auto-approve` is used
- Updates the **state file** after execution
```bash
terraform apply
terraform apply -auto-approve       # skip confirmation
terraform apply tfplan              # apply saved plan
```

#### `terraform destroy`
- **Destroys all resources** managed by Terraform in the current state
- Prompts for confirmation
- Useful for tearing down dev/test environments
```bash
terraform destroy
terraform destroy -target=azurerm_virtual_machine.myvm  # destroy specific resource
```

#### `terraform state`
- Manages the **Terraform state file** (`terraform.tfstate`)
- State tracks the mapping between config and real-world resources

```bash
terraform state list                         # list all resources in state
terraform state show azurerm_resource_group.rg  # show details of a resource
terraform state rm azurerm_resource_group.rg    # remove resource from state (doesn't destroy)
terraform state mv source destination           # rename/move resource in state
```

#### `terraform output`
- Displays values of **output variables** defined in config
- Useful for extracting resource attributes (IPs, URLs, IDs) after apply

```hcl
# Define in config
output "vm_public_ip" {
  value = azurerm_public_ip.myip.ip_address
}
```
```bash
terraform output              # show all outputs
terraform output vm_public_ip # show specific output
terraform output -json        # output in JSON format
```

### Terraform State
- State file (`terraform.tfstate`) is Terraform's source of truth
- Maps configuration to real-world resources
- **Remote state** (recommended for teams): store in Azure Blob Storage, AWS S3, Terraform Cloud
- **State locking** — prevents concurrent modifications (Azure uses blob leases)

```hcl
# Backend config for Azure remote state
terraform {
  backend "azurerm" {
    resource_group_name  = "tfstate-rg"
    storage_account_name = "tfstatestorage"
    container_name       = "tfstate"
    key                  = "prod.terraform.tfstate"
  }
}
```

### Terraform Key Concepts
| Concept | Description |
|---|---|
| **Provider** | Plugin to interact with cloud APIs (azurerm, aws, google) |
| **Resource** | Infrastructure object to manage (VM, RG, Storage) |
| **Variable** | Input values to make configs reusable |
| **Output** | Values exposed after apply |
| **Module** | Reusable, encapsulated configuration |
| **Data Source** | Read existing resources not managed by Terraform |
| **Workspace** | Separate state environments (dev, staging, prod) |

### Basic Azure Example
```hcl
provider "azurerm" {
  features {}
}

resource "azurerm_resource_group" "rg" {
  name     = "my-rg"
  location = "East US"
}

resource "azurerm_storage_account" "sa" {
  name                     = "mystorageaccount"
  resource_group_name      = azurerm_resource_group.rg.name
  location                 = azurerm_resource_group.rg.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
}
```

### Terraform Lifecycle
```
Write HCL → init → validate → plan → apply → (manage) → destroy
```

---

## 11. Azure Functions

### What is it?
Azure Functions is a **serverless compute service** that lets you run event-driven code without managing servers. You only pay for **execution time** (consumption plan).

### Key Characteristics
- **Event-driven** — triggered by HTTP, timer, queues, blobs, databases, etc.
- **Serverless** — no infrastructure management
- **Stateless by default** — each invocation is independent
- **Short-lived** — max timeout of 230 seconds on Consumption plan (configurable on Premium/Dedicated)

### Triggers (Common)
| Trigger | Description |
|---|---|
| **HTTP Trigger** | Triggered by HTTP request (REST API) |
| **Timer Trigger** | Triggered on a cron schedule |
| **Blob Trigger** | Triggered when a blob is created/modified |
| **Queue Trigger** | Triggered when a message arrives in Azure Storage Queue |
| **Service Bus Trigger** | Triggered by Service Bus messages |
| **Event Grid Trigger** | Triggered by Event Grid events |
| **Cosmos DB Trigger** | Triggered when documents change in Cosmos DB |

### Bindings
Bindings declaratively connect functions to other services — no boilerplate connection code needed.
- **Input binding** — read data from a source
- **Output binding** — write data to a destination

```json
// function.json example
{
  "bindings": [
    { "type": "httpTrigger", "direction": "in", "name": "req" },
    { "type": "blob", "direction": "out", "name": "outputBlob", "path": "output/{rand-guid}" }
  ]
}
```

### Hosting Plans
| Plan | Description | Cold Start |
|---|---|---|
| **Consumption** | Fully serverless, auto-scale, pay-per-execution | Yes |
| **Premium** | Pre-warmed instances, VNet integration, no cold start | No |
| **Dedicated (App Service)** | Runs on existing App Service Plan | No |

### Durable Functions
Extension for **stateful workflows** in a serverless environment.

- **Orchestrator function** — coordinates the workflow
- **Activity function** — the individual steps/tasks
- **Patterns:** Function chaining, fan-out/fan-in, async HTTP APIs, monitoring, human interaction

### Azure Functions vs App Service
| | Azure Functions | App Service |
|---|---|---|
| Execution model | Event-driven, short-lived | Always-on, long-running |
| Scaling | Automatic, per-execution | Manual or auto-scale |
| Billing | Per execution | Per hour |
| State | Stateless (unless Durable) | Stateful |
| Use case | Event processing, automation | Web apps, APIs |

---

## Quick Reference — Azure Services Cheatsheet

| Service | Category | Key Use Case |
|---|---|---|
| Azure VM | IaaS | Full control OS/runtime, lift-and-shift |
| Azure App Service | PaaS | Web apps, APIs, mobile backends |
| Azure Functions | Serverless | Event-driven, short tasks |
| Azure Blob Storage | Storage | Unstructured data, files, backups |
| Azure SQL Database | PaaS DB | Relational, transactional workloads |
| Azure Cosmos DB | PaaS DB | Global NoSQL, low-latency |
| Azure Monitor | Observability | Metrics, logs, alerts |
| Application Insights | APM | Web app performance monitoring |
| Terraform | IaC | Provision & manage Azure infra as code |
| VM Scale Sets | Compute | Horizontal auto-scaling of VMs |

---

## Common Interview Questions & Answers

**Q: What is the difference between IaaS, PaaS, and SaaS?**
> IaaS gives you raw infrastructure (VMs, storage, networking) — you manage OS and above. PaaS abstracts the infrastructure and runtime — you manage only the app and data. SaaS is fully managed software delivered over the internet.

**Q: When would you choose Cosmos DB over Azure SQL?**
> Choose Cosmos DB for globally distributed apps requiring low latency, flexible schema, and horizontal scale. Choose Azure SQL for structured relational data with complex joins, transactions, and reporting needs.

**Q: What happens if you run `terraform apply` twice with no changes?**
> Terraform compares the current state with the desired config. If nothing changed, it reports "No changes. Infrastructure is up-to-date." and makes no API calls.

**Q: What is a cold start in Azure Functions?**
> A cold start occurs when a function hasn't been invoked recently and Azure needs to spin up a new instance — adding latency to the first request. Use the **Premium plan** or **Durable Functions** with pre-warmed instances to avoid this.

**Q: Vertical vs Horizontal scaling — which is better?**
> Horizontal scaling is generally preferred for cloud-native apps because it has no upper limit, enables high availability, and supports zero-downtime deployments. Vertical scaling is simpler but has hardware limits and may require downtime.

**Q: What is the Terraform state file and why is it important?**
> The state file (`terraform.tfstate`) maps your HCL configuration to real-world resources. Without it, Terraform cannot know what already exists and would try to recreate everything. For teams, it must be stored remotely (e.g., Azure Blob Storage) with state locking enabled.
