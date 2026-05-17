# 🗂️ System Prompts vs User Prompts

---

## Overview

When interacting with an LLM via an API or chat interface, there are distinct **message roles** that serve different purposes. Understanding these roles is fundamental to building reliable AI applications.

---

## The Message Role System

Modern LLMs (especially those based on the **chat/instruct** format) accept structured conversations with distinct roles:

```json
[
  { "role": "system",    "content": "You are a helpful assistant..." },
  { "role": "user",      "content": "What is machine learning?" },
  { "role": "assistant", "content": "Machine learning is..." },
  { "role": "user",      "content": "Can you give an example?" }
]
```

The three main roles are:
1. **`system`** — Sets the stage (developer-controlled)
2. **`user`** — The human's messages
3. **`assistant`** — The model's responses

---

## System Prompts

### What is a System Prompt?

A **system prompt** is a special instruction block that appears **before the conversation begins**. It is written by the **developer or product team**, not the end user.

It defines:
- The model's **persona** and **identity**
- **Behavioral rules** and constraints
- **Context** about the application
- **Output format** requirements
- **Scope limitations** (what it should/shouldn't do)

### Characteristics
| Property | Description |
|---|---|
| Written by | Developer / operator |
| Visibility to user | Usually hidden |
| Persistence | Stays constant across the conversation |
| Priority | Generally higher trust than user messages |
| Position in context | Beginning of the conversation |

### Example System Prompts

#### Customer Support Bot
```
You are Aria, a friendly customer support assistant for ShopEasy, an e-commerce platform.

Your responsibilities:
- Help customers track orders, process returns, and resolve billing issues
- Always greet users warmly and use their first name if provided
- Escalate to a human agent if the issue cannot be resolved in 3 turns

Rules:
- Never discuss competitor products
- Do not make promises about refunds without checking policy
- Keep responses under 150 words
- Always end with "Is there anything else I can help you with?"

Tone: Professional, warm, patient
```

#### Coding Assistant
```
You are an expert software engineer specializing in Python and cloud infrastructure.

When answering:
1. Always provide working, tested code examples
2. Include brief inline comments for non-obvious logic
3. Mention time and space complexity for algorithms
4. Suggest best practices but don't over-engineer simple solutions
5. If a question is unclear, ask for clarification before answering

Format all code in markdown code blocks with the language specified.
```

#### Domain-Restricted Assistant
```
You are a legal research assistant for Johnson & Partners Law Firm.

You may ONLY discuss:
- General legal concepts and terminology
- Research techniques and case law analysis
- Document drafting assistance

You must NEVER:
- Provide specific legal advice to clients
- Speculate about case outcomes
- Discuss matters outside of legal topics

Always include: "This is for research purposes only and does not constitute legal advice."
```

---

## User Prompts

### What is a User Prompt?

A **user prompt** is the message sent by the **end user** in real time. It contains the actual question, task, or input the user wants the model to respond to.

### Characteristics
| Property | Description |
|---|---|
| Written by | End user |
| Visibility | Fully visible in the conversation |
| Persistence | Part of ongoing conversation history |
| Priority | Lower trust than system prompt (in most systems) |
| Position in context | After the system prompt |

### Types of User Messages
```
Simple question:     "What is photosynthesis?"
Task request:        "Write a cover letter for a data analyst role"
Clarification:       "Can you make that more concise?"
Multi-part:          "Summarize this article AND list the key claims"
Data input:          "Analyze this CSV: [data...]"
Follow-up:           "What about in the context of climate change?"
```

---

## System vs User: Side-by-Side Comparison

| Dimension | System Prompt | User Prompt |
|---|---|---|
| **Author** | Developer/operator | End user |
| **Purpose** | Configure model behavior | Request a specific task |
| **Timing** | Set once before conversation | Each conversation turn |
| **User can see?** | Usually no | Yes |
| **User can modify?** | No | Yes (it's their message) |
| **Trust level** | High | Lower |
| **Scope** | Global (applies to all turns) | Local (applies to that turn) |
| **Example content** | Persona, rules, format | Questions, tasks, data |

---

## The Assistant Role

The **`assistant`** role contains the model's previous responses. You can use this to:
1. Continue multi-turn conversations
2. **Pre-fill** the model's response (steer generation)

### Multi-turn Conversation
```json
[
  { "role": "system", "content": "You are a helpful tutor." },
  { "role": "user", "content": "What is the Pythagorean theorem?" },
  { "role": "assistant", "content": "The Pythagorean theorem states that..." },
  { "role": "user", "content": "Can you give me a practice problem?" }
]
```

### Pre-filling the Assistant Response
```json
[
  { "role": "user", "content": "List the planets in order from the sun." },
  { "role": "assistant", "content": "Here are the planets:\n1." }
]
```
The model will continue from where you left off.

---

## Conversation Architecture in Applications

### Standard Chat App Architecture
```
┌──────────────────────────────────────────────┐
│              APPLICATION                      │
│                                              │
│  System Prompt (hidden from user)            │
│  ┌────────────────────────────────────────┐  │
│  │ "You are HelperBot. Only discuss..."   │  │
│  └────────────────────────────────────────┘  │
│                  +                           │
│  Conversation History (grows each turn)      │
│  ┌────────────────────────────────────────┐  │
│  │ User: "Hello!"                         │  │
│  │ Assistant: "Hi there! How can I..."    │  │
│  │ User: "What's 2+2?"                    │  │
│  └────────────────────────────────────────┘  │
│                  +                           │
│  New User Message                            │
│  ┌────────────────────────────────────────┐  │
│  │ User: "Explain quantum computing"      │  │
│  └────────────────────────────────────────┘  │
│                  ↓                           │
│           → LLM API Call                     │
└──────────────────────────────────────────────┘
```

---

## Prompt Hierarchy and Trust Levels

Not all prompt content is treated equally. Modern aligned models follow a **trust hierarchy**:

```
1. HIGHEST TRUST  → Developer System Prompt
                    (Configures base behavior, operator rules)

2. MEDIUM TRUST   → User Messages
                    (User can adjust within operator-permitted limits)

3. LOWEST TRUST   → External Content
                    (Web pages, documents, retrieved data — potential injection risk)
```

### Practical Implications
- System prompts can **restrict** what users can do
- Users generally **cannot override** system prompt rules
- Developers can **grant additional permissions** to users via system prompt
  ```
  "The user has been verified as an adult. Adult content is permitted in this context."
  ```

---

## Best Practices for System Prompts

### Be Explicit and Specific
```
❌ Vague:   "Be helpful and professional"
✅ Specific: "Use formal language. Avoid contractions. Always cite sources.
              Responses should be 2-4 paragraphs unless asked for more."
```

### Define the Persona Clearly
```
"You are Max, a fitness coach assistant for FitLife app.
You specialize in strength training and nutrition for beginners.
Your tone is encouraging, motivational, and non-judgmental."
```

### Handle Edge Cases Proactively
```
"If the user asks about topics unrelated to fitness:
- Politely redirect: 'I'm focused on fitness and nutrition — let me help with that!'
- Do not engage with off-topic debates or requests"
```

### Specify Output Format Consistently
```
"Always structure responses as:
1. Direct answer (1-2 sentences)
2. Explanation (2-3 sentences)
3. Example or next step (1 sentence)"
```

---

## Common Pitfalls

| Pitfall | Problem | Solution |
|---|---|---|
| No system prompt | Model has no context or constraints | Always define role and scope |
| Contradictory instructions | Model gets confused | Be consistent, test edge cases |
| Overly long system prompt | Wastes context window, hard to maintain | Keep it focused and structured |
| Too restrictive | Model refuses valid requests | Test with real user scenarios |
| Assuming user sees system prompt | User is confused | Add user-facing instructions in first assistant message if needed |

---

## Quick Reference

```
SYSTEM PROMPT
├── Written by: Developer
├── Purpose: Configure persona, rules, format, scope
├── Visibility: Hidden from user
├── Trust: Highest
└── Tip: Be specific; test against edge cases

USER PROMPT
├── Written by: End user
├── Purpose: Ask questions, give tasks
├── Visibility: Full
├── Trust: Medium
└── Tip: Use few-shot examples for complex tasks

ASSISTANT ROLE
├── Contains: Model's previous responses
├── Used for: Conversation continuity, pre-filling
└── Tip: Include full history for context-aware responses
```

---

*Previous: [04 — Prompt Engineering](./04_Prompt_Engineering.md)*
