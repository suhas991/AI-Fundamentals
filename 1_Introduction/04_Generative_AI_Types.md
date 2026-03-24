# Generative AI: Types and Categories

## What Is Generative AI?

Generative AI (GenAI) includes models that create new content such as text, images, audio, video, code, and 3D assets by learning underlying data distributions.

## Major GenAI Categories

### 1. Language Generation (LLMs)

- Produces and transforms text
- Typical model lifecycle:
  - Base pretraining
  - Instruction tuning
  - Preference alignment (for helpfulness/safety)
- Common use cases: Q&A, summarization, drafting, reasoning support, coding help

### 2. Image Generation

- Text-to-image and image editing tasks
- Dominant techniques: diffusion models, latent consistency variants, and hybrid pipelines
- Capabilities: generation, style transfer, inpainting, outpainting

### 3. Video Generation

- Text-to-video and image-to-video synthesis
- Use cases: storyboarding, short-form creative content, simulation previews

### 4. Audio and Music Generation

- Text-to-speech, voice cloning, music and sound effect generation
- Use cases: narration, dubbing, accessibility, creative production

### 5. Vision-Language Generation

- Joint understanding of text and images
- Tasks: captioning, visual Q&A, document extraction, multimodal assistants

### 6. Code Generation

- Generates, edits, explains, and tests code
- Tasks: autocomplete, refactoring, bug fixing, documentation, test scaffolding

### 7. 3D and Spatial Generation

- Generates 3D objects or scenes from text/images
- Use cases: gaming assets, prototyping, digital twins, product design

### 8. Multimodal Foundation Models

- Accept and produce multiple modalities (text, image, audio, video)
- Enable cross-modal workflows such as image-grounded conversation or video reasoning

## GenAI by Output Type

| Output Type | Typical Tasks | Example Use Cases |
|-------------|---------------|-------------------|
| Text | Generation, rewrite, summarize | Knowledge assistants, support bots |
| Image | Create/edit visuals | Marketing, concept art, design ideation |
| Video | Generate clips/scenes | Storyboarding, ad prototypes |
| Audio | Speech/music synthesis | Voiceovers, accessibility tools |
| Code | Generate and refactor code | Developer productivity |
| 3D | Generate meshes/scenes | Games, AR/VR, product mockups |
| Synthetic Data | Data augmentation | Training and simulation |

## Common Training and Adaptation Approaches

### Foundation Pretraining

- Self-supervised learning on large-scale data
- Objectives include next-token prediction, masked modeling, and denoising

### Post-Training Alignment

- Supervised fine-tuning (SFT)
- Preference optimization/alignment methods

### Retrieval-Augmented Generation (RAG)

- Combines generation with external knowledge retrieval
- Improves factual grounding and freshness of responses

### Tool Use and Agents

- Models call external tools (search, code execution, APIs)
- Extends reliability and task completion for complex workflows

## Practical Concepts to Know

- **Prompt design**: Structuring instructions and context effectively
- **Token budget/context window**: Input and output limits
- **Sampling controls**: Temperature, top-p, and decoding strategy
- **Hallucination risk**: Plausible but incorrect outputs
- **Evaluation**: Task-specific quality, safety, and latency metrics

## Key Takeaways

1. GenAI is not only text; it spans many modalities.
2. Real-world systems combine pretrained models with alignment, retrieval, and tools.
3. Product quality depends on both model choice and system design.