## Artificial Intelligence - Fundamentals

What is Artificial Intelligence ? </br>
The capability of a machine to imitate intelligent human behaviour. </br>
Example : Chatbots & Virual Asssitants (chatGpt , Genmini)

Real World Examples of Artificial Intelligence 
- Healthcare Diagnostics
- Fraud detection in Finance
- E-Commerce Recomendations
- Sentiment Analysis for Market Research

What is Generative AI ? </br>
It is a category of Artificial Intelligence that can generate new content,text,images and sounds by learninf from a large dataset of existinf contents. </br>


### Prompt Engineering

The aim of prompt engineering is to strike a balance between providing sufficient guidance to the model while avoiding overly constraining its creativity and language generation capabilities.

Through careful manipulation of prompts, engineers can shape the behavior of the language model to produce more accurate, coherent, and contextually appropriate responses.

Traits for Prompt Engineers include:
- LLM Architecture Knowledge
- Making Ambiguous Problems Clear
- ID Core Principles That Translate Across Scenarios

Explore free models like 
Stable Diffusion - By hugging Face , ChatGpt , Gemini etc..

What can we do with Generative AI?

- Image Synthesis & Manipulation - Powerful tools for artists, designers, storytellers 
- Text Generation & Translation - Seamless communication and text services 
- Musical Composition - Avenues and assistance for creative expression

#### Image Synthesis 

Generative Adversarial Networks (GANs) is a learining model that can generate realistic data that resembles a specific trainig dataset,made of two main componets generator(generates new data) and discriminator(acts as a judge)

#### Text Generation and Translation

Text generation and translation rely on computational models, which are simplified simulations of real-world systems that can be studied using computers. These models allow researchers to analyze and predict system behavior without needing to interact with the actual system, saving time and cost. Neural Machine Translation (NMT), a key application of these models, uses neural networks trained on large datasets of translated text to recognize patterns, understand context, and produce more accurate and fluent translations compared to traditional methods.

#### Music Composition

Artificial intelligence, particularly through neural networks, enables music composition by training models on datasets of musical pieces to learn patterns, styles, and structures. These models generate original compositions by combining the learned elements, offering musicians new creative possibilities. Another approach, Variational Autoencoders (VAEs), uses probabilistic models to capture the underlying structure and variability in music, allowing for diverse compositions by manipulating latent variables. Additionally, Interactive Music Generation applications allow users to input preferences, combining AI with human creativity, fostering collaboration in music creation and exploration of new sounds and genres.

### Understanding the Fundamentals of NLP

Natural Language Processing (NLP) is a branch of artificial intelligence that focuses on enabling computers to understand and interact with human language. It involves techniques and algorithms to analyze, interpret, and generate human-like text, enabling applications such as chatbots, language translation, sentiment analysis, and information extraction.

In layman’s terms, this means that NLP  is a field of computer science that focuses on making computers understand and work with human language, like the way we speak or write. 

NLP Involves Teaching Computers To...

- Recognize Patterns in Language
- Understand Grammar & Syntax
- Detect Emotions or Sentiment Behind the Words
- Deals w/ Challenges Like Understanding Slang, Idioms, or Language Variations

**Tokenization** is the process of breaking text into smaller linguistic units called tokens (can be words, characters, subwords). This is a crucial step in NLP as it forms the basis for further analysis and processing of text data. 

We can break a piece of text down into smaller components, allowing us to pass individual words or groupings of words into multiple sequential functions that will analyze the meaning of the word given the context of information. 
 
Tokenization is the first step for more complex NLP tasks including:

- Sentiment Analysis
- Language Translation
- Text Summarization 

The quality and method of tokenization can significantly affect the performance of machine learning models. The way text is tokenized determines how the model will interpret and process the language data.

Example:
Imagine an NLP system designed to analyze customer reviews for sentiment analysis. The system uses a basic tokenization method that naively splits text based on spaces, without considering punctuation or other linguistic nuances.

A naive tokenization approach might tokenize this sentence as:

["The", "product's", "quality", "isn't", "bad,", "but", "the", "delivery", "was", "slow."]

Some issues that may arise are:

Misinterpretation of Negations: The token "isn't" could be incorrectly interpreted. Some simpler models might focus on "isn't" as a negative sentiment word, overlooking the negation's role in converting "bad" into a positive sentiment (i.e., not bad).

Incorrect Word Boundaries: "Product's" is tokenized as a single word, which might cause the model to miss the possessive nature of the word "product" and its relationship to "quality".

Tokenization is not one-size-fits-all and must be adapted to different languages and linguistic rules
![image](https://github.com/user-attachments/assets/fd66aa30-1eb2-487e-abe3-6c61b1cb8ea6)

**Word embeddings** are dense vector representations of words in a high-dimensional space. They capture semantic and syntactic relationships between words.

In layman’s terms, word embeddings are a way to represent words as points in a vast space, where each point's position is determined by the word's meaning and use. Imagine a map where similar words, like "happy" and "joyful," are close together, while different words, like "happy" and "sad," are farther apart. This method helps computers understand and process language by seeing how words relate to each other, much like we understand words by their associations and contexts.

Word embeddings are often pre-trained on large corpora (collection of text/audio organized into datasets). They are used to enhance the performance of NLP models by providing meaningful numerical representations of words. 

For example, during the training process, the algorithms look at the context in which words appear in sentences. They learn to predict the likelihood of a word occurring given its neighboring words or vice versa. As a result, words with similar meanings or usage tend to have similar vector representations, and their embeddings are closer together in the high-dimensional space.

In a well-trained word embedding space, words like "cat" and "dog" might have similar vector representations because they often appear in similar contexts (e.g., "I have a cat" or "I have a dog"). Similarly, words like "king" and "queen" would have similar representations because they are often used in similar contexts (e.g., "The king ruled the kingdom" or "The queen wore a crown").

![image](https://github.com/user-attachments/assets/2244f81d-c708-4be3-8343-35b9ca162489)

**corpus** refers to a large collection of text documents or spoken language data that is used as a basis for NLP tasks. Corpora (plural of corpus) are used for training, testing, and evaluating NLP models. They can be domain-specific (such as medical texts) or general-purpose (such as news articles or social media posts).

Corpora can include not just written texts but also transcriptions of spoken language, different languages, and dialects, which would provide a more comprehensive understanding of the diversity of data in NLP.

![image](https://github.com/user-attachments/assets/737d6e89-2d70-443e-bec5-aaede9176b1a)

### Subdomains of NLP

- Text Classification
- Named Entity Recognition (NER)
- Part-of-Speech Tagging (POS)
- Machine translation
- Information Extraction
- Question Answering
- Text Generation
- Sentiment Analysis
- Co-reference Resolution

### Introduction to Language Models

**Statistical Models** </br>
A statistical model in NLP is a computational algorithm used for a variety of language processing tasks, including but not limited to predicting the next word in a sentence. These models analyze corpora to learn the statistical properties of language, such as the frequency and patterns of word occurrences.
For instance, given a partial sentence like 'The quick brown fox,' the model uses its knowledge of English syntax and phrase structure, derived from the training data, to predict a likely continuation, such as 'jumps over the lazy dog.'
However, their applications extend beyond sentence completion, encompassing tasks like translation, sentiment analysis, and more. By understanding the statistical likelihood of word sequences and considering the broader context within which words appear, these models attempt to mimic aspects of human language understanding and generation. 

**N-Gram Model** </br>
N-gram models are a fundamental type of statistical language model used in NLP. These models predict the likelihood of a word appearing next in a sequence based on the previous 'N' words. Essentially, the 'N' in N-gram represents the number of words considered in each sequence. 
- Bigram Model
- Trigram Model
- N-Gram Model

### Neural Networks

A neural network is a tool in AI and machine learning which loosely mimics the structure of a brain in order to create a complex environment that can pass data between functions and methods in order to perform some action. This architecture is typically visualized as a network of nodes (representing functions or computations) connected by lines (indicating the flow of data).

#### Types

- Feedforward Model </br>
  Feedforward neural network models, also known as multi-layer perceptron (MLP) models, are a type of artificial neural network commonly used in various machine learning tasks, including natural language processing.
  Feedforward models process information in a sequential manner, evaluating it step by step without returning to previous evaluations

![image](https://github.com/user-attachments/assets/9dbdd13a-813e-4512-a815-daed10f4dc51)

- Recurrent Neural Networks </br>
  Recurrent Neural Networks (RNN) models process information sequentially and can remember past information through feedback loops. They are well-suited for tasks involving sequential data like text or time series data or natural language processing. They are designed to handle inputs of varying lengths and have the ability to capture temporal dependencies. Temporal Dependencies involve the understanding of how words or elements in a sequence relate to each other based on their order or position.

![image](https://github.com/user-attachments/assets/7e8c3a1d-37ac-409e-a917-e5f4a661f9a8)

- Long Short - Term Memory Model
  Long Short-Term Memory (LSTM) models are a variant of recurrent neural networks (RNNs) specifically designed to address the vanishing gradient problem and capture long-term dependencies in sequential data.
  Traditional RNNs suffer from the vanishing gradient problem, which hampers their ability to capture dependencies that are distant in time. LSTMs were introduced to alleviate this issue and enable the modeling of long-range dependencies in sequential data.
  LSTMs employ three specialized gates: the input gate, the forget gate, and the output gate. The gates control the flow of information into, out of, and within the memory cell, enabling the LSTM to regulate the information it retains and processes.

![image](https://github.com/user-attachments/assets/a8358044-bd1c-4c3e-a5d5-1b92acf0671b)

- Transformer Model </br>
  Transformer Models are a groundbreaking neural network architecture introduced in the paper "Attention is All You Need" by Vaswani et al. (2017). They have since become the state-of-the-art approach for many natural language processing tasks. Transformer models borrow concepts from feedforward, RNN and long short term memory models, but they introduce a fundamentally different approach to capturing dependencies and context in language processing. ChatGPT is an example of a chatbot using a transformer model (GPT).

Transformer models employ a self-attention mechanism and parallel processing, enabling them to capture long-range dependencies and handle large-scale sequential data effectively. 

![image](https://github.com/user-attachments/assets/2ed0b084-a666-4bd3-bc6f-f4b7ed881f12)

## Introduction to AI Agents





  


















