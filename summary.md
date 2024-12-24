# RAG

RAG is an AI framework that helps optimize the output of large language models or LLMs.

RAG combines retrieved information and generates natural language to create responses.

RAG consists of two main components: the retriever, the core of RAG, and the generator, which functions as a chatbot.

In RAG process:

The retriever encodes user-provided prompts and relevant documents into vectors, stores them in a vector database, and retrieves relevant context vectors based on the distance between the encoded prompt and documents.

The generator then combines the retrieved context with the original prompt to produce a response.

The Dense Passage Retrieval (or DPR) Context Encoder and its tokenizer focus on encoding potential answer passages or documents. This encoder creates embeddings from extensive texts, allowing the system to compare these with question embeddings to find the best match.

Facebook AI Similarity Search, also known as Faiss, is a library developed by Facebook AI Research that offers efficient algorithms for searching through large collections of high-dimensional vectors.

Faiss is essentially a tool to calculate the distance between the question embedding and the vector database of context vector embeddings.

The DPR question encoder and its tokenizer focus on encoding the input questions into fixed-dimensional vector representations, grasping their meaning and context to facilitate answering them.

# LangChain

LangChain provides an environment for building and integrating large language model (LLM) applications into external data sets and workflow.

LangChain simplifies the integration of language models like GPT-4 and makes it accessible for developers to build natural language processing or NLP applications.

The components of LangChain are:

Chains, agents, and retriever

LangChain-Core

LangChain-Community

Generative models understand and capture the underlying patterns and data distribution to resemble the given data sets. Generative models are applicable in generating images, text, and music, augmenting data, discovering drugs, and detecting anomalies.

Types of generative models are:

Gaussian mixture models (GMMs)

Hidden Markov models (HMMs)

Restricted Boltzmann machines (RBMs)

Variational autoencoders (VAEs)

Generative adversarial networks (GANs)

Diffusion models

In-context learning is a method of prompt engineering where task demonstrations are provided to the model as part of the prompt.

Prompts are inputs given to an LLM to guide it toward performing a specific task. They consist of instructions and context.

Prompt engineering is a process where you design and refine the prompts to get relevant and accurate responses from AI.

Prompt engineering has several advantages:

It boosts the effectiveness and accuracy of LLMs.

It ensures relevant responses.

It facilitates meeting user expectations.

It eliminates the need for continual fine-tuning.

A prompt consists of four key elements: instructions, context, input data, and output indicator.

Advanced methods for prompt engineering are: zero-shot prompting, few-shot prompting, chain-of-thought prompting, and self-consistency.

Prompt engineering tools facilitate interactions with LLMs.

LangChain uses “prompt templates,” which are predefined recipes for generating effective prompts for LLMs

An agent is a key component in prompt applications that can perform complex tasks across various domains using different prompts.

The language models in LangChain use text input to generate text output.

The chat model understands the questions or prompts and responds like a human.

The chat model handles various chat messages, such as:

HumanMessage

AIMessage

SystemMessage

FunctionMessage

ToolMessage

The prompt templates in LangChain translate the questions or messages into clear instructions.

An example selector instructs the model for the inserted context and guides the LLM to generate the desired output.

Output parsers transform the output from an LLM into a suitable format.

LangChain facilitates comprehensive tools for retrieval-augmented generation (RAG) applications, focusing on the retrieval step to ensure sufficient data fetching.

The “Document object” in LangChain serves as a container for data information, including two key attributes, such as page_content and metadata.

The LangChain document loader handles various document types, such as HTML, PDF, and code, from various locations.

LangChain in document retrieves relevant isolated sections from the documents by splitting them into manageable pieces.

LangChain embeds documents and facilitates various retrievers.

LangChain is a platform that embeds APIs for developing applications.

Chains in the LangChain is a sequence of calls. In chains, the output from one step becomes the input for the next step.

In LangChain, chains first define the template string for the prompt, then create a PromptTemplate using the defined template and create an LLMChain object name.

In LangChain, memory storage is important for reading and writing historical data.

Agents in LangChain are dynamic systems where a language model determines and sequences actions, such as predefined chains.

Agents integrate with tools such as search engines, databases, and websites to fulfill user requests.
