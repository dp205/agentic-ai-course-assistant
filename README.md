# agentic-ai-course-assistant
Agentic AI Course Assistant built using LangGraph, ChromaDB, and Streamlit. Supports context-aware Q&amp;A, retrieval-based responses, memory handling, and self-evaluation to ensure accurate, non-hallucinated answers from course materials.


An intelligent AI assistant built using LangGraph that helps students understand concepts from the Agentic AI course. The system uses Retrieval Augmented Generation (RAG), memory, and self-evaluation to provide accurate and context-aware answers.

Features

- 📚 Retrieval-based answers using ChromaDB (RAG)
- 🧠 Context-aware conversation with memory (thread_id)
- 🔀 Intelligent routing (retrieve / tool / skip)
- 🛠 Tool support (date, time, basic calculations)
- ✅ Self-evaluation to reduce hallucination
- 💬 Multi-turn conversation support
- 🌐 Streamlit-based interactive UI

Architecture

User Query  
→ Memory Node  
→ Router Node  
→ (Retrieval / Tool / Skip)  
→ Answer Node  
→ Evaluation Node  
→ Save Node  

Tech Stack

- Python  
- LangGraph  
- ChromaDB  
- SentenceTransformers  
- Groq LLM (LLaMA 3.1)  
- Streamlit  
## 📁 Project Structure
