# Infosys_KnownMap_AI

# 🧠 AI-KnowMap: Cross-Domain Knowledge Mapping with NLP & Graphs

AI-KnowMap is an open-source, AI-powered platform that transforms unstructured text into an interactive, explorable knowledge graph. It helps users discover semantic relationships across domains like science, history, technology, and philosophy using advanced NLP techniques and graph-based visualization.

---

## 🚀 Features

- 🔐 JWT-based user authentication  
- 📁 Upload datasets (CSV, Excel, JSON, TXT, PDF, DOCX, PNG, JPG)  
- 🧠 Named Entity Recognition (NER) and Relation Extraction  
- 📊 Dataset insights with interactive charts (Plotly)  
- 🔍 Semantic search using sentence embeddings  
- 🌐 Knowledge graph visualization (NetworkX + PyVis)  
- 🧩 Subgraph generation (ego graphs, k-hop BFS)  
- ⭐ Favorites and feedback system  
- 🛠️ Admin dashboard for monitoring and editing  
- 📦 Dockerized deployment for portability  

---

## 📦 Project Structure

```
AI-KnowMap/
├── app.py                  # Main Flask + Gradio app
└── README.md               # Project documentation
```

---

## 🧪 Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/AI-KnowMap.git
cd AI-KnowMap
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Run the app locally

```bash
python app.py
```

Access Gradio UI at: [http://localhost:7860](http://localhost:7860)

---

## 🐳 Docker Deployment

### 1. Build Docker image

```bash
docker build -t knowmap-app .
```

### 2. Run the container

```bash
docker run -p 8501:8501 knowmap-app
```
---

## 🧠 NLP Pipeline

- **NER**: spaCy or Hugging Face Transformers  
- **Relation Extraction**: Babelscape/rebel-large (text2text generation)  
- **Triple Format**: (Entity1, Relation, Entity2)  
- **Graph Storage**: NetworkX (in-memory) or Neo4j (optional)  

---

## 🔍 Semantic Search Flow

1. Embed node labels using SentenceTransformer (`all-MiniLM-L6-v2`)  
2. Compute cosine similarity with query  
3. Retrieve top-k matches  
4. Generate subgraph using ego_graph  
5. Visualize with PyVis (HTML) and NetworkX (PNG)  

---

## 🛠️ Admin Tools

- View total entities, relations, feedback  
- Edit, merge, delete triples  
- Monitor pipeline stats  
- Feedback review tab (CSV or DB)  

---

## 📬 Feedback System

- Users rate semantic search results (1–5)  
- Submit comments via Streamlit form  
- Stored in CSV or SQLite  
- Admins analyze trends and refine graph  

---

## 📈 Dataset Insights

- Bar/Pie charts for column distributions  
- Top 20 categories visualized with Plotly  
- Filter and smart search options  

---

## 📚 Milestones Covered

- ✅ Milestone 1: Authentication, Profile, Dataset Upload  
- ✅ Milestone 2: NER + Relation Extraction → Triples  
- ✅ Milestone 3: Graph Construction + Semantic Search + Visualization  
- ✅ Milestone 4: Admin Dashboard + Feedback + Docker Deployment  

---

## 📌 Future Scope

- Neo4j cloud integration  
- Multilingual support  
- Streamlit Cloud / Hugging Face Spaces deployment  
- Feedback-driven model retraining  
- Graph expansion with external APIs  

---

## 👤 Author

Built by PERANANDHA  
Location: Tamil Nadu, India  
Focus: Scalable, extensible knowledge graph platforms with semantic search and interactive UI  

---

## 📄 License

This project is open-source under the MIT License.
```
