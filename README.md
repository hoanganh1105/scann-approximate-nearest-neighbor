# ScaNN vs Brute-Force Approximate Nearest Neighbor Search

### Advanced Data Structures and Algorithms – Talent Program  
Ho Chi Minh City University of Technology (HCMUT)

---

## 📘 Overview
This project implements and evaluates **ScaNN (Scalable Nearest Neighbor# ScaNN: Efficient Approximate Nearest Neighbor Search

### Extended Coursework Project -- Data Structures & Algorithms (CO2003)

## 📌 Overview

This project explores **ScaNN (Scalable Nearest Neighbors)** --- a
state-of-the-art Approximate Nearest Neighbor (ANN) search algorithm
developed by Google.\
Our goal was to **study the theory**, **implement ScaNN**, **compare it
with brute-force and manual Product Quantization (PQ)**, and **extend
ScaNN into a multi-stage GraphRAG pipeline** for advanced semantic
retrieval.

This repository contains implementations, experiments, datasets, and
documentation from the project.

------------------------------------------------------------------------

## 👥 Team Members

  Name                  Student ID
  --------------------- ------------
  **Huỳnh Hoàng Anh**   2410078
  **Ngô Trung Tín**     2413506
  **Huỳnh Tấn Tiến**    2413468

Instructor: **Lê Thành Sách**\
Class: **TN01 -- Honors Program**

------------------------------------------------------------------------

## 🚀 Project Objectives

-   Understand ScaNN's architecture and retrieval pipeline.\
-   Build a real-world ANN search system using vector embeddings.\
-   Compare ScaNN with brute-force methods and manual PQ
    implementation.\
-   Evaluate performance based on latency, recall, and memory usage.\
-   Extend ScaNN into a **GraphRAG hybrid retrieval system** (semantic
    chunking, entity resolution, link prediction, fuzzy search,
    caching).

------------------------------------------------------------------------

## 🧠 ScaNN Architecture

### 🔹 1. Partitioning

Uses clustering (e.g., k-means) to reduce the search space.

### 🔹 2. Scoring with AVQ

Leverages **Anisotropic Vector Quantization**, optimized for **Maximum
Inner Product Search (MIPS)**.

### 🔹 3. Reordering

Performs exact similarity computation on top candidates to maximize
recall.

------------------------------------------------------------------------

## 🔄 Comparison with Other ANN Methods

ScaNN was benchmarked against: - **Brute-force** - **FAISS (IVF, PQ)** -
**HNSW** - **Annoy**

ScaNN demonstrated excellent **speed--recall trade-offs**, especially at
large scale.

------------------------------------------------------------------------

## 📂 Project Structure

    📦 scann-approximate-nearest-neighbor
     ┣ 📁 notebooks/          # Colab demos
     ┣ 📁 src/                # Core implementations
     ┣ 📁 data/               # Text data & embeddings
     ┣ 📁 pq_manual/          # Manual PQ implementation
     ┗ 📄 README.md

------------------------------------------------------------------------

## 🛠️ Implementation Details

### 1️⃣ Data Collection

News articles were gathered from Vietnamese RSS sources (VnExpress, Tuổi
Trẻ, Thanh Niên, ZingNews,...).

### 2️⃣ Embedding Generation

Used **Vietnamese-SBERT** to generate 768-dimensional semantic vectors.

### 3️⃣ ScaNN Index Construction

``` python
searcher = scann.scann_ops_pybind.builder(
    embeddings, 10, "dot_product"
).tree(
    num_leaves=32,
    num_leaves_to_search=16,
    training_sample_size=5000
).score_ah(
    2, anisotropic_quantization_threshold=0.25
).reorder(100).build()
```

### 4️⃣ Brute-force Baseline

Dot product computed across all vectors for exact results.

### 5️⃣ Manual Product Quantization

Includes: - Subspace training with k-means\
- Vector quantization\
- Asymmetric Distance Computation (ADC)\
- Reordering refinement

------------------------------------------------------------------------

## 📊 Experimental Results

### 🔹 Small Dataset (5K vectors)

  K     Brute-force (s)   ScaNN (s)   Speedup     Recall
  ----- ----------------- ----------- ----------- --------
  10    0.002436          0.000446    **5.46×**   0.974
  100   0.002313          0.000440    **5.26×**   0.883

### 🔹 Large Dataset (4M vectors, 128D)

  Method        Build Time   Memory
  ------------- ------------ ------------
  Brute-force   254 ms       1953 MB
  ScaNN         304 s        **489 MB**

  K    BF Time (ms)   ScaNN (ms)   Recall
  ---- -------------- ------------ --------
  1    2546           **30.88**    1.00
  50   2473           **25.05**    0.823

➡️ ScaNN achieves **\~80× faster queries** and **4× memory reduction**.

------------------------------------------------------------------------

## 🧩 Extended Work: GraphRAG + ScaNN

We implemented a **6-stage enhanced retrieval pipeline**:

### ✔ 1. Semantic Chunking

Adaptive chunking using ScaNN similarity thresholds instead of
fixed-size splitting.

### ✔ 2. Entity Resolution

Merges duplicated entities using cosine similarity \> 0.96.

### ✔ 3. Link Prediction

Adds missing graph edges when similarity ∈ (0.85, 0.96).

### ✔ 4. Hybrid ScaNN Index

Text chunks + entities unified in a single vector index.

### ✔ 5. Reference Tracking

Ensures every entity result can retrieve related text chunks.

### ✔ 6. Smart Semantic Cache

Two-layer cache: 1. ScaNN fast filter\
2. Cross-Encoder verification

------------------------------------------------------------------------

## 📈 Key Contributions

-   Complete implementation of ScaNN indexing pipeline.\
-   Manual PQ system built from scratch.\
-   Comprehensive performance benchmarks.\
-   Extended GraphRAG architecture for semantic search.\
-   Hybrid indexing for unified text + entity retrieval.

------------------------------------------------------------------------

## 🔮 Future Work

-   Full-text embedding instead of using only news titles.\
-   Deeper quantitative benchmarking for GraphRAG extension.\
-   GPU-based acceleration for ScaNN and PQ modules.\
-   Expanding dataset to multi-domain knowledge bases.

------------------------------------------------------------------------

## 📎 Resources

-   **GitHub Repository**\
    https://github.com/hoanganh1105/scann-approximate-nearest-neighbor

-   **Google Colab Notebooks**

    -   Text Retrieval\
    -   Vector Search\
    -   GraphRAG Extension

------------------------------------------------------------------------

## 📜 License

This project is for academic purposes under the CO2003 course\
at Ho Chi Minh City University of Technology (HCMUT).
s)** — a high-performance approximate nearest neighbor (ANN) search algorithm developed by **Google Research**.  
ScaNN combines **partitioning**, **vector quantization**, and **asymmetric hashing** to achieve efficient large-scale vector retrieval.  
We compare the performance of ScaNN against a baseline **brute-force** search to explore trade-offs between **accuracy**, **speed**, and **memory efficiency**.

---

## 🎯 Objectives
- Study and understand the architecture of **ScaNN** (partitioning → scoring → reordering).  
- Build an **ANN search system** using ScaNN in Python.  
- Implement a **brute-force search** as a performance baseline.  
- Conduct experiments to compare query latency, recall@K, and memory usage.  
- Visualize performance metrics through clear and reproducible plots.

---

## 🧠 Theoretical Background
ScaNN accelerates nearest neighbor search by reducing the number of distance computations required.  
Its three main components are:
1. **Partitioning** – clusters the dataset into smaller regions.  
2. **Scoring** – selects the most relevant partitions based on approximate distances.  
3. **Reordering** – re-ranks the candidates using exact distances for higher accuracy.  
Compared with other ANN methods like **Faiss**, **HNSW**, and **Annoy**, ScaNN achieves a strong balance between **speed** and **recall** for large vector databases.

---

## 🧩 Project Structure
```
scann-ann/
├── data/
│   ├── vectors.npy            # generated or preprocessed dataset
│   └── queries.npy            # sample query vectors
│
├── src/
│   ├── brute_force.py         # baseline linear search implementation
│   ├── scann_search.py        # ScaNN-based ANN search
│   ├── evaluate.py            # accuracy, timing, memory evaluation
│   ├── visualize.py           # performance plots
│   └── utils.py               # helper functions
│
├── notebooks/
│   └── demo.ipynb             # interactive demonstration and analysis
│
├── results/
│   ├── timing_plot.png
│   ├── recall_plot.png
│   └── memory_comparison.png
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation
### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/scann-ann.git
cd scann-ann
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
**Required libraries:**
```
scann
numpy
scikit-learn
sentence-transformers
matplotlib
```

---

## 🚀 Usage
### Run the experiments locally
```bash
python src/scann_search.py
```
or open the Jupyter notebook:
```bash
notebooks/demo.ipynb
```

### Example workflow
1. Generate or load vector datasets (e.g., from images or text embeddings).  
2. Build and query the ScaNN index.  
3. Run brute-force search for baseline comparison.  
4. Evaluate average query time and recall@K.  
5. Visualize the performance metrics.

---

## 📊 Expected Outputs
- **ScaNN** achieves significant speedup over brute-force with high recall.  
- Comparative plots:
  - Query time vs. K
  - Recall@K
  - Memory usage  
Example output:
```
Average query time:
  - Brute-force: 0.285 s
  - ScaNN:       0.012 s
Recall@10: 0.97
```

---

## 🧪 Experimental Setup
- **Dataset:** Random vectors or embeddings extracted via ResNet/SBERT  
- **Dimensions:** 128–768  
- **Queries:** 100 random vectors  
- **Evaluation metrics:**
  - Average query latency  
  - Recall@K (K = 5, 10, 20)  
  - Memory usage  
- **Hardware:** CPU-based testing (optionally GPU for embedding generation)

---

## 📈 Visualization
Performance results are plotted using `matplotlib`, including:
- Query time comparison between ScaNN and brute-force  
- Recall@K curves  
- Memory utilization histogram  

Example snippet:
```python
plt.plot(k_values, query_times_scann, label='ScaNN')
plt.plot(k_values, query_times_bruteforce, label='Brute-force')
plt.xlabel('K')
plt.ylabel('Average Query Time (s)')
plt.legend()
plt.show()
```

---

## 🧰 Tools and Environment
- **Language:** Python 3.10+  
- **Libraries:** ScaNN, NumPy, Matplotlib, Scikit-learn, Sentence-Transformers  
- **Supported Platforms:** Google Colab, Jupyter Notebook, or local Python environment

---

## 👥 Authors
- **Huỳnh Hoàng Anh**   
- **Ngô Trung Tín**   
- **Huỳnh Tấn Tiến**  

---

## 📜 License
This project is for **academic and educational purposes only**.  
© 2025, Ho Chi Minh City University of Technology – Faculty of Computer Science and Engineering.

---

## 🔗 Links
- 📘 [ScaNN Official Repository](https://github.com/google-research/google-research/tree/master/scann)  
- 📄 [Project Report (PDF)](report/report.pdf)  
- ▶️ [Google Colab Demo](https://colab.research.google.com/drive/your-demo-link)
