# SPICE & Scene-Graph Evaluation Dashboard

An interactive Streamlit application that combines the official SPICE metric with a conceptual scene-graph tuple visualizer. Evaluate your image captions end-to-end, inspect SPICE precision/recall/F₁, see live logs of the Java subprocess, and explore extracted (object–relation–object / object–attribute) scene-graph tuples as draggable force-directed graphs.

## 🚀 Features

- **SPICE Evaluation**  
  Invokes the SPICE-1.0 Java JAR in “detailed” mode to compute the official SPICE metric (precision, recall, F₁).
- **Live Logging**  
  Streams the SPICE subprocess stdout/stderr into a collapsible, scrollable terminal box in the UI.
- **Interactive Scene-Graph Visualization**  
  Extracts simple tuples from captions and renders them with PyVis: draggable nodes, curved edges, physics controls.
- **Extracted Tuple Preview**  
  View simplified lists of extracted tuples (candidate & references) in collapsible code blocks.

## 📦 Repository Structure

```
spice/
┣ SPICE-1.0/
┃ ┣ lib/                           ← SPICE’s dependencies
┃ ┣ spice-1.0.jar
┃ ┗ get\_stanford\_models.sh
┣ lib/                             ← Frontend JS/CSS assets (tom-select, vis-network)
┣ spice\_cache/                     ← Temp files: JSON in/out, HTML graphs, LMDB
┣ scene\_graph\_visualizer.py        ← PyVis graph-builder
┣ evaluator.py                     ← SPICE invocation + tuple extraction + WordNet matching
┣ app.py                           ← Streamlit frontend
┣ requirements.txt                 ← Python dependencies
┗ README.md                        ← This file
```

## 🛠️ Getting Started

Follow these steps to set up and run the dashboard locally.

### 1. Prerequisites

- **Java 8+** (to run the SPICE JAR)
- **Conda** or **Miniconda**
- **Git**

On Debian/Ubuntu you may need:

```bash
sudo apt update
sudo apt install -y default-jre unzip
```

### 2. Clone the Repository

```bash
git clone https://github.com/Gayanukaa/VLM-Playground.git
cd experiments/spice/
```

### 3. Download and Prepare SPICE

1. **Download SPICE-1.0**

   - From [panderson’s SPICE site](https://panderson.me/spice/)
   - Place the `spice-1.0.jar` and `lib/` folder under the `SPICE-1.0/` directory.

2. **Fetch Stanford CoreNLP Models**

   ```bash
   cd SPICE-1.0
   chmod +x get_stanford_models.sh
   ./get_stanford_models.sh
   ```

   This downloads the Stanford CoreNLP JARs (tokenizer, parser, NER, etc.) into `SPICE-1.0/lib/`.

### 4. Create & Activate Conda Environment

```bash
conda create -n env python=3.11.8 -y
conda activate env
```

### 5. Install Python Dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Running the Dashboard

From the project root:

```bash
streamlit run app.py
```

The app will open in your browser (usually at `http://localhost:8501`). Enter your **Candidate Caption** and one or more **Reference Captions** in the sidebar, then click **Evaluate**.

## 🧪 Manual SPICE Testing

If you want to test SPICE outside of Streamlit, you can run:

```bash
java -Xmx8G \
  -jar SPICE-1.0/spice-1.0.jar \
  sample_input.json \
  -detailed
```

This produces detailed SPICE output (`stdout` and a JSON file).


## 🐛 Troubleshooting

- **SPICE JAR not found**
  Ensure `SPICE-1.0/spice-1.0.jar` exists and is executable.

- **CoreNLP model errors / `NoClassDefFoundError`**
  You may encounter the issue described here:
  [https://github.com/Labbeti/aac-metrics/issues/7](https://github.com/Labbeti/aac-metrics/issues/7) </br>
  **Resolution:** Manually download the Stanford CoreNLP ZIP and extract it into `SPICE-1.0/lib/`:
  [http://nlp.stanford.edu/software/stanford-corenlp-full-2015-12-09.zip](http://nlp.stanford.edu/software/stanford-corenlp-full-2015-12-09.zip)

- **Missing NLTK data**
  NLTK auto-downloads WordNet & OMW on first run. To install manually:

  ```python
  import nltk
  nltk.download('wordnet')
  nltk.download('omw-1.4')
  ```

- **PyVis graphs not rendering**
  Ensure your network allows CDN loading, or switch `cdn_resources='local'` / `remote` in `scene_graph_visualizer.py`.


## 🤝 Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature-xxx`)
3. Commit your changes (`git commit -m "Add new feature"`)
4. Push to your fork (`git push origin feature-xxx`)
5. Open a Pull Request

## 📄 License

This project is released under the [MIT License](LICENSE).
