# VLM Qualitative Evaluation

This folder contains a set of Jupyter Notebooks for performing **qualitative analysis of Visual Language Models (VLMs)** across a variety of vision-language tasks. Conducted on four models:

- `Gemma 3 12B`
- `Molmo 7B`
- `Pixtral 12B`
- `LLaMA 3.2 11B Vision`

## 📊 Evaluation Scope

The following task categories were assessed:

1. **Image Captioning**
2. **Visual Question Answering (VQA)**
3. **Object Detection / Recognition**
4. **Scene & Context Awareness**
5. **Text-Image Matching**
6. **OCR / Text-in-Image**
7. **Commonsense & Logical Reasoning**
8. **Zero/Few-Shot Generalization**

## 🧪 Scripts and Evaluation Outputs

The following Jupyter Notebooks were used to generate **raw model outputs** for each task:

- `test_model_gemma3_12b.ipynb`
- `test_model_molmo_7b.ipynb`
- `test_model_pixtral_12b.ipynb`
- `test_model_llama3.2_11B.ipynb`

> ⚠️ **Note:** The qualitative analysis—including observations, comparisons, and table generation—was conducted manually based on these outputs.

## 🔐 API Requirements

```bash
  OPENROUTER_API_KEY="your-key-here"
  HUGGINGFACE_TOKEN="your-token-here"
  MISTRAL_API_KEY="your-key-here"
```

## 📌 Notes

- These evaluations are **qualitative** and do not use automatic benchmarks (e.g., CIDEr, SPICE).
- Ideal for insight generation or model behavior research.

### 📬 Contact

Built by [Gayanuka Amarasuriya](https://gayanukaa.github.io/).
