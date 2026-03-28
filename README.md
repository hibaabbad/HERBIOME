# 🌿 HERBIOME

**H**erbiers · **E**tiquettes · **R**econnaissance · **B**otanique · **I**mages · **O**ptimisation · **M**ultimodale · **E**tudes

> An AI-powered pipeline for automated detection, reading, and structured extraction of label information from European and French herbarium sheets.

---

## Overview

HERBIOME is an end-to-end pipeline that processes scanned herbarium specimen images and returns structured botanical metadata. It combines computer vision, OCR, and large language models to automate the digitization of herbarium collections, with a focus on European and French herbaria.

The pipeline performs five sequential steps:

1. **Component Detection** — YOLO model locates labels on the herbarium sheet (handwritten data, institutional labels, annotation labels)
2. **Text Region Detection** — CRAFT model identifies individual word/text regions within each detected label
3. **Reading Order Restoration** — DBSCAN clustering reconstructs the correct top-to-bottom, left-to-right reading order of detected word regions (see details below)
4. **Text Recognition** — TrOCR model reads and transcribes each word region in the restored order
5. **Structured Extraction** — GPT-4o-mini corrects OCR errors and returns structured JSON with botanical metadata

### Why YOLO + CRAFT?

Herbarium sheets contain heterogeneous content: pressed plant specimens, printed institutional labels, handwritten annotations, barcodes, and color references all coexist on the same sheet. A single general-purpose text detector applied to the full image would pick up noise from non-text regions or miss label areas entirely.

The two-stage approach solves this cleanly. **YOLO** acts as a semantic filter — it is trained specifically on herbarium sheet layouts and reliably isolates only the regions that carry label information, ignoring the plant specimen and other non-textual elements. **CRAFT** then operates within each cropped label region, where it excels at detecting individual word-level bounding boxes even across mixed handwritten and printed content. This combination ensures exhaustive coverage: YOLO guarantees no label region is missed at the sheet level, and CRAFT guarantees no word is missed within each label.

### Why TrOCR?

Herbarium labels present a particularly difficult OCR challenge. A single label can contain both printed institutional text and handwritten collector notes, often in 19th or early 20th century script. Standard OCR engines are typically optimized for one or the other and perform poorly when both are present on the same document.

TrOCR was selected because its transformer-based architecture handles this mixed-modality case robustly out of the box. Beyond the base model, the version used in HERBIOME was **fine-tuned on a private dataset of French and European herbarium labels**. This fine-tuning significantly improves recognition accuracy on the specific handwriting styles, abbreviations, and Latin botanical terminology found in these collections. The fine-tuned weights are not publicly distributed.

---

## Reading Order Restoration

CRAFT is a strong text detector but it does not guarantee that detected word bounding boxes are returned in any meaningful reading order — boxes can come out in arbitrary sequence depending on the image content. Feeding unordered word crops directly to TrOCR would produce transcribed text that is scrambled and unusable by the LLM.

To fix this, HERBIOME applies a **DBSCAN-based line reconstruction** step between CRAFT detection and TrOCR recognition:

1. **Line clustering** — the vertical center coordinate of each detected word bounding box is extracted, and DBSCAN is run on these Y values to group boxes that belong to the same text line. The epsilon parameter is set relative to the median word height across the label, making it adaptive to font size and scale variation across different specimens.

2. **Noise handling** — DBSCAN can label some boxes as noise (label `-1`) when they do not fit cleanly into any cluster. These are reassigned to the nearest existing line cluster based on vertical distance, or promoted to a new line if they are far from everything else.

3. **Sorting** — lines are sorted top-to-bottom by their mean Y position. Within each line, word boxes are sorted left-to-right by their X center coordinate.

4. **Index assignment** — each word is assigned a sequential `order` index reflecting its correct reading position. TrOCR then processes the crops in this order and the results are joined into a coherent full-text string per label component.

This approach handles the multi-line, mixed handwritten and printed layouts typical of European herbarium labels without requiring any additional trained model.

---

## Models

| Model | Role | Source |
|---|---|---|
| **YOLO (sheet-component)** | Herbarium sheet component detection | [https://huggingface.co/yeppeuda13/Yolo-hespi](https://huggingface.co/yeppeuda13/Yolo-hespi) |
| **TrOCR (Herbiome)** | Handwritten and printed text recognition | [https://huggingface.co/yeppeuda13/TrOCR_Herbiome](https://huggingface.co/yeppeuda13/TrOCR_Herbiome) |
| **GPT-4o-mini** | OCR correction and structured data extraction | [https://platform.openai.com/docs/models/gpt-4o-mini](https://platform.openai.com/docs/models/gpt-4o-mini) |

The CRAFT text detection model is loaded automatically via the `hezar` library (`hezarai/CRAFT`).

---

## Project Structure

```
herbiome/
├── assets/
├── src/
│   ├── pipeline.py        # Core HerbariumPipeline and HerbariumProcessor classes
│   └── utils.py           # File handling, validation, response formatting
├── app.py                 # FastAPI backend
├── streamlit_app.py       # Streamlit frontend
├── requirements.txt
├── .env                   # Environment variables (not committed)
└── README.md
```

---

## Environment Setup

Create a `.env` file at the root of the project with the following variables:

```env
# Path to the YOLO model on Hugging Face Hub
YOLO_MODEL_PATH=yeppeuda13/Yolo-hespi

# Path to the TrOCR model on Hugging Face Hub
TROCR_MODEL_PATH=yeppeuda13/TrOCR_Herbiome

# Your OpenAI API key (required for structured extraction)
OPENAI_API_KEY=sk-...

# Device to run models on: "cpu" or "cuda"
DEVICE=cpu
```

> **Note:** If `DEVICE=cuda` is set, make sure your environment has a compatible NVIDIA GPU with CUDA installed. For CPU-only machines, leave it as `cpu`.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/herbiome.git
cd herbiome

# Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Application

HERBIOME has two components that need to run simultaneously: the **FastAPI backend** and the **Streamlit frontend**.

### 1. Start the API (Backend)

```bash
uvicorn app:app --host localhost --port 8001 --reload
```

The API will be available at `http://localhost:8001`.

You can verify it is running by visiting `http://localhost:8001/health` in your browser.

### 2. Start the UI (Frontend)

Open a second terminal, activate the virtual environment, then run:

```bash
streamlit run streamlit_app.py
```

The Streamlit interface will open automatically in your browser at `http://localhost:8501`.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Root — API info |
| `GET` | `/health` | Health check and component validation |
| `POST` | `/process` | Full pipeline: detection + OCR + LLM structuring |
| `POST` | `/extract-text` | Detection + OCR only (no LLM) |
| `GET` | `/component-image/{image_id}` | Retrieve a cropped component image by ID |
| `GET` | `/supported-formats` | List of accepted image formats |

### Example: Full Processing

```bash
curl -X POST http://localhost:8001/process \
  -F "file=@specimen.jpg"
```

### Example Response

```json
{
  "status": "success",
  "message": "Image processed successfully",
  "data": {
    "filename": "specimen.jpg",
    "structured_data": {
      "specimen_family": "Iridaceae",
      "specimen_genus": "Iris",
      "country_country": "France",
      "locality_locality": "Pyrénées-Orientales, Col de Puymorens",
      "collect_date_collect_date": "06/1923",
      "collector_collector": "Gautier, G.",
      "rest_of_text": "Leg. in rupestribus"
    },
    "json_data": { ... }
  }
}
```

---

## Supported Image Formats

`.jpg` · `.jpeg` · `.png` · `.bmp` · `.tiff` · `.tif`

Recommended maximum file size: **10 MB**

---

## Processing Modes

The Streamlit UI offers two processing modes:

- **Full Processing (with LLM)** — runs the complete pipeline and returns structured botanical fields
- **Text Extraction Only** — runs detection and OCR only, returns raw transcribed text per component without LLM correction

---

## Extracted Fields

| Field | Description |
|---|---|
| `specimen_family` | Botanical family name (e.g. Iridaceae) |
| `specimen_genus` | Genus name (e.g. Iris) |
| `country_country` | Country of collection |
| `locality_locality` | Precise locality (town, region, etc.) |
| `collect_date_collect_date` | Collection date — format DD/MM/YYYY or MM/YYYY |
| `collector_collector` | Collector name — format Lastname, Firstname |
| `rest_of_text` | Any remaining text not classified above |

---

## Notes

- The pipeline is optimized for **European and French herbarium collections** and the LLM prompt is written in French to improve accuracy on French-language labels.
- On first startup the pipeline may take several minutes to download and initialize all models.
- The fine-tuned TrOCR weights were trained on private data from the IRD/UMMISCO laboratory and are not publicly shared.

---

## Acknowledgements

This research was partially funded by the **French National Research Agency** (Agence Nationale de la Recherche, ANR) under the **e-Col+ project** (ANR-21-ESRE-0053).

Development and fine-tuning data were produced in collaboration with the **IRD/UMMISCO laboratory**.
