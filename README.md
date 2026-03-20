# MIRFLICKR Big Data Pipeline

> Scalable image metadata extraction and EDA using Hadoop HDFS + PySpark

---

## Project Overview

This project builds a distributed data pipeline to ingest, store, and analyze a large-scale collection of unstructured image data from the [MIRFLICKR-1M dataset](https://press.liacs.nl/mirflickr/). Using Hadoop HDFS for distributed storage and PySpark for parallelized feature extraction, we transform 100,000+ raw images into a structured metadata catalog for Geospatial and Temporal EDA.

---

## Dataset

| Property | Detail |
|---|---|
| Source | MIRFLICKR-1M (LIACS Medialab, Leiden University) |
| License | Creative Commons |
| Subset used | Part 0 — ~100,000 images |
| Size | ~12.55 GB raw binary |
| Format | JPEG images + EXIF metadata |

### The 3 Vs

- **Volume** — 100,000+ images totalling ~12.55 GB stored in HDFS
- **Variety** — unstructured binary images combined with structured EXIF metadata (GPS, timestamps, camera specs)
- **Velocity** — batch ingestion and parallel processing via PySpark's `binaryFile` format

---

## Architecture

```
MIRFLICKR-1M Dataset
        │
        ▼
  Docker Cluster (docker-compose.yml)
        │
   ┌────┴────┐
   │  HDFS   │
   │NameNode │  ← metadata/namespace
   │DataNode │  ← raw image blocks
   └────┬────┘
        │
        ▼
  PySpark (binaryFile reader)
        │
        ▼
  Custom UDF + Pillow
  (EXIF extraction)
        │
        ▼
  Parquet / CSV
  (structured metadata)
        │
        ▼
  EDA Notebooks
  ┌──────────────────────┐
  │ Geospatial hotspots  │
  │ Device profiling     │
  │ Temporal trends      │
  │ Image quality dist.  │
  └──────────────────────┘
```

---

## Repository Structure

```
├── docker-compose.yml              # Hadoop/Spark cluster setup (NameNode + DataNode)
├── EDA.ipynb                       # Main EDA notebook
├── user-photos-EDA.ipynb           # User photo EDA (Bhavyasreekondi)
└── big-photos/
    ├── docker-compose.yml          # Cluster config (primary)
    ├── big-photos-eda.ipynb        # EDA for large batch
    ├── mirflickr_part0_structured_meta_eda.ipynb   # Structured metadata EDA
    └── mirflickr_thorough_eda_addon.ipynb           # Extended EDA
```

---

## Getting Started

### Prerequisites

- Docker + Docker Compose
- Python 3.8+
- PySpark 3.x
- Pillow

### 1. Start the Hadoop cluster

```bash
cd big-photos
docker-compose up -d
```

This spins up:
- **NameNode** — manages HDFS namespace and metadata
- **DataNode** — stores the actual image data blocks

### 2. Upload images to HDFS

```bash
# Copy dataset into HDFS
hdfs dfs -mkdir /mirflickr
hdfs dfs -put /path/to/images/ /mirflickr/
```

### 3. Run the extraction pipeline

Open `big-photos/big-photos-eda.ipynb` and run all cells. The notebook:
1. Reads raw images using PySpark's `binaryFile` format
2. Applies a custom UDF with Pillow to extract EXIF metadata
3. Outputs a structured Parquet/CSV file

### 4. Run EDA

Open any of the EDA notebooks to explore:
- `mirflickr_part0_structured_meta_eda.ipynb` — structured metadata analysis
- `mirflickr_thorough_eda_addon.ipynb` — extended EDA
- `user-photos-EDA.ipynb` — user photo analysis

---

## Key Analytical Questions

1. **Spatial Distribution** — Where are the geographic hotspots in the dataset?
2. **Device Profiling** — Which manufacturers (Apple, Samsung, Canon) dominate?
3. **Temporal Trends** — What are peak capture hours and seasonal patterns?
4. **Technical Quality** — What is the distribution of resolutions and ISO settings?

---

## Extracted Features

| Category | Fields |
|---|---|
| Hardware | Camera Make, Model, ISO, Focal Length |
| Geospatial | GPS Latitude, GPS Longitude (DMS → Decimal) |
| Temporal | Original capture timestamp |
| Image | Width, Height, Format |

---

## Team

| Role | Responsibility | Member |
|---|---|---|
| Data Engineer | Dataset ingestion, HDFS setup | |
| Processing Engineer | PySpark ETL, UDF, transformations | |
| Data Analyst | EDA, visualizations, insights | |
| System Designer | Architecture diagram, GitHub, docs | |

---

## References

- [MIRFLICKR Dataset](https://press.liacs.nl/mirflickr/) — LIACS Medialab, Leiden University
- Huiskes, M.J. & Lew, M.S. (2008). The MIR Flickr Retrieval Evaluation. ACM MIR '08.
