# VIGIL

## Project Description

VIGIL is a comprehensive framework designed to automate the collection, analysis, and visualization of cyber incident data. It provides tools to crawl for relevant information, process and label it using machine learning models, store the findings, and present them through an interactive dashboard and API. The goal is to offer a streamlined solution for monitoring and understanding the cyber threat landscape.

## Key Features

*   **Automated Data Collection**: Employs a crawler to gather cyber incident data from various sources.
*   **Machine Learning Analysis**: Utilizes ML models to classify and label collected articles, enabling efficient data processing.
*   **Data Processing Pipeline**: Features a robust pipeline (`crawl -> analyze -> store`) to manage the flow of data from collection to storage.
*   **Database Storage**: Stores processed incident data systematically for easy retrieval and analysis.
*   **REST API**: Provides an API (built with FastAPI) for programmatic access to incident data, sources, and search functionalities.
*   **Interactive Dashboard**: Offers a web-based dashboard (built with FastAPI and Jinja2) for visualizing incident data, trends, and model performance.
*   **Modular Design**: Composed of distinct components for crawling, data preparation, model training, classification, visualization, and UI, allowing for flexibility and scalability.
*   **Configurable**: Supports configuration through a `config.yml` file for various aspects like logging and component settings.
*   **Command-Line Interface**: Offers a suite of command-line tools for interacting with different parts of the framework (e.g., `vigil-crawl`, `vigil-train`, `vigil-ui`).

## User Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/waseemr02/virgil
   cd vigil
   ```

2. Setup your environment:
   It is recommended that you have uv installed (see [UV Repo](https://github.com/astral-sh/uv) for details).
   ```bash
   uv sync
   ```

Alternatively, you can use pip:
```bash
pip install .
```

## Running the Project

You can run the various components of VIGIL using the following commands. These commands are available after installing the project using `uv sync` or `pip install .`.

- `vigil`: Main entry point for the VIGIL application (if applicable, otherwise describe its function).
- `vigil-crawl`: Crawls for new cyber incident data.
- `vigil-label`: Labels collected articles.
- `vigil-prep`: Preprocesses data for model training.
- `vigil-train`: Trains a new model.
- `vigil-classify`: Classifies articles using a trained model.
- `vigil-visualize`: Generates visualizations of the data or model performance.
- `vigil-workflow`: Runs a predefined data processing workflow.
- `vigil-monitor`: Monitors the system or data sources.
- `vigil-scheduler`: Manages scheduled tasks.
- `vigil-ui`: Starts the user interface/dashboard.

For example, to start the UI, you would run:
```bash
vigil-ui
```

## Module Overview

The VIGIL project is organized into several key modules within the `src/vigil` directory, each responsible for a specific set of functionalities:

*   **`api`**: Implements the RESTful API (using FastAPI) for accessing and managing cyber incident data.
*   **`core`**: Contains the central data processing pipeline (`pipeline.py`), system monitoring (`monitoring.py`), and task scheduling (`scheduler.py`).
*   **`dashboard`**: Powers the web-based user interface (FastAPI and Jinja2) for visualizing data and trends.
*   **`data_collection`**: Handles fetching raw data using a web crawler (`crawler.py`) and extracting content (`content_extractor.py`).
*   **`database`**: Manages database interactions (SQLAlchemy ORM), including schema definitions (`models.py`), connections (`connection.py`), queries (`queries.py`), and URL tracking.
*   **`model`**: Encompasses machine learning tasks, including data preparation (`data_prep.py`), model training (`training.py`), and ML-specific visualizations (`model_viz.py`).
*   **`scripts`**: Contains standalone Python scripts for specific operations like running the UI, monitoring, or scheduling. These are often the entry points for CLI commands.
*   **`storage`**: Manages the file-based storage of crawled content, metadata, and other data artifacts (`data_store.py`).
*   **`tools`**: Provides command-line interface (CLI) tools for tasks like crawling, training models, classifying articles, and data preprocessing.

These modules work together to provide the comprehensive functionality of the VIGIL framework.
