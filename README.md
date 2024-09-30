## Install
```
python3 -m venv .venv
source .venv/bin/activate
pip install python-dotenv langsmith chainlit openai langchain langchain_community jq sentence-transformers langchain_huggingface faiss-cpu black pylint beautifulsoup4 serpapi google-search-results
```

## Setup
* add `.env` file
* The `data` directory should include all relevant data we want to include for rag

## Run the app
```
chainlit run app.py -w
```

## What is this?
Health Fact / Studies Analyzer

## Authors
* Tyler Law
* Jenny Kwan
* Sahil Agarwal
