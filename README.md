## Install
```
python3 -m venv .venv
source .venv/bin/activate
pip install python-dotenv langsmith chainlit openai llama_index
```

## Setup
* add `.env` file
* The `data` directory should include all relevant data we want to include for rag
* The `data_index_llama` is a cached index of the `data` directory. If the `data` changes, we should delete the `data_index_llama` directory to recreate a fresh index.

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
