import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import MarkdownTextSplitter
# from data_sources import LANGCHAIN_DATA
from langchain_community.document_loaders import JSONLoader, DirectoryLoader

load_dotenv()

data_folder = "data_by_file"
loader = DirectoryLoader(data_folder, glob='**/*.json', show_progress=False, loader_cls=JSONLoader, loader_kwargs = {'jq_schema':'{link: .link, content: .full_content}', 'text_content':False})
LANGCHAIN_DATA = loader.load()

def create_and_save_vectorstore():
    # Create text splitter
    text_splitter = MarkdownTextSplitter(chunk_size=20000, chunk_overlap=2000)
    splits = text_splitter.split_documents(LANGCHAIN_DATA)

    # Add metadata to each split
    for i, split in enumerate(splits):
        split.metadata["doc_id"] = i
        split.metadata["source"] = split.metadata.get("source", "unknown")

    # Create embedding model
    print("Creating embedding model...")
    embedding_model = HuggingFaceEmbeddings(
        # model_name="sentence-transformers/all-MiniLM-L6-v2"             # Lightweight and fast
        # model_name="sentence-transformers/all-MPNet-base-v2"            # Higher quality for semantic tasks
        model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"    # Optimized for QA
    )
    print("Embedding model created.")

    # Create FAISS index
    print("Creating FAISS index...")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embedding_model)
    print("FAISS index created.")

    # Save the vectorstore
    vectorstore.save_local("faiss_index")

if __name__ == "__main__":
    create_and_save_vectorstore()
    print("Vectorstore created and saved successfully.")
