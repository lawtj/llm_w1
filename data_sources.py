from llama_index.core import SimpleDirectoryReader
from langchain_community.document_loaders import TextLoader, JSONLoader, DirectoryLoader

# If true, use smaller ttm_data set (5 custom papers). Otherwise use full scraped data
GOLDEN_SOURCE = False
if GOLDEN_SOURCE:
    data_folder = "ttm_data"
else:
    data_folder = "data_by_file"

# For llama embedding: load context
LLAMA_DATA = SimpleDirectoryReader(data_folder).load_data()

# For langchain indexing: load context
if GOLDEN_SOURCE:
    loader = TextLoader(file_path=data_folder + "/scraped_markdown.md")
else:
    loader = DirectoryLoader(data_folder, glob='**/*.json', show_progress=False, loader_cls=JSONLoader, loader_kwargs = {'jq_schema':'.full_content', 'text_content':False})

LANGCHAIN_DATA = loader.load()

# Full TTM papers for feeding into system prompt for golden reference q&a:
with open("ttm_data/scraped_markdown.md", "r") as f:
    SCRAPED_MARKDOWN = f.read()
