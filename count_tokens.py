import tiktoken

# open markdown file
with open('scraped_markdown.md', 'r') as f:
    scraped_markdown = f.read()


def count_tokens(text, model="gpt-4o"):
    # Load the tokenizer for the model
    encoding = tiktoken.encoding_for_model(model)
    # Tokenize the text
    tokens = encoding.encode(text)
    return len(tokens)


print(count_tokens(scraped_markdown))