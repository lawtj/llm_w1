from langchain_community.document_loaders import TextLoader, JSONLoader, DirectoryLoader


# Full TTM papers for feeding into system prompt for golden reference q&a:
with open("ttm_data/scraped_markdown.md", "r") as f:
    TTM = f.read()

# Steroid summaries for feeding into system prompt for golden reference q&a:
with open("data/steroid_summaries.md", "r") as f:
    STEROID_SUMMARIES = f.read()

# Dialysis summaries for feeding into system prompt for golden reference q&a:
with open("data/dialysis_summaries.md", "r") as f:
    DIALYSIS_SUMMARIES = f.read()

GOLDEN_QUESTIONS = {
    'TTM':[
        "What is the optimal duration of TTM?",
        "What is the optimal temperature for TTM?",
        "What are the current controversies in the evidence for use of TTM?",
        "When should TTM be initiated?",
        "What is the indication for initiation of TTM?"
    ],
    'STEROID':[
        "What is an evidence based dose of steroids for sepsis?",
        "For what conditions is there evidence for the use of steroids in the ICU?",
        "What are the indications for the use of steroids in the ICU?",
        "What is the evidence for the use of steroids in respiratory conditions?",
        "When should steroids be initiated in a patient with sepsis?",
    ],
    'DIALYSIS':[
        "When should dialysis be initiated in a patient with acute kidney injury?",
        "Is early or late initiation of dialysis associated with better outcomes in the ICU?",
        "What are the current controversies in the use of dialysis in the ICU?",
        "What are the indications for the use of dialysis in the ICU?",
    ]
}