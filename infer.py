import ollama
# get arguments
import argparse
import os
import json
import sys
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores.qdrant import Qdrant
# from langchain_community.vectorstores import qdrant


# from langchain_community.llms.ollama import Ollama


username = sys.argv[1]
file_path = sys.argv[2]
print(username)
print(file_path)
system_prompt = """You are an Expert Doctor who analyses medical reports of patients, summarises it briefly using list and sections such as Overall Sumarry, Discharge Summary, Diaognosis, Treatment, Lab Reports and vitals wherever apply.
Use the following pieces of retrieved context to summarize the report. If you don't infer anything important, just say Everythin's Normal. 
Try to answer in 400 words maximum and keep the answer concise.
Give answer in Markdown Format.
"""


ollama_client = ollama.Client(host="http://ollama:11434")


embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
with open(file_path, "r") as f:
    data = json.load(f)

data_summary = []

for d in data:
    resp = ollama_client.generate(
        # model="mistral:latest",
        # prompt= "Summarize the following medical report: " + d,
        model="stablelm2:latest",
        prompt=system_prompt+ "Summarize the following medical report: " + d,
        system=system_prompt,
        )
    print(resp['response'])
    data_summary.append(resp["response"])




url = "http://healthcare-rag-qdrant-store:6333"
qdrant = Qdrant.from_texts(
    data_summary,
    embeddings,
    url=url,
    prefer_grpc=False,
    collection_name=username,
    force_recreate=True,
)


print(qdrant.search("Name of the Hospital?", search_type="similarity"))