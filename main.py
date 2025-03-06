#loading 
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from langchain_community.document_loaders import PyPDFLoader

file_path = "budget_speech.pdf"

loader = PyPDFLoader(file_path)
docs = loader.load()

#chunking

from langchain_text_splitters import RecursiveCharacterTextSplitter


text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)
chunks = text_splitter.split_documents(docs)

#print(type(chunks))

#embedding
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# We need to get the text content from chunks before embedding
texts = [str(doc.page_content) for doc in chunks]  # Convert to string explicitly
embedded_vectors = embeddings.embed_documents(texts)

#print(embedded_vectors)

#vectorstore

from langchain_community.vectorstores import FAISS

faissdb = FAISS.from_texts(texts, embedding=embeddings)

# # Save FAISS index
faissdb.save_local("faissdb")

print("FAISS vector store saved successfully as 'faissdb'")







# Part2

# retrieving from faiss



# Load the FAISS index
faissdb = FAISS.load_local("faissdb", embeddings,allow_dangerous_deserialization=True)

# Create a retriever
retriever = faissdb.as_retriever(search_kwargs={"k": 2})

#retrieved_chunks = retriever.invoke("how much money was allocated for the health sector in the budget speech?")
#print(docs)

query = input("Enter your query: ")
retrieved_chunks = retriever.invoke(query)

# Combine chunks into a single context string
context = "\n\n".join([chunk.page_content for chunk in retrieved_chunks])

# Define the query


# Create the final prompt
prompt = f"""
You are an expert AI assistant helping users extract information from retrieved documents.

Context:
{context}

Based on the provided context, answer the following question:
Question: {query}

Instructions:
- If the answer is found in the context, provide a detailed yet concise response.
- If the context lacks relevant information, state that the answer is not available.
- Do not generate information outside the given context.
- Maintain clarity and professionalism in your response.

Answer:
"""
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
ai_msg = llm.invoke(prompt)
print(ai_msg.content)