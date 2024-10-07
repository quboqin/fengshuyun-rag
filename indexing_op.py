#!/usr/bin/env python
# coding: utf-8

# # Rag From Scratch: Indexing
# 
# ![Screenshot 2024-03-25 at 8.23.02 PM.png](attachment:79718808-a305-4a64-8881-086508277324.png)
# 
# ## Preface: Chunking
# 
# We don't explicity cover document chunking / splitting.
# 
# For an excellent review of document chunking, see this video from Greg Kamradt:
# 
# https://www.youtube.com/watch?v=8OJC21T2SL4
# 
# ## Enviornment
# 
# `(1) Packages`

# In[ ]:


get_ipython().system(' pip install langchain_community tiktoken langchain-openai langchainhub chromadb langchain youtube-transcript-api pytube')


# `(2) LangSmith`
# 
# https://docs.smith.langchain.com/

# In[ ]:


import os
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = <your-api-key>


# `(3) API Keys`

# In[ ]:


os.environ['OPENAI_API_KEY'] = <your-api-key>


# ## Part 12: Multi-representation Indexing
# 
# Flow: 
# 
#  ![Screenshot 2024-03-16 at 5.54.55 PM.png](attachment:3eee1e62-6f49-4ca5-9d9b-16df2b6ffe06.png)
# 
# Docs:
# 
# https://blog.langchain.dev/semi-structured-multi-modal-rag/
# 
# https://python.langchain.com/docs/modules/data_connection/retrievers/multi_vector
# 
# Paper:
# 
# https://arxiv.org/abs/2312.06648

# In[1]:


from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()

loader = WebBaseLoader("https://lilianweng.github.io/posts/2024-02-05-human-data-quality/")
docs.extend(loader.load())


# In[2]:


import uuid

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

chain = (
    {"doc": lambda x: x.page_content}
    | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
    | ChatOpenAI(model="gpt-3.5-turbo",max_retries=0)
    | StrOutputParser()
)

summaries = chain.batch(docs, {"max_concurrency": 5})


# In[4]:


from langchain.storage import InMemoryByteStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever

# The vectorstore to use to index the child chunks
vectorstore = Chroma(collection_name="summaries",
                     embedding_function=OpenAIEmbeddings())

# The storage layer for the parent documents
store = InMemoryByteStore()
id_key = "doc_id"

# The retriever
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)
doc_ids = [str(uuid.uuid4()) for _ in docs]

# Docs linked to summaries
summary_docs = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(summaries)
]

# Add
retriever.vectorstore.add_documents(summary_docs)
retriever.docstore.mset(list(zip(doc_ids, docs)))


# In[5]:


query = "Memory in agents"
sub_docs = vectorstore.similarity_search(query,k=1)
sub_docs[0]


# In[6]:


retrieved_docs = retriever.get_relevant_documents(query,n_results=1)
retrieved_docs[0].page_content[0:500]


# Related idea is the [parent document retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever).

# ## Part 13: RAPTOR
# 
# Flow:
# 
# ![Screenshot 2024-03-16 at 6.16.21 PM.png](attachment:5ccfe50d-d22e-402b-86f6-b3afb0f06088.png)
# 
# Deep dive video:
# 
# https://www.youtube.com/watch?v=jbGchdTL7d0
# 
# Paper:
# 
# https://arxiv.org/pdf/2401.18059.pdf
# 
# Full code:
# 
# https://github.com/langchain-ai/langchain/blob/master/cookbook/RAPTOR.ipynb

# ## Part 14: ColBERT
# 
# RAGatouille makes it as simple to use ColBERT. 
# 
# ColBERT generates a contextually influenced vector for each token in the passages. 
# 
# ColBERT similarly generates vectors for each token in the query.
# 
# Then, the score of each document is the sum of the maximum similarity of each query embedding to any of the document embeddings:
# 
# See [here](https://hackernoon.com/how-colbert-helps-developers-overcome-the-limits-of-rag) and [here](https://python.langchain.com/docs/integrations/retrievers/ragatouille) and [here](https://til.simonwillison.net/llms/colbert-ragatouille).

# In[ ]:


get_ipython().system(' pip install -U ragatouille')


# In[9]:


from ragatouille import RAGPretrainedModel
RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")


# In[10]:


import requests

def get_wikipedia_page(title: str):
    """
    Retrieve the full text content of a Wikipedia page.

    :param title: str - Title of the Wikipedia page.
    :return: str - Full text content of the page as raw string.
    """
    # Wikipedia API endpoint
    URL = "https://en.wikipedia.org/w/api.php"

    # Parameters for the API request
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
    }

    # Custom User-Agent header to comply with Wikipedia's best practices
    headers = {"User-Agent": "RAGatouille_tutorial/0.0.1 (ben@clavie.eu)"}

    response = requests.get(URL, params=params, headers=headers)
    data = response.json()

    # Extracting page content
    page = next(iter(data["query"]["pages"].values()))
    return page["extract"] if "extract" in page else None

full_document = get_wikipedia_page("Hayao_Miyazaki")


# In[11]:


RAG.index(
    collection=[full_document],
    index_name="Miyazaki-123",
    max_document_length=180,
    split_documents=True,
)


# In[12]:


results = RAG.search(query="What animation studio did Miyazaki found?", k=3)
results


# In[13]:


retriever = RAG.as_langchain_retriever(k=3)
retriever.invoke("What animation studio did Miyazaki found?")

