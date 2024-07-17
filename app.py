api_key = "52142f3b-4ea1-4692-b5dc-8d90dc0df87f"
from langchain_community.retrievers import PineconeHybridSearchRetriever
import os
from pinecone import Pinecone, ServerlessSpec

cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'

spec = ServerlessSpec(cloud=cloud, region=region)
index_name = "hybride-search-pinecone"
pc = Pinecone(api_key=api_key)
# check if index already exists (it shouldn't if this is first time)
if index_name not in pc.list_indexes().names():
    # if does not exist, create index
    pc.create_index(
        index_name,
        dimension=1536,  # dimensionality of text-embedding-ada-002
        metric='dotproduct',
        spec=spec
    )
# connect to index
index = pc.Index(index_name)
print(index)
# view index stats
print(index.describe_index_stats())
import os
from dotenv import load_dotenv
load_dotenv()
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()