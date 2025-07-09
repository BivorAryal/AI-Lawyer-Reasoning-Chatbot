"""
# Step 1: upload and load raw PDF
# Step 2: Create chunks
# Step 3: Setup embedding models (use deepseek R1 with ollama)
# Step 4: Index Documents (Store embedding in FAISS V.Store)
"""

from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter,Language
from langchain_huggingface import HuggingFaceEmbeddings

# Step 1: upload and load raw PDF
def load_pdf():
    loader = DirectoryLoader(
    path = 'books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
    ) 
    docs = loader.load() 
    return docs
docs = load_pdf()

# Step 2: Create chunks
def create_chucks(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, # character content
        chunk_overlap = 150, # tells you how many character will overlaps between 2 chunks
        separators=["\n\n", "\n", "(?<=\\. )", " ", ""]
        )
    text_chunks= splitter.split_documents(docs) # split document
    return text_chunks
text_chunks = create_chucks(docs)

# Step 3: Setup embedding models (use deepseek R1 with ollama)
def get_embedding_model():
  embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'
  )
  return embedding_model

embedding_model = get_embedding_model() # sentence transformer
test_embedding = embedding_model.embed_query(text_chunks[0].page_content)

# Step 4: Index Documents (Store embedding in FAISS V.Store)
from langchain_community.vectorstores import FAISS

DB_FAISS_PATH = "db_faiss"
def create_faiss_vectorstore(text_chunks, embedding_model):
    """Creates and saves FAISS vectorstore"""
    # Create and save in one step
    faiss_db = FAISS.from_documents(
        documents=text_chunks,
        embedding=embedding_model
    )
    faiss_db.save_local(DB_FAISS_PATH)
    print(f"✅ Saved FAISS index to {DB_FAISS_PATH}")
    return faiss_db
faiss_db = create_faiss_vectorstore(text_chunks, embedding_model)

print(f"""
Document Processing Complete:
- Pages loaded: {len(docs)}
- Text chunks: {len(text_chunks)} (sample: {len(text_chunks[0].page_content)} chars)
- Embedding dims: {len(test_embedding)} (sample: {test_embedding[:5]})
- Vectors stored: {faiss_db.index.ntotal}
""")