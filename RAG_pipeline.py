'''
Step 1: Setup LLM
Step 2: Retrive Docs
Step 3: Question Answer

'''

# Step 1: Setup LLM
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

def load_llm():
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        task="text-generation",
        max_new_tokens=150,
        temperature=1.2,
        do_sample=True,
        top_p=0.95,
        repetition_penalty=1.2
        )
    model = ChatHuggingFace(llm=llm)
    return model
model = load_llm()
parser = StrOutputParser()

# Step 2: Retrive Docs
from vector_database import faiss_db

def retrieve_docs(query):
    return faiss_db.similarity_search(query)

def get_context(documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    return context

# Step 3: Question Answer

#create system prompt
custom_prompt_template = """
You must ONLY use the following context to answer the question. 
If the context doesn't contain the answer, say "I don't know".

Context: {context}
Question: {question}
Answer:
"""

from langchain_core.prompts import ChatPromptTemplate
def answer_query(documents, model, query):
    context = get_context(documents)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    
    chain = prompt | model | parser
    
    return chain.invoke({"question": query, "context": context})

# Checking if everrything works

questions = "How would you prioritize features for a new product launch?"
retrieved_docs = retrieve_docs(questions)
print("AI Lawyer: ", answer_query(retrieved_docs,
                                   model,
                                   questions))