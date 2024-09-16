from operator import itemgetter
import os
import torch
from huggingface_hub import login
from transformers import BitsAndBytesConfig
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch

class DHARA:
    def __init__(self, hf_token: str, model_id: str, embedding_model_name: str):
        # Login to Hugging Face
        login(hf_token)
        
        # Set up quantization for the model
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
        )
        
        # Initialize the LLM model pipeline
        self.llm = HuggingFacePipeline.from_model_id(
            model_id=model_id,
            task="text-generation",
            pipeline_kwargs=dict(
                max_new_tokens=512,
                do_sample=False,
                repetition_penalty=1.03,
            ),
            model_kwargs={"quantization_config": self.quantization_config},
        )
        self.chat_model = ChatHuggingFace(llm=self.llm)

        # Initialize the prompt template
        self.prompt_template = PromptTemplate.from_template("""
            You are a helpful, respectful, and honest legal research assistant. Answer the question using the context given to you.

            Your goal is to provide accurate legal research, relevant case law, statutory interpretation, and insights into legal principles and precedents, always maintaining a focus on legal accuracy and ethical standards. Only answer to legal queries. On any queries that are not legal just say that you are a legal research assistant and can only help with legal queries.

            Make sure to keep your response short and to the point.

            Answer the question based on the context below.

            Context: {context}

            Question: {question}

            """)

        # Initialize embeddings
        self.embedding_model_name = embedding_model_name
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        
        # Set up the parser for output
        self.parser = StrOutputParser()

    def load_text_files(self, directory_path: str):
        """Load .txt documents from a directory."""
        documents = []
        for filename in os.listdir(directory_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(directory_path, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    doc = Document(page_content=content, metadata={"source": file_path})
                    documents.append(doc)
        return documents

    def create_document_embeddings(self, documents):
        """Generate document embeddings using HuggingFace Embeddings."""
        document_texts = [doc.page_content for doc in documents]
        return self.embeddings.embed_documents(document_texts)

    def build_vectorstore(self, documents):
        """Build a vector store for document retrieval."""
        return DocArrayInMemorySearch.from_documents(documents, embedding=self.embeddings)
    
    def inspect(state):
        """Print the state passed between Runnables in a langchain and pass it on"""
        print(state)
        return state

    def create_chain(self, retriever, prompt, question):
        """Create and execute the retrieval chain."""
        retrieved_docs = itemgetter("question") | retriever
        chain = (
            {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
            | prompt
            | self.chat_model
            | self.parser
        )
        return chain.invoke({"question": question}), retrieved_docs
    
    def run_query(self, documents, question):
        """Run a query against loaded documents using the RAG pipeline."""
        # Create embeddings and vector store for retrieval
        vectorstore = self.build_vectorstore(documents)
        retriever = vectorstore.as_retriever()
        
        # Run the chain with a given question
        response, retrieved_docs = self.create_chain(retriever, self.prompt_template, question)
        return response.split("[/INST]")[-1].strip(), retrieved_docs.invoke({"question": question})