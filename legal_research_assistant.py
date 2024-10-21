import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
import google.generativeai as genai
from py2neo import Graph

class DHARA:
    def __init__(self):
        genai.configure(api_key=os.environ['GEMINI_API_KEY'])
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.parser = StrOutputParser()
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.base_prompt = '''
            You are a helpful, respectful, and honest legal research assistant. Answer the question using the provided context.

            Your goal is to provide accurate legal research, including relevant case law, statutory interpretation, and insights into legal principles and precedents. Always maintain a focus on legal accuracy and ethical standards, and answer only legal queries. If the query is non-legal, politely state that you can only assist with legal queries.

            Answer the question based on the context provided. If the context lacks relevant information, answer based on your legal knowledge, but clearly mention the absence of relevant context. Use case summaries and information inferred from legal principles to form your answer.

            In the context, you will find both case summaries and inferred insights from a legal knowledge graph (KG), which includes relationships between:

                •	Case (legal cases)
                •	Statute (legal statutes and provisions)
                •	Issue (key legal issues)
                •	Precedent (legal precedents)
                •	Doctrine (legal doctrines and principles)
                •	Jurisdiction (legal jurisdictions)

            These relationships are based on citations, applicable statutes, doctrines, and jurisdictional relevance. Seamlessly integrate this information into your answer without explicitly mentioning the KG, but ensure it enriches your reasoning. Only cite and include the most relevant cases and information that directly apply to the legal question.

            At the end of your response, briefly list the relevant parties involved in the cases you cited, along with the dates, don't mention it if other cases are not relevant. Ensure this information is taken from the context provided.
        '''

        self.graph = Graph(os.environ['NEO4J_URI'], auth=(os.environ['NEO4J_USER'], os.environ['NEO4J_PASSWORD']))
        self.cypher_query_model = genai.GenerativeModel('gemini-1.5-flash')

    def generate_cypher_query(self, user_query: str):
        """
        Generates a Cypher query based on the user's input query.
        Placeholder for now: A model can be fine-tuned later to improve this process.
        """
        # Example prompt template for query generation
        prompt = f"""Generate a Cypher query to retrieve relevant cases from a legal knowledge graph for the following legal query. There should be nothing else in your response other than the Cypher query
                        The KG consists of nodes such as:
                        - Case (representing legal cases)
                        - Statute (representing legal statutes and provisions)
                        - Issue (representing key legal issues)
                        - Precedent (representing legal precedents cited)
                        - Doctrine (representing legal doctrines and principles)
                        - Jurisdiction (representing legal jurisdictions)

                        These nodes are connected by relationships like [:CITES] for cases citing other cases, [:PERTAINS_TO] for cases involving certain statutes or doctrines, [:WITHIN] for jurisdictions, [:DEALS WITH] for cases dealing with certain key issues, and [:FALLS UNDER] for cases falling under a certain legal doctrine.
                        The query should be really simple otherwise the results are coming empty from the graph. Keep it simple and focused on the query.

                        Absolutely do not include anything else, not even a single extra symbol or word. Just the Cypher query in plain text. Don't even include something like '''cypher that you can include at the start of the response, I will be transeffering your response directly to the graph query.

                        :\n\n{user_query}"""
        
        # Use the placeholder LLM model for generating Cypher queries
        if self.cypher_query_model:
            cypher_query = self.cypher_query_model.generate_content(prompt)
        else:
            # For now, generate a basic query that searches for matching cases
            cypher_query = f"MATCH (c:Case) WHERE c.title CONTAINS '{user_query}' RETURN c"

        # Extract the Cypher query from the generated response
        cypher_list = cypher_query.text.split("\n")
        query = " ".join(cypher_list).replace("```cypher ", "").replace("```", "").replace('cypher', '').strip()
        print("Generated Cypher Query:", query)
        return query

    def query_knowledge_graph(self, user_query: str):
        """
        Query the Neo4j knowledge graph using a Cypher query based on user input.
        """
        cypher_query = self.generate_cypher_query(user_query)
        result = self.graph.run(cypher_query).data()
        return result
    
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

    def split_documents_into_chunks(self, documents):
        """Split documents into smaller chunks for efficient retrieval."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=100000,  # Set the size of each chunk
            chunk_overlap=200  # Overlap between chunks for context preservation
        )
        chunks = splitter.split_documents(documents)
        return chunks

    def create_document_embeddings(self, documents):
        """Generate document embeddings using HuggingFace Embeddings."""
        document_texts = [doc.page_content for doc in documents]
        return self.embeddings.embed_documents(document_texts)

    def build_vectorstore(self, documents):
        """Build a FAISS vector store for document retrieval."""
        # First split documents into chunks
        chunked_documents = self.split_documents_into_chunks(documents)
        
        # Generate embeddings for the chunked documents
        vector_store = FAISS.from_documents(chunked_documents, self.embeddings)
        return vector_store

    def inspect(self, state):
        """Print the state passed between Runnables in a langchain and pass it on."""
        print(state)
        return state

    def create_chain(self, retriever, prompt, question):
        """Create and execute the retrieval chain."""
        # Retrieve documents based on the question
        retrieved_docs = itemgetter("question") | retriever
        kg_results = self.query_knowledge_graph(question)
        # Create a full prompt by concatenating the context and the question
        print("KG Result: ", str(kg_results))
        context = retrieved_docs.invoke({"question": question})
        full_prompt = f"{prompt}\n\nContext: {context}\nKG Results: {str(kg_results)}\nQuestion: {question}"

        # Generate response using the GenAI model
        response = self.model.generate_content(full_prompt)

        return response, retrieved_docs
    
    def run_query(self, documents, question):
        """Run a query against loaded documents using the RAG pipeline."""
        # Create embeddings and vector store for retrieval
        vectorstore = self.build_vectorstore(documents)
        retriever = vectorstore.as_retriever()
        
        # Run the chain with a given question
        response, retrieved_docs = self.create_chain(retriever, self.base_prompt, question)
        return response.text, retrieved_docs.invoke({"question": question})