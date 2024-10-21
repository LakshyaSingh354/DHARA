from legal_research_assistant import DHARA
import os

# app = modal.App("dhara-retrieval")

# image = (
#     modal.Image.debian_slim(python_version="3.10")
#     .pip_install(['torch', 'torchvision', 'transformers', 'sentence-transformers', 'huggingface_hub'])
#     .pip_install('torchaudio')
#     .pip_install(['langchain', 'langchain_core', 'langchain_community', 'langchain_huggingface', 'einops', 'accelerate', 'bitsandbytes', 'scipy'])
#     .pip_install(['xformers', 'sentencepiece', 'docarray'])
# )

# @app.cls(gpu="any", image=image)
# class Model:
#     @modal.build()
#     def __init__(self):
#         print("Loading model...")
#         self.assistant = DHARA(
#             hf_token=os.environ["HF_TOKEN"], 
#             model_id="mistralai/Mistral-7B-Instruct-v0.2",
#             embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"
#         )
#         print("Model loaded successfully")

#     def get_assistant(self):
#         return self.assistant


# # Ensure that the assistant is loaded only once and reused
# model_instance = None


# @app.function(gpu="any", image=image, mounts=[modal.Mount.from_local_dir("data_summary", remote_path="/root/data_summary")])
# @modal.web_endpoint()
# def get_response(question: str):
#     global model_instance
#     if model_instance is None:
#         model_instance = Model()  # Initialize model class

#     # Use the already-loaded assistant for inference
#     assistant = model_instance.get_assistant()
#     # Load documents
#     documents = assistant.load_text_files('/root/data_summary')
#     print("Documents loaded successfully")

#     # Run the query
#     response, retrieved_docs = assistant.run_query(documents, question)

#     return {"response": response}

assistant = DHARA()
    
# Specify the directory where your text files are stored
directory_path = 'data_summary'

# Load the text files from the directory
documents = assistant.load_text_files(directory_path)

# A question about arbitration law in india
question = "A construction company entered into a contract with a government agency for the construction of a bridge. The contract specified that the arbitration proceedings would be held in Mumbai. However, the government agency filed an application under Section 34 of the Arbitration and Conciliation Act, 1996 before the Delhi High Court challenging the arbitral award. The construction company argues that the Delhi High Court lacks jurisdiction as the seat of arbitration was designated as Mumbai. Determine the relevant legal principles and jurisdictional considerations in this case."

# Run the query using the loaded documents
response, retrieved_docs = assistant.run_query(documents, question)

# Output the results
print("Generated Response:", response)
# print("Retrieved Documents:", retrieved_docs)