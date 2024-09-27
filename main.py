from legal_research_assistant import DHARA
import modal
import os

app = modal.App("dhara-retrieval")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(['torch', 'torchvision', 'transformers', 'sentence-transformers', 'huggingface_hub'])
    .pip_install('torchaudio')
    .pip_install(['langchain', 'langchain_core', 'langchain_community', 'langchain_huggingface', 'einops', 'accelerate', 'bitsandbytes', 'scipy'])
    .pip_install(['xformers', 'sentencepiece', 'docarray'])
)

@app.cls(gpu="any", image=image)
class Model:
    @modal.build()
    def __init__(self):
        print("Loading model...")
        self.assistant = DHARA(
            hf_token=os.environ["HF_TOKEN"], 
            model_id="mistralai/Mistral-7B-Instruct-v0.2",
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        print("Model loaded successfully")

    def get_assistant(self):
        return self.assistant


# Ensure that the assistant is loaded only once and reused
model_instance = None


@app.function(gpu="any", image=image, mounts=[modal.Mount.from_local_dir("data_summary", remote_path="/root/data_summary")])
@modal.web_endpoint()
def get_response(question: str):
    global model_instance
    if model_instance is None:
        model_instance = Model()  # Initialize model class

    # Use the already-loaded assistant for inference
    assistant = model_instance.get_assistant()
    # Load documents
    documents = assistant.load_text_files('/root/data_summary')
    print("Documents loaded successfully")

    # Run the query
    response, retrieved_docs = assistant.run_query(documents, question)

    return {"response": response}