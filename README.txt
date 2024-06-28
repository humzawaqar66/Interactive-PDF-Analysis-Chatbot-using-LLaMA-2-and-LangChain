Interactive PDF Analysis Chatbot using LLaMA 2 and LangChain

Overview
This project involves the development of an advanced chatbot utilizing LLaMA 2 and LangChain for interactive PDF analysis. The chatbot leverages robust natural language processing (NLP) techniques to load documents, split text, and generate embeddings, enabling efficient information retrieval and detailed data analysis. OpenAI's transformers and custom prompt engineering are employed to generate contextually accurate responses for various financial queries. The project also enhances user interaction through dynamic text generation and visual representation of data using PyPDF2 and pdf2image.

Features
Interactive PDF Analysis: Load and analyze PDF documents for detailed information extraction.
Advanced NLP Techniques: Efficient document loading, text splitting, and embedding generation.
Accurate Query Responses: Utilize OpenAI's transformers and custom prompt engineering for precise answers to financial queries.
Dynamic Text Generation: Enhance user interaction with real-time text generation.
Data Visualization: Visualize data from PDF documents using PyPDF2 and pdf2image.
Installation
Prerequisites
Python 3.10+
CUDA-enabled GPU (if using GPU for acceleration)
Install Dependencies
Clone the repository:

sh
Copy code
git clone https://github.com/yourusername/interactive-pdf-chatbot.git
cd interactive-pdf-chatbot
Install required packages:

sh
Copy code
pip install -r requirements.txt
Usage
Download the necessary PDF documents:

sh
Copy code
!mkdir pdfs
!gdown 1v-Rn1FVU1pLTAQEgm0N9oB6cExMoebZr -O pdfs/tesla-earnings-report.pdf
!gdown 1Xc890jrQvCExAkryVWAttsv1DBLdVefN -O pdfs/nvidia-earnings-report.pdf
!gdown 1Epz-SQ3idPpoz75GlTzzomag8gplzLv8 -O pdfs/meta-earnings-report.pdf
Run the Jupyter Notebook to load the model and analyze the PDFs:

sh
Copy code
jupyter notebook chat_with_multiple_pdfs_by_llama_2_and_langchain.ipynb
Execute the notebook cells to interact with the chatbot and retrieve information from the PDF documents.

Code Snippets
Model Loading and Initialization
python
Copy code
import torch
from auto_gptq import AutoGPTQForCausalLM
from transformers import AutoTokenizer, TextStreamer, pipeline

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def load_model_llama(path, device):
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    model = AutoGPTQForCausalLM.from_quantized(path, device=device, use_safetensors=True, trust_remote_code=True, quantize_config=None)
    return tokenizer, model

model_name_or_path = "TheBloke/Llama-2-7B-Chat-GPTQ"
tokenizer, model = load_model_llama(model_name_or_path, DEVICE)
Document Loading and Embedding Generation
python
Copy code
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

pdf_folder_path = "pdfs"

loader = PyPDFDirectoryLoader(pdf_folder_path)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=400)
docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large", model_kwargs={"device": DEVICE})
db = Chroma.from_documents(docs, embeddings)
Querying the Chatbot
python
Copy code
from langchain.chains import RetrievalQA
from langchain import HuggingFacePipeline, PromptTemplate

DEFAULT_SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant..."

def generate_prompt(prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    return f"""
[INST] <>
{system_prompt}
<>

{prompt} [/INST]
""".strip()

text_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    temperature=0,
    top_p=0.95,
    repetition_penalty=1.15,
    streamer=TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True),
)

llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0})

SYSTEM_PROMPT = "Use the following pieces of context to answer the question at the end..."

template = generate_prompt(
    """
{context}

Question: {question}
""",
    system_prompt=SYSTEM_PROMPT,
)

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
)

result = qa_chain("What is the per share revenue for Meta during 2023?")
print(result["source_documents"][0].page_content)
Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or suggestions.

License
This project is licensed under the MIT License. See the LICENSE file for details.
