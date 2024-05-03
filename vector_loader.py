#import Essential dependencies

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


#create a new file named vectorstore in your current directory.
if __name__=="__main__":
        DB_FAISS_PATH = 'vectorstore/db_faiss'
        loader=PyPDFLoader("./random machine learing pdf.pdf")
        docs=loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
        embeddings_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embed_model = HuggingFaceEmbeddings(model_name=embeddings_model_name)

        # vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings(api_key="Enter your api key"))
        vectorstore = FAISS.from_documents(documents=splits, embedding=HuggingFaceEmbeddings(model_name=embeddings_model_name))
        vectorstore.save_local(DB_FAISS_PATH)
        print("..")