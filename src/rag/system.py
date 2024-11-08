from llama_index.core import (
    SimpleDirectoryReader,
    GPTVectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.llms.openai import OpenAI
from llama_index.core.settings import Settings
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings


class RAGSystem:
    """
    A class for a Retrieval Augmented Generation (RAG) system.
    """
    def __init__(
        self,
        model_name="gpt-4o-mini",
        num_output=512,
        max_chunk_overlap=20,
        temperature=0,
        data_dir='data/documents',
        persist_dir='data/index_storage',
        chunk_size=512,
        embedding_model="text-embedding-3-small"
    ):
        load_dotenv()
        self.data_dir = data_dir
        self.persist_dir = persist_dir

        # Load documents from the data directory
        self.documents = SimpleDirectoryReader(self.data_dir).load_data()

        # Set up the language model
        self.llm = OpenAI(
            temperature=temperature,
            model_name=model_name,
            max_tokens=num_output
        )

        # Configure settings instead of ServiceContext
        Settings.llm = self.llm
        Settings.chunk_size = chunk_size
        Settings.chunk_overlap = max_chunk_overlap


        Settings.embedding_model = OpenAIEmbeddings(model=embedding_model)
        self.index = None
        self.query_engine = None

    def build_index(self):
        # Build the index
        self.index = GPTVectorStoreIndex.from_documents(self.documents, show_progress=True)

    def save_index(self):
        # Save the index
        if self.index is not None:
            self.index.storage_context.persist(persist_dir=self.persist_dir)
        else:
            print("Index has not been built yet.")

    def load_index(self):
        # Load the index
        storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
        self.index = load_index_from_storage(storage_context)
        self.query_engine = self.index.as_query_engine()

    def query(self, query_text, return_type='sources'):
        """
        Query the RAG system.
        
        Args:
            query_text (str): The query text
            return_type (str): Type of response to return
                - 'sources': Only return retrieved source texts
                - 'response': Only return LLM response
                - 'all': Return both (default)
        """
        if self.query_engine is None:
            print("Index has not been loaded. Call load_index() first.")
            return None
        
        response = self.query_engine.query(query_text)
        
        if return_type == 'sources':
            return "\n".join([node.node.text for node in response.source_nodes])
        elif return_type == 'response':
            return str(response)
        else:  # 'all'
            sources = "\n".join([node.node.text for node in response.source_nodes])
            return {
                'response': str(response),
                'sources': sources
            }