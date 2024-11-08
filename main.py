from src.rag import RAGSystem
from src.evaluation import evaluate_rag_system
from src.llms import gpt_4o_mini, ministral_3b, gemini_1_5_flash, llama_3_2_1B

def main():
    # Initialize the RAG system
    rag = RAGSystem(
        model_name="gpt-4o-mini",
        data_dir='data/documents',
        persist_dir='data/index_storage'
    )
    
    # Build the index
    rag.build_index()
    
    # # Save the index for future use
    rag.save_index()
    
    # Load the index (Only needed if you didn't build and saved the index before)
    rag.load_index()
    
    # Optional: Test a query
    response = rag.query("Taylor Swift birthplace?")
    print(response)

    # Test the RAG system
    average_score = evaluate_rag_system(
        rag_query_fn=rag.query,
        test_set_path="data/test.csv",
        batch_size=20,
        num_tests=100,
        evaluator_model=gemini_1_5_flash
    )

    print(f"Average Score: {average_score:.2f}%")

if __name__ == "__main__":
    main()