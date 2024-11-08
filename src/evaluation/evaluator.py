"""This is the testing function for the RAG system."""
import os
import pandas as pd
from typing import Callable, List, Tuple, Dict, Any, Optional
import csv
from datetime import datetime
from tqdm import tqdm
import asyncio
from src.llms import ministral_3b

EVALUATION_FORMAT = """
Score (0 or 1): [your score]
Explanation: [your explanation]
"""
DEFAULT_SCORE = "0"
DEFAULT_EXPLANATION = "Failed to parse evaluation"

SCORE_KEYWORDS = ['score', 'score:']
EXPLANATION_KEYWORDS = ['explanation', 'explanation:']

def parse_evaluation(evaluation_text: str) -> Tuple[str, str]:
    """Parse the evaluation text and return score and explanation.

    Args:
        evaluation_text: Raw evaluation text to parse

    Returns:
        Tuple[str, str]: Score and explanation
    """
    if not evaluation_text:
        return DEFAULT_SCORE, DEFAULT_EXPLANATION

    try:
        lines = [line.lower().strip() for line in evaluation_text.strip().split('\n')]
        score = DEFAULT_SCORE
        explanation = DEFAULT_EXPLANATION

        for line in lines:
            if any(keyword in line for keyword in SCORE_KEYWORDS):
                score_text = ''.join(filter(str.isdigit, line))
                if score_text in ['0', '1']:
                    score = score_text
                    
            if any(keyword in line for keyword in EXPLANATION_KEYWORDS):
                try:
                    explanation = line.split(':', 1)[1].strip()
                    if explanation:
                        explanation = explanation.capitalize()
                except IndexError:
                    continue
                    
        if score not in ['0', '1']:
            print(f"Invalid score format detected. Raw text: {evaluation_text}")
            score = DEFAULT_SCORE
            
        if explanation == DEFAULT_EXPLANATION:
            print(f"Failed to find explanation. Raw text: {evaluation_text}")

        return score, explanation

    except Exception as e:
        print(f"Error parsing evaluation: {e}")
        print(f"Raw evaluation text:\n{evaluation_text}")
        return DEFAULT_SCORE, f"Error in evaluation parsing: {str(e)}"

async def evaluate_batch(
    questions: List[Dict[str, Any]],
    rag_query_fn: Callable[[str], str],
    evaluator_model: Callable[[str], str],
    batch_size: int = 5,
    pbar: Optional[tqdm] = None
) -> List[Dict[str, Any]]:
    """Process a batch of questions concurrently."""
    
    # Initialize results list
    results = []
    
    async def process_question(row):
        try:
            # Get RAG response
            rag_response = rag_query_fn(row['question'])
            
            eval_prompt = f"""Evaluate if the RAG system's response includes the expected answer.

Question: {row['question']}
Expected Answer: {row['answer']}
RAG System Response: {rag_response}

Instructions:
1. Compare the RAG response with the expected answer
2. Score 1 if the RAG response includes the expected answer
3. Score 0 if it does not include the expected answer
4. Provide a brief explanation for your score

You must respond using EXACTLY this format:
Score: [0 or 1]
Explanation: [your brief explanation]

Do not include any other text in your response."""
            
            # Get evaluation
            evaluation = evaluator_model(eval_prompt)
            score, explanation = parse_evaluation(evaluation)
            
            return {
                'question_id': row['id'],
                'question_num': row['question_num'],
                'question': row['question'],
                'expected_answer': row['answer'],
                'rag_response': rag_response,
                'score': score,
                'explanation': explanation,
                'source_file': row['filename']
            }
        except Exception as e:
            return {
                'question_id': row['id'],
                'question_num': row['question_num'],
                'question': row['question'],
                'expected_answer': row['answer'],
                'rag_response': 'ERROR',
                'score': '0',
                'explanation': f'Error during evaluation: {str(e)}',
                'source_file': row['filename']
            }

    # Process questions in batches
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]
        batch_results = await asyncio.gather(
            *[process_question(row) for row in batch]
        )
        results.extend(batch_results)
        
        if pbar:
            pbar.update(len(batch))
            pbar.set_description(f"Processing batch {i//batch_size + 1}/{len(questions)//batch_size + 1}")
    
    return results

def evaluate_rag_system(
    rag_query_fn: Callable[[str], str],
    test_set_path: str,
    output_path: Optional[str] = None,
    batch_size: int = 5,
    num_tests: Optional[int] = None,
    evaluator_model: Callable[[str], str] = ministral_3b
) -> float:
    """
    Evaluate RAG system using a test set and save results.
    
    Args:
        rag_query_fn: Function that takes a question and returns RAG response
        test_set_path: Path to CSV file containing test questions
        output_path: Path to save evaluation results (optional)
        batch_size: Number of questions to process concurrently
        num_tests: Number of tests to run (optional, runs all tests if None)
        evaluator_model: Model to use for evaluation
        
    Returns:
        float: Average score across all evaluations
        
    Raises:
        FileNotFoundError: If test_set_path doesn't exist
        ValueError: If batch_size < 1 or num_tests < 1
    """
    # Add input validation
    if not os.path.exists(test_set_path):
        raise FileNotFoundError(f"Test set file not found: {test_set_path}")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if num_tests is not None and num_tests < 1:
        raise ValueError("num_tests must be at least 1")
        
    # Read test set
    test_df = pd.read_csv(test_set_path)
    if num_tests is not None:
        test_df = test_df.head(num_tests)
    questions = test_df.to_dict('records')
    
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"rag_evaluation_results_{timestamp}.csv"
    
    pbar = tqdm(total=len(questions), desc="Evaluating questions")
    
    try:
        results = asyncio.run(
            evaluate_batch(
                questions,
                rag_query_fn,
                evaluator_model,
                batch_size,
                pbar
            )
        )
        
        # Calculate average score
        total_score = 0
        valid_scores = 0
        for result in results:
            try:
                score = float(result['score'])
                total_score += score
                valid_scores += 1
            except (ValueError, TypeError):
                continue
                
        total_score = sum(int(result['score']) for result in results if result['score'] in ['0', '1'])
        valid_scores = sum(1 for result in results if result['score'] in ['0', '1'])
        average_score = (total_score / valid_scores) * 100 if valid_scores > 0 else 0.0
        
        output_path = f"results/{output_path}"

        if not os.path.exists("results"):
            os.makedirs("results")

        # Save results
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            dict_writer = csv.DictWriter(f, fieldnames=results[0].keys())
            dict_writer.writeheader()
            dict_writer.writerows(results)
        
        pbar.close()
        print(f"\nEvaluation completed. Results saved to: {output_path}")
        
        return average_score
        
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        raise
    finally:
        pbar.close()