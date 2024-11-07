"""This script processes all documents in a folder, generates multiple test questions and answers for each, and saves the results to a CSV file."""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import pandas as pd
import os
import glob
import concurrent.futures
import time
from tqdm import tqdm

class TestGenerator:
    def __init__(self, model="gpt-4o-mini", temperature=0):
        load_dotenv()
        self.llm = ChatOpenAI(model=model, temperature=temperature)

    def create_tests(self, context: str, id: str, num_tests: int = 3):
        """Create multiple test sets for the RAGCheck project.
        Args:
            context: The context to create the test set from.
            id: The id of the test set.
            num_tests: Number of question-answer pairs to generate.
        Returns:
            str: The LLM's response containing the questions and answers.
        """
        prompt_template = PromptTemplate.from_template("""
        task:
        You are given a task to create a test set for a RAGCheck project.
        Your goal is to create {num_tests} question and answer pairs based on the provided context.

        requirements:
        - The questions must be answerable using the context.
        - The answers should be directly from the context.
        - Both questions and answers should match the context's language.

        context:
        {context}

        id: {id}

        output_format:
        Please provide exactly {num_tests} question-answer pairs in the following format:

        id: {{id}}
        question_1: <question>
        answer_1: <answer>
        question_2: <question>
        answer_2: <answer>
        ...etc up to question_{num_tests}/answer_{num_tests}

        IMPORTANT: Only output the answer in the specified format.
        """)
        chain = prompt_template | self.llm

        # Implement retry logic with exponential backoff
        max_retries = 5
        retry_delay = 2  # initial delay in seconds
        for attempt in range(max_retries):
            try:
                response = chain.invoke({"context": context, "id": id, "num_tests": num_tests})
                return response
            except Exception as e:
                error_message = str(e)
                if 'rate_limit_exceeded' in error_message:
                    # Parse the recommended retry time from the error message
                    wait_time = 60  # default wait time in seconds
                    if 'Please try again in' in error_message:
                        try:
                            wait_time_str = error_message.split('Please try again in')[1].split('.')[0]
                            wait_time = float(wait_time_str.strip())
                        except:
                            pass
                    print(f"Rate limit exceeded. Waiting for {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                else:
                    # For other errors, you may want to retry or handle differently
                    print(f"Error: {e}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
        raise Exception(f"Failed to complete after {max_retries} attempts.")

    def parse_response(self, response_text):
        """Parses the LLM response to extract questions and answers.
        Args:
            response_text: The text response from the LLM.
        Returns:
            list: A list of dictionaries containing id, question_num, question, and answer.
        """
        lines = response_text.strip().split('\n')
        data = []
        current_id = ''
        current_question_num = ''
        current_question = ''
        current_answer = ''
        for line in lines:
            if line.startswith('id:'):
                current_id = line.replace('id:', '').strip()
            elif line.startswith('question_'):
                current_question_num = line.split(':',1)[0].replace('question_', '').strip()
                current_question = line.split(':',1)[1].strip()
            elif line.startswith('answer_'):
                answer_num = line.split(':',1)[0].replace('answer_', '').strip()
                if answer_num == current_question_num:
                    current_answer = line.split(':',1)[1].strip()
                    data.append({
                        'id': current_id,
                        'question_num': current_question_num,
                        'question': current_question,
                        'answer': current_answer
                    })
                    # Reset current_question and current_answer
                    current_question = ''
                    current_answer = ''
        return data

    def process_file(self, args):
        """Process a single file to generate test questions and answers.
        Args:
            args: A tuple containing file_path, id_counter, num_tests.
        Returns:
            list: Parsed data from the LLM response.
        """
        file_path, id_counter, num_tests = args
        filename = os.path.basename(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            context = f.read()

        # Call create_tests
        response = self.create_tests(context, str(id_counter), num_tests=num_tests)

        # Parse the response
        parsed_data = self.parse_response(response.content.strip())
        for item in parsed_data:
            item['filename'] = filename
        return parsed_data

    def main(self):
        input_folder = 'documents/'  # Replace with your folder path
        output_file = 'output.csv'
        num_tests_per_file = 20

        files = glob.glob(os.path.join(input_folder, '*'))
        
        print(f"Processing {len(files)} files...")
        data = []

        # Prepare arguments for processing files
        args_list = [(file_path, idx + 1, num_tests_per_file) for idx, file_path in enumerate(files)]

        # Process files with controlled concurrency and progress bar
        max_workers = 2
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = list(tqdm(
                executor.map(self.process_file, args_list),
                total=len(files),
                desc="Generating tests",
                unit="file"
            ))
            
            for parsed_data in futures:
                try:
                    data.extend(parsed_data)
                except Exception as e:
                    print(f'Error processing file: {e}')

        # Create DataFrame and save to CSV
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        print(f'Data saved to {output_file}')

if __name__ == "__main__":
    generator = TestGenerator()
    generator.main()