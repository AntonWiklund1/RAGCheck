import os
import wikipedia
from tqdm import tqdm
import sys
from typing import List, Optional
from multiprocessing import Pool, cpu_count
from functools import partial
import tiktoken

encoding = tiktoken.encoding_for_model("gpt-4o-mini")

# add path to src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

topics = [
    "Python (programming language)",
    "Artificial intelligence",
    "Climate Change",
    "World War II",
    "Quantum Mechanics",
    "Human Rights",
    "History of the Internet",
    "Neuroscience",
    "Global Financial Crisis of 2008",
    "Renewable Energy",
    "Cryptocurrency and Blockchain",
    "Ancient Egypt",
    "History of Mathematics",
    "Evolutionary Biology",
    "International Space Station",
    "Philosophy of Mind",
    "Renaissance Art",
    "History of the United Nations",
    "Genetic Engineering",
    "Artificial Neural Networks",
    "Psychology of Learning",
    "Political Philosophy",
    "Astronomy",
    "Robotics",
    "Epidemiology",
    "History of Medicine",
    "Cognitive Science",
    "Philosophy of Science",
    "Public Health",
    "Space-Time Continuum",
    "Sustainable Development Goals (SDGs)",
    "Internet Privacy",
    "International Monetary Fund (IMF)",
    "Ancient Greek Mythology",
    "History of Computing",
    "Artificial Intelligence Ethics",
    "Biotechnology",
    "Cybersecurity",
    "Digital Transformation",
    "E-commerce",
    "Environmental Policy",
    "Feminism",
    "Genomics",
    "History of Art",
    "History of Physics",
    "Internet of Things (IoT)",
    "Linguistics",
    "Nanotechnology",
    "Oceanography",
    "Paleontology",
    "Quantum Computing",
    "Renewable Resources",
    "Social Media Impact",
    "Sociology",
    "Supply Chain Management",
    "Sustainable Agriculture",
    "Telemedicine",
    "Urban Development",
    "Water Conservation",
    "Wildlife Conservation",
    "World Health Organization (WHO)",
    "Zoology",
    "Augmented Reality",
    "Blockchain Technology",
    "Cultural Anthropology",
    "Data Science",
    "Ecology",
    "Geopolitics",
    "History of Astronomy",
    "History of Chemistry",
    "History of Engineering",
    "History of Literature",
    "History of Music",
    "History of Philosophy",
    "History of Psychology",
    "History of Sociology",
    "History of Technology",
    "History of Theatre",
    "History of Warfare",
    "History of Western Civilization",
    "History of World Religions",
    "Marine Biology",
    "Meteorology",
    "Nuclear Physics",
    "Renewable Energy Technologies",
    "Sustainable Energy",
    "Theoretical Physics",
    "Urban Planning",
    "World Trade Organization (WTO)",
    "Taylor Swift",
    "George W. Bush",
    "European debt crisis"
]


def setup_environment(output_folder: str) -> None:
    """Create necessary directories if they don't exist."""
    os.makedirs(output_folder, exist_ok=True)

def get_safe_filename(title: str) -> str:
    """Convert a title to a safe filename."""
    return "".join(c for c in title if c.isalnum() or c in (' ', '_', '-')).rstrip()

def fetch_wikipedia_content(topic: str) -> Optional[tuple[str, str]]:
    """
    Fetch content from Wikipedia for a given topic.
    Returns tuple of (title, content) if successful, None otherwise.
    """
    try:
        search_results = wikipedia.search(topic)
        if not search_results:
            tqdm.write(f"No results found for '{topic}'.\n")
            return None

        page = wikipedia.page(search_results[0], auto_suggest=False)
        return (page.title, page.content)

    except wikipedia.exceptions.DisambiguationError as e:
        tqdm.write(f"The topic '{topic}' is ambiguous. Possible options include:")
        for option in e.options[:5]:  # Show only first 5 options
            tqdm.write(f"- {option}")
        tqdm.write("")
    except wikipedia.exceptions.PageError:
        tqdm.write(f"The page '{topic}' does not exist.\n")
    except Exception as e:
        tqdm.write(f"An error occurred: {e}\n")
    
    return None

def clean_content(content: str) -> str:
    """Clean the Wikipedia content by removing unnecessary whitespace and formatting artifacts."""
    # Split into lines
    lines = content.split('\n')
    
    # Clean and filter lines
    cleaned_lines = []
    for line in lines:
        # Skip empty or whitespace-only lines
        if not line.strip():
            continue
            
        # Skip lines that are just formatting artifacts
        if line.strip() in ['', '\t', '		']:
            continue
            
        # Skip lines that are just brackets or separators
        if set(line.strip()) <= set('[]_='):
            continue
            
        # Clean the line
        cleaned_line = line.strip()
        
        # Add the cleaned line if it's not empty
        if cleaned_line:
            cleaned_lines.append(cleaned_line)
    
    # Join lines back together with proper spacing
    # Add extra newline before section headers
    final_content = []
    for i, line in enumerate(cleaned_lines):
        if line.startswith('===') or line.startswith('=='):
            if i > 0:  # Don't add extra newline if it's the first line
                final_content.append('')  # Add blank line before section
            final_content.append(line)
        else:
            final_content.append(line)
    
    return '\n'.join(final_content)

def save_content(output_folder: str, title: str, content: str) -> None:
    """Save content to a file in the specified folder."""
    # Clean the content before saving
    cleaned_content = clean_content(content)

    # Tokenize the content
    tokens = encoding.encode(cleaned_content)
    num_tokens = len(tokens)

    # If content exceeds token limit, truncate it
    if num_tokens > 100000:
        tqdm.write(f"Content for '{title}' exceeds token limit ({num_tokens} tokens). Truncating...")
        # Decode only the first 100k tokens back to text
        cleaned_content = encoding.decode(tokens[:100000])
    
    filename = os.path.join(output_folder, f"{get_safe_filename(title)}.txt")
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(cleaned_content)

def process_single_topic(topic: str, output_folder: str) -> None:
    """Process a single topic and save its content."""
    try:
        result = fetch_wikipedia_content(topic)
        if result:
            title, content = result
            save_content(output_folder, title, content)
            return True
    except Exception as e:
        tqdm.write(f"Error processing topic '{topic}': {str(e)}")
        return False

def scrape_wikipedia_topics(topics: List[str], output_folder: str, num_processes: int = None) -> None:
    """Main function to scrape Wikipedia topics in parallel."""
    # Create output directory
    setup_environment(output_folder)
    
    # If num_processes is not specified, use number of CPU cores minus 1
    if num_processes is None:
        num_processes = max(1, cpu_count() - 1)
    
    print(f"Scraping Wikipedia topics with {num_processes} processes")
    # Create a partial function with the output_folder parameter
    process_topic = partial(process_single_topic, output_folder=output_folder)
    
    # Create a process pool and map the topics
    with Pool(processes=num_processes) as pool:
        # Use tqdm to show progress
        list(tqdm(
            pool.imap_unordered(process_topic, topics),
            total=len(topics),
            desc=f"Processing topics using {num_processes} processes",
            unit="topic"
        ))

if __name__ == "__main__":
    # Set the language to English (change if needed)
    wikipedia.set_lang("en")
    
    output_folder = 'data/documents'
    
    scrape_wikipedia_topics(topics, output_folder)