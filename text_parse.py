"""
Text File Parser

This script processes text files located in a specified input directory. 
For each text file, it performs a series of transformations including:
1. Stripping Markdown syntax from the text.
2. Splitting the cleaned text into sentences.
3. Adding Prompt Eng For GKC-CI annotation.

The results are saved as CSV files in a specified output directory. Each CSV file will contain:
- A column named "prompt" with the modified sentences.
- A column named "completion" initialized as empty.
10/28/24
J. Chanenson
"""

import os, re, csv, yaml
from nltk.tokenize import sent_tokenize 

def main():
    # Load configuration from production.yml
    with open("prod.yml", "r") as file:
        config = yaml.safe_load(file)

    input_dir = config["raw_text_dir"] 
    output_dir = config["processed_text_dir"]  

    # Ensure the data directory exists
    validate_data_directory(input_dir)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Loop through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):  # Process only text files
            file_path = os.path.join(input_dir, filename)
            process_text_file(file_path, output_dir)


def process_text_file(file_path, output_dir):
    """
    Process a single text file, apply transformations, and save results to a CSV file.

    Parameters:
    - file_path (str): The path to the text file to process.
    - output_dir (str): The directory where the output CSV file will be saved.
    """
    # Read the text file
    with open(file_path, 'r', encoding='utf-8', errors="ignore") as file:
        content = file.read()
        
    # Process the content using the defined functions
    cleaned_content = stripMarkdown(content)
    chunks = splitIntoSent(cleaned_content)
    annotated_chunks = addPromptEngAnnotations(chunks)

    # Prepare data for CSV with only annotated chunks
    annotated_list = [{'prompt': annotated, 'completion': ''} for annotated in annotated_chunks]

    # Construct a unique filename for the output CSV
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    output_file_path = os.path.join(output_dir, f"{base_filename}_output.csv")

    # Save the processed results to a CSV file
    with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=['prompt', 'completion'])
        csv_writer.writeheader()  # Write the header
        csv_writer.writerows(annotated_list)  # Write the rows

def stripMarkdown(markdownContent):
    """
    Strips common Markdown elements from a given string containing Markdown content.
    Helper function of getCost. 

    Parameters:
    - markdownContent (str): The content string with Markdown formatting.

    Returns:
    - str: The content with Markdown elements removed.
    """
    
    content = markdownContent

    # ### REMOVE THE BLOCK QUOTE AT THE TOP OF THE FILE ##
    # # This will match everything starting from the first block quote to the newline just before the first top-level heading.
    # pattern = r'^(?:>.*\n)+\n(?=# )'

    # # Replace the matched block quote with an empty string
    # content = re.sub(pattern, '', content, flags=re.MULTILINE)

    # Links: ![alt_text](url) or [link_text](url)
    content = re.sub(r'!\[.*?\]\(.*?\)', '', content)  # Images
    content = re.sub(r'\[.*?\]\(.*?\)', '', content)   # Links

    # Headers: # or ## or ### etc.
    # content = re.sub(r'#+', '', content)

    # Emphasis: *italic* or **bold** or ***bolditalic***
    content = re.sub(r'\*+', '', content)

    # Lists: - item or * item or + item
    # content = re.sub(r'^[-\*+]\s+', '', content, flags=re.M)

    # Inline Code: `code`
    content = re.sub(r'`', '', content)

    # Blockquotes: > quote
    content = re.sub(r'^>\s+', '', content, flags=re.M)

    # Horizontal lines: --- or *** or - - -
    content = re.sub(r'^[\-_*]\s*[\-_*]\s*[\-_*]\s*$', '', content, flags=re.M)

    # Code blocks: ```code```
    content = re.sub(r'```.*?```', '', content, flags=re.S)

    return content

def splitIntoSent(text):
    """
    Splits the given text into sentences and cleans each sentence.

    Parameters:
    - text (str): The input text to be split into sentences.

    Returns:
    - list: A list of cleaned sentences. Each sentence is represented as a string.
    
    The function ignores any sentences that are not successfully cleaned by 
    `clean_single_sentence` and does not include them in the output list.
    """

    from nltk.tokenize import sent_tokenize
    sentences = sent_tokenize(text)
    
    ret_lst = []

    for line in sentences:
        foo = clean_single_sentence(line)
        
        if foo:
            ret_lst.append(foo)
        else:
            # print(f"skipped at split stage {line}")
            pass
    
    return ret_lst

def clean_single_sentence(sentence):
    """
    Clean a single sentence string. 
    If the sentence is just a number, return None.

    Args:
    - sentence (str): The sentence string to be cleaned.

    Returns:
    - str or None: Cleaned sentence or None if the sentence is just a number.
    """

    # Enhanced pattern to match bullet points, newline characters, spaces, numbers with optional (escaped) periods, hash symbols, multiple pipes with spaces, and the pipe symbol at the start of a string
    pattern_start = r"^(?:[•*+\-#|]|\n|\s|\\\.?|(?:\|\s*)+)+|(?:^\d+\\.?)"
    
    # Pattern for the end of the string to match patterns like ## 1. or single newlines followed by a number and an optional (escaped) period.
    pattern_end = r"((\n\s*)+(## \d+\\?\.?$)|(\n \d+\\?\.?$))"
    
    # Pattern specifically for strings like 'Marketing\n   9.' or 'Who we are\n   2.' or 'Who we are\n   2\\.'
    pattern_specific = r"(.+?)\s*\n\s*\d+\\?\.?(\s*)$"

    # Use re.sub to clean the sentence from start, end, and specific patterns
    cleaned_sentence = re.sub(pattern_start, "", sentence)
    cleaned_sentence = re.sub(pattern_end, "", cleaned_sentence)
    cleaned_sentence = re.sub(pattern_specific, r"\1", cleaned_sentence).strip()
       
    # If after cleaning, the sentence is not just a number (with or without a period, escaped or not) and not empty, return the cleaned sentence
    if cleaned_sentence and not (cleaned_sentence.replace('.', '').isdigit() or cleaned_sentence.replace('\\.', '').isdigit()):
        return cleaned_sentence
    
    return None

def addPromptEngAnnotations(sentList):
    """
    Generate a list of formatted excerpts for GKC-CI annotation based on the provided sentence list and predefined parameters.
    
    Parameters:
    - sentList (list of str): A list of sentences that need GKC-CI annotation.
    
    Returns:
    - list of str: A list containing formatted strings requesting GKC-CI annotations for each parameter and sentence combination.
    """
    params = ['Sender', 'Subject', 'Consequence', 'Modality', 'Recipient', 
              'Condition', 'Aim', 'Attribute']
    
    annotatedList = []

    for line in sentList:
        cleaned_sentence = clean_single_sentence(line)
        
        # If the cleaned sentance is garbage we move to the next item
        if not cleaned_sentence:
            print(f"skipped {line}")
            continue
        
        # For every parameter in params
        for param in params:
            # Create the desired format and append it to the annotatedList
            # annotatedList.append(f"For the following excerpt, provide the GKC-CI annotation of \'{param}\': [\'{line}\']")
            annotatedList.append(f"Annotate: [\'{line}\']\n\'{param}\'-->")

    
    return annotatedList

def validate_data_directory(directory_path):
    """
    Validates a given directory path to ensure it meets specific conditions:
    
    1. The directory exists.
    2. The directory is not empty.
    3. The directory contains at least one `.txt` file.

    If any of these conditions are not met, the function will print an error message and exit the program.
    """
    
    # Check if the directory exists
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist.")
        os._exit(1)

    # Check if the directory is empty
    if not os.listdir(directory_path):
        print(f"Error: Directory '{directory_path}' is empty.")
        os._exit(1)

    # Check if the directory contains any .txt files
    if not any(file.endswith('.txt') for file in os.listdir(directory_path)):
        print(f"Error: Directory '{directory_path}' does not contain any .txt files.")
        os._exit(1)


if __name__ == "__main__":
    main()
