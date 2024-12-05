""""
File that pulls from previous benchmarking scripts to measure how well GPT3.5-Turbo is doing

J. Chanenson with contributions and full functions by Madison Pickering
8/23/23
Touch Up: 10/28/2024 & 11/20/2024
"""
import pandas as pd
from tqdm import tqdm
from tiktoken import get_encoding
import re, os, openai, datetime, time, yaml

#####THIS MUST GO HERE

# Fetch the encoding method for GPT4/GPT3.5-Turbo only once
encoding = get_encoding("cl100k_base")
#####THIS MUST GO HERE

# Load configuration from production.yml
with open("prod.yml", "r") as file:
    config = yaml.safe_load(file)

MODEL_NAME = config["MODEL_NAME"]
PRODUCTION_DIR = config["processed_text_dir"]
CORPUS_PRODUCTION_DIR = config["corpus_text_dir"]
OUTPUT_DIR = config["OUTPUT_DIR"]
PLC_DBO = config["PLC_DBO"]

# Set the summary file path based on the loaded configuration
summary_file_path = os.path.join(OUTPUT_DIR, config["SUMMARY_FILE_NAME"])


# Speed limit on tokens per min. The official limit is 10k
TOK_LIMIT = 15000 

DELAY = 0 # in seconds

def main():

    # 0. Set up throttler to keep us under OpenAI's token per min quota   
    global throttler
    throttler = Throttler()

    # 1. Read in the CSV files 
    # Ensure the directory exists
    if not os.path.exists(PRODUCTION_DIR):
        print(f"The directory '{PRODUCTION_DIR}' does not exist!")
        raise ValueError
    
    #2. Filter files, if applicable 
    directory_path = PRODUCTION_DIR 
    # If using corpus, keep track of completed files and only pull a subset of longitudinal policies
    if PLC_DBO:
        directory_path = CORPUS_PRODUCTION_DIR
        if os.path.exists(summary_file_path):
            fileList = select_files(directory_path, verbose=False, completed_file_path=summary_file_path)
        else:
            fileList = select_files(directory_path, verbose=False)
    # else, just pull the file paths 
    else:
        fileList = list_full_file_paths(directory_path) #robust to various file names


    # Loop through each file in the directory
    for i, filePath in enumerate(fileList): 
        filename_with_extension = os.path.basename(filePath)
        fileName = os.path.splitext(filename_with_extension)[0]
        if os.path.isfile(filePath):
            print(f"\n(idx:{i} | {i}/{len(fileList)}) Running {filePath}...")

            df = pd.read_csv(filePath)      

            ##3. get results
            log_results, stats_dict = runProductionFile(MODEL_NAME, df, fileName)

            ##4. log results of LLM output

            logOutput(log_results, os.path.join(OUTPUT_DIR, 'logs'), f"{fileName}_results")
            
            export_to_csv(stats_dict, os.path.join(OUTPUT_DIR, 'csvs'), f"{fileName}_results.csv", summary_file=summary_file_path)
     
def list_full_file_paths(directory):
    """
    Walks through the given directory and returns a list of full file paths.

    params 
        directory: Path to the directory to walk through.
    return
        List of full file paths.
    """
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

def select_files(directory, verbose = False, completed_file_path = False):
    """
    For a given directory, selects one file per year for each organization (like "eff", "buzzfeed", "boa", etc.) 
    with a preference for the 'A' version.
    
    If the 'completed_file_path' is provided, it reads a CSV from that path and filters out filenames
      that have already been completed.

    Parameters:
    - directory (str): The path to the directory containing the files.
    - verbose (bool, optional): If set to True, provides additional logging (currently not implemented).
    - completed_file_path (Union[str, bool], optional): The path to the CSV file containing completed filenames.
      If False, no filtering against completed filenames is done.

    
    Returns:
    - list: full path for each file
    """
    
    # Get the list of filenames from the directory
    filenames = sorted(os.listdir(directory))

    # Use a nested dictionary to keep track: organization -> year -> filename
    selected_files = {}

    for filename in filenames:
        # Split filename to extract organization, year, and version
        parts = filename.split('-')
        organization = parts[0]
        year = parts[1]
        version = parts[2].split('.')[0]

        # Initialize organization in the dictionary if not present
        if organization not in selected_files:
            selected_files[organization] = {}

        # If the year is not in the organization dictionary, or if the year is in the organization dictionary but the version is 'A', update the dictionary
        if year not in selected_files[organization] or version == 'A':
            selected_files[organization][year] = filename

    # Create a list to store full file paths
    full_file_paths = []

    # Iterate over the selected files and build the full path
    for organization, files in selected_files.items():
        for year, filename in files.items():
            full_path = os.path.join(directory, filename)
            full_file_paths.append(full_path)


    if verbose:
        for organization, files in selected_files.items():
            print(f"Selected files for {organization}:")
            for year, filename in files.items():
                print(filename)
            print()

    if completed_file_path:
        # Read in the CSV into a DataFrame
        df = pd.read_csv(completed_file_path)

        # Extract the filenames from the full paths
        file_names = [os.path.basename(path.replace('\\', '/')) for path in full_file_paths]

        completed =[e.replace("_results", "") for e in df['Source Filename'].tolist()]

        # Remove items from full_file_paths that already exist in the "Filename" column
        filtered_paths = [path for path, name in zip(full_file_paths, file_names) if name not in completed]
        
        print(f"Processing the following files: {filtered_paths}\nAll other files in '{directory}' are already logged in the summary file ({completed_file_path}).")
        
        return filtered_paths


    return full_file_paths
           

def queryTurbo(prompt, model):
    """
    Queries GPT3.5-Turbo and returns the response

    Parameters
    - prompt (str): The prompt or question to be sent to the model.
    - model (str): The model ID or name to be used for querying.

    Returns 
    - output from model
    """
    response = openai.ChatCompletion.create(
        model=model, 
        messages=prompt, 
        temperature=1, 
        max_tokens=500
    )

    # Add the response tokens so the throttler has info needed to do its job
    global throttler
    throttler.addResponseTokensToCount(response["usage"]["completion_tokens"])

    return response["choices"][0]["message"]["content"]

def getOpenAICompletion(prompt, model="Foo"):
    """
    Just to handle the wide types of open AI models
    Parameters:
    - prompt (str): the raw prompt that we want sent to the AI model
    - model (str): the string for the ft model; If we leave the argument blank then
                   we can send requests to GPT4
    Returns 
    - response (str): the response from the model
    """
    global OPEN_AI_MODEL_TYPE
    if "gpt-3.5-turbo" in model:
        OPEN_AI_MODEL_TYPE = "GPT3_5-Turbo"
        text = generateTurboChat(prompt)

        # If you want throtteling
        message = processTextForOPENAI(text, throttler)

        response = queryTurbo(message, model)
    else:
        raise ValueError


    # Pause so we don't hit our token limit of 10k tokens/min
    # Really needed for fewshot 
    # if DELAY > 0:
    #     time.sleep(DELAY)
    return response

# Create a class to keep our requests under the limit
class Throttler:
    def __init__(self):
        # Initialize the throttler by resetting token counts and timestamps.
        self.reset()

    def reset(self):
        # Resets the token count and start time.
        self.tokenCount = 0
        self.startTime = time.time()
    
    def addResponseTokensToCount(self, responseTokensCount):
        self.tokenCount += responseTokensCount

    def canSend(self, tokensCount):
        """
        Check if we can send the text based on its token count 
        without exceeding the 10,000 tokens-per-minute limit.
        """
        # If more than a minute has passed since the last check, reset the counters.
        if time.time() - self.startTime > 60:
            self.reset()

        # Check if sending the new text will keep us under the limit.
        if self.tokenCount + tokensCount <= TOK_LIMIT:
            self.tokenCount += tokensCount
            # print(f"\nWe are at {self.tokenCount} tokens")
            return True
        else:
            print(f"\nWe are at {self.tokenCount} tokens. We must wait.")
            return False

    def waitTime(self):
        # Calculate the amount of time (in seconds) we need to wait before the next send can occur.
        return max(0, 60 - (time.time() - self.startTime))

def countTokens(text):
    """
    Count how many tokens are in the provided text using the specified model's encoding.
    """
    # # Fetch the encoding method for GPT4/GPT3.5-Turbo
    # encoding = get_encoding("cl100k_base")

    # Tokenize the text and return the token count.
    return len(encoding.encode(text))

def processTextForOPENAI(text, throttler):
    """
    Processes the text, ensuring it's safe to send to GPT-4 based on the throttling restrictions.
    """
    # Calculate token count for the provided text.
    tokensCount = countTokens(str(text))

    # If the token count exceeds 9500, raise an error.
    if tokensCount > 9500:
        raise ValueError("The provided text exceeds the token limit!")

    # Wait until it's safe to send the text.
    while not throttler.canSend(tokensCount):
        time.sleep(throttler.waitTime())

    # Once it's safe, return the text.
    # print("Time to submit the job")
    return text

def runProductionFile(model, df, fileName):
    outputs_to_log = []
    stats_dict = initialize_dict()

    for value in tqdm(df['prompt']):
        outputs_to_log.append(f"prompt = {value}")
        
        try:
            completion = getOpenAICompletion(prompt=value, model=model) 
            if len(completion.strip()) == 0:
                completion = "LLM_ERROR" 

            #2. Parse the completion by trimming off "x-"'s, whitespace
            completion = completion.split("x-")[0]
            completion = completion.strip()

            # update the stats
            stats_dict = update_dict(completion, stats_dict)

            if len(completion) == 0:
                completion = "LLM_ERROR" 
            outputs_to_log.append(f"completion = {completion}")
        except:
            print(f"Something went wrong in {fileName}. It has to do with:\n {value}\n\ncompletion given {completion}")
            outputs_to_log.append(f"completion = ERROR")
    
    return outputs_to_log, stats_dict

def logOutput(outputs_to_log, directory, docName):
    # logfile = f"production_results/{docName}".replace(".csv","")
    logfile = os.path.join(directory, docName)

    ensurePathExists(logfile)
    with open(logfile, "w", encoding='utf-8') as f:
        for line in outputs_to_log:
            if ("prompt = " in line):
                f.write(f"\n\n{line}\n")
            else:
                f.write(f"{line}\n")
    print(f"File written to {logfile}")

def initialize_dict():
    """
    Initialize a dictionary with a set of default keys. Each key is initialized with
    a sub-dictionary containing counters for 'value_count' and 'N/A_count'.
    
    Returns:
        dict: Initialized dictionary with default keys and counters.
    """
    keys = ['Sender', 'Subject', 'Consequence', 'Modality', 'Recipient', 'Condition', 'Aim', 'Attribute']
    return {key: {'value_count': 0, 'N/A_count': 0} for key in keys}

def update_dict(input_string, data_dict):
    """
    Update the dictionary based on the input string. If the tag in the input string is not 
    recognized, a warning is printed and the dictionary remains unchanged. Recognized tags
    will have their corresponding counters updated.
    
    Args:
        input_string (str): The input string in format "tag: value".
        data_dict (dict): The dictionary to be updated.
    
    Returns:
        dict: Updated dictionary.
    """
    ##Old error prone way
    # tag, value = [s.strip() for s in input_string.split(":")]

    # # Check if tag is in the default set
    # if tag not in data_dict:
    #     print(f"Warning: {tag} is not a recognized tag and will not be added.")
    #     return data_dict

    # Check if there's a colon that's NOT inside square brackets
    match = re.match(r'^(\S+):', input_string)
    

    if match:
        tag, value = [s.strip() for s in input_string.split(":", 1)]
    else:
        split_data = input_string.split(None, 1)  # Split only once based on whitespace
        tag = split_data[0]
        value = split_data[1] if len(split_data) > 1 else ''
        value = value.strip(":").strip()

    try:
        if value.upper() == "N/A":
            data_dict[tag]['N/A_count'] += 1
        else:
            data_dict[tag]['value_count'] += 1
    except:
        print(f"We had a dict update issue Tag: {tag}\nInput: {input_string}")

    return data_dict

def export_to_csv(data_dict, directory='', filename='output.csv', summary_file='summary.csv'):
    """
    Exports data to a specified CSV and appends a summary to another CSV.
    
    Args:
    - data_dict (dict): Data with tags as keys and count dicts as values.
    - directory (str): Directory to save the CSV file in.
    - filename (str): Target CSV file for full data.
    - summary_file (str): Summary CSV file for first row of new data.
    
    Returns:
    None.
    """
    # Create the full path for the filename
    full_path = os.path.join(directory, filename)
    ensurePathExists(full_path)
    ensurePathExists(summary_file)
    
    # Convert data_dict to DataFrame
    df = pd.DataFrame(data_dict)
    
    # Sort header
    df = df.sort_index(axis=1)

    # Save to the specified CSV
    df.to_csv(full_path)
    print(f"File written to {full_path}")
    
    # Prepare summary data with source filename
    summary_data = df.iloc[[0]]
    summary_data.insert(0, 'Source Filename', filename)

    # Check and update the summary CSV
    try:
        if os.path.exists(summary_file):
            existing_data = pd.read_csv(summary_file, index_col=0)
            combined_data = pd.concat([existing_data, summary_data])
        else:
            combined_data = summary_data

        combined_data.to_csv(summary_file)
        print(f"Updated {summary_file}")
    except Exception as e:
        print(f"An error occurred: {e}")



################## PROMPT GENERATION CODE -- NO TOUCH ##################
systemMessage = ("You are an assistant that understands Helen Nissenbaum's theory of "
                     "Contextual integrity (CI) and the governance of knowledge commons framework "
                     "(GKC). This framework is abbreviated as GKC-CI. You reply with brief, "
                     "to-the-point answers with no elaboration.")

def generateTurboChat(prompt):
    """
    Helper function for other files like benchmarking and ft_turbo
    Paramters
    - prompt (str): the prompt we want to wrap in so GPT3.5 Turbo can accept the message

    Returns:
    - chatMessage: chat object that GPT3.5 Turbo likes
    """
    chatMessage = []
    chatMessage.append({"role": "system", "content": systemMessage})
    userMessage = createUserMessage(prompt)
    chatMessage.append({"role": "user", "content": userMessage})

    return chatMessage

def createUserMessage(text):
    """
    Replaces a given string pattern with a specified output format.
    
    Given an input string with the pattern "Annotate: ... -->", this function
    will replace it with something better for a chat system

    Args:
    - text (str): The input string to be processed.

    Returns:
    - str: The processed string.
    """
    # Flatten the text by removing newlines, tabs, etc.
    # flatText = ' '.join(text.split()) #orginal
    flatText = '\n\n'.join([' '.join(paragraph.split()) for paragraph in text.split('\n\n')])

    match = re.match(r'Annotate:(.*?) ([a-zA-Z-]+)-->$', flatText) 
    
    if not match:
        try:
            text_partial = text.split(":",1)[1]
            param, content = extract_and_remove_word_with_arrow(text_partial)
        except:
             print(f"Fatal error! We were unable to processes the prompt into something good for chatGPT. \nThe issue was:\n{flatText} ")
             os._exit(1)
    else:
        content, param = match.groups()

    if not content:
        print("++++++++++++++++++++No Content!++++++++++++++++++++++++++++")
        print(f"tag:{param}, content: {content}\n*************\nText: {text}\n*********\nFlat: {flatText}\n-------------")

    return f'For the following excerpt, provide the GKC-CI annotation of \'{param}\': {content}'

def extract_and_remove_word_with_arrow(text):
    """
    Extracts a tag with format 'word -->' from the given text and returns the tag 
    without the arrow and the modified text without the tag.
    
    Args:
    - text (str): The input string from which to extract the tag and generate modified text.
    
    Returns:
    - tuple: A tuple containing two strings:
        1. The tag without the arrow ('-->'). 
        2. The modified text without the tag.
           If no tag is found, the first item in the tuple is None.
    """
    lines = text.split('\n')
    modified_lines = []
    tag = None

    for line in lines:
        words = line.split()
        current_tag = next((word for word in words if '-->' in word), None)
        if current_tag and not tag:
            tag = current_tag
        modified_line = ' '.join([word for word in words if '-->' not in word])
        modified_lines.append(modified_line)

    modified_text = '\n'.join(modified_lines)
    if tag:
        return tag.replace('-->', ''), modified_text.strip()
    else:
        return None, modified_text.strip()

def ensurePathExists(path_to_file):
    """
    Ensures that a path to the file exists and creates one if not
    Parameters
    - path_to_file (str): the path to the file
    Returns 
    - Nothing
    """

    # Extract the directory path from the full file path
    directory = os.path.dirname(path_to_file)

    # Check if the directory exists, if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)
        # print(f"Directory {directory} created.")
    else:
        pass
        # print("Directory already exists.")


if __name__ == "__main__":
    main()
    # checkIt()
