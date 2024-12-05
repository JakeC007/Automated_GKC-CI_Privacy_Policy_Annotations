"""
Contains every function needed to manipulate data from the sql database
Assumes that the db is in this folder
J. Chanenson
8/15/23
"""
import os, pickle, sqlite3, tldextract, tiktoken, re, yaml
from nltk.tokenize import sent_tokenize, word_tokenize 
from datetime import date
import pandas as pd
from tranco import Tranco
from pprint import pprint

# Uncomment below if you need to download punkt
# import nltk
# nltk.download('punkt')

## Globals
# Tokenizer, make it global so it only loads in once
print("Loading in Tokenizer from tiktoken...")
encoding = tiktoken.get_encoding("cl100k_base")

def inputFromList():
    """
    Returns a predefined list that the user can edit directly in this script.
    
    Returns:
        list: The user-edited list from the Python file.
    """
    # Input what files you want to find in the db and then read in 
    # Load the YAML file
    # with open(file_path, 'r') as file:
    #     config = yaml.safe_load(file)
    
    # # Get List
    # input_list = config.get('input_list', [])
    
    global INPUT_LIST

    return INPUT_LIST

def dataInput(RELEASE_DB_NAME):
    """
    Input function to define how we ingest the data for chunking
    """
    print("Loading in data...")
    filteredDF = getFilteredDF(RELEASE_DB_NAME)
    
    print("\n\n DATA INPUT:")
    print("Where do you want to get your data from:")
    print("1. Input from User List (all entries in corpus)")
    print("2. Input from User List (most current entry in corpus)")
    print("3. Input from Tranco (all entries in corpus)")
    print("4. Input from Tranco (most current entry in corpus)")



    choice = int(input("Choice: "))

    if choice == 1:
        siteList = inputFromList()
        df = filterDomainsExact(filteredDF, siteList, current = False)
    elif choice == 2:
        siteList = inputFromList()
        df = filterDomainsExact(filteredDF, siteList, current = True)
    elif choice == 3:
        numSites = int(input("Enter the number of top sites you'd like to retrieve from Tranco: "))
        df = getTranco(filteredDF, numSites, current = False)
    elif choice == 4:
        numSites = int(input("Enter the number of top sites you'd like to retrieve from Tranco: "))
        df = getTranco(filteredDF, numSites, current = True)
    else:
        print("Invalid choice!")

    print("Filtered the data!")
    
    print("\n\n DATA PROCESSING:")
    print("What do you want to do with the data:")
    print("1. Get Token Counts and Cost")
    print("2. Process Data Into LLM input")
    print("3. Print Domains In Tranco Rank Order To DF")
    print("4. Export df with policy texts")
    print("5. [Viz] Export policy texts (for viz)")
    choice = int(input("Choice: "))
    if choice == 1:
        df2 = convertText(df, getCost=True) #2nd arg prevents writing to file
        getCost(df2)
    elif choice == 2:
        convertText(df)
    elif choice == 3:
        df['Filename'] = df.apply(create_file_name, axis=1)
        df = df.reset_index(drop=True)
        # Select the 'fileName' column and save it to CSV
        df[['Filename']].to_csv('filenames.csv', index=False)
    elif choice == 4:
        df2 = convertText(df, getCost=True)
        fname = input("Filename? ")
        df2.to_csv(f"{fname}.csv", index=False, encoding='utf-8')
    elif choice == 5:
        export_policy_for_viz(df)
        # subdirName = input("New dir name? ")
        # if is_valid_dir_name(subdirName):
        #     export_policy_for_viz(df, subdirName)
        # else:
        #     print(f"'{subdirName}' is not a valid directory name.")
    else:
        print("Invalid choice!")
    print("Done!")
    os._exit(1)
    return None

 
    
def createNewSubdir():
    """
    Creates and returns a path to a new subdirectory named after today's date (formatted as YYYY-MM-DD).
    If the directory already exists, it simply returns the path without creating a new one.

    Parameters:
    None

    Returns:
    - str: Path to the new (or existing) subdirectory named after today's date.

    """
    
    # Get today's date and format it as YYYY-MM-DD
    todayDate = date.today().strftime('%Y-%m-%d')
    
    # Get the directory of the current script
    currentDir = os.path.dirname(os.path.abspath(__file__))

    # Go one directory up and then to 'production_data'
    parentDir = os.path.dirname(currentDir)
    productionCsvsDir = os.path.join(parentDir, "policy_text\corpus")

    # Join the productionCsvsDir with today's date
    newSubdir = os.path.join(productionCsvsDir, todayDate)
    
    # Create new subdir if it doesn't exist
    if not os.path.exists(newSubdir):
        os.makedirs(newSubdir)

    return newSubdir

def handleLongSentence(sentence, maxTokens):
    """"
    Handle sentences that exceed the maxTokens limit by splitting them into 
    smaller chunks using LLM-friendly tokenization.
    
    Args:
    - sentence (str): The sentence to be handled.
    - maxTokens (int): Maximum number of tokens for each chunk.

    Returns:
    - List[str]: List of chunks derived from the long sentence.
    """
    words = word_tokenize(sentence)
    chunks = []
    
    # This list will temporarily hold words for the current chunk.
    currentChunk = []
    
    # Index to keep track of our position in the words list.
    wordIndex = 0

    # Continue until all words from the sentence have been processed.
    while wordIndex < len(words):
        # Calculate the space left in the current chunk.
        spaceLeft = maxTokens - countTokens(''.join(currentChunk))
        
        # Get a segment of words that can fit into the space left in the current chunk.
        segment = words[wordIndex:wordIndex + spaceLeft]
        
        for word in segment:
            # Check if the word is a punctuation or if the current chunk is empty.
            # If so, append the word without adding a space before it.
            if word in ["!", ",", ".", "?", ";", ":", "(", ")", "{", "}", "[", "]", "'", "\"", "—", "-", "..."] or not currentChunk:
                currentChunk.append(word)
            # Otherwise, add a space before appending the word.
            else:
                currentChunk.append(' ' + word)

        # Move the word index forward by the number of words we've just processed.
        wordIndex += spaceLeft
        
        # If the current chunk reaches the max token limit or we've processed all words, finalize the chunk.
        if countTokens(''.join(currentChunk)) >= maxTokens or wordIndex >= len(words):
            # Add the current chunk to the chunks list after removing any leading or trailing spaces.
            chunks.append(''.join(currentChunk).strip())
            
            # Reset the current chunk for further processing.
            currentChunk = []

    return chunks

def splitIntoChunks(text, maxTokens=175):
    """
    New method of splitting text into chunks for the LLM.
    Unlike the original method, this method splits on paragraphs 
    and removes section headers for clarity. Hence its just paragrah 
    bodies up until the maxToken amount at which point it is split  
    """
    # Split the text into paragraphs using the newline character
    paragraphs = text.split('\n')
    
    # Filter out markdown headings
    paragraphs = [p for p in paragraphs if not re.match(r'^#+\s', p)]

    chunks = []

    # Process each paragraph
    for paragraph in paragraphs:
        # Count the tokens in the current paragraph
        paragraphTokens = countTokens(paragraph)

        # If the paragraph exceeds maxTokens, split it
        if paragraphTokens > maxTokens:
            currentChunk = []
            # Tokenize the paragraph into sentences
            sentences = sent_tokenize(paragraph)
            for sentence in sentences:
                # Count the tokens in the current sentence
                sentenceTokens = countTokens(sentence)
                
                # If the sentence exceeds maxTokens on its own
                if sentenceTokens > maxTokens:
                    if len(currentChunk) > 0:
                        # Join the sentences in the current chunk into a string
                        combined_text = ' '.join(currentChunk).strip()
                        # Only append if the combined text is not an empty or whitespace-only string
                        if combined_text:
                            chunks.append(combined_text)
                        currentChunk = []

                    # Handle very long sentences and ensure they don't contain only whitespace
                    extendedSentences = handleLongSentence(sentence, maxTokens)
                    chunks.extend([sent for sent in extendedSentences if sent.strip()])
                
                # If adding the sentence to the current chunk doesn't exceed the max tokens
                elif countTokens(' '.join(currentChunk) + ' ' + sentence) <= maxTokens:
                    currentChunk.append(sentence)
                
                # If the sentence causes the current chunk to exceed the max tokens
                else:
                    # Join the sentences in the current chunk into a string
                    combined_text = ' '.join(currentChunk).strip()
                    # Only append if the combined text is not an empty or whitespace-only string
                    if combined_text:
                        chunks.append(combined_text)
                    currentChunk = [sentence]

            # Process any remaining sentences in the current chunk after all sentences are processed
            if len(currentChunk) > 0:
                # Join the sentences in the current chunk into a string
                combined_text = ' '.join(currentChunk).strip()
                # Only append if the combined text is not an empty or whitespace-only string
                if combined_text:
                    chunks.append(combined_text)
        else:
            # If paragraph does not exceed maxTokens and is not empty, directly add it to the chunks
            # Only append if the paragraph is not an empty or whitespace-only string
            if paragraph.strip():
                chunks.append(paragraph.strip())

    return chunks


def splitIntoSent(text):
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
        # # print(line)
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


def addAnnotations(chunkList):
    """
    Append "Annotate:" to the beginning and "--->" to the end of each item in the chunk list.
    Does this 8 times for the 9 CI-GKC parameters: 'Sender', 'Subject', 'Consequence', 'Modality', 'Recipient', 
             'Condition', 'Aim', 'Attribute'

    Args:
    - chunkList (List[str]): List of text chunks.

    Returns:
    - List[str]: List of annotated text chunks.
    """
    # List of parameters
    params = ['Sender', 'Subject', 'Consequence', 'Modality', 'Recipient', 
              'Condition', 'Aim', 'Attribute']
    
    # List to hold the annotated chunks
    annotatedList = []

    # For every chunk in the chunkList
    for chunk in chunkList:
        # For every parameter in params
        for param in params:
            # Create the desired format and append it to the annotatedList
            annotatedList.append("Annotate:" + " " + chunk + " " + param + "--->")
    
    return annotatedList

def convertText(df, getCost=False):
    """
    Processes a given dataframe containing policy text data to extract annotated chunks and saves each 
    entry as a separate CSV file with custom naming.

    Parameters:
    - df (DataFrame): The input dataframe with columns 'policy_text', 'domain', 'year', and 'phase'.

    Returns:
    - None: This function saves the processed data to CSV files and does not return any value.
    """
    # Applying stripMarkdown first to clean documents
    df['cleaned_content'] = df['policy_text'].apply(stripMarkdown)

    # Splitting the text from the file into chunks
    df['chunks'] = df['cleaned_content'].apply(splitIntoSent)

    # Annotating chunks with BOS and EOS tokens and param tags
    # df['annotated'] = df['chunks'].apply(addAnnotations)   
    df['annotated'] = df['chunks'].apply(addPromptEngAnnotations)   

    df.reset_index(drop=True, inplace=True)

    for index, annotatedList in enumerate(df['annotated']):
        # Get info for file-name
        domainName = df.loc[index, 'domain'].replace('.', '_')
        year = df.loc[index, 'year']
        phase = df.loc[index, 'phase']

        # Create file name
        fileName = f"{domainName}-{year}-{phase}.csv"

        # Create file path
        csvFilePath = os.path.join("policy_text\corpus", fileName)
        
        ## Uncomment if you want dated folders for each run
        # newSubdir = createNewSubdir()
        # csvFilePath = os.path.join(newSubdir, fileName)

        
        # Convert the list to a DataFrame and rename the column
        annotatedDf = pd.DataFrame(annotatedList, columns=['prompt'])
        
        # Add a 'completion' column with empty values
        annotatedDf['completion'] = ''
        
        # Exit early if we are just computing for cost
        if getCost:
            return df

        # Save to CSV with UTF-8 encoding
        annotatedDf.to_csv(csvFilePath, index=False, encoding='utf-8')

def export_policy_for_viz(df, subdirName= "visualizer/viz_text"):
    """
    Cleans policy texts in the DataFrame and exports each to a .txt file named by domain, year, and phase in a specified subdirectory.

    Parameters:
    df (pandas.DataFrame): DataFrame with 'policy_text', 'domain', 'year', and 'phase' columns.
    subdirName (str): Subdirectory name under "zviz" for saving .txt files.

    Returns:
    None
    """
    # Applying stripMarkdown first to clean documents
    df['cleaned_content'] = df['policy_text'].apply(stripMarkdown)

    df.reset_index(drop=True, inplace=True)

    for index, policy_text in enumerate(df['cleaned_content']):
        # Get info for file-name
        domainName = df.loc[index, 'domain'].replace('.', '_')
        year = df.loc[index, 'year']
        phase = df.loc[index, 'phase']

        # Create file name
        fileName = f"{domainName}-{year}-{phase}.txt"

        # Create file path
        # newSubdir = os.path.join("visualizer", subdirName)
        newSubdir = subdirName
        os.makedirs(newSubdir, exist_ok=True)
        txtFilePath = os.path.join(newSubdir, fileName)

        # Write the policy text to the file
        with open(txtFilePath, 'w', encoding = 'utf-8') as file:
            file.write(policy_text)
   
def countTokens(plainText):
    """
    Counts the tokens in a given plaintext string. 
    Using the GPT-3 token counter because its fast.
    Helper function of getCost. 

    Parameters:
    - plainText (str): Plaintext content.

    Returns:
    - int: Number of tokens in the content.
    """
    return len(encoding.encode(str(plainText))) 

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

    ## Jake: leaving this in because we finetuned on lists
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

def getCost(df, keepCleanText = False, cost = 0.0120):
    """
    Gets the token counts and then cost of corpus documents
        Parameters:
    - df (pd.DataFrame): The DataFrame to filter. Expected to have a 'domain' column.
    - keepCleanText (bool): True keeps the cleaned text, false deletes the column.
    - cost (int): cost of computing 1k tokens; 0.12 is GPT3 finetune as of aug 2023
    
    Returns:
    - df of ['domain', 'year', 'phase','token_count', 'cost']
    """
    # Applying stripMarkdown first
    # df['cleaned_content'] = df['policy_text'].apply(stripMarkdown)

    # # Applying countTokens on the cleaned content
    # df['token_count'] = df['cleaned_content'].apply(countTokens)

    df['token_count'] = df['annotated'].apply(countTokens)

    # Creating the 'cost' column
    df['cost'] = (df['token_count']/1000) * cost

    if keepCleanText:
        # Remove 'cleaned_content' column
        df.drop('cleaned_content', axis=1, inplace=True)
    
    # Compute Sums
    totalTokens = df['token_count'].sum()
    totalCost = df['cost'].sum() #total cost for imput tokens
    outputTokenEst = ((len(df['cost'])*500)/1000)*0.0160 # (num requests * avg response token length)/1k * output price per 1k
    print(f"Number of Documents {df.shape[0]:,}")
    print(f"Total tokens: {totalTokens:,}")
    print(f"Input Token cost: ${totalCost:,.2f}")
    print(f"Total cost: ${totalCost+outputTokenEst:,.2f}") 
    pprint(df)
    return df[['domain', 'year', 'phase','token_count', 'cost']]

def getTranco(df, numSites, current = True):
    """
    gets a data frame of all the matching tranco sites
    Parameters:
    - df (pd.DataFrame): The DataFrame to filter. Expected to have a 'domain' column.
    - numSites (int): the top N sites 
    - current (bool): Pass through for next function; when true it only grabs the most current policy from the df; false grab all
    
    Returns:
    - df of matching files
    """
    # Get Tranco rankings 
    t = Tranco(cache=True, cache_dir='.tranco')
    latest_list = t.list()
    
    siteLst = latest_list.top(numSites)

    print(siteLst)

    return filterDomainsExact(df, siteLst, current = current)

def filterDomainsExact(df, domainList, current = False):
    """
    Filters a DataFrame based on a list of domains for exact domain matches.

    Parameters:
    - df (pd.DataFrame): The DataFrame to filter. Expected to have a 'domain' column.
    - domainList (list): List of domain strings to search for.
    - current (bool): when true it only grabs the most current policy from the df; false grab all
    
    Returns:
    - pd.DataFrame: A DataFrame containing rows that match the provided domain list.
    """
    filteredDf = pd.DataFrame()
    
    for domain in domainList:
        
        # Check for the exact domain match
        exactMatch = df.query('domain == @domain')

        if not exactMatch.empty and current == True:
            # Sort by year and phase to prioritize 'B' and then get the top record
            latestEntry = exactMatch.sort_values(by=['year', 'phase'], ascending=[False, False]).iloc[0]
            # Add to df
            filteredDf = pd.concat([filteredDf, pd.DataFrame([latestEntry])])
        elif not exactMatch.empty: # grab them all
            filteredDf = pd.concat([filteredDf, exactMatch])
        else:
            print(f"Unable to find {domain}")

    return filteredDf

def filterDomains(df, domainList):
    """
    Filters a DataFrame based on a list of domains. The search is prioritized:
    1. Exact domain matches.
    2. Domains with different TLDs.
    3. Subdomains related to the domain.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame to filter. Expected to have a 'domain' column.
    - domainList (list): List of domain strings to search for.
    
    Returns:
    - pd.DataFrame: A DataFrame containing rows that match the provided domain list.
    """
    filteredDf = pd.DataFrame()
    
    for domain in domainList:
        extracted = tldextract.extract(domain)
        baseDomain = extracted.domain
        
        # Check for the exact domain match
        exactMatch = df.query('domain == @domain')
        if not exactMatch.empty:
            filteredDf = pd.concat([filteredDf, exactMatch])
        
        # If not found, search for the same domain with different TLDs
        elif not df.query('domain.str.contains(@baseDomain + ".") and domain != @domain and not domain.str.contains(r"\.' + baseDomain + r'\.")').empty:
            differentTld = df.query('domain.str.contains(@baseDomain + ".") and domain != @domain and not domain.str.contains(r"\.' + baseDomain + r'\.")')
            filteredDf = pd.concat([filteredDf, differentTld])
        
        # If still not found, search for subdomains
        else:
            subdomainPattern = fr".+\.{baseDomain}\.[a-z0-9\-]+$"
            subdomains = df.query('domain.str.match(@subdomainPattern)')
            filteredDf = pd.concat([filteredDf, subdomains])

    # return filteredDf.drop_duplicates()
    return filteredDf

def getFilteredDF(RELEASE_DB_NAME, both = False):
    """
    Checks if the pickle files 'filteredDF.pkl' and 'chronoDF.pkl' exist in the directory of the script.
    If either file does not exist, it calls the `generateDF` function to create the files.
    Then, it returns both files.

    Files Checked:
        - filteredDF.pkl
        - chronoDF.pkl
    Parameters:
    - both (bool): When false file only returns filteredDF; when true it returns both

    Returns:
        - filteredDF (pandas.DataFrame)
        - chronoDF (pandas.DataFrame)
    """
    # Get the directory of the current script
    currentDir = os.path.dirname(os.path.realpath(__file__))  

    # Just grab the full data set
    if not both:
        df1Exists = os.path.exists(os.path.join(currentDir, 'filteredDF.pkl'))
        getFilteredDF
        if not df1Exists:
            generateDF(RELEASE_DB_NAME)
        
        return pd.read_pickle(os.path.join(currentDir, 'filteredDF.pkl'))

    ## Assuming we want both files...  
    # Check if each pickle file exists in the script's directory
    df1Exists = os.path.exists(os.path.join(currentDir, 'filteredDF.pkl'))
    df2Exists = os.path.exists(os.path.join(currentDir, 'chronoDF.pkl'))

    # Call generateDF function if either pickle df doesn't exist
    if not (df1Exists and df2Exists):
        generateDF(RELEASE_DB_NAME)
    
    # Read in data frames
    filteredDF = pd.read_pickle(os.path.join(currentDir, 'filteredDF.pkl'))
    chronoDF = pd.read_pickle(os.path.join(currentDir, 'chronoDF.pkl'))

    return filteredDF, chronoDF

def generateDF(RELEASE_DB_NAME):
    """
    Creates the data frames and filters it. Then it writes them to file

    much of this code was take from the Release-DB-Read-Demo.ipynb script

    """
    # RELEASE_DB_NAME = "release_db.sqlite"
    conn = sqlite3.connect(RELEASE_DB_NAME)

    # Merge everything into this dataframe
    df = pd.read_sql_query("SELECT * FROM policy_snapshots", conn)

    sites_df = pd.read_sql_query("SELECT * FROM sites", conn)
    policy_texts_df = pd.read_sql_query("SELECT * FROM policy_texts", conn)
    alexa_ranks_df = pd.read_sql_query("SELECT * FROM alexa_ranks", conn)

    # Left join with policy text table
    df = pd.merge(df, policy_texts_df, how="left", left_on="policy_text_id", right_on="id")

    # Left join with sites table
    df = pd.merge(df, sites_df, how="left", left_on="site_id", right_on="id")

    # Left join with alexa ranks table
    df = pd.merge(df, alexa_ranks_df, how="left", on=['site_id', 'year', 'phase'])

    # Filter the DataFrame
    df_filtered = df[['domain', 'year', 'phase', "policy_text_id", "policy_text"]]

    # Group by 'domain' and aggregate
    df_time_summary = df_filtered.groupby('domain').agg(
        count=('domain', 'size'),
        years=('year', list),
        phases=('phase', list)
    ).reset_index() 

    # Pickle the dfs in the script dir

    currentDir = os.path.dirname(os.path.realpath(__file__)) 

    # Check if filteredDF.pkl already exists
    if not os.path.exists(os.path.join(currentDir, 'filteredDF.pkl')):
        with open(os.path.join(currentDir, 'filteredDF.pkl'), 'wb') as file:
            pickle.dump(df_filtered, file)
    else:
        print("filteredDF.pkl already exists!")

    # Check if chronoDF.pkl already exists
    if not os.path.exists(os.path.join(currentDir, 'chronoDF.pkl')):
        with open(os.path.join(currentDir, 'chronoDF.pkl'), 'wb') as file:
            pickle.dump(df_time_summary, file)
    else:
        print("chronoDF.pkl already exists!")
    
    return None


def create_file_name(row):
    # Function to create the file name
    domainName = row['domain'].replace('.', '_')
    year = row['year']
    phase = row['phase']
    return f"{domainName}-{year}-{phase}.csv"

def is_valid_dir_name(subdirName):
    """
    Validates the input directory name.

    This function checks the following constraints to ensure the input string
    can be used as a valid directory name:
    1. It does not contain any invalid characters: <>:\"/\\|?*
    2. It is not a reserved name on Windows systems (e.g., CON, PRN, AUX, NUL, COM1, etc.).
    3. It does not exceed the maximum length of 255 characters.

    Parameters:
    subdirName (str): The directory name input provided by the user.

    Returns:
    bool: True if the directory name is valid, False otherwise.
    """
    # Define invalid characters
    invalid_chars = r'[<>:"/\\|?*]'

    # Check for invalid characters
    if re.search(invalid_chars, subdirName):
        print("Invalid directory name. It contains one or more of the following characters: <>:\"/\\|?*")
        return False

    # Check for reserved names (specific to Windows)
    reserved_names = [
        "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", 
        "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4", 
        "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"
    ]
    if subdirName.upper() in reserved_names:
        print("Invalid directory name. It is a reserved name.")
        return False

    # Check length
    if len(subdirName) > 255:
        print("Invalid directory name. It is too long.")
        return False

    return True


if __name__ == '__main__':
    file_path = "pull_policy_from_corpus\db_config.yml"
    global INPUT_LIST

    # Load the YAML file
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    
    db_path = config.get('RELEASE_DB_NAME', '')
    INPUT_LIST = config.get('input_list', [])

    dataInput(db_path)
    os._exit(0)