""""
File that pulls from previous benchmarking scripts to measure how well GPT3.5-Turbo is doing

J. Chanenson with contributions and full functions by Madison Pickering
8/23/23; slight cleanup 12/4/24
"""
import pandas as pd
from tqdm import tqdm
from tiktoken import get_encoding
import re, ast, traceback, os, openai, datetime, time, yaml


################## DO NOT TOUCH ##################
encoding = get_encoding("cl100k_base") # Fetch the encoding method only once
#################################################


#################### GLOBALS ####################
## Load configuration
with open("benchmark/benchmark_config.yaml", "r") as file:
    config = yaml.safe_load(file)

MODEL_NAME = config['MODEL_NAME']
OUTPUT_DIR = config['OUTPUT_DIR']

RUN_SUBSET = config['RUN_SUBSET']
N_EXAMPLES = config['N_EXAMPLES']

FILEPATH_TEST = "benchmark/test_data_full.csv" #standard benchmark


## Book Keeping For Benchmark Stats
END_Y_STR = "x-x-x"
LABELS = ["PERFECT_MATCH", "MATCH", "MATCH_ERROR", "PARSE_FAILURE"]
AIM_PERFORMANCE = [0, 0, 0, 0]
ATTRIBUTE_PERFORMANCE = [0, 0, 0, 0]
SUBJECT_PERFORMANCE = [0, 0, 0, 0]
SENDER_PERFORMANCE = [0, 0, 0, 0]
RECIPIENT_PERFORMANCE = [0, 0, 0, 0]
MODALITY_PERFORMANCE = [0, 0, 0, 0]
CONDITION_PERFORMANCE = [0, 0, 0, 0]
CONSEQUENCE_PERFORMANCE = [0, 0, 0, 0]
TRANSMISSION_PRINCIPLE = [0, 0, 0, 0]
NEG_PERFORMANCE = [0, 0, 0, 0]
CURRENT_DATETIME = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def main():
    # #1. load the testing data
    
    # Set up throttler to keep us under OpenAI's token per min quota   
    global throttler
    throttler = Throttler()

    global alldata
    if RUN_SUBSET:
        alldata = get_test_df().sample(N_EXAMPLES, random_state=12, replace = False)
    else:
        alldata = get_test_df()

    ##3. benchmark
    log = benchmark(model=MODEL_NAME)

    ##4. log results of benchmarking
    logOutput(outputs_to_log=log)

    ##5. export param stats
    exportParamStats()

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
    return response["choices"][0]["message"]["content"]

def queryGPT3(prompt, model):
    """
    Queries GPT3 and returns the response

    Parameters
    - prompt (str): The prompt or question to be sent to the model.
    - model (str): The model ID or name to be used for querying.

    Returns 
    - output from model
    """
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=0, #should probably be 1
        top_p=0.9,
        stop=["]}x-x-x"]
    )
    return response["choices"][0]["text"]

def queryGPT4(prompt):
    """
    Queries GPT3 and returns the response

    Parameters
    - prompt (str): The prompt or question to be sent to the model.
    - model (str): The model ID or name to be used for querying.

    Returns 
    - output from model
    """
    
    completion = openai.ChatCompletion.create(
      model="gpt-4",
      messages=prompt,
      max_tokens=500
    )

    # Add the response tokens so the throttler has info needed to do its job
    global throttler
    throttler.addResponseTokensToCount(completion["usage"]["completion_tokens"])

    return completion.choices[0].message.content 

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
        message = processTextForOPENAI(text, throttler)
        response = queryTurbo(message, model)
    else:
        raise ValueError("model not turbo")

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
        if self.tokenCount + tokensCount <= 9000:
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

    # Tokenize the text and return the token count.
    return len(encoding.encode(text))

def processTextForOPENAI(text, throttler):
    """
    Processes the text, ensuring it's safe to send to GPT-4 based on the throttling restrictions.
    """
    # Calculate token count for the provided text.
    tokensCount = countTokens(str(text))

    # If the token count exceeds 6500, raise an error.
    if tokensCount > 6500:
        raise ValueError("The provided text exceeds the 6,500 token limit!")

    # Wait until it's safe to send the text.
    while not throttler.canSend(tokensCount):
        time.sleep(throttler.waitTime())

    # Once it's safe, return the text.
    # print("Time to submit the job")
    return text

def exportParamStats():
    """
    Exports the collected param stats to CSV
    """
    data = {
        "LABELS": LABELS,
        "AIM_PERFORMANCE": AIM_PERFORMANCE,
        "ATTRIBUTE_PERFORMANCE": ATTRIBUTE_PERFORMANCE,
        "SUBJECT_PERFORMANCE": SUBJECT_PERFORMANCE,
        "SENDER_PERFORMANCE": SENDER_PERFORMANCE,
        "RECIPIENT_PERFORMANCE": RECIPIENT_PERFORMANCE,
        "MODALITY_PERFORMANCE": MODALITY_PERFORMANCE,
        "CONDITION_PERFORMANCE": CONDITION_PERFORMANCE,
        "CONSEQUENCE_PERFORMANCE": CONSEQUENCE_PERFORMANCE,
        "TRANSMISSION_PRINCIPLE": TRANSMISSION_PRINCIPLE,
        "NEG_PERFORMANCE": NEG_PERFORMANCE
    }

    df = pd.DataFrame(data) 
    # Write out to file
    model_formatted = MODEL_NAME.replace(":", "_")
    logfile = os.path.join(OUTPUT_DIR, f"benchmarking_log_{CURRENT_DATETIME}_{model_formatted}.csv")
    ensurePathExists(logfile)
    df.to_csv(logfile, index=False, encoding='utf-8')  

####MADISON CODE DO NOT TOUCH. MATCHES SLURM_TEST AS OF 8/23/23###########################################################################
def get_test_df():
    ''' reads in a small number of single files as a df and returns it'''
    testfile = FILEPATH_TEST.split("/")[-1]
    TEST_DIR = FILEPATH_TEST[:-len(testfile)]
    # print(f"testfile={testfile}, TEST_DIR={TEST_DIR}")
    files = [testfile]
    # print(f"Reading contents of {files}...")
    alldata = None
    for file in files:
        # print(f"filepath = {TEST_DIR + file}")
        df = pd.read_csv(TEST_DIR + file)
        if (alldata is None):
            alldata = df
        else:
            alldata = pd.concat(objs=[alldata, df])

    return alldata
 
def updateErrorList(groundtruth, completion, key, prompt, currentlist, log=None, neglist=None):
    '''
    conveinence function: given the ground truth (y), completion (y'), key (aim, attribute, ...),
        prompt (X), and currentlist (AIM_PERFORMANCE, SENDER_PERFORMANCE...)
    parse whatever error may exist and update currentlist appropriately

    log is a list used to log outputs to a file
    neglist is a list used to log results for negative examples
    '''
    #0. Determine which list we should update
    update_neg = False
    updatelist = currentlist
    neg_groundtruth = key + " N/A"
    if (groundtruth == neg_groundtruth):
        updatelist = neglist
        update_neg = True
   
    #1. check if theres a parse failure
    prefix = completion[0: len(key)]
    if (prefix != key):
        log.append(f"PARSE ERROR: missing key. key={key}, completion prefix={prefix}")
        updatelist[3] = updatelist[3] + 1
        return

    #2. Check if we can short-circut for a match error
    #2.1 format groundtruth, outputs into list; if list sizes are unequal, then match error
    ymatch = groundtruth[len(key) + 1:]
    if (ymatch[0] != "["):
        ymatch = [ymatch]
    else:
        ymatch = ast.literal_eval(ymatch)
    completion_match = completion[len(key) + 1:]
    if (len(completion_match) == 0 or completion_match[0] != "["):
        completion_match = [completion_match]
    else: #2.1.1: fix malformed lists
        if (completion_match[-2:] != "']"):
            completion_match += "']"
        elif(completion_match[-1:] != "]"):
            completion_match += "]"
        try:
            completion_match = ast.literal_eval(completion_match)
        except SyntaxError as s: #parse failure that we couldnt fix
            log.append(f"PARSE ERROR: could not evaluate the list when read in. key={key}, completion prefix={prefix}")
            updatelist[3] = updatelist[3] + 1
            return

    if (len(completion_match) != len(ymatch)):
        updatelist[2] = updatelist[2] + 1
        log.append(f"MATCH ERROR: ymatch={ymatch}; completion_match={completion_match}")
        return

    #2.2 list sizes are equal--for each element in ymatch, the completion needs to have all the words in ymatch
    allmatches = True
    for element in ymatch:
        words = element.split(" ")
        reggy = ".*"
        for word in words:
            reggy += word + ".*"
        matchExists = False
        reggy = reggy.replace(")", "\)")
        reggy = reggy.replace("(", "\(")
        reggy = reggy.replace("[", "\[")
        reggy = reggy.replace("]", "\]")
        reggy = reggy.replace("^", "\^")
        reggy = reggy.replace("$", "\$")
        #log.append(f"built regex: {reggy}")
        for somestr in completion_match:
            if (re.search(reggy, somestr) is not None):
                matchExists = True
                break
        if (matchExists == False):
            allmatches = False
            break
    if (allmatches == False):
        updatelist[2] = updatelist[2] + 1
        log.append(f"MATCH ERROR: ymatch={ymatch}; completion_match={completion_match}")
        return
    
    #3. all elements of y are also in the completion...check if we have a perfect match!
    perfectMatches = True
    if (update_neg):
        if (completion.strip() != neg_groundtruth):
            perfectMatches = False
    else:
        for somestr in completion_match:
            if (somestr not in prompt):
                perfectMatches = False
                break
    if (perfectMatches):
        updatelist[0] = updatelist[0] + 1
        log.append(f"PERFECT MATCH: ymatch={ymatch}, completion_match={completion_match}")
        return
    
    #4. We have a normal match: all elements of y are also in the completion, but not all elements of y are also in prompt.
    updatelist[1] = updatelist[1] + 1
    log.append(f"MATCH: ymatch={ymatch}, completion_match={completion_match}")


def benchmark(model=None, tokenizer=None):
    ''' benchmarks the model. calls updateErrorList as a subroutine
        returns the log of the prompt/completions and how it was categorized
    '''
    prog = tqdm(range(len(alldata)))
    outputs_to_log = []
    for i in range(0, len(alldata)):
        
        #1. Loop thru all examples, prompting the model for completions
        X, y = alldata.iloc[i]
        y = y.strip(END_Y_STR)
        y = y.strip() #get rid of any extra whitespace
        outputs_to_log.append(f"prompt = {X}")
        outputs_to_log.append(f"ground truth = {y}")
        
        try:
            # completion = getCompletion(prompt=X, model=model, tokenizer=tokenizer)
            completion = getOpenAICompletion(prompt=X, model=model) 
            if len(completion.strip()) == 0:
                completion = "LLM_ERROR" #this will get marked as a parse error
            '''
            print(f"prompt={X}")
            print(f"completion={completion}")
            '''
            #2. Parse the completion by trimming off "x-"'s, whitespace
            completion = completion.split("x-")[0]
            completion = completion.strip()
            if len(completion) == 0:
                completion = "LLM_ERROR" #this will get marked as a parse error
            outputs_to_log.append(f"completion={completion}")

            #3. Determine key type so we can record errors correctly
            # remove brackets which are causing uncatchable parse errors
            completion = completion.replace("['", "").replace("']", "").replace('["', '').replace('"]', '')
            y = y.replace("['", "").replace("']", "").replace('["', '').replace('"]', '')

            if (y[0:4] == "Aim:"):
                updateErrorList(groundtruth=y, completion=completion, key="Aim:", prompt=X, currentlist=AIM_PERFORMANCE, log=outputs_to_log, neglist=NEG_PERFORMANCE)
            elif (y[0:10] == "Attribute:"):
                updateErrorList(groundtruth=y, completion=completion, key="Attribute:", prompt=X, currentlist=ATTRIBUTE_PERFORMANCE, log=outputs_to_log, neglist=NEG_PERFORMANCE)
            elif (y[0:8] == "Subject:"):
                updateErrorList(groundtruth=y, completion=completion, key="Subject:", prompt=X, currentlist=SUBJECT_PERFORMANCE, log=outputs_to_log, neglist=NEG_PERFORMANCE)
            elif (y[0:7] == "Sender:"):
                updateErrorList(groundtruth=y, completion=completion, key="Sender:", prompt=X, currentlist=SENDER_PERFORMANCE, log=outputs_to_log, neglist=NEG_PERFORMANCE)
            elif (y[0:10] == "Recipient:"):
                updateErrorList(groundtruth=y, completion=completion, key="Recipient:", prompt=X, currentlist=RECIPIENT_PERFORMANCE, log=outputs_to_log, neglist=NEG_PERFORMANCE)
            elif (y[0:9] == "Modality:"):
                updateErrorList(groundtruth=y, completion=completion, key="Modality:", prompt=X, currentlist=MODALITY_PERFORMANCE, log=outputs_to_log, neglist=NEG_PERFORMANCE)
            elif (y[0:10] == "Condition:"):
                updateErrorList(groundtruth=y, completion=completion, key="Condition:", prompt=X, currentlist=CONDITION_PERFORMANCE, log=outputs_to_log, neglist=NEG_PERFORMANCE)
            elif (y[0:12] == "Consequence:"):
                updateErrorList(groundtruth=y, completion=completion, key="Consequence:", prompt=X, currentlist=CONSEQUENCE_PERFORMANCE, log=outputs_to_log, neglist=NEG_PERFORMANCE)
            elif (y[0:23] == "Transmission-Principle:"):
                updateErrorList(groundtruth=y, completion=completion, key="Transmission-Principle:", prompt=X, currentlist=TRANSMISSION_PRINCIPLE, log=outputs_to_log, neglist=NEG_PERFORMANCE)
            else:
                raise Exception(f"couldnt parse label y={y}")
            '''
            if i == 2:
                break
            '''
            #print(i)
            
        except openai.InvalidRequestError as e:
            print(f"ERROR! too long for index i={i}. Skipping over this example...")
            traceback.print_exc()
        except Exception as e:
            print(f"ERROR! connection error (manually verify this is the case) for i={i} Skipping over this example...")
            traceback.print_exc()
        finally:
            prog.update(1)
    return outputs_to_log

def logOutput(outputs_to_log=None):
    model_formatted = MODEL_NAME.replace(":", "_") if ":" in MODEL_NAME else MODEL_NAME
    logfile = os.path.join(OUTPUT_DIR, f"benchmarking_log_{CURRENT_DATETIME}_{model_formatted}")
    ensurePathExists(logfile)
    with open(logfile, "w", encoding='utf-8') as f:
        for line in outputs_to_log:
            if ("prompt = " in line):
                f.write(f"\n\n{line}\n")
            else:
                f.write(f"{line}\n")
    print(f"File written to {logfile}")

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
        print(f"Directory {directory} created.")



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




if __name__ == "__main__":
    main()







