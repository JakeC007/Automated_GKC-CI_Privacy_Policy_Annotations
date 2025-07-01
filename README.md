# GKC – CI Automated Parameter Annotation Tool

## Overview

This repository is the companion artifact to the paper *Automating Governing Knowledge Commons and Contextual Integrity (GKC-CI)  Privacy Policy Annotations with Large Language Models* which will appear in PETS 2025.

This repository provides tools for automating parameter annotation using OpenAI’s API. It is designed to parse policy text files and output a summary in `.csv` format along with detailed logs. Please follow the setup and usage instructions below to get started.



**N.B.:** If you are looking for the 456 annotated privacy polices please see the [Privacy Policy Annotations](https://github.com/JakeC007/Automated_GKC-CI_Privacy_Policy_Annotations/releases/tag/data) release.

## DIR Overview
```bash
+---benchmark # contains code needed to benchmark the finetuned model
+---create_model # contains code needed to create the finetuned model
|   \---data
|           
+---policy_text # holds policy texts 
|   +---corpus # where dbOperations.py's processed policy text files are saved
|   +---processed # where text_parse.py's processed policy text files are saved    
|   \---raw # save raw policy text files here and then run text_parse.py
|           
+---pull_policy_from_corpus # contains code to extract processed policy text files from Princeton-Leuven Longitudinal Corpus
|       
+---results # where output from the LLM is saved
\---visualizer # contains code and data to run the visualizer
    +---viz_annotation # store annotation files here (LLM output)  
    \---viz_text # store processed text files here (pull from policy_text subdirs)
```

## Enviroment Setup

1. **Install Dependencies**  
   - Run the following command to install the required Python packages: `pip install -r requirements.txt` 
   - **Note:** we assume that Python has `tkinter` installed as part of Python's standard library. Some Linux distros may not have it installed by default and you will need to install it. On Ubuntu/Debian run `sudo apt-get install python3-tk`.

2. **Configure OpenAI API Key**  
   - Obtain your API key from [OpenAI](https://platform.openai.com/account/api-keys).
   - Set the API key as an environment variable in your shell (see instructions below).

### Setting the API Key as an Environment Variable

To securely configure your OpenAI API key:

Note if you have `~/.bashrc` instead of `~/.bash_profile`. Just swap out `~/.bashrc` for`~/.bash_profile` if that is the case.

1. Run the following command, replacing `yourkey` with your actual API key:
   `echo "export OPENAI_API_KEY='yourkey'" >> ~/.bash_profile`
2. Load the updated shell configuration: `source ~/.bash_profile`
3. Verify that the variable was set correctly: `echo $OPENAI_API_KEY`

## First Time Set Up

1. **Finetune The Model and Create a Model ID**  
   Run `create_model\ft_turbo.ipynb` to create and retrieve your model ID.

2. **Update the Model Configuration**  
   Open `prod.yml` and update the `MODEL_NAME` field with your generated model ID.

3. **(Optional) Benchmark Your New Model**
   - Open `benchmark\benchmark_config.yaml` and update the `MODEL_NAME` field with your generated model ID. 
   - Run the model via `benchmark\benchmark_model.py`. 
   - The `.csv` file produced from the script can be used to calculate accuracy. 

## Usage – Bring Your Own Policy Files

1. **Prepare Policy Text Files**  
   Place your policy text files in the `policy_text/raw` directory.

2. **Parse Data**  
   Use `text_parse.py` to parse the policy data.

3. **Run the Model**  
   Execute the main script in `production.py` to process your files.

   - This will generate a `.csv` file with parameter counts and a detailed log file for each policy processed.

## Usage – Pull Privacy Policies From the Princeton-Leuven Longitudinal Corpus of Privacy Policies

> **Note**: If you plan to use the [Princeton-Leuven Longitudinal Corpus of Privacy Policies](https://privacypolicies.cs.princeton.edu/), you’ll need to set up additional configurations and download the [dataset in SQLite format](https://privacypolicies.cs.princeton.edu/data).

1. **Download the Database**  
   Request to download the `.db` file (*release_db.sqlite.xz*) from the corpus website. Link is [here](https://privacypolicies.cs.princeton.edu/data). Then unzip the db.

2. **Configure Database Settings**  
   Update the `db_config.yml` file within the `pull_policy_from_corpus` directory with the path to your downloaded database.

3. **Run the Script**  
   Execute `dbOperations.py` within the `pull_policy_from_corpus` folder to pull the privacy policies from the corpus into your dataset.
   Note:
      - To pull entries from a custom user-defined list of domains you must specify your input list in the `db_config.yml` file.
      - To pull your enteries from the [Tranco list](https://tranco-list.eu/) no customization is needed. Simply run the file.
      - When asked about data processing be sure to select *Process Data Into LLM Input* to get the processed policy texts.

3. **Run the Model**  
   Execute the main script in `production.py` to process your files.

   - This will generate a `.csv` file with parameter counts and a detailed log file for each policy processed.


### Options Explanation for `dbOperations.py`

1. **User List**: 
   - Choose **1** to load all longitudional entries from a custom user-defined list -- defined in `pull_policy_from_corpus\db_config.yml`.
   - Choose **2** to load only the most recent entry from the custom user-defined list -- defined in `pull_policy_from_corpus\db_config.yml`.
   
2. **Tranco List**:
   - Choose **3** to retrieve a specified number of top websites from Tranco (all entries).
   - Choose **4** to retrieve only the most recent top websites from Tranco.

After selecting a data source, the script filters and loads the data accordingly.

### Data Processing Options

After filtering the data, you’ll have options on how to proceed:
- **Token Counts and Cost**: Calculates the token count and associated costs for the dataset.
- **Process Data Into LLM Input**: Prepares data to be used as input for a language model.
- **Print Domains in Tranco Rank Order**: Outputs the domains in the Tranco rank order into a dataframe.
- **Export df with Policy Texts**: Exports the current `df` with the included policy texts.
- **Export Policy Texts for Visualization**: Outputs policy texts for external visualization or analysis.

## `production.py` Output

- **CSV File**: Provides a summary of parameter counts for each policy.
- **Log File**: Contains detailed, line-by-line output for each parsed policy.

## Visualization

The `visualizer` folder contains two key subdirectories:

- **`viz_annotation`**: This folder should contain the "log files" generated by the LLM. These files are typically named in the format `XXXX_results` (e.g., `adobe_com-2011-B_results`). 
- **`viz_text`**: This folder should hold the processed policy files. These files can be from the corpus or from the policy files you use (the processed versions should be in `policy_text\processed`). **Important**: Do not use raw policy files as they may have incorrect formatting, which could lead to errors during visualization.

### Using the Visualizer

1. **Configure Files**: 
   The visualizer is configured using the `viz_config.yml` file. In this configuration file, you will specify the paths to the text and annotation files. 
   - Set the `text_file` to the path of your processed text file from the `viz_text` folder.
   - Set the `annotation_file` to the corresponding log file from the `viz_annotation` folder.
   
   - Set the `mode` to `single` to run one visualization without multithreading.
   - Set the `mode` to `multiple` to run two visualizations in parallel using multithreading.

2. **Run the Visualization**: 
   Once the YAML file is properly configured, run the visualization script (`vizit.py`). This will generate the visualizations based on the selected files.

### Explanation of Highlights Output File

In addition to generating visualizations, the tool will also produce an explanation of highlights file (`explanation-of-highlights.txt`). This feature provides:
- The **sentence** where the annotation is found.
- The **substring** that corresponds to the parameter.
- The **parameter name** associated with that annotation.

This helps provide context to the visualizations by linking the highlighted text to specific annotations.

## Parameter Analysis at Scale 

To analyze how parameters change at scale and over time, use the summary file output by `production.py` as the basis for creating figures or graphs. The `analysis` folder in the [data release](https://github.com/JakeC007/Automated_GKC-CI_Privacy_Policy_Annotations/releases/tag/data) contains sample scripts used to generate the figures in Section 7. However, creating the figures/graphs based on *user generated* summary files is the user's responsibility.


## Citation

If you use this repository in your work, please cite our project:

The PETS citation (preferred) is below.

**Bibtex Citation:**

```bibtex
@article{chanenson2025,
  title = {Automating {{Governing Knowledge Commons}} and {{Contextual Integrity}} ({{GKC-CI}}) {{Privacy Policy Annotations}} with {{Large Language Models}}},
  author = {Chanenson, Jake and Pickering, Madison and Apthrope, Noah},
  year = {2025},
  journal = {Proceedings on Privacy Enhancing Technologies},
  issn = {2299-0984},
  urldate = {2025-04-29},
  note = {\url{https://petsymposium.org/popets/2025/popets-2025-0062.php}}
}
```

**IEEE Formatted Citation** 

> J. Chanenson, M. Pickering, and N. Apthrope, “Automating Governing Knowledge Commons and Contextual Integrity (GKC-CI) Privacy Policy Annotations with Large Language Models,” *Proceedings on Privacy Enhancing Technologies*, 2025, Accessed: Apr. 28, 2025. [Online]. Available: https://petsymposium.org/popets/2025/popets-2025-0062.php

## Troubleshooting

### Common Issues on Ubuntu

- **Command Not Found Error**  
   If you encounter an error stating that the `openai` command was not found:

   1. Verify that `~/.local/bin/` is in your `$PATH`:
      ```bash
      echo $PATH | grep -q "~/.local/bin" || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc
      ```
      This ensures `~/.local/bin/` is added to your `$PATH` if it isn't already, and reloads your shell configuration.

   2. Verify the installation:
      ```bash
      pip3 show openai
      ```
   
   3. If installed, ensure the `openai` command is accessible:
      ```bash
      alias openai=~/.local/bin/openai
      ```
   
   4. To make this change permanent, add the alias to your `~/.bashrc` file:
      ```bash
      echo "alias openai=~/.local/bin/openai" >> ~/.bashrc
      ```
