# Artifact Appendix

Paper title: Automating Governing Knowledge Commons and Contextual Integrity (GKC-CI)  Privacy Policy Annotations with Large Language Models

Artifacts HotCRP Id: 2

Requested Badge: Available and Functional 

## Description
Our artifact contains code that will fine-tune an LLM (GPT 3.5 Turbo) using the exact same configuration that was used to produce our best performing model. Because LLM training is non-deterministic, we cannot guarantee that our model will be reproduced exactly from this training process. However, we anticipate that the reproduced artifact will have extremely similar performance to the one in our paper. 

Moreover, we provide code to then uses the fine-tuned LLM as a tool to annotate GKC-CI tags in privacy policy texts which is the core contribution we are offering up for review. To that end, we also provide our visualizer that will take the annotations and highlight the policy text.

In addition, we provide a dataset of the annotation files for the `longitudinal`, `top n`, and `bottom n` polices which we analyze in Section 7 of our paper. This dataset has its own README where you can learn more about the dataset, what it contains, and how the annotation format works so that future researchers can use the data in their own projects. 


### Security/Privacy Issues and Ethical Concerns (All badges)
None.

## Basic Requirements (Only for Functional and Reproduced badges)
We require a stable internet connection, approximately 50 GB of hard drive space, and at minimum a single core 1 GHz cpu, 4 GB RAM on a local machine to reproduce the artifact. However, access to more powerful compute is required to reproduce the model, which we discuss in more detail in the following section. 

### Hardware Requirements
Our artifact requires access to a pod of powerful GPUs, as well as the proprietary GPT 3.5 Turbo model’s weights. These can be accessed using the API key provided in the HOTCRP submission site.

### Software Requirements
- The code was developed on a windows machine but it should run on a UNIX system with no issues. 
- The software packages needed can be found in the `requirements.txt` file in the artifacts repo. In addition, you'll need to install the python3-tk package for visualization.
- You will need access to OpenAI's compute resources; see the Hardware Requirements section.


### Estimated Time and Storage Consumption
- Fine-tuning the model happens non-locally as it uses OpenAI's compute. We estimate that it will take less than 3 hours.

- To run the full benchmark task it will take it will take somewhere between 2-3 hours.

- To run the parameter annotation task on one policy it will take less than an hour. There will be serveral of these.

- We estimate the storage consumption to be less than 50 GB
    - The majority of this storage is for the Princeton-Leuven Longitudinal Corpus of Privacy Policies – which is a project we utilize to pull privacy policies for analysis in Section 7 of the paper.  
    - For the data that we provide the majority of this storage is for the dataset artifact and is less than 1 GB.


## Environment 
1. Clone the repository from GitHub using the following command:

```
git clone https://github.com/JakeC007/Automated_GKC-CI_Privacy_Policy_Annotations.git
```

2. Ensure that all necessary dependencies are installed. 

```
pip install -r requirements.txt
```

3. For setting up the environment, make sure you have Python 3.x installed and configured on your system. Further, on Ubuntu or Debian, you'll need to install the python3-tk package for visualization. You can do this by running 

```
sudo apt-get install python3-tk
```

### Accessibility (All badges)
You may access the code here: https://github.com/JakeC007/Automated_GKC-CI_Privacy_Policy_Annotations

The most recent commit should be sufficient.


### Set up the environment (Only for Functional and Reproduced badges)
1. Clone repo and get tkinter

```bash
git clone https://github.com/JakeC007/Automated_GKC-CI_Privacy_Policy_Annotations.git
sudo apt-get install python3-tk
```
This should clone the repo and ensure you have tkinter. 

2. Create a new virtual environment 
3. Run `pip install -r requirements.txt`
4. Set the API key as an environment variable in your shell (see README Setup section for details)
5. Ensure that you have updated `MODEL_NAME` in both `prod.yml` and `benchmark/benchmark_config.yaml` with the one provided to you in the HOTCRP submission.

### Testing the Environment (Only for Functional and Reproduced badges)
N/A

## Artifact Evaluation (Only for Functional and Reproduced badges)

### Main Results and Claims
1. We provide datasets to fine-tune an OpenAI model to perform a GKC-CI parameter annotation task.
2. Our fine-tuned model can then be used to perform a GKC-CI parameter annotation task.
3. The output of our fine-tuned model can be visualized superimposed on the policy document using our visualizer.

### Experiments
<!-- List each experiment the reviewer has to execute. Describe:
- How to execute it in detailed steps.
- What the expected result is.
- How long it takes and how much space it consumes on disk (approximately).
- Which claim and results it supports, and how. -->

#### Experiment 1: Fine-Tuning GPT-3.5 Turbo

##### Expected result:
The output is the fine-tuned LLM. This will appear in the form of a Model ID.
##### How long it takes and how much space it consumes on disk (approximately)
This should consume a negligible amount of space on disk and run on OpenAI's servers for less than 3 hours.
##### Which claim and results it supports, and how.
This supports our claim that we provide the code needed for users to fine-tune their own model at home.
##### How to execute the experiment in detailed steps:

1. **Open the fine-tuning notebook**:
   ```bash
   jupyter notebook ft_turbo.ipynb
   ```

2. **Adjust the number of epochs**:
   - In the *Create fine-tune job* section, change `"n_epochs": 25` to `"n_epochs": 1`.

3. **Run the entire notebook**.

*Expected Output*: The final cell will display your model ID. You can ignore this, as we've provided the model ID for our top-performing model on hotCRP.

#### Experiment 2: Running the Benchmark Script

##### Expected result:
A directory `benchmark/bench_result` that will contain a `.csv` file with the parameter counts and a log file. 
#####  How long it takes and how much space it consumes on disk (approximately)
It will take less than an hour for the script to benchmark on a reduced dataset. Full benchmarking can take 2-6 hours.
Running the script creates less than 2 MB of data.
##### Which claim and results it supports, and how.
This is our benchmarking script mentioned in Section 5 of the paper.

##### How to execute the experiment in detailed steps:

1. **Navigate to the root directory of the repo**.

2. **Edit the YAML file**:
   Edit `benchmark/benchmark_config.yaml` and set:
   - `RUN_SUBSET` to `True`
   - `N_EXAMPLES` to 500

   In doing so, the benchmark will only run 500 examples, which should be sufficient for the Functional badge. We limit the number of examples to ensure that our limited funds will last for all of the experiments.

3. **Run the script**:
   ```bash
   python benchmark/benchmark_model.py
   ```
*Expected Output*: After running, there will be a directory `benchmark/bench_result` that will contain a `.csv` file with the parameter counts and a log file. The `.csv` file may be used to compute the accuracy of the model, but that is out of scope for the "functional" badge.

#### Experiment 3: Performing Policy Annotation A La Carte
##### Expected result:
An annotation file and CSV file in the `results` folder.
##### How long it takes and how much space it consumes on disk (approximately)
It will take less than an hour for the script to annotate a policy file.
Running the script creates less than 2 MB of data.
##### Which claim and results it supports, and how.
This is part of the tool we promised to release in our paper at various points, including the abstract and the conclusion.

##### How to execute the experiment in detailed steps:

1. **Navigate to the root directory of the repo**.

2. **Run the text parse script**:
   ```bash
   python text_parse.py
   ```

*Expected Output*: `.csv` files will appear in `policy_text/processed`, containing the same company names as those in `policy_text_raw`.

3. **Run the policy annotation script**:
   ```bash
   python production.py
   ```

*Expected Output*: This will create files in the `results` folder:
- `results/csvs/adobe2011-B_output_results.csv` (summary of parameter counts for each policy)
- `results/logs/adobe2011-B_output_results.csv` (detailed, line-by-line output for each parsed policy)
- `results/summary.csv` (contains summary stats for large-scale parameter analysis)

#### Experiment 4: Performing Policy Annotation on the Princeton-Leuven Longitudinal Corpus of Privacy Policies
##### Expected result:
Produce an annotation file and CSV file in the `results` folder by pulling data from the Princeton-Leuven Longitudinal Corpus of Privacy Policies.
##### How long it takes and how much space it consumes on disk (approximately)
It will take less than an hour for the script to annotate a policy file. However, the time to download the 3 GB database zip depends on your internet connection.
Downloading and unzipping the corpus will take up ~48 GB; the script creates less than 2 MB of data.
##### Which claim and results it supports, and how.
This is part of the tool we promised to release in our paper at various points, including the abstract and the conclusion. This pipeline is how we completed the at-scale annotation discussed in Section 7.

##### How to execute the experiment in detailed steps:

1. **Download the database**  
   Request to download the `.db` file (*release_db.sqlite.xz*) from the corpus website. Link is [here](https://privacypolicies.cs.princeton.edu/data). Then unzip the DB.

2. **Configure database settings**  
   Update the `db_config.yml` file within the `pull_policy_from_corpus` directory with the path to your downloaded database.

3. **Run the DB parse script**:
   ```bash
   python pull_policy_from_corpus/dbOperations.py
   ```
   - In the terminal, it will say "Loading in data...". On the first run, this can take 10-15 minutes as data is loaded in and `.pckl` files are created. Subsequent runs (which are outside the scope) take less than 5 minutes.
   - In the terminal, answer the DATA INPUT question by entering `4`.
   - In the terminal, answer the number of top sites you'd like to retrieve by entering `1`.
   - In the terminal, answer the DATA PROCESSING question by entering `2`.

*Expected Output*: `policy_text/corpus` should now have new CSV file inside.

4. **Modify production's `yaml` file**  
   In `prod.yml`, change `PLC_DBO` to `True`.

5. **Run the policy annotation script**:
   ```bash
   python production.py
   ```

*Expected Output*: This will populate files in `results/csvs`, `results/logs`, and write rows in `results/summary.csv`. Be on the lookout for terminal output stating "Processing the following files." This explains which files are going to be run, as the files not listed as being processed in `policy_text/corpus` are already logged in the summary file with their parameter outputs.

#### Experiment 5: Running the Visualizer
The following will:
- Display in a GUI an annotated policy overlaid on the policy text.
- It will take less than a minute for the script to generate the GUI, and then the script will run until you close the window.
- Running the script takes up no extra hard disk space.
- This demos the tool discussed in Section 7.3.

1. **Navigate to the root directory of the repo**.

2. **Run the script**:
   ```bash
   python visualizer/vizit.py
   ```

*Expected Output*: A GUI with an annotated policy overlaid on the policy text will appear.

3. **Close the GUI windows**.

#### Experiment 6: Replicating the Section 7 Graphs
##### Expected result:
Graphs like those in Section 7. See the table below for a mapping between fig number and filename.

| Figure Number | File Name                                  |
|---------------|--------------------------------------------|
| Fig 6         | variance.png                              |
| Fig 7         | paramdensity.png                          |
| Fig 8         | popularity-param-count.png                |
|               | popularity-param-ratio.png                |


##### How long it takes and how much space it consumes on disk (approximately)
It will take less than a minute for each script to run.
The generated graphs will only take up ~2.80 MB of space.

##### Which claim and results it supports, and how.
Counts for the GKC-CI parameters longitudinally, with particular focus on the variance 
of individual parameter types as well as the density of parameters.

##### How to execute the experiment in detailed steps:
1. **Download the data artifact to your machine**  
   It is [here](https://github.com/JakeC007/Automated_GKC-CI_Privacy_Policy_Annotations/releases/tag/data).

2. **Navigate to the root directory of the data artifact**:
   ```bash
   cd /path/to/data-artifact
   ```

3. **Run the analysis scripts**:

   - **Run `crossindustry-figures.py`**:
     ```bash
     python analysis/crossindustry-figures.py
     ```
     *Expected Output*: `variance.png` and `paramdensity.png`.

   - **Run `longitudinal-figures.py`**:
     ```bash
     python analysis/longitudinal-figures.py
     ```
     *Expected Output*: `longitudinal.png`.

   - **Run `popularity-stats-figs.py`**:
     ```bash
     python analysis/popularity-stats-figs.py
     ```
     *Expected Output*: `popularity-param-count.png` and `popularity-param-ratio.png`.

## Limitations (Only for Functional and Reproduced badges)
As noted elsewhere, resources are limited, particularly in terms of funding. To stay within our budget, we have reduced the number of epochs for fine-tuning, limited the parameter annotation tasks for the benchmark, and restricted the number of documents pulled from the Princeton-Leuven Longitudinal Corpus of Privacy Policies. These decisions ensure we have enough funds to demonstrate that the various Python scripts run successfully. Since our focus is on functionality rather than reproducibility, we believe these limitations are acceptable.

Further, none of the results of our paper are possible to be reproduced exactly for two reasons. Firstly, LLM training occurs non-deterministically largely due to differences in GPU instruction set architectures, as discussed in more detail in https://arxiv.org/pdf/2403.09603. Secondly, we use OpenAI’s default generation parameters to produce our results, which at the time of writing had a non-zero temperature. This means that model generation happens non-deterministically, and thus, annotation happens non-deterministically. That being said, although our results may not be reproduced exactly, we anticipate that anyone will be able to produce a model which has very similar performance as the one we describe in our paper, particularly in aggregate.

We do not provide the code to fine-tune the open-source models due to the excessive computational costs required to run those models. Further, we believe our main artifact of interest is our best-performant model. 


## Notes on Reusability (Only for Functional and Reproduced badges)
Individuals may choose to use our provided data and code to produce a model capable of annotating privacy policies other than the policies that we provide in our repository. Further, individuals may choose to use our data to fine-tune models other than the ones we investigate in our study, such as GPT-4-o or Gemini 1.5 Pro. Finally, individuals may choose to use our data to investigate the extent to which Retrieval Augmented Generation (RAG) based approaches are possible for GKC-CI annotation.
