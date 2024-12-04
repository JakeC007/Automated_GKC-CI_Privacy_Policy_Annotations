"""
Viz of privacy policy given LLM log ouputs and outputs.
Exports a file called `explanation-of-highlights.txt` to explain highlights.
7/22/24
J. Chanenson
"""

"""
as the full privacy policies with text color-highlighted by annotated parameter, 
or side-by-side comparisons of sentences that have changed between 
longitudinal policy versions with the parameters highlighted
"""

import re, os, yaml
import tkinter as tk
from tkinter import scrolledtext, messagebox
import threading

# # Dispatch table for keyword to color mapping
# KEYWORD_COLOR_MAP = {
#     'Aim': 'blue',
#     'Attribute': 'orange',
#     'Sender': 'purple',
#     'Subject': 'brown',
#     'Consequence': 'red',
#     'Modality': 'green',
#     'Recipient': 'cyan',
#     'Condition': 'magenta'
# }

# REVERSE_MAP = {
#     'blue': 'Aim',
#     'orange': 'Attribute',
#     'purple': 'Sender',
#     'brown': 'Subject',
#     'red': 'Consequence',
#     'green': 'Modality',
#     'cyan': 'Recipient',
#     'magenta': 'Condition'
# }

# Dispatch table for keyword to color mapping
KEYWORD_COLOR_MAP = {
    'Aim': '#1f77b4',       # Blue
    'Attribute': '#ff7f0e', # Orange
    'Sender': '#2ca02c',    # Green
    'Subject': '#d62728',   # Red
    'Consequence': '#9467bd', # Purple
    'Modality': '#8c564b',  # Brown
    'Recipient': '#e377c2', # Pink
    'Condition': '#7f7f7f'  # Gray
}

REVERSE_MAP = {
    '#1f77b4': 'Aim',
    '#ff7f0e': 'Attribute',
    '#2ca02c': 'Sender',
    '#d62728': 'Subject',
    '#9467bd': 'Consequence',
    '#8c564b': 'Modality',
    '#e377c2': 'Recipient',
    '#7f7f7f': 'Condition'
}

def write_list_to_file(file_path, data_list):
    """
    Writes the contents of a list to a text file, each item on a new line.

    Args:
        file_path (str): The path to the file where the list should be written.
        data_list (list of str): The list of strings to write to the file.
    """
    with open(file_path, 'w') as file:
        for item in data_list:
            file.write(item)

def read_text_file(file_path):
    """Read the content of a text file.

    Args:
        file_path (str): Path to the text file.

    Returns:
        str: Content of the text file.
    """
    with open(file_path, 'r') as file:
        return file.read()

def read_annotation_file(file_path):
    """Read the annotation file.

    Args:
        file_path (str): Path to the annotation file.

    Returns:
        list: Lines of the annotation file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()

def parse_annotations(annotation_lines):
    """Parse the annotations and return a list of tuples with the text segment, keyword segment, and color.

    Args:
        annotation_lines (list): Lines of the annotation file.

    Returns:
        list: List of tuples containing text segment, keyword segment, and color.
    """
    annotations = []
    text_segment = None
    keyword = None

    for line in annotation_lines:
        # Skip newlines 
        line = line.strip()  # Remove leading/trailing whitespace
        if line == "":
            continue

        if line.startswith("prompt ="):
            # Extract text segment from the 'Annotate' line
            match_text = re.match(r"prompt = Annotate: \['(.*?)'\]", line)
            if match_text:
                text_segment = match_text.group(1)
            else:
            #if regex fails do it manually
                text_segment = line.split(":")[1].strip().strip("[]").strip("'")

        elif '-->' in line:
            keyword = line.split("-->")[0].strip(' \t\n\r\'"')
        
        elif line.startswith("completion ="):
            try:
                keyword_segment = line.split(":")[1].strip().strip("[]").strip("'")
            except:
                #no keyword segment then we reset
                text_segment = None
                keyword = None
                continue
            
            # Segment is not N/A then we have an annotation
            if keyword_segment != "N/A" and text_segment and keyword and keyword_segment in text_segment:
                color = KEYWORD_COLOR_MAP.get(keyword, 'black')  # Default color is black if keyword not found
                annotations.append((text_segment, keyword_segment, color))
                # Reset text_segment and keyword for the next annotation
                text_segment = None
                keyword = None
            else:
                # it was N/A and we reset
                text_segment = None
                keyword = None
    return annotations

def color_text(text_widget, text, annotations):
    """Apply colors to the specified sections of the text in the text widget.

    Args:
        text_widget (tk.Text): Text widget to display colored text.
        text (str): The entire text content.
        annotations (list): List of annotations with text segment, keyword segment, and color.
    """
    text_widget.delete(1.0, tk.END)  # Clear existing text
    text_widget.insert(tk.END, text)
    text_widget.update_idletasks()  # Ensure the text is inserted before calculating indices
    
    output_list = []

    for annotation in annotations:
        text_segment, keyword_segment, color = annotation
        param = REVERSE_MAP.get(color, "NONE")
        
        output_list.append(f"Full Segment: {text_segment}\nTagged: {keyword_segment}\n{param}\n")
        output_list.append("=="*20)
        output_list.append("\n\n\n")

        
        try:
            # Find the start and end indices of the text segment
            start_idx = text.index(text_segment)
            end_idx = start_idx + len(text_segment)
            segment_text = text[start_idx:end_idx]
            
            # Find the indices of the keyword segment within the text segment
            keyword_start_idx = segment_text.index(keyword_segment)
            keyword_end_idx = keyword_start_idx + len(keyword_segment)

            global_start_idx = start_idx + keyword_start_idx
            global_end_idx = start_idx + keyword_end_idx

            # Convert indices to tk.Text format
            start_index = text_widget.index(f"1.0 + {global_start_idx}c")
            end_index = text_widget.index(f"1.0 + {global_end_idx}c")

            # Apply color to the keyword segment
            text_widget.tag_add(keyword_segment, start_index, end_index)
            text_widget.tag_config(keyword_segment, foreground=color)
        except ValueError:
            # Display an error message if the keyword segment is not found within the text segment
            print(f"Keyword segment '{keyword_segment}' not found within the specified text segment.")
            continue
    write_list_to_file("visualizer/explanation-of-highlights.txt", output_list)

def substr_precedence(tuples_list):
    """
    Sorts a list of tuples in place based on the first element primarily, and if two or more tuples 
    have the same first element, sorts them by the length of the second element in ascending order.

    Parameters:
    tuples_list (list): A list of tuples where each tuple consists of three elements.
    - prompt str
    - tagged str 
    - color 

    Returns:
    list: The same list of tuples, sorted based on the specified criteria.
    """
    return sorted(tuples_list, reverse=True, key=lambda x: (x[0], len(x[1])))


def main(text_file, annotation_file):
    text = read_text_file(text_file)
    annotation_lines = read_annotation_file(annotation_file)
    annotations = parse_annotations(annotation_lines)
    annotations = substr_precedence(annotations)
   
    # Create the main window
    root = tk.Tk()
    file_title = os.path.basename(text_file)
    root.title(f"Text Annotation Viewer: {file_title}")
    
    # Create a scrolled text widget
    text_widget = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=100, height=30)
    text_widget.pack(expand=True, fill='both')
    

    # Create a frame for the legend and add it to the window
    legend_frame = tk.Frame(root)
    legend_frame.pack(side=tk.TOP, fill='x', padx=10, pady=10)

    # Add the legend items
    for keyword, color in KEYWORD_COLOR_MAP.items():
        color_label = tk.Label(legend_frame, text=keyword, bg=color, fg='white', padx=10, pady=5)
        color_label.pack(side=tk.LEFT, padx=2)


    # Color the text based on the annotations
    color_text(text_widget, text, annotations)

    # Control font size; add to gui in future dev
    size = 12
    text_widget.tag_configure('default', font=("Helvetica", size))
    text_widget.configure(font=("Helvetica", size))
    
    # Start the GUI event loop
    root.mainloop()

def load_config(file_path="config.yaml"):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def run_visualization(text_file, annotation_file):
    main(text_file, annotation_file)  # Assuming `main` is defined to handle the visualization

def run_program(config):
    mode = config.get("mode", "single")
    visualizations = config.get("visualizations", [])
    print(visualizations[0]["text_file"], visualizations[0]["annotation_file"])

    if mode == "single":
        # Run only the first visualization without threading
        if visualizations:
            text_file = visualizations[0]["text_file"]
            annotation_file = visualizations[0]["annotation_file"]
            run_visualization(text_file, annotation_file)
    elif mode == "multiple" and len(visualizations) > 1:
        # Run two visualizations in parallel using threading
        thread1 = threading.Thread(target=lambda: main(visualizations[0]["text_file"], visualizations[0]["annotation_file"]))
        thread2 = threading.Thread(target=lambda: main(visualizations[1]["text_file"], visualizations[1]["annotation_file"]))

        # Start both threads
        thread1.start()
        thread2.start()

        # Wait for both threads to complete
        thread1.join()
        thread2.join()

if __name__ == "__main__":
    config = load_config("visualizer/viz_config.yml")
    run_program(config)