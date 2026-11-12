import os
import re
import argparse
import heapq
from collections import defaultdict
from tqdm import tqdm


def read_file(file_path):
    """
    Reads the contents of a file and counts the frequencies of tokens.
    
    Args:
        file_path (str): Path to the file to read.
        
    Returns:
        dict: Dictionary mapping tokens to their frequencies.
    """
    token_freqs = defaultdict(int)

    # Open the file in reading mode with UTF-8 encoding
    with open(file_path, "r", encoding='utf-8') as file:
        text = file.read()

        # Tokenize the text using regular expressions
        tokens = re.findall(r"[^\W\d_]+|['’](?=\w)", text, re.UNICODE)
        
        # Remove empty tokens and convert to lowercase
        tokens = [token.lower() for token in tokens if token.strip()]

        # Count the frequencies of tokens
        for token in tokens:
            token_freqs[token] += 1

    # Close the file after reading
    file.close()

    return token_freqs


def generate_ngrams(token_freqs, ns=[1, 2, 3], top_n=300):
    """
    Generate N-grams from a dictionary of token frequencies.
    
    Args:
        token_freqs (dict): Dictionary mapping tokens to their frequencies.
        ns (list): List of N-gram sizes to generate.
        top_n (int): Number of top N-grams to keep.
        
    Returns:
        list: List of top N-grams in descending order of frequency.
    """
    ngram_counts = defaultdict(int)

    # Iterate over each token and its frequency
    for token, freq in token_freqs.items():
        # Generate N-grams for each N in the given list
        for n in ns:
            # Pad the token with underscores at the beginning and end
            padded_token = "_" + token + "_" * (n - 1)

            # Generate N-grams for the padded token
            for j in range(len(padded_token) - n + 1):
                ngram = padded_token[j:j+n]
                ngram_counts[ngram] += freq

    # Remove N-grams that consist only of underscores
    ngram_counts = {k: v for k, v in ngram_counts.items() if not k.replace('_', '').strip() == ''}

    # Sort the N-gram counts in descending order and keep the top N-grams
    sorted_ngrams = heapq.nlargest(top_n, ngram_counts, key=ngram_counts.get)

    return sorted_ngrams


def process_files(input_folder, output_folder, file_ending, ns, top_n):
    """
    Process the files in a directory, generating N-grams for each language
    and saving the N-gram profiles in a separate file.
    
    Args:
        input_folder (str): Path to the directory containing the files.
        output_folder (str): Path to the directory where the profiles will be saved.
        file_ending (str): File ending for the training files, could be extension or any other conventional ending for training.
        ns (list): List of N-gram sizes to generate.
        top_n (int): Number of top N-grams to keep.
    """
    language_data = defaultdict(list)

    # Loop through each folder in the input_folder directory
    for language in os.listdir(input_folder):
        language_path = os.path.join(input_folder, language)

        # Check if the path is a directory
        if os.path.isdir(language_path):
            file_paths = []

            # Loop through each file in the language folder
            for file_name in os.listdir(language_path):
                # Check if the file matches the provided file extension
                if file_name.endswith(file_ending):
                    # Create the full file path
                    file_path = os.path.join(language_path, file_name)
                    file_paths.append(file_path)

            # Use tqdm to track the progress of file processing
            with tqdm(total=len(file_paths), desc=f"Processing {language}") as pbar:
                for file_path in file_paths:
                    # Read the data for each file separately
                    token_freqs = read_file(file_path)

                    # Generate N-grams for the token frequencies
                    n_grams = generate_ngrams(token_freqs, ns, top_n)

                    # Append the N-grams to the language's data list
                    language_data[language].extend(n_grams)

                    pbar.update(1)

    # Create the output_folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Write the combined N-grams of files for each language into a separate file
    for language, n_grams in language_data.items():
        # Create a file path for the language
        file_path = os.path.join(output_folder, f"{language}_profile.txt")

        # Open the file in writing mode with UTF-8 encoding
        with open(file_path, "w", encoding="utf-8") as file:
            # Write the combined N-grams for the language
            for n_gram in n_grams:
                file.write(f"{n_gram}\n")

    tqdm.write("Training completed successfully!")


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Language Identification Training")

    # Add arguments for input folder, output folder, file ending, ns, and top_n
    parser.add_argument("-input", metavar="INPUT_FOLDER", help="Folder containing the training data")
    parser.add_argument("-output", metavar="OUTPUT_FOLDER", help="Folder to save the language profiles")
    parser.add_argument("-ending", metavar="FILE_ENDING", help="File ending for the training files", default='train.txt')
    parser.add_argument("-ns", metavar="N_GRAMS_LIST", nargs="+", type=int, help="List of N-gram sizes to generate", default=[1, 2, 3])
    parser.add_argument("-topn", metavar="TOP_N", type=int, help="Number of top N-grams to keep", default=300)

    # Parse the command-line arguments
    args = parser.parse_args()

    # Check if the input folder is provided
    if not args.input:
        parser.error("Please provide the input folder.")

    # Check if the output folder is provided
    if not args.output:
        parser.error("Please provide the output folder.")

    # Get the input folder, output folder, file ending, ns, and top_n from the arguments
    input_folder = args.input
    output_folder = args.output
    file_ending = args.ending
    ns = args.ns
    top_n = args.topn

    # Perform language identification training
    process_files(input_folder, output_folder, file_ending, ns, top_n)