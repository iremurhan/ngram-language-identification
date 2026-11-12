import os
import argparse
import logging
from train import read_file, generate_ngrams

# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def calculate_distance(document_profile, language_profile, top_n):
    """
    Calculate the out-of-place measure distance between the document and a given language profile.
    
    Args:
        document_profile (list): The profile of the document.
        language_profile (list): The profile of a particular language.
        top_n (int): Number of top N-grams considered in the profiles.

    Returns:
        int: The calculated distance.
    """
    distance = 0
    language_ngram_indices = {ngram: index for index, ngram in enumerate(language_profile)}
    document_ngram_indices = {ngram: index for index, ngram in enumerate(document_profile)}

    for ngram in language_profile:
        if ngram in document_ngram_indices:
            index = document_ngram_indices[ngram]
            language_index = language_ngram_indices[ngram]
            distance += abs(index - language_index)
        else:
            distance += top_n
    return distance

def identify_language(document_path, language_profiles_folder):
    """
    Identify the most probable language of a document by comparing its N-gram profile with existing language profiles.
    
    Args:
        document_path (str): Path to the document to be identified.
        language_profiles_folder (str): Path to the directory where the language profiles are stored.

    Returns:
        str: The identified language.

    Raises:
        FileNotFoundError: If the document or language profiles directory does not exist.
    """
    if not os.path.exists(document_path):
        logging.error(f"The document at path {document_path} does not exist.")
        raise FileNotFoundError(f"The document at path {document_path} does not exist.")

    if not os.path.exists(language_profiles_folder):
        logging.error(f"The language profiles directory at path {language_profiles_folder} does not exist.")
        raise FileNotFoundError(f"The language profiles directory at path {language_profiles_folder} does not exist.")

    top_n_is_extracted = False
    language_distances = {}

    token_freqs = read_file(document_path)
    document_profile = generate_ngrams(token_freqs)

    file_names = [file_name for file_name in os.listdir(language_profiles_folder) if file_name.endswith("profile.txt")]

    for file_name in file_names:
        language = os.path.splitext(file_name)[0]
        language_profile_path = os.path.join(language_profiles_folder, file_name)
        with open(language_profile_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            language_profile = [line.strip() for line in lines]
        
        if not top_n_is_extracted:
            top_n = len(lines)
            top_n_is_extracted = True

        distance = calculate_distance(document_profile, language_profile, top_n)
        language_distances[language] = distance

    identified_language = min(language_distances, key=language_distances.get)
    return identified_language.replace("_profile", "")

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Language Identification Testing")

    # Add arguments for document path and language profiles folder
    parser.add_argument("-test", metavar="DOCUMENT", help="Path to the document for language identification", required=True)
    parser.add_argument("-profiles", metavar="PROFILES_FOLDER", help="Folder containing the language profiles", required=True)

    # Parse the command-line arguments
    args = parser.parse_args()

    # Get the document path and language profiles folder from the arguments
    document_path = args.test
    language_profiles_folder = args.profiles

    # Perform language identification on the given document
    try:
        identified_language = identify_language(document_path, language_profiles_folder)
        logging.info(f"The identified language is: {identified_language}")
    except FileNotFoundError as e:
        logging.error(str(e))
    except Exception as e:
        logging.error(f"An error occurred during language identification: {e}")
