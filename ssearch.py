import argparse
import os
import sys
from sentence_transformers import SentenceTransformer
import numpy as np


def load_model(device):

    if device:
        model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device=device
        )
        return model

    # Make sure to use cpu and not gpu
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model


def calculate_embedding(model, sentences):
    sentence_embeddings = model.encode(sentences)
    # Normalize embeddings
    sentence_embeddings = sentence_embeddings / np.linalg.norm(
        sentence_embeddings, axis=1, keepdims=True
    )
    return sentence_embeddings


def calculate_similarity(query_embedding, file_embeddings):
    # calculate similarity using dot product (cosine similarity)
    similarity = np.dot(file_embeddings, query_embedding)
    return similarity


def get_files(search_dir, recursive):
    files = []
    if recursive:
        for root, _, filenames in os.walk(search_dir):
            for filename in filenames:
                if os.path.isfile(os.path.join(root, filename)):
                    files.append(os.path.join(root, filename))
    else:
        for filename in os.listdir(search_dir):
            if os.path.isfile(os.path.join(search_dir, filename)):
                files.append(os.path.join(search_dir, filename))
    return files


def parse_args():

    # Define the usage of the script and the arguments it accepts indicating the type of the argument and if it is required or not
    usage = """
    python ssearch.py search_dir query [--OPTION VALUE] [--OPTION VALUE] ... 
    """

    description = "Search for files based on similarity of the search query with the file's name and optionally, content"
    parser = argparse.ArgumentParser(usage=usage, description=description)

    # Positional arguments (required)
    parser.add_argument("search_dir", type=str, help="Directory to search in")
    parser.add_argument("query", type=str, help="Search query")

    # Optional arguments
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.3,
        help="Threshold for similarity [0-1]. Default is 0.3. Files with similarity greater than threshold are shown",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        help="Device to use. Default is automatic selection. Use 'cpu' to force cpu and 'cuda' to force gpu.",
    )
    parser.add_argument(
        "-c",
        "--content",
        type=bool,
        default=False,
        help="Search in content. Default is False. If set to True, search in content as well as name",
    )
    parser.add_argument(
        "--max-content-size",
        type=int,
        default=1000,
        help="If searching in content, maximum size of content to search in. Default is 1000 bytes",
    )
    parser.add_argument(
        "-m",
        "--max-results",
        type=int,
        default=25,
        help="Maximum number of results to show. Default is 25",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        type=bool,
        default=False,
        help="Search recursively in subdirectories. Default is False",
    )
    args = parser.parse_args()
    return args, parser


def validate_args(args):
    if args.threshold < 0 or args.threshold > 1:
        print("Threshold should be between 0 and 1")
        return False

    if args.device and args.device not in ["cpu", "cuda"]:
        print("Device should be 'cpu' or 'cuda'")
        return False

    if args.device == "cuda" and not torch.cuda.is_available():
        print("Cuda is not available on this machine")
        return False

    if args.search_dir is None:
        print("Query is missing")
        return False

    if args.query is None:
        print("Query is missing")
        return False

    if args.max_content_size < 0:
        print("Max content size should be greater than 0")
        return False

    if args.max_results < 0:
        print("Max results should be greater than 0")
        return False

    # If all checks pass, return True
    return True


if __name__ == "__main__":

    # Parse the arguments
    args, arg_parser = parse_args()

    # Validate the arguments
    if not validate_args(args):
        # Print usage and help
        arg_parser.print_help()
        exit(1)

    # Block print statements from transformers
    sys.stderr = open(os.devnull, "w")  # Redirect stdout to /dev/null
    sys.stdout = open(os.devnull, "w")  # Redirect stderr to /dev/null

    # Load the model
    model = load_model(args.device)

    # Unblock print statements
    sys.stderr = sys.__stderr__  # Reset redirection
    sys.stdout = sys.__stdout__  # Reset redirection

    # Gather all files in the search directory
    files = get_files(args.search_dir, args.recursive)

    # Get the last part of the path as the file name
    file_names = [os.path.basename(file) for file in files]

    # Generate embeddings for the query and the files' names (last part of the path)
    query_embedding = calculate_embedding(model, [args.query])[0]
    file_embeddings = calculate_embedding(model, file_names)
    similarities = calculate_similarity(query_embedding, file_embeddings)

    # Filter files based on similarity threshold
    results = [
        (file, similarity)
        for file, similarity in zip(files, similarities)
        if similarity >= args.threshold
    ]

    # Sort the results based on similarity
    results_capped = sorted(results, key=lambda x: x[1], reverse=True)[
        : args.max_results
    ]

    # Print the results with each result on a new line
    for result in results_capped:
        print(result[0])

    # Print a new line
    print()

    results_capped_count = len(results_capped)
    results_count = len(results)
    files_count = len(files)

    # Print a message if the results are capped or if some files were filtered out
    if results_capped_count < files_count or results_capped_count > args.max_results:
        if results_count > args.max_results:
            print(
                f"Showing only the top {args.max_results} results (out of {results_count}). You can increase the number of results using the --max-results option"
            )
        if results_count < files_count:
            print(
                f"{files_count - results_count} files were filtered out because they did not meet the similarity threshold of {args.threshold}"
            )
