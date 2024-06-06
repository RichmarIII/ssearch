# SSearch: File Search by Semantic Similarity

SSearch is a command-line tool that searches for files based on the semantic similarity of the search query with the file's name and optionally, the content. It uses machine learning for generating embeddings and calculating similarity.

## Features

- Search files by name semantic similarity.
- Optional content-based search.
- Recursive directory search.
- Customizable similarity threshold.
- Device selection for model inference (CPU/GPU).
- Limit the number of results.

## Installation

To use SSearch, you need to install the required Python packages. It's recommended to create a virtual environment first:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python ssearch.py search_dir query [--OPTION VALUE] [--OPTION VALUE] ...
```

### Positional Arguments

- `search_dir`: Directory to search in.
- `query`: Search query.

### Optional Arguments

- `-t`, `--threshold`: Threshold for semantic similarity [0-1]. Default is 0.3. Files with similarity greater than the threshold are shown.
- `-d`, `--device`: Device to use. Default is automatic selection. Use 'cpu' to force CPU and 'cuda' to force GPU.
- `-c`, `--content`: Search in content. Default is `False`. If set to `True`, search in content as well as name.
- `--max-content-size`: If searching in content, maximum size of content to search in. Default is 1000 bytes.
- `-m`, `--max-results`: Maximum number of results to show. Default is 25.
- `-r`, `--recursive`: Search recursively in subdirectories. Default is `False`.

## Examples

### Basic Search

Search for files in the `documents` directory with names semantically similar to "report":

```bash
python ssearch.py documents "report"
```

### Search with Custom Threshold

Search for files with names semantically similar to "report" and a similarity threshold of 0.5:

```bash
python ssearch.py documents "report" --threshold 0.5
```

### Content-based Search

Search for files with names or content semantically similar to "report":

```bash
python ssearch.py documents "report" --content True
```

### Recursive Search

Search for files in `documents` and all its subdirectories:

```bash
python ssearch.py documents "report" --recursive True
```

### Specify Device

Force the use of CPU for inference:

```bash
python ssearch.py documents "report" --device cpu
```

## How It Works

1. **Model Loading**: The machine learning model is loaded based on the specified device (CPU/GPU).
3. **File Gathering**: The script gathers file paths from the specified directory. If recursive search is enabled, it gathers files from subdirectories as well.
4. **Embedding Calculation**: Embeddings are calculated for the query and the file names (or contents, if enabled).
5. **Similarity Calculation**: Cosine similarity is calculated between the query embedding and the file embeddings.
6. **Filtering and Sorting**: Files are filtered based on the semantic similarity threshold and sorted by similarity. The results are then printed.

## Requirements

All dependencies are listed in `requirements.txt`. You can install them using `pip`:

```bash
pip install -r requirements.txt
```

## Contributing

Feel free to open issues or submit pull requests on [GitHub](https://github.com/yourusername/sssearch).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE.md) file for details.

---

SSearch is a versatile tool to search for files based on semantic similarity, providing options for content-based search and recursive directory traversal. Its flexibility in configuring the search parameters makes it suitable for various use cases.
