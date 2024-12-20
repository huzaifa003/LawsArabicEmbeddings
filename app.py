from ai_utils import process_docs


def main(input_folder, chunks_folder, vector_db_dir):
    process_docs(input_folder, chunks_folder, vector_db_dir)   
    return

if __name__ == "__main__":
    main('Folders', 'Chunks', 'VectorDB')