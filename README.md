# Document Ranking Information Retrieval System using Vector Space Model

This project is implementation of a Vector Space Model (VSM) for building an information retrieval system. The VSM is encapsulated within a class named VSM, designed with methods including constructor, set_query, generate_document_index, generate_query_index, get_all_words, generate_tf_document, generate_tf_idf_document, generate_tf_query, generate_tf_idf_query, and similarity.

## Features:
- **Document and Query Processing**: Parses documents and queries, tokenizes them, and calculates term frequencies (tf).
- **Cosine Similarity**: Computes the similarity between documents and queries using cosine similarity.
- **Document-Query Indexing**: Constructs tf, dft, and tf-idf vectors for documents and queries.
- **User Interaction**: Allows users to input queries and receive ranked document results based on relevance.

## Corpus Document Description

The corpus consists of 201,000 rows of training data obtained from the Hugging Face website. These data are used to predict whether a text is a machine-paraphrased essay, with columns such as label, dataset, and method. Only the text column is retained for analysis.

Data is formatted as TSV and read using the Python Pandas library. Word counts per document are computed and visualized in a histogram, showcasing a right-skewed distribution with many documents containing fewer than 300 words. Statistical data on word count per document is provided, indicating a range of word counts and identifying outliers.

Documents containing 200 to 300 words are selected for further processing, amounting to 28,223 documents. Non-Latin characters are removed during document processing. Finally, 100 documents are chosen for efficient document retrieval.
