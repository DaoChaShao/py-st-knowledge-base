**INTRODUCTION**  
This application is designed to help you understand how to structure and organize the Knowledge Base. Familiarity with
the tech skills required to build a knowledge base.

1. The application is built using [Streamlit](https://docs.streamlit.io/).
2. Transforming the current data into a structured format, which are sentences in a Python list.
3. Embedding the sentences into vectors, then cluster them into groups based on Kmeans or other algorithms.
4. Building a simple relation network to show the relationship between different clusters and their sentences
   throughout [streamlit-agraph on Pypi](https://pypi.org/project/streamlit-agraph/)
   or [streamlit_agraph on GitHub](https://github.com/ChrisDelClea/streamlit-agraph).
5. Of course, you also can use
   the [Pyvis on Pypi](https://pypi.org/project/pyvis/), [Pyvis on GitHub](https://github.com/WestHealth/pyvis)
   or [Pyvis on its official website](https://pyvis.readthedocs.io/en/latest/) to visualize the relationship.
6. Besides, you also can use the [Neo4j](https://neo4j.com/) to store the relationship between different clusters and
   their sentences.
7. Embedding the query into a vector, then find the most similar sentence in the knowledge base.
8. Building a simple search engine to search for the most similar sentence in the knowledge base.
    - First, find the cluster that is most similar.
    - Secondly, find the most similar sentence in the cluster.
    - Finally, return the most similar sentence or top N based on the different algorithm.
9. One embedding model is [OpenAI](https://platform.openai.com/docs/pricing)'s `text-embedding-3-small`, a powerful
   language model. However, the most important reason for using it is its low price among `text-embedding-3-large`,
   `text-embedding-ada-002` and itself.
10. The other embedding model is [Sentence-Transformers](https://www.sbert.net/), which is a Python framework for
    state-of-the-art sentence, paragraph, and image embeddings.
    The [model](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html) is `all-MiniLM-L6-v2`.

**INSTRUCTIONS**

1. Clone the repository to your local machine.
2. Install the required dependencies with the command `pip install -r requirements.txt`.
3. Run the application with the command `streamlit run main.py` command.
4. Or you can try the application by visiting the following
   link: [![Static Badge](https://img.shields.io/badge/Open%20in%20Streamlit-Daochashao-red?style=for-the-badge&logo=streamlit&labelColor=white)](https://knowledge-base-2.streamlit.app/)

**LICENSE**

This application is licensed under the [BSD-3-Clause License](LICENSE). You can click the link to read the license.