from pandas import DataFrame
from streamlit import (empty, data_editor, sidebar, spinner,
                       subheader, markdown, session_state, write)

from utilis.models import api_key_checker, OpenAIEmbedder, HuggingFaceEmbedder
from utilis.graphs import network
from utilis.tools import (params, COMPARES, cluster_kmeans, n_clusters_ss,
                          knowledge_base_builder, QUERIES, search_top_one)

empty_messages: empty = empty()

if "open_clusters" not in session_state:
    session_state.open_clusters = {}

if "hug_clusters" not in session_state:
    session_state.hug_clusters = {}

category, embed, var = params()

if not embed:
    empty_messages.info("Please select the type of embedding model.")
else:
    subheader("Sentences for COMPARES")
    df_compares = DataFrame(data={"sentences": COMPARES})
    data_editor(df_compares, hide_index=True, disabled=True, use_container_width=True)
    subheader("Sentences for QUERIES")
    df_queries = DataFrame(data={"inputs": QUERIES})
    data_editor(df_queries, hide_index=True, disabled=True, use_container_width=True)

    match category:
        case "OpenAI":
            if not api_key_checker(var):
                empty_messages.error("Invalid API key. Please enter a valid API key.")
            else:
                if sidebar.button("Open Embed", type="primary", help="Click to embed the sentences"):
                    with spinner("Embedding...", show_time=True):
                        embedder = OpenAIEmbedder(var)
                        embeddings = embedder.client(COMPARES, embed, 512)

                        n_clusters: int = n_clusters_ss(embeddings, len(COMPARES))
                        labels, session_state.open_clusters = cluster_kmeans(embeddings, COMPARES,
                                                                             num_clusters=n_clusters)

                if session_state.cluster_open:
                    # network(session_state.open_clusters)
                    # for label, sentences in session_state.cluster_open.items():
                    #     subheader(f"Cluster {label}")
                    #     for sentence in sentences:
                    #         markdown(f"- {sentence}")
                    #     markdown("---")

                    subheader("The Network of the Knowledge Base")
                    knowledge_base = knowledge_base_builder(embeddings, labels, n_clusters, COMPARES)
                    network(knowledge_base["clusters"])

                    feedback: str = search_top_one(embedder, embed, QUERIES, knowledge_base)

                    if feedback:
                        subheader("Feedback given by the function `search_top_one`")
                        df_feedback = DataFrame(data={"sentences": [feedback]})
                        data_editor(df_feedback, hide_index=True, disabled=True, use_container_width=True)
                        empty_messages.success("OpenAI Embedding completed successfully.")
                    else:
                        empty_messages.error("No feedback found.")
        case "Hugging Face":
            if sidebar.button("Hug Embed", type="primary", help="Click to embed the sentences"):
                with spinner("Embedding...", show_time=True):
                    embedder = HuggingFaceEmbedder(embed)
                    embeddings = embedder.client(COMPARES)

                    session_state.hug_clusters = cluster_kmeans(embeddings, COMPARES)

            if session_state.cluster_hug:
                network(session_state.hug_clusters)
                # for label, sentences in session_state.cluster_open.items():
                #     subheader(f"Cluster {label}")
                #     for sentence in sentences:
                #         markdown(f"- {sentence}")
                #     markdown("---")
                empty_messages.success("Hugging Face Embedding completed successfully.")
