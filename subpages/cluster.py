from pandas import DataFrame
from streamlit import (empty, data_editor, sidebar, spinner,
                       subheader, markdown, session_state)

from utilis.models import api_key_checker, OpenAIEmbedder, HuggingFaceEmbedder
from utilis.graphs import network
from utilis.tools import params, SENTENCES, cluster_kmeans, n_clusters_ss

empty_messages: empty = empty()

if "cluster_open" not in session_state:
    session_state.cluster_open = {}

if "cluster_hug" not in session_state:
    session_state.cluster_hug = {}

category, embed, var = params()

if not embed:
    empty_messages.info("Please select the type of embedding model.")
else:
    df = DataFrame(data={"sentences": SENTENCES})
    data_editor(df, hide_index=True, disabled=True, use_container_width=True)

    match category:
        case "OpenAI":
            if not api_key_checker(var):
                empty_messages.error("Invalid API key. Please enter a valid API key.")
            else:
                if sidebar.button("Open Embed", type="primary", help="Click to embed the sentences"):
                    with spinner("Embedding...", show_time=True):
                        embedder = OpenAIEmbedder(var)
                        embeddings = embedder.client(SENTENCES, embed, 512)

                        n_clusters: int = n_clusters_ss(embeddings, len(SENTENCES))
                        session_state.cluster_open = cluster_kmeans(embeddings, SENTENCES, num_clusters=n_clusters)

                if session_state.cluster_open:
                    network(session_state.cluster_open)
                    # for label, sentences in session_state.cluster_open.items():
                    #     subheader(f"Cluster {label}")
                    #     for sentence in sentences:
                    #         markdown(f"- {sentence}")
                    #     markdown("---")
                    empty_messages.success("OpenAI Embedding completed successfully.")
        case "Hugging Face":
            if sidebar.button("Hug Embed", type="primary", help="Click to embed the sentences"):
                with spinner("Embedding...", show_time=True):
                    embedder = HuggingFaceEmbedder(embed)
                    embeddings = embedder.client(SENTENCES)

                    session_state.cluster_hug = cluster_kmeans(embeddings, SENTENCES)

            if session_state.cluster_hug:
                network(session_state.cluster_hug)
                # for label, sentences in session_state.cluster_open.items():
                #     subheader(f"Cluster {label}")
                #     for sentence in sentences:
                #         markdown(f"- {sentence}")
                #     markdown("---")
                empty_messages.success("Hugging Face Embedding completed successfully.")
