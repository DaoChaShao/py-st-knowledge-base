from numpy import array
from pandas import DataFrame
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from streamlit import (sidebar, header, segmented_control, selectbox,
                       caption, text_input, slider, line_chart)
from time import perf_counter

from utilis.models import api_key_checker


class Timer(object):
    """ A simple timer class to measure the elapsed time.

    :param precision: the number of decimal places to round the elapsed time
    :param description: the description of the timer
    """

    def __init__(self, precision: int = 5, description: str = None):
        self._precision: int = precision
        self._description: str = description
        self._start: float = 0.0
        self._end: float = 0.0
        self._elapsed: float = 0.0

    def __enter__(self):
        self._start = perf_counter()
        print(f"{self._description} has been started.")
        return self

    def __exit__(self, *args):
        self._end = perf_counter()
        self._elapsed = self._end - self._start

    def __repr__(self):
        if self._elapsed != 0.0:
            return f"{self._description} took {self._elapsed:.{self._precision}f} seconds."
        return f"{self._description} has NOT been started."


SENTENCES: list[str] = [
    "Transformer 是一种深度学习模型。",
    "它用于自然语言处理任务。",
    "自注意力机制是 Transformer 的核心。",
    "训练时使用 Adam 优化器。",
    "Transformer 在机器翻译中表现良好。",
    "BERT 是基于 Transformer 的预训练模型。",
    "GPT 也使用 Transformer 结构。",
    "苹果是一种水果。",
    "苹果公司推出了新的 iPhone。",
    "iPhone 15 Pro 采用了钛合金设计。",
    "智能手机市场竞争激烈。",
    "华为是一家高科技公司。",
    "华为 P50 手机采用了华为自研芯片。",
]


def paragraph(chunks: dict[int, list[str]]):
    """ Return the chunks as a list of paragraphs

    :return: a list of paragraphs
    """
    return [" ".join(chunk) for chunk in chunks.values()]


def params() -> tuple[str, str, str] | tuple[None, None, None]:
    """ Return the parameters for the embedding model """
    with sidebar:
        header("Embedding Parameters")
        options_seg: list[str] = ["OpenAI", "Hugging Face"]
        category: str = segmented_control("Segmentation Model Type", options_seg, selection_mode="single", disabled=0,
                                          help="Select a model type for the segmentation.")
        caption(f"The selected model type is **{category}**.")

        match category:
            case "OpenAI":
                options_box: list[str] = ["text-embedding-3-small"]
                model = selectbox("Model", options_box, 0, placeholder="Choose an option",
                                  help="Select a model for the chunking.")
                caption(f"The selected model is **{model}**.")
                api_key: str = text_input("API Key", max_chars=200, type="password",
                                          placeholder="Enter your API key.", help="Enter your OpenAI API key.")
                if not api_key_checker(api_key):
                    caption("**INVALID** API key. Please enter a valid API key.")
                else:
                    caption("API key is **VALID**.")
                return category, model, api_key
            case "Hugging Face":
                options_box: list[str] = ["all-MiniLM-L6-v2"]
                model = selectbox("Model", options_box, 0, placeholder="Choose an option",
                                  help="Select a model for the chunking.")
                caption(f"The selected model is **{model}**.")
                threshold = slider("Threshold", min_value=0.5, max_value=1.5, value=1.0, step=0.1,
                                   help="Select a threshold for the Agglomerative Clustering.")
                caption(f"The selected threshold is **{threshold}**.")
                return category, model, threshold
        return None, None, None


def n_clusters_plot(embeddings: list, max_: int, seed_: int = None):
    sse: list[float] = []
    values = list(range(2, max_))
    for i in values:
        kmeans = KMeans(n_clusters=i, random_state=seed_)
        kmeans.fit(embeddings)
        sse.append(kmeans.inertia_)

    cluster: dict[str, list[float]] = {"Number of Clusters": values, "SSE": sse}
    df: DataFrame = DataFrame(cluster)
    line_chart(df, x="Number of Clusters", y="SSE")


def n_clusters_ss(embeddings: list, max_: int) -> int:
    """ Return the optimal number of clusters using the Silhouette Score

    :param embeddings: the embeddings of the sentences
    :param max_: the maximum number of clusters
    :return: the optimal number of clusters
    """
    # Initialize the best silhouette score
    best_score: float = -1
    # Initialize the best number of clusters
    best_k: int = 0
    values = list(range(2, max_))

    for k in values:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(embeddings)
        score = silhouette_score(embeddings, kmeans.labels_)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k


def cluster_kmeans(embeddings: list, sentences: list[str], num_clusters: int = 5) -> dict:
    """ Cluster the embeddings using K-Means Clustering

    :param embeddings: the embeddings of the sentences
    :param sentences:  to cluster the embeddings
    :param num_clusters: the number of clusters
    :return: a dictionary of clusters
    """
    # Use K-Means clustering to cluster the embeddings
    kmeans = KMeans(n_clusters=num_clusters, random_state=None)
    kmeans.fit(embeddings)

    # create a dictionary, where the keys are the cluster labels and the values are the corresponding sentences
    clusters = {}
    for index, label in enumerate(kmeans.labels_):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(sentences[index])
    return clusters


def cluster_agglomerate(embeddings: list, sentences: list[str], threshold: float = 1.0) -> dict:
    # Cluster the embeddings using Agglomerative Clustering with a threshold of 1.0 and average linkage
    agglomerate = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, linkage="average")
    # Use the fit_predict method to cluster the embeddings and return the cluster labels
    labels = agglomerate.fit_predict(embeddings)

    # create a dictionary, where the keys are the cluster labels and the values are the corresponding sentences
    clusters = {}
    for index, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(sentences[index])
    return clusters


def cluster_dbscan(embeddings: list, sentences: list[str], eps: float = 0.5, min_samples: int = 5) -> dict:
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(embeddings)

    # create a dictionary, where the keys are the cluster labels and the values are the corresponding sentences
    clusters = {}
    for index, label in enumerate(labels):
        if label != -1:  # ignore outliers
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(sentences[index])
    return clusters


def cluster_sc(embeddings: list, sentences: list[str], num_clusters: int = 5) -> dict:
    spectral = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors')
    labels = spectral.fit_predict(embeddings)

    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(sentences[i])
    return clusters


def cluster_gmm(embeddings: list, sentences: list[str], num_clusters: int = 5) -> dict:
    gmm = GaussianMixture(n_components=num_clusters)
    labels = gmm.fit_predict(embeddings)

    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(sentences[i])
    return clusters


def knowledge_base_builder(embeddings: list, labels: list[int], num_clusters: int, sentences: list[str]):
    centroids = array([embeddings[labels == i].mean(axis=0) for i in range(num_clusters)])

    knowledge_base: dict[str, list | dict] = {
        "centroids": centroids,
        "clusters": {i: [] for i in range(num_clusters)},
    }

    for i, label in enumerate(labels):
        knowledge_base["clusters"][label].append(sentences[i])

    return knowledge_base
