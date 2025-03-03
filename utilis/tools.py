from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from time import perf_counter
from typing import List, Dict


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


SENTENCES: List[str] = [
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


class Chunker(object):

    def __init__(self, model: str = "paraphrase-distilroberta-base-v1", threshold: float = 1.0) -> None:
        """ Initialize the hyperparameters of the Chunker object

        :param model: the name of the model to use for the chunking, default is "all-MiniLM-L6-v2"
        :param threshold: the threshold for the Agglomerative Clustering, default is 1.0, the range is (0.5, 1.5)
        """
        self._model: str = model
        self._threshold: float = threshold
        self._chunks: Dict[int, List[str]] = {}

    def chunk(self, sentences: List[str]) -> Dict[int, List[str]]:
        """ Chunk the sentences using Agglomerative Clustering

        :param sentences: a list of sentences to be chunked
        :return: a dictionary, where the keys are the cluster labels and the values are the corresponding sentences
        """
        # Reset the chunks dictionary avoiding conflicts with previous chunking results
        self._chunks = {}

        # Load the model from Hugging Face model hub
        model = SentenceTransformer(self._model)

        # Compute the embeddings for the sentences, the dimensionality of the embeddings is 384
        embeddings = model.encode(sentences)

        # Cluster the embeddings using Agglomerative Clustering with a threshold of 1.0 and average linkage
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=self._threshold, linkage="average")
        # Use the fit_predict method to cluster the embeddings and return the cluster labels
        clusters = clustering.fit_predict(embeddings)

        for index, category in enumerate(clusters):
            if category not in self._chunks:
                self._chunks[category] = []
            self._chunks[category].append(sentences[index])
        return self._chunks

    def paragraph(self):
        """ Return the chunks as a list of paragraphs

        :return: a list of paragraphs
        """
        if not self._chunks:
            raise ValueError("No chunks found. Please run `chunk(sentences)` first.")

        return [" ".join(chunk) for chunk in self._chunks.values()]
