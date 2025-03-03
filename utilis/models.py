from openai import OpenAI
from sentence_transformers import SentenceTransformer


def api_key_checker(api_key: str, category: str = "OpenAI") -> bool:
    """ Check the API key format
            - The length of an OpenAI API key is 164-digit key

    :param api_key: enter the API key of the deepseek
    :param category: the category of the API key
    :return: True if the API key is valid
    """
    if api_key.startswith("sk-"):
        if category == "OpenAI" and len(api_key) == 164:
            return True
        elif category == "DeepSeek" and len(api_key) == 35:
            return True
    return False


class OpenAICompleter(object):
    """ OpenAI Completer API Wrapper """

    def __init__(self, api_key: str, temperature: float = 0.7, top_p: float = 0.9) -> None:
        """ Initialize the OpenAI Hyperparameter Tuning API

        :param api_key: str: The API key for the OpenAI API
        :param temperature: float: The temperature for the completion
        :param top_p: float: The top-p for the completion
        """
        self._api_key = api_key
        self._temperature = temperature
        self._top_p = top_p

    def client(self, content: str, prompt: str, model: str) -> str:
        """ Initialize the OpenAI Completion API

        :param content: str: The input text to be completed
        :param prompt: str: The prompt to complete the input text
        :param model: str: The model to use for the completion
        :return: str: The completed text
        """
        client = OpenAI(api_key=self._api_key, base_url="https://api.openai.com/v1")

        messages = [
            {"role": "system", "content": content},
            {"role": "user", "content": prompt},
        ]

        completion = client.chat.completions.create(
            model=model,
            store=False,
            messages=messages,
            stream=False,
            temperature=self._temperature,
            top_p=self._top_p,
        )
        return completion.choices[0].message.content


class OpenAIEmbedder(object):

    def __init__(self, api_key: str) -> None:
        """ Initialize the OpenAI Embeddings API

        :param api_key: str: The API key for the OpenAI API
        """
        self._api_key = api_key

    def client(self, prompt: list, model: str, dimensions: int = 1024) -> list:
        """ Initialize the OpenAI Embeddings API
                - dimensions: 256、512、1024、1536

        :param dimensions: int: The number of dimensions for the embedding
        :param model: str: The model to use for the embedding
        :param prompt: list: The input text to be embedded
        :return: None
        """
        client = OpenAI(api_key=self._api_key, base_url="https://api.openai.com/v1")

        response = client.embeddings.create(
            input=prompt,
            model=model,
            dimensions=dimensions,
            encoding_format="float",
            timeout=3,
        )

        return [item.embedding for item in response.data]


class HuggingFaceEmbedder(object):

    def __init__(self, model: str) -> None:
        """ Initialize the Hugging Face Embeddings API

        :param model: str: The model to use for the embedding
        """
        self._model = model

    def client(self, sentences: list) -> list:
        # Load the model from Hugging Face model hub
        model = SentenceTransformer(self._model)

        # Compute the embeddings for the sentences, the dimensionality of the embeddings is 384
        embeddings = model.encode(sentences)
        return embeddings.tolist()
