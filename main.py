from pprint import pprint
from typing import List, Dict

from utilis.tools import SENTENCES, Chunker


def main():
    """ streamlit run main.py """
    chunker = Chunker()

    chunks: Dict[int, List[str]] = chunker.chunk(SENTENCES)
    pprint(chunks)

    paras = chunker.paragraph()
    for para in paras:
        print(para)


if __name__ == "__main__":
    main()
