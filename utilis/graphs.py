#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/3/3 22:14
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   graphs.py
# @Desc     :   

from enum import unique, StrEnum
from streamlit_agraph import agraph, Node, Edge, Config


@unique
class Color(StrEnum):
    """ Color Enum """
    RED: str = "#FFAAAA"
    YELLOW: str = "#FFEEAA"
    BLUE: str = "#88EEFE"
    GREEN: str = "#AAFFAA"
    GRAY: str = "#525352"


@unique
class Shape(StrEnum):
    """ Shape Enum """
    # IMAGE: str = "image"
    # CIRCULAR_IMAGE: str = "circularImage"
    DIAMOND: str = "diamond"
    DOT: str = "dot"
    STAR: str = "star"
    TRIANGLE: str = "triangle"
    TRIANGLE_DOWN: str = "triangleDown"
    HEXAGON: str = "hexagon"
    SQUARE: str = "square"


def network(clusters: dict) -> agraph:
    """ Create a network graph using the agraph function from the streamlit_agraph library

    :param clusters: a dictionary where the keys are the cluster labels and the values are the corresponding sentences
    :return: a network graph
    """
    nodes: list[Node] = []
    edges: list[Edge] = []

    for label, sentences in clusters.items():
        node = Node(id=f"{label}", label=f"",
                    color=Color.GRAY, shape=Shape.HEXAGON,
                    size=10, font={"size": 12})
        nodes.append(node)

        for sentence in sentences:
            node = Node(id=sentence, label=sentence,
                        color=Color.RED, shape=Shape.DOT,
                        size=30, font={"size": 16})
            nodes.append(node)

            edge = Edge(source=sentence, target=f"{label}", arrow_to=True, label="grouped",
                        width=2, color=Color.GRAY)
            edges.append(edge)

    config = Config(width="100%", height=400, physics=True, directed=True, hierarchical=True)

    return agraph(nodes=nodes, edges=edges, config=config)
