"""计算图抽取：从真实执行轨迹重建一张带证据的计算图。

用法：
    from nanovllm_selector.extract import extract_graph
    g = extract_graph("tmp/models/Qwen3-0.6B")
    print(g.op_counts())
"""

from .schema import SCHEMA_VERSION, Edge, Graph, Node


def __getattr__(name):
    if name == "extract_graph":
        from .api import extract_graph

        return extract_graph
    raise AttributeError(name)

__all__ = ["extract_graph", "Graph", "Node", "Edge", "SCHEMA_VERSION"]
