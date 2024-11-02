from typing import Dict


def parse_retriever_input(params: Dict):
    return params["messages"][-1].content
