from dbmeta_app.api.model import PromptItem, PromptItemType
from dbmeta_app.vector_db.milvus import QueryExample, get_hits
import logging


def get_query_example_prompt_item(query: str, db: str) -> PromptItem:
    logging.info("get_query_example_prompt_item")

    data = get_hits(query, db)

    # Format into a human-readable LLM prompt
    formatted_examples = []

    for i, example in enumerate(data):
        request = example.request.strip()
        response = example.response.strip()

        # Format the example for LLM input
        formatted_example = (
            f"### Example #{i + 1}:\n"
            f"**User Request:** {request}\n\n"
            f"**Generated SQL:**\n```\n{response}\n```"
        )

        formatted_examples.append(formatted_example)

    # Combine all examples into a single LLM input string
    llm_prompt = "\n\n".join(formatted_examples)
    logging.info("get_query_example_prompt_item done")

    return PromptItem(
        text=llm_prompt,
        prompt_item_type=PromptItemType.query_example,
        score=100_000,
    )


def get_query_examples(query: str, db: str) -> list[QueryExample]:
    res = get_hits(query, db)
    return res
