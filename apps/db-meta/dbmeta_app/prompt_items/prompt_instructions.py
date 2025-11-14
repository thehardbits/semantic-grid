import pathlib

from dbmeta_app.api.model import PromptItem, PromptItemType
from dbmeta_app.config import get_settings
from dbmeta_app.prompt_assembler.prompt_packs import assemble_effective_tree, load_yaml
import logging


def get_prompt_instructions_item(profile: str) -> PromptItem:
    logging.info("get_prompt_instructions_item")
    settings = get_settings()
    repo_root = pathlib.Path(settings.packs_resources_dir).resolve()
    client = settings.client
    env = settings.env
    tree = assemble_effective_tree(repo_root, profile, client, env)

    instructions = load_yaml(tree, "resources/prompt_instructions.yaml")["profiles"][
        profile
    ]

    # Format into a human-readable LLM prompt
    llm_prompt = "\n\n### Additional Instructions:\n" + "\n".join(
        f"- {instruction}" for instruction in instructions
    )
    logging.info("get_prompt_instructions_item done")

    return PromptItem(
        text=llm_prompt,
        prompt_item_type=PromptItemType.instruction,
        score=100_000,
    )


def get_prompt_instructions(profile: str) -> list[str]:
    # Load the YAML-like structure (assuming it's stored in a file)
    settings = get_settings()
    repo_root = pathlib.Path(settings.packs_resources_dir).resolve()
    client = settings.client
    env = settings.env
    tree = assemble_effective_tree(repo_root, profile, client, env)

    instructions = load_yaml(tree, "resources/prompt_instructions.yaml")["profiles"][
        profile
    ]

    return instructions
