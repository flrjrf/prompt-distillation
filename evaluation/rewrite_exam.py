import asyncio
import re
import time
from pathlib import Path
from xml.etree.ElementTree import parse as xml_parse, Element, SubElement, tostring
from xml.dom import minidom

from openai import AsyncOpenAI

from core.llm import LLM
from core.messages import Message, Role, merge_messages
from core.model_configs import get_model_config
from core.utils import generate_extra_body
from evaluation.utils import async_wrapper


def prettify(elem: Element) -> str:
    rough_string = tostring(elem, "utf-8")
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="")


def main(
    exam_file: str,
    base: str = "Qwen/Qwen3-4B-Instruct-2507",
    vllm_hostname: str = "0.0.0.0",
    temperature: float = 0.1,
) -> None:
    cfg = get_model_config(base)
    opening_msg = Message(Role.SYSTEM, cfg.system_message)
    llm = LLM(base, opening_message=opening_msg)

    base_url = f"http://{vllm_hostname}:8000/v1"
    vllm_client = AsyncOpenAI(base_url=base_url, api_key="EMPTY")
    extra_body = generate_extra_body(base)

    exam_path = Path(exam_file)
    if not exam_path.exists():
        print(f"{exam_path} does not exist – aborting.")
        return

    output_path = exam_path.with_name(exam_path.stem + "_rewritten.xml")
    if output_path.exists():
        print(f"{output_path} already exists – aborting.")
        return

    tree = xml_parse(exam_path)
    root = tree.getroot()

    # Collect all lessons/exercises — support both formats:
    # 1) <lessons><lesson><material>ctx</material><exercise>q</exercise></lesson>
    # 2) <exercises_with_answers><exercise_with_answers><messages><message role="user"><TIPS>ctx</TIPS>question</message>
    items = []  # (parent_element_to_edit, context, question)
    if root.tag == "lessons":
        for lesson in root.iter("lesson"):
            material_el = lesson.find("material")
            exercise_el = lesson.find("exercise")
            if material_el is None or exercise_el is None:
                continue
            context = material_el.text or ""
            question = exercise_el.text or ""
            items.append((exercise_el, context, question))
    else:
        # exercises_with_answers format
        for exercise in root.iter("exercise_with_answers"):
            msg_el = exercise.find(".//message[@role='user']")
            if msg_el is None or msg_el.text is None:
                continue
            text = msg_el.text
            tips_match = re.search(r"<TIPS>(.*?)</TIPS>", text, re.DOTALL)
            if not tips_match:
                continue
            context = tips_match.group(1)
            question = text[tips_match.end():].strip()
            items.append((msg_el, tips_match.end(), question, context))

    print(f"Found {len(items)} exercises", flush=True)

    # Build rewrite prompts
    prompts = []
    for _, context, question in items:
        prompt = f"""Here is a piece of text:
{context}

Here is a question related to the text:
{question}

Please re-write the question such that it can be fully understood and it makes sense without access to the text. Output the new question inside <question> xml tags, like this:

<question>Rewritten question</question>"""

        messages = [Message(Role.USER, prompt)]
        messages = merge_messages(messages)
        prompts.append(llm.messages_to_prompt(messages))

    print(f"Number of prompts: {len(prompts)}", flush=True)
    start_time = time.time()
    answers = asyncio.run(
        async_wrapper(vllm_client, cfg.vllm_model, prompts, extra_body, temperature, max_tokens=500)
    )
    end_time = time.time()
    print(f"Generation time: {end_time - start_time:.4f} s", flush=True)
    assert len(prompts) == len(answers)

    # Rewrite the exercises in the XML tree
    for item, answer in zip(items, answers):
        matches = re.findall(r"<question>(.*?)</question>", answer)
        if len(matches) == 1 and len(matches[0]):
            rewritten = matches[0]
        else:
            print(f"Answer {answer[:100]} invalid. Keeping original")
            continue

        if len(item) == 3:
            # lessons format: (exercise_el, context, question)
            exercise_el = item[0]
            exercise_el.text = rewritten
        else:
            # exercises_with_answers format: (msg_el, tips_end, question, context)
            msg_el, tips_end, orig_q, context = item
            tips_text = msg_el.text[:tips_end]
            msg_el.text = tips_text + rewritten

    with open(output_path, "w", encoding="utf-8") as f:
        try:
            f.write(prettify(root))
        except Exception:
            f.write(tostring(root, "unicode"))
    print(f"Saved rewritten exam in {output_path}")


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)
