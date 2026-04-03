# %%
import asyncio
import os
import time
import warnings
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict

from openai import AsyncOpenAI

from core import DATA_PATH
from core.file_naming import generate_augmented_filename, generate_lesson_filename, generate_exam_filename
from core.llm import LLM
from core.messages import Message, Role
from core.model_configs import create_model_flags, get_model_config
from core.utils import generate_extra_body
from curriculum.lesson import read_lessons, Lesson, Exercise
from curriculum.exercise_with_answers import ExerciseWithAnswers, Choice, xml_dump
from evaluation.utils import async_wrapper
from training.utils import clean_xml_content


def generate_prompt(
    llm: LLM,
    lesson: Lesson,
    max_total_tokens: int,
    max_new_tokens: int,
) -> Tuple[List[Tuple[str, int]], List[Exercise]]:
    """Generate prompts and exercises from a lesson."""
    prompts = []
    exercises = []
    lesson.create_exercise_prompts(verbose=False)

    for ex in lesson.exercises:
        n_tokens_prompt = len(llm.tokenize(ex.student_prompt))
        max_tokens_to_generate = max_total_tokens - n_tokens_prompt

        if max_new_tokens > 0:
            max_tokens_to_generate = min(max_new_tokens, max_tokens_to_generate)

        if max_tokens_to_generate <= 10:
            warnings.warn(f"Too few tokens left for the answer: {max_tokens_to_generate}.", stacklevel=2)
        elif max_tokens_to_generate <= 0:
            raise ValueError(f"Too many tokens in the prompt: {n_tokens_prompt}, while the limit is {max_total_tokens}.")

        prompt = ex.teacher_prompt.replace("&lt;", "<").replace("&gt;", ">")

        prompts.append((prompt, max_tokens_to_generate))
        exercises.append(ex)

    return prompts, exercises


def process_answers(llm: LLM, exercise: Exercise, answers: List[str]) -> ExerciseWithAnswers:
    """Process answers for an exercise."""
    answer_choices = []

    for answer in answers:
        if not isinstance(answer, str):
            answer = answer.text

        tokens = llm.tokenize(answer)
        terminators = llm.get_terminators()
        truncated = bool(tokens[0, -1] not in terminators)
        answer = llm.decode(tokens[:, :-1])
        choice = Choice(answer, truncated)
        answer_choices.append(choice)

    messages = [Message(Role.USER, exercise.teacher_prompt_with_tips_tags)]
    return ExerciseWithAnswers(
        messages, 
        answer_choices, 
        model_answer=exercise.model_answer, 
        grading_str=exercise.grading_str
    )


def save_to_xml(lesson_id: str, exercises_with_answers: List[ExerciseWithAnswers],
                temperature: float, n_choices: int, model_flags: Dict[str, bool]):
    """Save exercises with answers to XML file."""
    root = ET.Element("exercises_with_answers")
    ET.SubElement(root, "temperature", value=str(temperature))

    for ex in exercises_with_answers:
        ex.to_xml(root)

    fname = generate_augmented_filename(lesson_id, n_choices, temperature, model_flags)
    path = DATA_PATH / fname

    with open(path, "w") as file:
        xml_dump(root, file)

    print(f"Saved to {path}")


def setup_models(base: str, vllm_hostname: str = "0.0.0.0") -> Tuple[LLM, AsyncOpenAI]:
    """Setup LLM and OpenAI client."""
    config = get_model_config(base)
    opening_message = Message(Role.SYSTEM, config.system_message)

    llm = LLM(base, opening_message=opening_message)
    client = AsyncOpenAI(base_url=f"http://{vllm_hostname}:8000/v1", api_key="EMPTY")

    return llm, client


async def _generate_teacher_answer(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    extra_body: Dict,
    temperature: float,
    max_tokens: int,
    index: int,
    total: int,
    n_choices: int = 1,
    system_message: str = "",
) -> List[str]:
    """Generate teacher answer(s) for a single prompt via chat completions."""
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})

    rsp = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        n=n_choices,
        stream=False,
        extra_body=extra_body,
    )
    return [choice.message.content for choice in rsp.choices]


def main(
    base: str = "Qwen/Qwen3-4B-Instruct-2507",
    generate_lesson: bool = False,
    generate_exam: bool = False,
    lesson_num_choices: int = 1,
    exam_num_choices: int = 1,
    lesson_temp: float = 1.5,
    exam_temp: float = 0.25,
    max_total_tokens: int = 1024,
    max_new_tokens: int = 500,
    dataset_family: str = "squadshifts",
    dataset: str = "nyt",
    variant: str = "default",
    question_model: str = "Qwen3-4B-Instruct-2507",
    max_items: int = 1000,
    train_questions: int = 30,
    question_temperature: float = 1.5,
    chunk_size: int = 500,
    verbose: bool = False,
    vllm_hostname: str = "0.0.0.0",
):
    assert not (generate_lesson and generate_exam), "The code doesn't support generating lesson and exam simultaneously"
    # Setup models
    llm, client = setup_models(base, vllm_hostname)
    cfg = get_model_config(base)
    model_flags = create_model_flags(base)

    # Setup processing modes
    if generate_lesson:
        xml_name = generate_lesson_filename(
            dataset_family, dataset, variant, question_model, train_questions, question_temperature, max_items
        )
    if generate_exam:
        xml_name = generate_exam_filename(
            dataset_family, dataset, variant, max_items
        )

    temperature = lesson_temp if generate_lesson else exam_temp
    num_choices = lesson_num_choices if generate_lesson else exam_num_choices

    # Setup extra body
    extra_body = generate_extra_body(base)
    print(f"Processing {xml_name}", flush=True)

    # Read lessons
    try:
        lessons = read_lessons(xml_name)
    except ET.ParseError:
        cleaned_xml_filename = clean_xml_content(xml_name)
        lessons = read_lessons(cleaned_xml_filename)

    # Generate prompts and exercises
    prompts = []
    exercises = []
    print(f"Number of lessons: {len(lessons)}", flush=True)

    for lesson_id, lesson in lessons.items():
        p, e = generate_prompt(llm, lesson, max_total_tokens, max_new_tokens)
        prompts += p
        exercises += e

    assert len(prompts) == len(exercises)
    print(f"Number of prompts: {len(prompts)}", flush=True)

    # Generate answers
    start_time = time.time()
    prompts_only = [p for p, _ in prompts]

    answers = asyncio.run(
        async_wrapper(
            client,
            cfg.vllm_model,
            prompts_only,
            extra_body,
            temperature,
            max_total_tokens,
            batch_size=chunk_size,
            custom_fnc=_generate_teacher_answer,
            custom_fnc_extra_kwargs={
                "n_choices": num_choices,
                "system_message": cfg.system_message,
            },
        )
    )

    end_time = time.time()
    print(f"Generation time: {end_time - start_time:.4f} s", flush=True)

    assert len(prompts) == len(exercises) == len(answers)
    assert len(answers[0]) == num_choices

    # Process answers
    exercises_with_answers = []
    for ex, ans in zip(exercises, answers):
        exercises_with_answers.append(process_answers(llm, ex, ans))

    assert len(exercises_with_answers) == len(exercises)

    # Save results
    save_to_xml(
        lesson_id.rsplit('_', 1)[0], 
        exercises_with_answers, 
        temperature, 
        num_choices,
        model_flags,
    )


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)
