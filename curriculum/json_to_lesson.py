"""Convert a JSON/JSONL file with {id, context, question, answer} into exercises_with_answers XML.

Matches the format in data/ (e.g. qasper_train_default_20_test_t0.25.xml):
  <exercises_with_answers>
    <temperature value="0.25" />
    <exercise_with_answers>
      <messages>
        <message role="user">
          <TIPS>context text</TIPS>question text
        </message>
      </messages>
      <answer_choices>
        <choice truncated="false">answer text</choice>
      </answer_choices>
    </exercise_with_answers>
  </exercises_with_answers>

Usage:
    python curriculum/json_to_lesson.py --input data.jsonl --output data/custom_t0.25.xml
"""

import argparse
import json
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom


def prettify(elem: Element) -> str:
    return minidom.parseString(tostring(elem, "utf-8")).toprettyxml(indent="")


def main(input: str, output: str = "output.xml", temperature: float = 0.25):
    items = []
    with open(input, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if content.startswith("["):
        items = json.loads(content)
    else:
        items = [json.loads(line) for line in content.splitlines() if line.strip()]
    print(f"Loaded {len(items)} items")

    root = Element("exercises_with_answers")
    SubElement(root, "temperature", value=str(temperature))

    for item in items:
        context = item["context"]
        question = item["question"]
        answer = item.get("answer", "")

        ex = SubElement(root, "exercise_with_answers")
        messages = SubElement(ex, "messages")
        msg = SubElement(messages, "message", role="user")

        # <TIPS> as a proper child element, question as its tail text
        tips = SubElement(msg, "TIPS")
        tips.text = context
        tips.tail = question

        choices = SubElement(ex, "answer_choices")
        choice = SubElement(choices, "choice", truncated="false")
        choice.text = answer

    xml_output = prettify(root)
    with open(output, "w", encoding="utf-8") as f:
        f.write(xml_output)
    print(f"Wrote {len(items)} exercises to {output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="output.xml")
    p.add_argument("--temperature", type=float, default=0.25)
    main(**vars(p.parse_args()))