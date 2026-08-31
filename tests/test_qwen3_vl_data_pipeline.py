import ast
import unittest
from pathlib import Path
from typing import Any, Dict, List

import torch


TRAIN_SCRIPT = (
    Path(__file__).parents[1]
    / "models"
    / "Qwen3-VL"
    / "05-Qwen3-VL-30B-A3B-Instruct  Lora 可视化微调案例 - LaTexOCR"
    / "train_qwen3_vl.py"
)


def load_data_pipeline():
    source = TRAIN_SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    nodes = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.ClassDef)
            and node.name == "Qwen3VLDataCollator"
        )
        or (
            isinstance(node, ast.FunctionDef)
            and node.name == "process_func"
        )
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "PROMPT_TEXT"
                for target in node.targets
            )
        )
    ]
    namespace = {
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "torch": torch,
        "process_vision_info": lambda messages: (["image"], None),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(TRAIN_SCRIPT), "exec"), namespace)
    return namespace["Qwen3VLDataCollator"], namespace["process_func"]


Qwen3VLDataCollator, process_func = load_data_pipeline()


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [20, 21], "attention_mask": [1, 1]}


class FakeProcessor:
    def __init__(self, include_mm_token_type_ids=True):
        self.include_mm_token_type_ids = include_mm_token_type_ids

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return "prompt"

    def __call__(self, **kwargs):
        result = {
            "input_ids": [[10, 11, 12]],
            "attention_mask": [[1, 1, 1]],
            "pixel_values": torch.ones((2, 3)),
            "image_grid_thw": [[1, 1, 2]],
        }
        if self.include_mm_token_type_ids:
            result["mm_token_type_ids"] = [[0, 1, 0]]
        return result


def make_feature(input_ids, mm_token_type_ids=None):
    feature = {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * (len(input_ids) - 1) + [input_ids[-1]],
        "pixel_values": torch.ones((1, 3)),
        "image_grid_thw": [1, 1, 1],
    }
    if mm_token_type_ids is not None:
        feature["mm_token_type_ids"] = mm_token_type_ids
    return feature


class Qwen3VLDataPipelineTest(unittest.TestCase):
    def test_process_func_preserves_mm_types_and_extends_text_tokens(self):
        result = process_func(
            {"image": object(), "text": "answer"},
            FakeTokenizer(),
            FakeProcessor(),
        )

        self.assertEqual([10, 11, 12, 20, 21, 2], result["input_ids"])
        self.assertEqual([0, 1, 0, 0, 0, 0], result["mm_token_type_ids"])
        self.assertEqual(len(result["input_ids"]), len(result["mm_token_type_ids"]))

    def test_collator_pads_mm_types_with_text_type(self):
        features = [
            make_feature([10, 11, 12], [0, 1, 0]),
            make_feature([13, 14], [0, 0]),
        ]

        batch = Qwen3VLDataCollator(FakeTokenizer())(features)

        self.assertEqual((2, 3), tuple(batch["mm_token_type_ids"].shape))
        self.assertEqual([[0, 1, 0], [0, 0, 0]], batch["mm_token_type_ids"].tolist())
        self.assertEqual(batch["input_ids"].shape, batch["mm_token_type_ids"].shape)

    def test_legacy_processor_remains_supported(self):
        result = process_func(
            {"image": object(), "text": "answer"},
            FakeTokenizer(),
            FakeProcessor(include_mm_token_type_ids=False),
        )
        self.assertNotIn("mm_token_type_ids", result)

        batch = Qwen3VLDataCollator(FakeTokenizer())(
            [make_feature([10, 11]), make_feature([12])]
        )
        self.assertNotIn("mm_token_type_ids", batch)

    def test_collator_rejects_mixed_or_misaligned_mm_types(self):
        collator = Qwen3VLDataCollator(FakeTokenizer())

        with self.assertRaisesRegex(ValueError, "全部包含或全部不包含"):
            collator(
                [
                    make_feature([10, 11], [0, 1]),
                    make_feature([12, 13]),
                ]
            )

        with self.assertRaisesRegex(ValueError, "长度必须一致"):
            collator([make_feature([10, 11], [0])])


if __name__ == "__main__":
    unittest.main()
