import ast
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN = ROOT / "models" / "Qwen2" / "05-Qwen2-7B-Instruct Lora 微调.md"
NOTEBOOK = ROOT / "models" / "Qwen2" / "05-Qwen2-7B-Instruct Lora.ipynb"


def _from_pretrained_calls(source):
    tree = ast.parse(source)
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "from_pretrained"
    ]


def _keyword(call, name):
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


class Qwen2LoraLoadingRegression(unittest.TestCase):
    def test_markdown_training_and_inference_loading_contract(self):
        source = MARKDOWN.read_text(encoding="utf-8")
        blocks = re.findall(r"```python\n(.*?)```", source, flags=re.DOTALL)
        calls = [call for block in blocks for call in _from_pretrained_calls(block)]
        model_calls = [call for call in calls if _keyword(call, "torch_dtype") is not None]
        self.assertGreaterEqual(len(model_calls), 2)
        self.assertIsNone(_keyword(model_calls[0], "device_map"))
        self.assertIsNotNone(_keyword(model_calls[1], "device_map"))

    def test_notebook_keeps_auto_mapping_for_inference_only(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        model_cells = [
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if "AutoModelForCausalLM.from_pretrained" in "".join(cell.get("source", []))
        ]
        self.assertEqual(len(model_cells), 2)
        training_call = [
            call for call in _from_pretrained_calls(model_cells[0])
            if _keyword(call, "torch_dtype") is not None
        ][0]
        inference_call = [
            call for call in _from_pretrained_calls(model_cells[1])
            if _keyword(call, "torch_dtype") is not None
        ][0]
        self.assertIsNone(_keyword(training_call, "device_map"))
        self.assertIsNotNone(_keyword(inference_call, "device_map"))

    def test_original_failure_is_reproducible_with_a_meta_parameter(self):
        try:
            import torch
            from torch import nn
        except ImportError as error:
            self.skipTest(str(error))
        module = nn.Module()
        module.register_parameter("weight", nn.Parameter(torch.empty(1, device="meta")))
        with self.assertRaises(NotImplementedError):
            module.to("cpu")


if __name__ == "__main__":
    unittest.main()
