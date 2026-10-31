import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN = ROOT / "models" / "LLaMA3" / "04-LLaMA3-8B-Instruct Lora 微调.md"
NOTEBOOK = ROOT / "models" / "LLaMA3" / "LLaMA3-8B-Instruct Lora.ipynb"


def _calls(source):
    tree = ast.parse(source)
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "from_pretrained"
        and any(keyword.arg == "torch_dtype" for keyword in node.keywords)
    ]


def _has_keyword(call, name):
    return any(keyword.arg == name for keyword in call.keywords)


class Llama3LoraLoadingRegression(unittest.TestCase):
    def test_markdown_training_and_inference_loading_contract(self):
        source = MARKDOWN.read_text(encoding="utf-8")
        blocks = source.split("```python\n")[1:]
        model_calls = [call for block in blocks for call in _calls(block.split("```", 1)[0])]
        self.assertGreaterEqual(len(model_calls), 2)
        self.assertFalse(_has_keyword(model_calls[0], "device_map"))
        self.assertTrue(_has_keyword(model_calls[1], "device_map"))

    def test_notebook_keeps_auto_mapping_for_inference_only(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cells = [
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if "AutoModelForCausalLM.from_pretrained" in "".join(cell.get("source", []))
        ]
        self.assertEqual(len(cells), 2)
        training_call = _calls(cells[0])[0]
        inference_call = _calls(cells[1])[0]
        self.assertFalse(_has_keyword(training_call, "device_map"))
        self.assertTrue(_has_keyword(inference_call, "device_map"))

    def test_original_meta_parameter_failure_is_reproducible(self):
        import torch
        from torch import nn

        module = nn.Module()
        module.register_parameter("weight", nn.Parameter(torch.empty(1, device="meta")))
        with self.assertRaises(NotImplementedError):
            module.to("cpu")


if __name__ == "__main__":
    unittest.main()
