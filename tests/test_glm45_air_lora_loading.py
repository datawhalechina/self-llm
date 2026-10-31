import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN = ROOT / "models" / "GLM-4.5-Air" / "03-GLM-4.5-Air-Lora 及 Swanlab 可视化微调.md"
NOTEBOOK = ROOT / "models" / "GLM-4.5-Air" / "03-GLM-4.5-Air-Lora 及 Swanlab 可视化微调.ipynb"
MODEL_ID = "/model/ModelScope/ZhipuAI/GLM-4.5-Air"


def model_calls(source):
    tree = ast.parse(source)
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "from_pretrained"
        and any(keyword.arg == "torch_dtype" for keyword in node.keywords)
    ]


def keyword(call, name):
    return next((item.value for item in call.keywords if item.arg == name), None)


def model_path(call):
    return call.args[0] if call.args else keyword(call, "pretrained_model_name_or_path")


class GLM45AirLoadingRegression(unittest.TestCase):
    def test_markdown_uses_glm_path_and_trainer_owned_training_placement(self):
        source = MARKDOWN.read_text(encoding="utf-8")
        blocks = source.split("```python\n")[1:]
        calls = [call for block in blocks for call in model_calls(block.split("```", 1)[0])]
        self.assertGreaterEqual(len(calls), 2)
        self.assertEqual(ast.literal_eval(model_path(calls[0])), MODEL_ID)
        self.assertIsNone(keyword(calls[0], "device_map"))
        self.assertIsNotNone(keyword(calls[1], "device_map"))
        self.assertIn(f"mode_path = '{MODEL_ID}'", source)

    def test_notebook_keeps_auto_mapping_for_inference_only(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cells = [
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if "AutoModelForCausalLM.from_pretrained" in "".join(cell.get("source", []))
        ]
        self.assertEqual(len(cells), 2)
        training_call, inference_call = model_calls(cells[0])[0], model_calls(cells[1])[0]
        self.assertIsNone(keyword(training_call, "device_map"))
        self.assertIsNotNone(keyword(inference_call, "device_map"))

    def test_original_meta_parameter_failure_is_reproducible(self):
        import torch
        from torch import nn

        module = nn.Module()
        module.register_parameter("weight", nn.Parameter(torch.empty(1, device="meta")))
        with self.assertRaises(NotImplementedError):
            module.to("cpu")


if __name__ == "__main__":
    unittest.main()
