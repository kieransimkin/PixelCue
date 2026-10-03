import ast
from pathlib import Path


def test_joycaption_methods_are_class_children_not_nested_functions():
    source = (Path(__file__).parents[1] / "src" / "pixelcue" / "model.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "JoyCaption4Bit"
    )
    names = {
        node.name for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert {"ensure_downloaded", "load", "tags_for_image"} <= names
