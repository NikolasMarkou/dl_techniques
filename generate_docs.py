#!/usr/bin/env python3
"""
Documentation Generator for DL-Techniques Library

Run this script from your project root to generate comprehensive documentation.
"""

import os
import sys
import ast
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import json


@dataclass
class ComponentInfo:
    """Information about a code component."""
    name: str
    type: str  # 'class', 'function', 'module'
    docstring: Optional[str]
    file_path: str
    line_number: int
    signature: Optional[str] = None
    bases: List[str] = None  # For classes
    decorators: List[str] = None
    is_public: bool = True
    init_args: Optional[List[str]] = None  # For Keras layer classes


@dataclass
class ModuleInfo:
    """Information about a module."""
    name: str
    path: str
    docstring: Optional[str]
    components: List[ComponentInfo]
    imports: List[str]
    category: str  # 'models', 'layers', 'losses', etc.


class DLTechniquesDocGenerator:
    """
    Documentation generator for the dl_techniques library.

    This class analyzes the codebase and generates comprehensive documentation
    showing the structure and contents of the library.
    """

    def __init__(self, root_path: str = "src/dl_techniques", output_dir: str = "docs"):
        """
        Initialize the documentation generator.

        Args:
            root_path: Path to the dl_techniques source code
            output_dir: Directory to save generated documentation
        """
        self.root_path = Path(root_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Component categories based on your project structure
        self.categories = {
            'models': 'Complete model architectures ready for training',
            'layers': 'Individual neural network layers and building blocks',
            'losses': 'Loss functions for different tasks',
            'metrics': 'Evaluation metrics',
            'initializers': 'Weight initialization strategies',
            'regularizers': 'Regularization techniques',
            'constraints': 'Weight constraints',
            'optimization': 'Training optimization utilities',
            'utils': 'Utility functions and helpers',
            'analyzer': 'Model analysis and evaluation tools',
            'weightwatcher': 'Advanced model quality analysis',
            'visualization': 'Visualization utilities'
        }

        self.modules: Dict[str, ModuleInfo] = {}
        self.category_index: Dict[str, List[str]] = defaultdict(list)

    def analyze_codebase(self) -> None:
        """Analyze the entire codebase and extract component information."""
        print("🔍 Starting codebase analysis...")

        py_files = list(self.root_path.rglob("*.py"))
        print(f"📁 Found {len(py_files)} Python files")

        analyzed_count = 0
        for py_file in py_files:
            if "__pycache__" in str(py_file) or py_file.name.startswith("."):
                continue

            try:
                module_info = self._analyze_file(py_file)
                if module_info:
                    self.modules[module_info.name] = module_info
                    self.category_index[module_info.category].append(module_info.name)
                    analyzed_count += 1
                    if analyzed_count % 10 == 0:
                        print(f"  ✅ Analyzed {analyzed_count} modules...")
            except Exception as e:
                print(f"  ⚠️ Failed to analyze {py_file}: {e}")

        print(f"📊 Analysis complete: {len(self.modules)} modules across {len(self.category_index)} categories")

    def _analyze_file(self, file_path: Path) -> Optional[ModuleInfo]:
        """Analyze a single Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            print(f"⚠️ Could not read {file_path} due to encoding issues")
            return None

        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            print(f"⚠️ Syntax error in {file_path}: {e}")
            return None

        # Extract module info
        module_name = self._get_module_name(file_path)
        category = self._get_category(file_path)
        docstring = ast.get_docstring(tree)

        components = []
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                comp_info = self._extract_class_info(node, file_path)
                components.append(comp_info)
            elif isinstance(node, ast.FunctionDef):
                comp_info = self._extract_function_info(node, file_path)
                components.append(comp_info)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.extend(self._extract_imports(node))

        return ModuleInfo(
            name=module_name,
            path=str(file_path),
            docstring=docstring,
            components=components,
            imports=imports,
            category=category
        )

    def _get_module_name(self, file_path: Path) -> str:
        """Get the module name from file path."""
        rel_path = file_path.relative_to(self.root_path)
        parts = list(rel_path.parts[:-1])  # Remove filename
        if rel_path.stem != "__init__":
            parts.append(rel_path.stem)
        return ".".join(parts) if parts else rel_path.stem

    def _get_category(self, file_path: Path) -> str:
        """Determine the category of a file based on its path."""
        rel_path = file_path.relative_to(self.root_path)
        if len(rel_path.parts) > 0:
            first_part = rel_path.parts[0]
            if first_part in self.categories:
                return first_part
        return "utils"

    def _extract_class_info(self, node: ast.ClassDef, file_path: Path) -> ComponentInfo:
        """Extract information about a class."""
        bases = [self._ast_to_string(base) for base in node.bases]
        decorators = [self._ast_to_string(dec) for dec in node.decorator_list]

        # Check if this is a Keras layer by examining base classes
        is_keras_layer = self._is_keras_layer(bases)
        init_args = None

        if is_keras_layer:
            init_args = self._extract_init_args(node)

        return ComponentInfo(
            name=node.name,
            type="class",
            docstring=ast.get_docstring(node),
            file_path=str(file_path),
            line_number=node.lineno,
            bases=bases,
            decorators=decorators,
            is_public=not node.name.startswith("_"),
            init_args=init_args
        )

    def _extract_function_info(self, node: ast.FunctionDef, file_path: Path) -> ComponentInfo:
        """Extract information about a function."""
        # Build signature
        args = []
        for arg in node.args.args:
            args.append(arg.arg)
        signature = f"{node.name}({', '.join(args)})"

        decorators = [self._ast_to_string(dec) for dec in node.decorator_list]

        return ComponentInfo(
            name=node.name,
            type="function",
            docstring=ast.get_docstring(node),
            file_path=str(file_path),
            line_number=node.lineno,
            signature=signature,
            decorators=decorators,
            is_public=not node.name.startswith("_")
        )

    def _extract_imports(self, node: ast.AST) -> List[str]:
        """Extract import statements."""
        imports = []
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append(f"{module}.{alias.name}" if module else alias.name)
        return imports

    def _is_keras_layer(self, bases: List[str]) -> bool:
        """Check if a class inherits from keras.layers.Layer."""
        keras_layer_indicators = [
            'keras.layers.Layer',
            'Layer',
            'layers.Layer',
            'BaseConv',
            'BaseDepthwiseConv',
            'BaseLayer',
        ]

        for base in bases:
            if any(indicator in base for indicator in keras_layer_indicators):
                return True
        return False

    def _extract_init_args(self, class_node: ast.ClassDef) -> List[str]:
        """Extract __init__ method arguments from a class."""
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef) and node.name == "__init__":
                args = []
                for arg in node.args.args:
                    if arg.arg != "self":  # Skip 'self' parameter
                        # Check if argument has a default value
                        arg_name = arg.arg

                        # Try to get type annotation if available
                        if arg.annotation:
                            try:
                                type_hint = self._ast_to_string(arg.annotation)
                                arg_name = f"{arg_name}: {type_hint}"
                            except:
                                pass  # If we can't parse the annotation, just use the name

                        # Check for default values
                        defaults_offset = len(node.args.args) - len(node.args.defaults)
                        arg_index = node.args.args.index(arg)

                        if arg_index >= defaults_offset:
                            default_index = arg_index - defaults_offset
                            try:
                                default_value = self._ast_to_string(node.args.defaults[default_index])
                                arg_name = f"{arg_name} = {default_value}"
                            except:
                                arg_name = f"{arg_name} = ..."

                        args.append(arg_name)

                # Handle **kwargs if present
                if node.args.kwarg:
                    args.append(f"**{node.args.kwarg.arg}")

                return args

        return []  # No __init__ method found

    def _ast_to_string(self, node: ast.AST) -> str:
        """Convert AST node to string representation."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._ast_to_string(node.value)}.{node.attr}"
        elif isinstance(node, ast.Constant):
            return repr(node.value)
        else:
            try:
                return ast.unparse(node)
            except:
                return str(node)

    def generate_documentation(self) -> None:
        """Generate all documentation files."""
        print("📝 Generating documentation...")

        self._generate_main_index()
        self._generate_category_pages()
        self._generate_component_reference()
        self._generate_json_index()

        print(f"✅ Documentation generated in {self.output_dir}")

    def _generate_main_index(self) -> None:
        """Generate the main index page."""
        content = []
        content.append("# DL-Techniques Library Documentation")
        content.append(
            "\nA comprehensive deep learning techniques library built on Keras 3.x with TensorFlow 2.18.0 backend.")
        content.append("\n## Library Overview")

        # Statistics
        total_modules = len(self.modules)
        total_classes = sum(len([c for c in mod.components if c.type == "class" and c.is_public])
                            for mod in self.modules.values())
        total_functions = sum(len([c for c in mod.components if c.type == "function" and c.is_public])
                              for mod in self.modules.values())
        total_keras_layers = sum(len([c for c in mod.components
                                      if c.type == "class" and c.is_public and c.init_args])
                                 for mod in self.modules.values())

        content.append(f"\n- **{total_modules}** modules")
        content.append(f"- **{total_classes}** public classes")
        content.append(f"- **{total_keras_layers}** Keras layers")
        content.append(f"- **{total_functions}** public functions")
        content.append(f"- **{len(self.category_index)}** component categories")

        # Categories overview
        content.append("\n## Component Categories")
        content.append("\nThe library is organized into the following categories:")

        for category, description in self.categories.items():
            module_count = len(self.category_index.get(category, []))
            if module_count > 0:
                content.append(f"\n### [{category.title()}](categories/{category}.md) ({module_count} modules)")
                content.append(f"{description}")

        # Quick navigation
        content.append("\n## Documentation Navigation")
        content.append("- 🔍 [Component Reference](component_reference.md) - All components organized by type")
        content.append("- 📁 [Category Pages](categories/) - Components organized by purpose")
        content.append("- 📊 [JSON Index](module_index.json) - Machine-readable library index")

        # Getting started
        content.append("\n## Getting Started")
        content.append("\n### Installation")
        content.append("```python")
        content.append("# Install dependencies")
        content.append("pip install keras==3.8.0 tensorflow==2.18.0")
        content.append("```")

        content.append("\n### Basic Usage")
        content.append("```python")
        content.append("import keras")
        content.append("from dl_techniques.models import ConvNeXtV2")
        content.append("from dl_techniques.layers import SwinTransformerBlock")
        content.append("from dl_techniques.losses import ClipContrastiveLoss")
        content.append("")
        content.append("# Create a model")
        content.append("model = ConvNeXtV2(num_classes=10)")
        content.append("")
        content.append("# Use custom layers")
        content.append("transformer_block = SwinTransformerBlock(dim=96, num_heads=3)")
        content.append("```")

        self._write_file("README.md", "\n".join(content))

    def _generate_category_pages(self) -> None:
        """Generate documentation pages for each category."""
        categories_dir = self.output_dir / "categories"
        categories_dir.mkdir(exist_ok=True)

        for category, description in self.categories.items():
            modules = [self.modules[name] for name in self.category_index.get(category, [])]
            if not modules:
                continue

            content = []
            content.append(f"# {category.title()}")
            content.append(f"\n{description}")
            content.append(f"\n**{len(modules)} modules in this category**")

            # Group by subcategory if applicable
            subcategories = defaultdict(list)
            for module in modules:
                parts = module.name.split(".")
                subcat = parts[1] if len(parts) > 1 else "core"
                subcategories[subcat].append(module)

            for subcat, submodules in sorted(subcategories.items()):
                if len(subcategories) > 1:
                    content.append(f"\n## {subcat.title()}")

                for module in sorted(submodules, key=lambda x: x.name):
                    content.append(f"\n### {module.name}")
                    if module.docstring:
                        # Take first line of docstring
                        first_line = module.docstring.split('\n')[0].strip()
                        if first_line:
                            content.append(f"{first_line}")

                    # List public classes with __init__ args for Keras layers
                    classes = [c for c in module.components if c.type == "class" and c.is_public]
                    if classes:
                        content.append(f"\n**Classes:**")
                        for cls in classes:
                            if cls.init_args:  # Keras layer with constructor args
                                content.append(f"\n- `{cls.name}` - Keras Layer")
                                if cls.docstring:
                                    first_line = cls.docstring.split('\n')[0].strip()
                                    if first_line:
                                        content.append(f"  {first_line}")
                                content.append(f"  ```python")
                                args_str = ", ".join(cls.init_args[:3])  # Show first 3 args to keep it concise
                                if len(cls.init_args) > 3:
                                    args_str += ", ..."
                                content.append(f"  {cls.name}({args_str})")
                                content.append(f"  ```")
                            else:  # Regular class
                                class_name = f"`{cls.name}`"
                                content.append(f"- {class_name}")

                    # List public functions
                    functions = [c for c in module.components if c.type == "function" and c.is_public]
                    if functions:
                        func_list = ", ".join(f"`{c.name}`" for c in functions[:5])
                        if len(functions) > 5:
                            func_list += f" (and {len(functions) - 5} more)"
                        content.append(f"\n**Functions:** {func_list}")

                    content.append(f"\n*📁 File: `{module.path}`*")

            self._write_file(f"categories/{category}.md", "\n".join(content))

    def _generate_component_reference(self) -> None:
        """Generate component reference organized by type."""
        content = []
        content.append("# Component Reference")
        content.append("\nAll public components in the DL-Techniques library organized by type.")

        # Collect all components by type
        all_classes = []
        all_functions = []

        for module in self.modules.values():
            for comp in module.components:
                if comp.is_public:
                    if comp.type == "class":
                        all_classes.append((comp, module))
                    elif comp.type == "function":
                        all_functions.append((comp, module))

        # Classes section
        content.append(f"\n## Classes ({len(all_classes)})")

        # Group classes by category
        classes_by_category = defaultdict(list)
        for comp, module in all_classes:
            classes_by_category[module.category].append((comp, module))

        for category in sorted(classes_by_category.keys()):
            content.append(f"\n### {category.title()} Classes")
            for comp, module in sorted(classes_by_category[category], key=lambda x: x[0].name):
                content.append(f"\n#### `{comp.name}`")
                content.append(f"**Module:** `{module.name}`")

                if comp.docstring:
                    # Take first paragraph of docstring
                    lines = comp.docstring.split('\n')
                    first_para = []
                    for line in lines:
                        line = line.strip()
                        if not line and first_para:
                            break
                        if line:
                            first_para.append(line)
                    if first_para:
                        content.append(f"\n{' '.join(first_para)}")

                # Show __init__ arguments for Keras layers
                if comp.init_args:
                    content.append(f"\n**Constructor Arguments:**")
                    content.append("```python")
                    args_str = ",\n    ".join(comp.init_args)
                    content.append(f"{comp.name}(\n    {args_str}\n)")
                    content.append("```")

                if comp.bases and comp.bases != ['object']:
                    content.append(f"\n*Inherits from: {', '.join(f'`{b}`' for b in comp.bases)}*")

                content.append(f"\n*📁 {comp.file_path}:{comp.line_number}*")

        # Functions section
        content.append(f"\n## Functions ({len(all_functions)})")

        functions_by_category = defaultdict(list)
        for comp, module in all_functions:
            functions_by_category[module.category].append((comp, module))

        for category in sorted(functions_by_category.keys()):
            content.append(f"\n### {category.title()} Functions")
            for comp, module in sorted(functions_by_category[category], key=lambda x: x[0].name):
                content.append(f"\n#### `{comp.signature}`")
                content.append(f"**Module:** `{module.name}`")

                if comp.docstring:
                    # Take first paragraph of docstring
                    lines = comp.docstring.split('\n')
                    first_para = []
                    for line in lines:
                        line = line.strip()
                        if not line and first_para:
                            break
                        if line:
                            first_para.append(line)
                    if first_para:
                        content.append(f"\n{' '.join(first_para)}")

                content.append(f"\n*📁 {comp.file_path}:{comp.line_number}*")

        self._write_file("component_reference.md", "\n".join(content))

    def _generate_json_index(self) -> None:
        """Generate machine-readable JSON index."""
        index_data = {
            "library": "dl_techniques",
            "version": "1.0.0",
            "description": "A comprehensive deep learning techniques library built on Keras 3.x",
            "statistics": {
                "total_modules": len(self.modules),
                "total_classes": sum(len([c for c in mod.components if c.type == "class" and c.is_public])
                                     for mod in self.modules.values()),
                "total_keras_layers": sum(len([c for c in mod.components
                                               if c.type == "class" and c.is_public and c.init_args])
                                          for mod in self.modules.values()),
                "total_functions": sum(len([c for c in mod.components if c.type == "function" and c.is_public])
                                       for mod in self.modules.values()),
                "categories": len([cat for cat in self.category_index.keys() if self.category_index[cat]])
            },
            "categories": {cat: desc for cat, desc in self.categories.items()
                           if cat in self.category_index and self.category_index[cat]},
            "modules": {}
        }

        for module in self.modules.values():
            index_data["modules"][module.name] = {
                "path": module.path,
                "category": module.category,
                "docstring": module.docstring,
                "public_classes": [comp.name for comp in module.components
                                   if comp.type == "class" and comp.is_public],
                "public_functions": [comp.name for comp in module.components
                                     if comp.type == "function" and comp.is_public],
                "keras_layers": [
                    {
                        "name": comp.name,
                        "init_args": comp.init_args,
                        "docstring": comp.docstring
                    }
                    for comp in module.components
                    if comp.type == "class" and comp.is_public and comp.init_args
                ],
                "total_components": len([c for c in module.components if c.is_public])
            }

        self._write_file("module_index.json", json.dumps(index_data, indent=2))

    def _write_file(self, filename: str, content: str) -> None:
        """Write content to a file."""
        file_path = self.output_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  📝 Generated {filename}")


def main():
    """Main entry point for the documentation generator."""
    print("🚀 DL-Techniques Documentation Generator")
    print("=" * 50)

    # Check if source directory exists
    root_path = "src/dl_techniques"
    if not Path(root_path).exists():
        print(f"❌ Source directory '{root_path}' not found!")
        print("Please run this script from your project root directory.")
        sys.exit(1)

    output_dir = "docs"
    print(f"📂 Source: {root_path}")
    print(f"📂 Output: {output_dir}")
    print()

    generator = DLTechniquesDocGenerator(root_path, output_dir)
    generator.analyze_codebase()
    print()
    generator.generate_documentation()

    print("\n" + "=" * 50)
    print("✅ Documentation generated successfully!")
    print(f"📁 Output directory: {Path(output_dir).absolute()}")
    print(f"🌐 Open {output_dir}/README.md to get started")
    print("\nGenerated files:")
    print("  📚 README.md - Main documentation index")
    print("  📋 component_reference.md - All components by type")
    print("  📁 categories/ - Components by category")
    print("  📊 module_index.json - Machine-readable index")


if __name__ == "__main__":
    main()