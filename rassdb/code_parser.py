"""Tree-sitter based code parser for extracting semantic code chunks.

This module provides code parsing functionality using Tree-sitter to extract
meaningful code chunks like functions, classes, and methods from source files.
"""

import tree_sitter
from pathlib import Path
from typing import List, Dict, Optional, Any, NamedTuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class CodeChunk:
    """Represents a semantic code chunk extracted from source code.

    Attributes:
        content: The actual code content.
        chunk_type: Type of chunk (e.g., 'function', 'class', 'method').
        start_line: Starting line number in the source file (1-indexed).
        end_line: Ending line number in the source file (1-indexed).
        name: Optional name of the chunk (e.g., function name).
        metadata: Additional metadata about the chunk.
    """

    content: str
    chunk_type: str
    start_line: int
    end_line: int
    name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class LanguageConfig(NamedTuple):
    """Configuration for a programming language parser."""

    module_name: str
    node_types: Dict[str, str]  # Maps Tree-sitter node types to our chunk types
    extensions: List[str]


class CodeParser:
    """Parser for extracting semantic code chunks using Tree-sitter.

    This parser supports multiple programming languages and falls back to
    simple heuristic parsing when Tree-sitter is not available.
    """

    # Language configurations
    LANGUAGE_CONFIGS = {
        "python": LanguageConfig(
            module_name="tree_sitter_python",
            node_types={
                "function_definition": "function",
                "class_definition": "class",
                "method_definition": "method",
            },
            extensions=[".py", ".pyi"],
        ),
        "javascript": LanguageConfig(
            module_name="tree_sitter_javascript",
            node_types={
                "function_declaration": "function",
                "function_expression": "function",
                "arrow_function": "function",
                "class_declaration": "class",
                "method_definition": "method",
            },
            extensions=[".js", ".jsx", ".mjs"],
        ),
        "typescript": LanguageConfig(
            module_name="tree_sitter_typescript",
            node_types={
                "function_declaration": "function",
                "function_expression": "function",
                "arrow_function": "function",
                "class_declaration": "class",
                "method_definition": "method",
                "interface_declaration": "interface",
                "type_alias_declaration": "type",
            },
            extensions=[".ts", ".tsx"],
        ),
        "java": LanguageConfig(
            module_name="tree_sitter_java",
            node_types={
                "method_declaration": "method",
                "class_declaration": "class",
                "interface_declaration": "interface",
                "constructor_declaration": "constructor",
            },
            extensions=[".java"],
        ),
        "cpp": LanguageConfig(
            module_name="tree_sitter_cpp",
            node_types={
                "function_definition": "function",
                "class_specifier": "class",
                "struct_specifier": "struct",
            },
            extensions=[".cpp", ".cc", ".cxx", ".hpp", ".h", ".c++"],
        ),
        "c": LanguageConfig(
            module_name="tree_sitter_c",
            node_types={
                "function_definition": "function",
                "struct_specifier": "struct",
            },
            extensions=[".c", ".h"],
        ),
        "rust": LanguageConfig(
            module_name="tree_sitter_rust",
            node_types={
                "function_item": "function",
                "impl_item": "impl",
                "struct_item": "struct",
                "enum_item": "enum",
                "trait_item": "trait",
            },
            extensions=[".rs"],
        ),
        "go": LanguageConfig(
            module_name="tree_sitter_go",
            node_types={
                "function_declaration": "function",
                "method_declaration": "method",
                "type_declaration": "type",
            },
            extensions=[".go"],
        ),
    }

    def __init__(self) -> None:
        """Initialize the code parser."""
        self.parsers: Dict[str, tree_sitter.Parser] = {}
        self.languages: Dict[str, tree_sitter.Language] = {}
        self._extension_map: Dict[str, str] = {}
        self._init_languages()

    def _init_languages(self) -> None:
        """Initialize Tree-sitter languages with robust API handling."""
        # Build extension map
        for lang_name, config in self.LANGUAGE_CONFIGS.items():
            for ext in config.extensions:
                self._extension_map[ext] = lang_name

        # Try to initialize each language parser
        for lang_name, config in self.LANGUAGE_CONFIGS.items():
            try:
                module = __import__(config.module_name)

                # Special handling for TypeScript which has a different API
                if lang_name == "typescript":
                    # TypeScript module exports language_typescript instead of language()
                    if hasattr(module, "language_typescript"):
                        lang_capsule = module.language_typescript()
                        language = tree_sitter.Language(lang_capsule)
                    else:
                        # Skip TypeScript if it doesn't have the expected API
                        logger.debug(f"Skipping {lang_name} parser - incompatible API")
                        continue
                else:
                    # Get the language capsule and wrap it
                    lang_capsule = module.language()
                    language = tree_sitter.Language(lang_capsule)

                parser = tree_sitter.Parser(language)

                self.languages[lang_name] = language
                self.parsers[lang_name] = parser
                logger.debug(f"✓ {lang_name} parser initialized successfully")

            except ValueError as e:
                # Skip languages with version incompatibility
                if "Incompatible Language version" in str(e):
                    logger.debug(
                        f"Skipping {lang_name} parser - version incompatibility: {e}"
                    )
                else:
                    logger.warning(f"Could not initialize {lang_name} parser: {e}")
            except AttributeError as e:
                # Skip languages with API changes
                logger.debug(f"Skipping {lang_name} parser - API incompatibility: {e}")
            except Exception as e:
                logger.warning(f"Could not initialize {lang_name} parser: {e}")

    def detect_language(self, file_path: str) -> Optional[str]:
        """Detect language from file extension.

        Args:
            file_path: Path to the source file.

        Returns:
            Language name or None if not detected.
        """
        ext = Path(file_path).suffix.lower()
        return self._extension_map.get(ext)

    def parse_file(self, file_path: str, content: str) -> List[CodeChunk]:
        """Parse a file and extract code chunks.

        Args:
            file_path: Path to the source file.
            content: Content of the file.

        Returns:
            List of extracted code chunks.
        """
        language = self.detect_language(file_path)

        if not language:
            logger.debug(f"Unknown language for {file_path}, using simple parsing")
            return self._simple_parse(content, None)

        # If we have a parser for this language, use it
        if language in self.parsers:
            try:
                return self._parse_with_tree_sitter(content, language)
            except Exception as e:
                logger.warning(f"Tree-sitter parsing failed for {file_path}: {e}")

        # Fallback to simple parsing
        return self._simple_parse(content, language)

    def _parse_with_tree_sitter(
        self, content: str, language: str, max_chunk_size: int = 1500
    ) -> List[CodeChunk]:
        """Parse using Tree-sitter with chunk size management.

        Args:
            content: Source code content.
            language: Programming language name.
            max_chunk_size: Maximum size of a chunk in characters.

        Returns:
            List of extracted code chunks.
        """
        parser = self.parsers[language]
        config = self.LANGUAGE_CONFIGS[language]
        tree = parser.parse(bytes(content, "utf8"))

        chunks = []
        lines = content.split("\n")

        def extract_chunks(
            node: tree_sitter.Node, depth: int = 0, parent_class: Optional[str] = None
        ) -> None:
            """Recursively extract chunks from the syntax tree."""
            current_class = parent_class

            # Check if this node is a class
            if node.type in [
                "class_definition",
                "class_declaration",
                "class_specifier",
            ]:
                # Extract class name
                for child in node.children:
                    if child.type in ["identifier", "property_identifier"]:
                        current_class = lines[child.start_point[0]][
                            child.start_point[1] : child.end_point[1]
                        ]
                        break

            if node.type in config.node_types:
                start_line = node.start_point[0]
                end_line = node.end_point[0]
                chunk_content = "\n".join(lines[start_line : end_line + 1])

                # Check if chunk is too large
                if len(chunk_content) > max_chunk_size:
                    # If node has semantic children, recursively chunk them
                    semantic_children = [
                        child
                        for child in node.children
                        if child.type in config.node_types
                    ]

                    if semantic_children:
                        # Process children instead
                        for child in semantic_children:
                            extract_chunks(child, depth + 1, current_class)
                        return
                    else:
                        # No semantic children, split by lines while preserving context
                        # This handles cases like very long functions without subfunctions
                        chunk_lines = lines[start_line : end_line + 1]
                        current_chunk = []
                        current_size = 0
                        chunk_start = start_line + 1

                        # Track the original chunk boundaries
                        original_start = start_line + 1
                        original_end = end_line + 1

                        for i, line in enumerate(chunk_lines):
                            line_size = len(line) + 1  # +1 for newline
                            if (
                                current_size + line_size > max_chunk_size
                                and current_chunk
                            ):
                                # Save current chunk
                                chunks.append(
                                    CodeChunk(
                                        content="\n".join(current_chunk),
                                        chunk_type=config.node_types[node.type]
                                        + "_part",
                                        start_line=chunk_start,
                                        end_line=chunk_start + len(current_chunk) - 1,
                                        name=name if "name" in locals() else None,
                                        metadata={
                                            "language": language,
                                            "depth": depth,
                                            "part": True,
                                            "original_start_line": original_start,
                                            "original_end_line": original_end,
                                            "original_type": config.node_types[
                                                node.type
                                            ],
                                        },
                                    )
                                )
                                current_chunk = [line]
                                current_size = line_size
                                chunk_start = start_line + i + 1
                            else:
                                current_chunk.append(line)
                                current_size += line_size

                        # Don't forget the last chunk
                        if current_chunk:
                            chunks.append(
                                CodeChunk(
                                    content="\n".join(current_chunk),
                                    chunk_type=config.node_types[node.type] + "_part",
                                    start_line=chunk_start,
                                    end_line=start_line + len(chunk_lines),
                                    name=name if "name" in locals() else None,
                                    metadata={
                                        "language": language,
                                        "depth": depth,
                                        "part": True,
                                        "original_start_line": original_start,
                                        "original_end_line": original_end,
                                        "original_type": config.node_types[node.type],
                                    },
                                )
                            )
                        return

                # Try to extract name
                name = None
                for child in node.children:
                    if child.type in ["identifier", "property_identifier"]:
                        name = lines[child.start_point[0]][
                            child.start_point[1] : child.end_point[1]
                        ]
                        break

                metadata = {
                    "language": language,
                    "depth": depth,
                    "node_type": node.type,  # Original TreeSitter node type
                }

                # Add parent class for methods
                chunk_type = config.node_types[node.type]
                if chunk_type == "method" and current_class:
                    metadata["parent_class"] = current_class
                elif chunk_type == "class" and name:
                    metadata["class_name"] = name

                # Add function/method name to metadata
                if name and chunk_type in ["function", "method"]:
                    metadata["function_name"] = name

                # Add any docstring if we can find it
                docstring = self._extract_docstring(node, lines, language)
                if docstring:
                    metadata["docstring"] = docstring

                chunks.append(
                    CodeChunk(
                        content=chunk_content,
                        chunk_type=chunk_type,
                        start_line=start_line + 1,
                        end_line=end_line + 1,
                        name=name,
                        metadata=metadata,
                    )
                )

            # Recurse into children with updated parent class
            for child in node.children:
                extract_chunks(child, depth + 1, current_class)

        extract_chunks(tree.root_node)
        return chunks

    def _extract_docstring(
        self, node: tree_sitter.Node, lines: List[str], language: str
    ) -> Optional[str]:
        """Extract docstring from a function or class node if available.

        Args:
            node: The TreeSitter node.
            lines: Source code lines.
            language: Programming language.

        Returns:
            The docstring if found, None otherwise.
        """
        # Language-specific docstring extraction
        if language == "python":
            # Look for the first string literal in the body
            for child in node.children:
                if child.type == "block":
                    for stmt in child.children:
                        if stmt.type == "expression_statement":
                            for expr_child in stmt.children:
                                if expr_child.type == "string":
                                    start_line = expr_child.start_point[0]
                                    end_line = expr_child.end_point[0]
                                    docstring_lines = lines[start_line : end_line + 1]
                                    return "\n".join(docstring_lines).strip()
                    break
        elif language in ["javascript", "typescript", "java", "cpp", "c"]:
            # Look for JSDoc or similar comment blocks before the node
            if node.start_point[0] > 0:
                # Check the line before for comment
                prev_line = lines[node.start_point[0] - 1].strip()
                if prev_line.startswith("/**") or prev_line.startswith("///"):
                    # Found a doc comment, extract it
                    doc_lines = []
                    line_idx = node.start_point[0] - 1
                    while line_idx >= 0:
                        line = lines[line_idx].strip()
                        if (
                            line.startswith("/**")
                            or line.startswith("/*")
                            or line.startswith("*")
                            or line.startswith("///")
                        ):
                            doc_lines.insert(0, line)
                            if line.startswith("/**"):
                                break
                        else:
                            break
                        line_idx -= 1
                    if doc_lines:
                        return "\n".join(doc_lines)

        return None

    def _simple_parse(
        self, content: str, language: Optional[str], max_chunk_size: int = 1500
    ) -> List[CodeChunk]:
        """Simple parsing fallback for when Tree-sitter is not available.

        Args:
            content: Source code content.
            language: Programming language name.
            max_chunk_size: Maximum size of a chunk in characters.

        Returns:
            List of extracted code chunks.
        """
        lines = content.split("\n")

        # Language-specific simple parsing
        if language == "python":
            return self._simple_parse_python(lines, language, max_chunk_size)
        elif language in ["javascript", "typescript"]:
            return self._simple_parse_javascript(lines, language, max_chunk_size)
        else:
            # Generic chunking by empty lines
            return self._chunk_by_paragraphs(lines, language, max_chunk_size)

    def _simple_parse_python(
        self, lines: List[str], language: str, max_chunk_size: int = 1500
    ) -> List[CodeChunk]:
        """Simple Python parsing based on indentation and keywords.

        Args:
            lines: Lines of code.
            language: Programming language name.

        Returns:
            List of extracted code chunks.
        """
        chunks = []
        current_chunk = []
        current_type = None
        current_name = None
        current_class = None
        start_line = 0

        for i, line in enumerate(lines):
            # Check for function or class definition
            if line.strip().startswith(("def ", "class ", "async def ")):
                # Save previous chunk if exists
                if current_chunk:
                    metadata = {"language": language}
                    # Add parent class for methods
                    if (
                        current_type == "function"
                        and current_class
                        and line.startswith((" ", "\t"))
                    ):
                        metadata["parent_class"] = current_class
                        chunk_type = "method"
                    else:
                        chunk_type = current_type or "code"

                    chunks.append(
                        CodeChunk(
                            content="\n".join(current_chunk),
                            chunk_type=chunk_type,
                            start_line=start_line + 1,
                            end_line=i,
                            name=current_name,
                            metadata=metadata,
                        )
                    )

                # Extract name
                parts = line.strip().split()
                if len(parts) > 1:
                    name_part = parts[1]
                    current_name = name_part.split("(")[0]
                else:
                    current_name = None

                # Start new chunk
                current_chunk = [line]
                if line.strip().startswith("class "):
                    current_type = "class"
                    current_class = current_name  # Track current class
                else:
                    current_type = "function"
                    # If this is indented, it's likely a method
                    if not line.startswith((" ", "\t")):
                        current_class = (
                            None  # Reset class context for top-level functions
                        )

                start_line = i
            elif current_chunk:
                # Continue current chunk if indented
                if line.strip() == "" or line.startswith((" ", "\t")):
                    current_chunk.append(line)
                else:
                    # End current chunk
                    metadata = {"language": language}
                    # Add parent class for methods
                    if current_type == "function" and current_class:
                        metadata["parent_class"] = current_class
                        chunk_type = "method"
                    else:
                        chunk_type = current_type or "code"

                    chunks.append(
                        CodeChunk(
                            content="\n".join(current_chunk),
                            chunk_type=chunk_type,
                            start_line=start_line + 1,
                            end_line=i,
                            name=current_name,
                            metadata=metadata,
                        )
                    )
                    current_chunk = []
                    current_type = None
                    current_name = None
                    # Reset class context when we're back at top level
                    if not line.startswith((" ", "\t")):
                        current_class = None

        # Don't forget the last chunk
        if current_chunk:
            metadata = {"language": language}
            # Add parent class for methods
            if current_type == "function" and current_class:
                metadata["parent_class"] = current_class
                chunk_type = "method"
            else:
                chunk_type = current_type or "code"

            chunks.append(
                CodeChunk(
                    content="\n".join(current_chunk),
                    chunk_type=chunk_type,
                    start_line=start_line + 1,
                    end_line=len(lines),
                    name=current_name,
                    metadata=metadata,
                )
            )

        return chunks

    def _simple_parse_javascript(
        self, lines: List[str], language: str, max_chunk_size: int = 1500
    ) -> List[CodeChunk]:
        """Simple JavaScript/TypeScript parsing based on braces.

        Args:
            lines: Lines of code.
            language: Programming language name.

        Returns:
            List of extracted code chunks.
        """
        chunks = []
        current_chunk = []
        current_type = "code"
        current_name = None
        start_line = 0
        brace_count = 0

        for i, line in enumerate(lines):
            # Check for function or class
            if any(keyword in line for keyword in ["function ", "class ", "=>"]):
                if brace_count == 0 and current_chunk:
                    # Save previous chunk
                    chunks.append(
                        CodeChunk(
                            content="\n".join(current_chunk),
                            chunk_type=current_type,
                            start_line=start_line + 1,
                            end_line=i,
                            name=current_name,
                            metadata={"language": language},
                        )
                    )
                    current_chunk = []
                    start_line = i

                # Try to extract name
                if "function " in line:
                    parts = line.split("function ")
                    if len(parts) > 1:
                        name_part = parts[1].split("(")[0].strip()
                        current_name = name_part if name_part else None
                    current_type = "function"
                elif "class " in line:
                    parts = line.split("class ")
                    if len(parts) > 1:
                        name_part = parts[1].split()[0].strip("{")
                        current_name = name_part if name_part else None
                    current_type = "class"
                elif "=>" in line:
                    current_type = "function"
                    # Try to get arrow function name
                    if "const " in line or "let " in line or "var " in line:
                        parts = line.split("=")[0].strip().split()
                        if parts:
                            current_name = parts[-1]

            current_chunk.append(line)

            # Count braces
            brace_count += line.count("{") - line.count("}")

            # If braces are balanced and we have content, save chunk
            if brace_count == 0 and current_chunk and i > start_line + 2:
                chunks.append(
                    CodeChunk(
                        content="\n".join(current_chunk),
                        chunk_type=current_type,
                        start_line=start_line + 1,
                        end_line=i + 1,
                        name=current_name,
                        metadata={"language": language},
                    )
                )
                current_chunk = []
                current_type = "code"
                current_name = None
                start_line = i + 1

        # Last chunk
        if current_chunk:
            chunks.append(
                CodeChunk(
                    content="\n".join(current_chunk),
                    chunk_type=current_type,
                    start_line=start_line + 1,
                    end_line=len(lines),
                    name=current_name,
                    metadata={"language": language},
                )
            )

        return chunks

    def _chunk_by_paragraphs(
        self, lines: List[str], language: Optional[str], max_chunk_size: int = 1500
    ) -> List[CodeChunk]:
        """Generic chunking by paragraphs (empty lines).

        Args:
            lines: Lines of code.
            language: Programming language name.

        Returns:
            List of extracted code chunks.
        """
        chunks = []
        current_chunk = []
        start_line = 0

        for i, line in enumerate(lines):
            if line.strip():
                current_chunk.append(line)
            else:
                if current_chunk:
                    chunks.append(
                        CodeChunk(
                            content="\n".join(current_chunk),
                            chunk_type="code_block",
                            start_line=start_line + 1,
                            end_line=i,
                            metadata={"language": language or "unknown"},
                        )
                    )
                    current_chunk = []
                    start_line = i + 1

        # Last chunk
        if current_chunk:
            chunks.append(
                CodeChunk(
                    content="\n".join(current_chunk),
                    chunk_type="code_block",
                    start_line=start_line + 1,
                    end_line=len(lines),
                    metadata={"language": language or "unknown"},
                )
            )

        # If no chunks were created, make one big chunk
        if not chunks and lines:
            chunks.append(
                CodeChunk(
                    content="\n".join(lines),
                    chunk_type="file",
                    start_line=1,
                    end_line=len(lines),
                    metadata={"language": language or "unknown"},
                )
            )

        return chunks
