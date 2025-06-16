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

    def _parse_with_tree_sitter(self, content: str, language: str) -> List[CodeChunk]:
        """Parse using Tree-sitter.

        Args:
            content: Source code content.
            language: Programming language name.

        Returns:
            List of extracted code chunks.
        """
        parser = self.parsers[language]
        config = self.LANGUAGE_CONFIGS[language]
        tree = parser.parse(bytes(content, "utf8"))

        chunks = []
        lines = content.split("\n")

        def extract_chunks(node: tree_sitter.Node, depth: int = 0) -> None:
            """Recursively extract chunks from the syntax tree."""
            if node.type in config.node_types:
                start_line = node.start_point[0]
                end_line = node.end_point[0]
                chunk_content = "\n".join(lines[start_line : end_line + 1])

                # Try to extract name
                name = None
                for child in node.children:
                    if child.type in ["identifier", "property_identifier"]:
                        name = lines[child.start_point[0]][
                            child.start_point[1] : child.end_point[1]
                        ]
                        break

                chunks.append(
                    CodeChunk(
                        content=chunk_content,
                        chunk_type=config.node_types[node.type],
                        start_line=start_line + 1,
                        end_line=end_line + 1,
                        name=name,
                        metadata={"language": language, "depth": depth},
                    )
                )

            # Recurse into children
            for child in node.children:
                extract_chunks(child, depth + 1)

        extract_chunks(tree.root_node)
        return chunks

    def _simple_parse(self, content: str, language: Optional[str]) -> List[CodeChunk]:
        """Simple parsing fallback for when Tree-sitter is not available.

        Args:
            content: Source code content.
            language: Programming language name.

        Returns:
            List of extracted code chunks.
        """
        lines = content.split("\n")

        # Language-specific simple parsing
        if language == "python":
            return self._simple_parse_python(lines, language)
        elif language in ["javascript", "typescript"]:
            return self._simple_parse_javascript(lines, language)
        else:
            # Generic chunking by empty lines
            return self._chunk_by_paragraphs(lines, language)

    def _simple_parse_python(self, lines: List[str], language: str) -> List[CodeChunk]:
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
        start_line = 0

        for i, line in enumerate(lines):
            # Check for function or class definition
            if line.strip().startswith(("def ", "class ", "async def ")):
                # Save previous chunk if exists
                if current_chunk:
                    chunks.append(
                        CodeChunk(
                            content="\n".join(current_chunk),
                            chunk_type=current_type or "code",
                            start_line=start_line + 1,
                            end_line=i,
                            name=current_name,
                            metadata={"language": language},
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
                else:
                    current_type = "function"
                start_line = i
            elif current_chunk:
                # Continue current chunk if indented
                if line.strip() == "" or line.startswith((" ", "\t")):
                    current_chunk.append(line)
                else:
                    # End current chunk
                    chunks.append(
                        CodeChunk(
                            content="\n".join(current_chunk),
                            chunk_type=current_type or "code",
                            start_line=start_line + 1,
                            end_line=i,
                            name=current_name,
                            metadata={"language": language},
                        )
                    )
                    current_chunk = []
                    current_type = None
                    current_name = None

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(
                CodeChunk(
                    content="\n".join(current_chunk),
                    chunk_type=current_type or "code",
                    start_line=start_line + 1,
                    end_line=len(lines),
                    name=current_name,
                    metadata={"language": language},
                )
            )

        return chunks

    def _simple_parse_javascript(
        self, lines: List[str], language: str
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
        self, lines: List[str], language: Optional[str]
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
