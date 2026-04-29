from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set, Tuple
import importlib
import re

from rank_bm25 import BM25Okapi
from tree_sitter import Language, Parser
import tree_sitter_python as tspython

def _build_parser(language: str) -> "Parser":
    if Parser is None or Language is None:
        raise RuntimeError("tree_sitter is required. Install with `pip install tree-sitter`.")
    PY_LANGUAGE = Language(tspython.language())

    lang_obj = _load_language(language)
    return Parser(PY_LANGUAGE)
    
class CodeUtils:
    """Utilities for API extraction and CodeBLEU-based filtering."""

    def __init__(
        self,
        # middle: str,
        language: str = "python",
        stdlib_modules: Optional[Set[str]] = None,
        parser: Optional["Parser"] = None,
    ) -> None:
        # self.middle = middle.strip()
        self.language = language
        self._stdlib_modules = stdlib_modules or _default_stdlib_modules()
        self._parser = parser or _build_parser(language)
        # self.middle_api = self.extract_external_apis(middle)

    def extract_external_apis(self, code: str) -> Set[str]:
        """Parse code with tree-sitter and extract external calls + member accesses."""
        tree = _parse_code(self._parser, code)
        if tree is None:
            return set()

        root = tree.root_node
        source_bytes = code.encode("utf8", errors="ignore") if isinstance(code, str) else code
        alias_map, imported_bases = _collect_imports(root, source_bytes)
        defined_callables = _collect_defined_callables(root, source_bytes)

        apis: Set[str] = set()
        for node in _walk_nodes(root):
            if node.type == "call":
                func_node = node.child_by_field_name("function")
                if func_node is None:
                    continue
                api_name = _resolve_call_name(func_node, alias_map, imported_bases, source_bytes)
                if (
                    api_name
                    and not _is_stdlib_api(api_name, self._stdlib_modules)
                    and not _is_internal_call(api_name, defined_callables)
                ):
                    apis.add(api_name)
            elif node.type == "attribute":
                # Also keep external member access, e.g. obj.member (not only obj.method()).
                attr_name = _resolve_attribute_name(node, alias_map, imported_bases, source_bytes)
                if attr_name and not _is_stdlib_api(attr_name, self._stdlib_modules):
                    apis.add(attr_name)
        return apis

    def extract_apis(self, code: str) -> Set[str]:
        """
        Extract both:
        - external_apis: APIs this code segment calls/accesses from outside
        - provided_apis: APIs this code segment provides for external use
        """
        tree = _parse_code(self._parser, code)
        if tree is None:
            return {"external_apis": set(), "provided_apis": set()}

        root = tree.root_node
        source_bytes = code.encode("utf8", errors="ignore") if isinstance(code, str) else code
        provided_apis = _collect_provided_apis(root, source_bytes)
        external_apis = self.extract_external_apis(code)
        return set(list(provided_apis)+list(external_apis))

    def codebleu_similarity(
        self,
        code_a: str,
        code_b: str,
        lang: Optional[str] = None,
        weights: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
    ) -> float:
        """Compute CodeBLEU similarity between two code strings."""
        try:
            from codebleu import calc_codebleu
        except Exception as exc:
            raise RuntimeError(
                "codebleu is required. Install with `pip install codebleu` "
                "and the appropriate tree-sitter language package."
            ) from exc

        result = calc_codebleu(
            [code_a],
            [code_b],
            lang=lang or self.language,
            weights=weights,
            tokenizer=None,
        )
        if isinstance(result, dict):
            if "codebleu" in result:
                return float(result["codebleu"])
            if "code_bleu" in result:
                return float(result["code_bleu"])
        return float(result)  # pragma: no cover - unexpected result shape


    def filter_by_middle(
        self,
        middle:str,
        codes: List[str],
        lang: Optional[str] = None
    ) -> List[str]:
        """
        Keep BM25 top-k code strings to middle, or those sharing external APIs.
        """
        if not middle or not any(codes):
            return []
        
        tokenized_codes = [_bm25_tokenize(code) for code in codes]
        if tokenized_codes and any(tokenized_codes):
            bm25 = BM25Okapi(tokenized_codes)
        else:
            return []
        bm25_top = bm25.get_top_n(_bm25_tokenize(middle), codes, n=1)
        # bm25_top = []
        # middle_api = self.extract_external_apis(middle)
        # for code in codes:
        #     if not code:
        #         continue
        #     # if code in bm25_top or middle in code:
        #     #     kept.append(code)
        #     #     continue
        #     apis = self.extract_external_apis(code)
        #     if apis & middle_api and code not in bm25_top:
        #         bm25_top.append(code)
        return bm25_top
    # def filter_by_middle(
    #     self,
    #     middle:str,
    #     codes: List[str],
    #     lang: Optional[str] = None,
    #     weights: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
    # ) -> List[str]:
    #     """
    #     Keep code strings whose Jaccard similarity to middle is >= z
    #     or that share at least one external API with middle.
    #     """
    #     if not codes:
    #         return []
    #     middle_clean = _strip_comments(middle)
    #     if not middle_clean.strip():
    #         return []
    #     middle_tokens = set(_bm25_tokenize(middle_clean))
    #     kept: List[str] = []
    #     middle_api = self.extract_external_apis(middle_clean)
    #     for code in codes:
    #         if not code:
    #             continue
    #         code_clean = _strip_comments(code)
    #         if not code_clean.strip():
    #             continue

    #         code_tokens = set(_bm25_tokenize(code_clean))
    #         sim = _jaccard_similarity(middle_tokens, code_tokens)
    #         # if sim >= 0.3:
    #         #     kept.append(code)
    #         #     continue
    #         # class_filtered = self._filter_class_by_middle_api(code, middle_api)
    #         # if class_filtered is not None:
    #         #     kept.append(class_filtered)
    #         #     continue
    #         apis = self.extract_apis(code_clean)
    #         if apis & middle_api:
    #             kept.append(code)
    #     return kept
        # if kept:
        #     return kept
        # else:
        #     return codes

    def _filter_class_by_middle_api(self, code: str, middle_api: Set[str]) -> Optional[str]:
        """
        If `code` is a class snippet, keep only:
        - __init__ (always kept, but not used for overlap comparison)
        - methods whose external APIs overlap with `middle_api`
        Returns rebuilt class code if any non-__init__ method overlaps, else None.
        """
        if not code or not middle_api:
            return None
        tree = _parse_code(self._parser, code)
        if tree is None:
            return None

        root = tree.root_node
        source_bytes = code.encode("utf8", errors="ignore")
        class_nodes: List[Tuple[object, object]] = []
        import_prefix: List[str] = []

        for node in getattr(root, "named_children", []):
            inner = node
            if node.type == "decorated_definition":
                maybe_inner = _decorated_inner_definition(node)
                if maybe_inner is not None:
                    inner = maybe_inner
            if inner.type == "class_definition":
                class_nodes.append((node, inner))
                continue
            if inner.type in {"import_statement", "import_from_statement"}:
                import_prefix.append(_node_text(node, source_bytes).rstrip())
                continue
            if _is_docstring_statement(node, source_bytes):
                import_prefix.append(_node_text(node, source_bytes).rstrip())
                continue
            return None

        if not class_nodes:
            return None

        filtered_classes: List[str] = []
        has_overlap_method = False
        for class_outer, class_node in class_nodes:
            filtered_class, class_has_overlap = self._rebuild_class_with_overlap_methods(
                class_outer,
                class_node,
                source_bytes,
                middle_api,
            )
            if filtered_class:
                filtered_classes.append(filtered_class)
            has_overlap_method = has_overlap_method or class_has_overlap

        if not has_overlap_method or not filtered_classes:
            return None

        prefix = [chunk for chunk in import_prefix if chunk]
        return "\n\n".join(prefix + filtered_classes)

    def _rebuild_class_with_overlap_methods(
        self,
        class_outer,
        class_node,
        source_bytes: bytes,
        middle_api: Set[str],
    ) -> Tuple[Optional[str], bool]:
        body = class_node.child_by_field_name("body")
        if body is None:
            return None, False

        class_header = source_bytes[class_outer.start_byte : body.start_byte].decode(
            "utf8", errors="ignore"
        ).rstrip()
        if not class_header:
            return None, False

        kept_methods: List[str] = []
        has_overlap_method = False
        for child in getattr(body, "named_children", []):
            method_outer = child
            method_inner = child
            if child.type == "decorated_definition":
                maybe_inner = _decorated_inner_definition(child)
                if maybe_inner is not None:
                    method_inner = maybe_inner

            if method_inner.type not in {"function_definition", "async_function_definition"}:
                continue

            method_name = _node_text(method_inner.child_by_field_name("name"), source_bytes)
            method_code = _node_text(method_outer, source_bytes).rstrip()
            if not method_name or not method_code:
                continue

            if method_name == "__init__":
                kept_methods.append(method_code)
                continue

            method_apis = self.extract_external_apis(method_code)
            if method_apis & middle_api:
                kept_methods.append(method_code)
                has_overlap_method = True

        if not has_overlap_method:
            return None, False
        if not kept_methods:
            return None, False

        class_code = class_header + "\n" + "\n\n".join(kept_methods)
        return class_code, True


def _strip_comments(code: str) -> str:
    if not code:
        return ""
    text = str(code)
    # Remove Python triple-quoted blocks and C/Java block comments first.
    text = re.sub(r'"""[\\s\\S]*?"""', "", text)
    text = re.sub(r"'''[\\s\\S]*?'''", "", text)
    text = re.sub(r"/\\*[\\s\\S]*?\\*/", "", text)
    # Remove single-line comments beginning with #.
    lines = []
    for line in text.splitlines():
        lines.append(re.sub(r"#.*$", "", line))
    return "\n".join(lines)


def _is_docstring_statement(node, source_bytes: bytes) -> bool:
    if node is None or node.type != "expression_statement":
        return False
    text = _node_text(node, source_bytes).strip()
    if not text:
        return False
    prefixes = ("r", "u", "b", "f", "R", "U", "B", "F")
    while text and text[0] in prefixes:
        text = text[1:]
    return text.startswith(("'''", '"""', "'", '"'))


def _jaccard_similarity(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _build_parser(language: str) -> "Parser":
    if Parser is None or Language is None:
        raise RuntimeError("tree_sitter is required. Install with `pip install tree-sitter`.")
    PY_LANGUAGE = Language(tspython.language())

    lang_obj = _load_language(language)
    return Parser(PY_LANGUAGE)


def _load_language(language: str) -> "Language":
    # Preferred: tree_sitter_<lang> packages, e.g., tree_sitter_python
    try:
        lang_module = importlib.import_module(f"tree_sitter_{language}")
    except Exception:
        lang_module = None

    if lang_module is not None:
        if hasattr(lang_module, "language"):
            return Language(lang_module.language())
        if hasattr(lang_module, "LANGUAGE"):
            lang_obj = getattr(lang_module, "LANGUAGE")
            if isinstance(lang_obj, Language):
                return lang_obj
            return Language(lang_obj)

    # Fallback: tree_sitter_languages package
    try:
        from tree_sitter_languages import get_language
    except Exception as exc:
        raise RuntimeError(
            "Failed to load tree-sitter language. Install `tree-sitter-<lang>` "
            "(e.g., tree-sitter-python) or `tree-sitter-languages`."
        ) from exc

    return get_language(language)


def _parse_code(parser: "Parser", code: str):
    if code is None:
        return None
    if isinstance(code, str):
        code_bytes = code.encode("utf8", errors="ignore")
    elif isinstance(code, (bytes, bytearray)):
        code_bytes = bytes(code)
    else:
        return None

    try:
        return parser.parse(code_bytes)
    except Exception as e:
        print(e)
        return None


def _walk_nodes(root) -> Iterable:
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        # reverse to keep approximate source order
        children = list(getattr(node, "children", []))
        stack.extend(reversed(children))


def _collect_imports(root, source_bytes: bytes) -> Tuple[Dict[str, str], Set[str]]:
    alias_map: Dict[str, str] = {}
    imported_bases: Set[str] = set()

    def add_import(module_path: str, alias: Optional[str]) -> None:
        module_path = (module_path or "").strip()
        if not module_path:
            return
        base = module_path.split(".")[0]
        if base:
            imported_bases.add(base)
        if alias:
            alias_map[alias] = module_path
        else:
            alias_map.setdefault(base, base)

    for node in _walk_nodes(root):
        if node.type == "import_statement":
            name_node = node.child_by_field_name("name")
            names = _extract_import_names(name_node, source_bytes)
            if not names:
                for child in getattr(node, "named_children", []):
                    names.extend(_extract_import_names(child, source_bytes))
            for name, alias in names:
                add_import(name, alias)
        elif node.type == "import_from_statement":
            module_node = node.child_by_field_name("module_name")
            module_path = _node_text(module_node, source_bytes)
            if module_path:
                imported_bases.add(module_path.split(".")[0])

            name_node = node.child_by_field_name("name")
            names = _extract_import_names(name_node, source_bytes)
            for name, alias in names:
                full = f"{module_path}.{name}" if module_path else name
                add_import(full, alias or name)

    return alias_map, imported_bases


def _extract_import_names(node, source_bytes: bytes) -> List[Tuple[str, Optional[str]]]:
    if node is None:
        return []
    if node.type == "aliased_import":
        name_node = node.child_by_field_name("name")
        alias_node = node.child_by_field_name("alias")
        name = _node_text(name_node, source_bytes)
        alias = _node_text(alias_node, source_bytes) if alias_node else None
        return [(name, alias)] if name else []
    if node.type == "dotted_name":
        name = _node_text(node, source_bytes)
        return [(name, None)] if name else []
    if node.type == "wildcard_import":
        return []

    names: List[Tuple[str, Optional[str]]] = []
    for child in getattr(node, "named_children", []):
        names.extend(_extract_import_names(child, source_bytes))
    return names


def _resolve_call_name(
    func_node,
    alias_map: Dict[str, str],
    imported_bases: Set[str],
    source_bytes: bytes,
) -> Optional[str]:
    full_name = _node_full_name(func_node, source_bytes)
    if not full_name:
        return None

    parts = full_name.split(".")
    base = parts[0]
    # if base in {"self", "cls", "super"}:
    #     return None

    if base in alias_map:
        resolved_base = alias_map[base]
        if resolved_base != base:
            if len(parts) > 1:
                return resolved_base + "." + ".".join(parts[1:])
            return resolved_base
        return full_name

    if base in imported_bases:
        return full_name

    # Fallback: keep attribute calls even without explicit imports (e.g., vae.encode)
    # This is a heuristic to capture external APIs referenced via variables.
    if "." in full_name:
        return full_name

    # Keep unresolved bare calls as potential external APIs; later filter internal ones.
    return full_name


def _resolve_attribute_name(
    attr_node,
    alias_map: Dict[str, str],
    imported_bases: Set[str],
    source_bytes: bytes,
) -> Optional[str]:
    full_name = _node_full_name(attr_node, source_bytes)
    if not full_name or "." not in full_name:
        return None

    parts = full_name.split(".")
    base = parts[0]
    # if base in {"self", "cls", "super"}:
    #     return None

    if base in alias_map:
        resolved_base = alias_map[base]
        if resolved_base != base:
            return resolved_base + "." + ".".join(parts[1:])
        return full_name

    if base in imported_bases:
        return full_name

    # Fallback: keep dotted member access even without explicit import.
    return full_name


def _collect_defined_callables(root, source_bytes: bytes) -> Set[str]:
    """Collect callable names defined in this snippet (functions/classes)."""
    callables: Set[str] = set()
    for node in _walk_nodes(root):
        if node.type in {"function_definition", "async_function_definition", "class_definition"}:
            name_node = node.child_by_field_name("name")
            name = _node_text(name_node, source_bytes)
            if name:
                callables.add(name)
        elif node.type == "decorated_definition":
            defn = _decorated_inner_definition(node)
            if defn and defn.type in {"function_definition", "async_function_definition", "class_definition"}:
                name_node = defn.child_by_field_name("name")
                name = _node_text(name_node, source_bytes)
                if name:
                    callables.add(name)
    return callables


def _is_internal_call(api_name: str, defined_callables: Set[str]) -> bool:
    """True if call is an internal snippet-local callable (exclude from external APIs)."""
    if not api_name:
        return False
    if "." in api_name:
        return False
    return api_name in defined_callables


def _collect_provided_apis(root, source_bytes: bytes) -> Set[str]:
    """
    APIs provided by the code snippet:
    - function snippet: function name
    - class snippet: class name, class methods, class members, instance members (self.x)
    """
    provided: Set[str] = set()

    for node in _walk_nodes(root):
        current = node
        if current.type == "decorated_definition":
            current = _decorated_inner_definition(current)
            if current is None:
                continue

        if current.type in {"function_definition", "async_function_definition"}:
            if _is_inside_class(current):
                continue
            func_name = _node_text(current.child_by_field_name("name"), source_bytes)
            if func_name:
                provided.add(func_name)
        elif current.type == "class_definition":
            class_name = _node_text(current.child_by_field_name("name"), source_bytes)
            if not class_name:
                continue
            provided.add(class_name)
            _collect_class_apis(current, class_name, source_bytes, provided)

    return provided


def _collect_class_apis(class_node, class_name: str, source_bytes: bytes, out: Set[str]) -> None:
    body = class_node.child_by_field_name("body")
    if body is None:
        return
    for child in getattr(body, "named_children", []):
        inner = child
        if inner.type == "decorated_definition":
            inner = _decorated_inner_definition(inner)
            if inner is None:
                continue

        if inner.type in {"function_definition", "async_function_definition"}:
            method_name = _node_text(inner.child_by_field_name("name"), source_bytes)
            if method_name:
                out.add(f"{class_name}.{method_name}")
            for member_name in _collect_self_members_in_function(inner, source_bytes):
                out.add(f"{class_name}.{member_name}")
        elif inner.type in {"assignment", "augmented_assignment", "annotated_assignment"}:
            for class_member in _collect_assigned_identifiers(inner, source_bytes):
                out.add(f"{class_name}.{class_member}")


def _collect_self_members_in_function(func_node, source_bytes: bytes) -> Set[str]:
    members: Set[str] = set()
    for node in _walk_nodes(func_node):
        if node.type not in {"assignment", "augmented_assignment", "annotated_assignment"}:
            continue
        for target in _assignment_targets(node):
            if target is None or target.type != "attribute":
                continue
            obj_node = target.child_by_field_name("object")
            attr_node = target.child_by_field_name("attribute")
            obj_name = _node_full_name(obj_node, source_bytes)
            attr_name = _node_text(attr_node, source_bytes)
            if obj_name == "self" and attr_name:
                members.add(attr_name)
    return members


def _collect_assigned_identifiers(assign_node, source_bytes: bytes) -> Set[str]:
    names: Set[str] = set()
    for target in _assignment_targets(assign_node):
        if target is None:
            continue
        if target.type == "identifier":
            name = _node_text(target, source_bytes)
            if name:
                names.add(name)
    return names


def _assignment_targets(assign_node) -> List:
    targets: List = []
    if assign_node is None:
        return targets

    if assign_node.type == "assignment":
        left = assign_node.child_by_field_name("left")
        if left is not None:
            if left.type in {"pattern_list", "tuple", "list"}:
                targets.extend(getattr(left, "named_children", []))
            else:
                targets.append(left)
        return targets

    if assign_node.type == "annotated_assignment":
        left = assign_node.child_by_field_name("left")
        if left is not None:
            targets.append(left)
        return targets

    if assign_node.type == "augmented_assignment":
        left = assign_node.child_by_field_name("left")
        if left is not None:
            targets.append(left)
        return targets

    return targets


def _decorated_inner_definition(node):
    for child in getattr(node, "named_children", []):
        if child.type in {"function_definition", "async_function_definition", "class_definition"}:
            return child
    return None


def _is_inside_class(node) -> bool:
    cur = getattr(node, "parent", None)
    while cur is not None:
        if cur.type == "class_definition":
            return True
        cur = getattr(cur, "parent", None)
    return False


def _node_full_name(node, source_bytes: bytes) -> Optional[str]:
    if node is None:
        return None
    if node.type == "call":
        func_node = node.child_by_field_name("function")
        return _node_full_name(func_node, source_bytes)
    if node.type == "identifier":
        return _node_text(node, source_bytes)
    if node.type == "subscript":
        value_node = node.child_by_field_name("value")
        return _node_full_name(value_node, source_bytes)
    if node.type == "attribute":
        obj_node = node.child_by_field_name("object")
        attr_node = node.child_by_field_name("attribute")
        obj_name = _node_full_name(obj_node, source_bytes)
        attr_name = _node_text(attr_node, source_bytes)
        if obj_name and attr_name:
            return f"{obj_name}.{attr_name}"
    return None


def _node_text(node, source_bytes: bytes) -> str:
    if node is None:
        return ""
    try:
        text = node.text
        if text:
            if isinstance(text, bytes):
                return text.decode("utf8", errors="ignore")
            return str(text)
    except Exception:
        pass
    try:
        return source_bytes[node.start_byte : node.end_byte].decode("utf8", errors="ignore")
    except Exception:
        return ""


def _default_stdlib_modules() -> Set[str]:
    try:
        import sys

        return set(sys.stdlib_module_names)
    except Exception:
        return {
            "abc",
            "argparse",
            "asyncio",
            "base64",
            "collections",
            "contextlib",
            "csv",
            "dataclasses",
            "datetime",
            "enum",
            "functools",
            "hashlib",
            "heapq",
            "inspect",
            "io",
            "itertools",
            "json",
            "logging",
            "math",
            "os",
            "pathlib",
            "random",
            "re",
            "shlex",
            "statistics",
            "string",
            "subprocess",
            "sys",
            "threading",
            "time",
            "typing",
            "types",
            "unittest",
            "uuid",
        }


def _is_stdlib_api(api_name: str, stdlib_modules: Set[str]) -> bool:
    if not api_name:
        return False
    base = api_name.split(".")[0]
    return base in stdlib_modules


_BM25_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _bm25_tokenize(text: str) -> List[str]:
    return _BM25_TOKEN_RE.findall((text or "").lower())



