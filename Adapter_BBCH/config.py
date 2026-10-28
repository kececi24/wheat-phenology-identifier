"""Load a flat, typed YAML configuration using the training parser's schema."""

from __future__ import annotations

import math


def load_config_defaults(parser, path):
    try:
        import yaml
    except ImportError:
        parser.error("--config requires PyYAML; install Adapter_BBCH/requirements.txt")

    class UniqueKeyLoader(yaml.SafeLoader):
        def construct_mapping(self, node, deep=False):
            mapping = {}
            for key_node, value_node in node.value:
                key = self.construct_object(key_node, deep=deep)
                if not isinstance(key, str):
                    raise ValueError("configuration keys must be strings")
                if key in mapping:
                    raise ValueError(f"duplicate configuration key: {key}")
                mapping[key] = self.construct_object(value_node, deep=deep)
            return mapping

    try:
        with open(path, encoding="utf-8") as handle:
            values = yaml.load(handle, Loader=UniqueKeyLoader)
    except (OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
        parser.error(f"cannot load config {path}: {exc}")
    if not isinstance(values, dict):
        parser.error(f"config {path} must contain a flat mapping of training options")

    actions = {action.dest: action for action in parser._actions
               if action.dest not in {"help", "config"}}
    defaults = {}
    for key, value in values.items():
        name = key.replace("-", "_")
        if name not in actions:
            parser.error(f"config {path}: unknown training option {key!r}")
        if name in defaults:
            parser.error(f"config {path}: duplicate training option {name!r}")
        action = actions[name]
        expected = action.type or (bool if isinstance(action.default, bool) else str)

        def convert(item):
            if expected is bool:
                valid = type(item) is bool
            elif expected is int:
                valid = type(item) is int
            elif expected is float:
                # SafeLoader can read 3e-4 as a string; accept numeric strings.
                valid = type(item) in (int, float, str)
            else:
                valid = isinstance(item, str)
            if not valid:
                parser.error(f"config {path}: {key} must contain {expected.__name__} values")
            try:
                converted = expected(item)
            except (TypeError, ValueError, OverflowError):
                parser.error(f"config {path}: invalid value for {key}: {item!r}")
            if expected is float and not math.isfinite(converted):
                parser.error(f"config {path}: {key} must be finite")
            if action.choices is not None and converted not in action.choices:
                parser.error(f"config {path}: {key} must be one of {list(action.choices)}")
            return converted

        if value is None:
            if action.default is not None or action.nargs is not None:
                parser.error(f"config {path}: {key} cannot be null")
        elif action.nargs == "+":
            if not isinstance(value, list) or not value:
                parser.error(f"config {path}: {key} must be a nonempty YAML list")
            value = [convert(item) for item in value]
        else:
            value = convert(value)
        defaults[name] = value
    return defaults
