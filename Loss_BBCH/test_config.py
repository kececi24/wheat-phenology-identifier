"""Remote configuration checks; no data/model downloads required."""
from pathlib import Path

import pytest

from Loss_BBCH.config import parse_args


@pytest.mark.parametrize("name,budget", [("config.yaml", 100), ("config50.yaml", 50), ("config25.yaml", 25)])
def test_supplied_configs_and_override(name, budget):
    path = Path(__file__).parent / "config" / name
    args = parse_args(["--config", str(path), "--seed", "44"])
    assert args.label_percentage == budget
    assert args.seed == 44
    assert args.fold_seed == args.label_seed == 42
    assert args.methods == ["hard_ce", "soft_ce", "ordinal", "transition"]


@pytest.mark.parametrize("body", ["path_weight: 1\npath_weight: 2\n", "unknown: 1\n", "epochs: true\n"])
def test_bad_yaml_rejected(tmp_path, body):
    path = tmp_path / "config.yaml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(SystemExit):
        parse_args(["--config", str(path)])


@pytest.mark.parametrize("options", [["--path-weight", "nan"], ["--label-percentage", "0"], ["--methods", "all", "soft_ce"]])
def test_bad_cli_rejected(options):
    with pytest.raises(SystemExit):
        parse_args(options)
