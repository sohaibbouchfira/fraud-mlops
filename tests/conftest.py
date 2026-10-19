import sys, pathlib
import pytest, yaml
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from make_data import generate


@pytest.fixture(scope="session")
def params():
    return yaml.safe_load(open(pathlib.Path(__file__).resolve().parents[1] / "params.yaml"))


@pytest.fixture(scope="session")
def df():
    return generate(n_samples=4000, seed=0)      # small in-memory data: fast, no DVC remote needed
