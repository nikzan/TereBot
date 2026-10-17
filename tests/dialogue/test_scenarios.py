import pytest
from dialogue_scenarios import ScenarioError, all_scenarios, load, run


@pytest.mark.parametrize("path", all_scenarios(), ids=lambda p: p.stem)
def test_scenario(path):
    try:
        run(load(path))
    except ScenarioError as e:
        error = str(e)
    else:
        return
    pytest.fail(f"сценарий записан не по формату: {error}", pytrace=False)
