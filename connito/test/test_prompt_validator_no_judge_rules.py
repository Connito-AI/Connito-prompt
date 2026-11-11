"""Without judge_rules.json, as in the public release: the config imports and loads, the judge is refused, and the
rules hash with the judge off is the one production's cycles carry.

The judge's defaults then come from a file that is absent, so they are None: that relies on pydantic not validating
defaults. A fresh process each time, so the config module is imported with the file hidden from it.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

HIDE_RULES = """
import pathlib
_read_text = pathlib.Path.read_text
def read_text(self, *args, **kwargs):
    if self.name == "judge_rules.json":
        raise FileNotFoundError(self)
    return _read_text(self, *args, **kwargs)
pathlib.Path.read_text = read_text
"""

CHECK = HIDE_RULES + """
import json
from pathlib import Path
from connito.prompt_validator import config, snapshot
from connito.test.test_prompt_validator_snapshot import _base

out = {"rules": bool(config.JUDGE_RULES_DATA), "judge_rules": snapshot.JUDGE_RULES}
out["miner_defaults"] = config.MinerPromptsCfg().model_dump(mode="json")
pv = config.PromptValidatorCfg(prompt_source="dataless")
out["off"] = snapshot.rules_sha256(pv, _base())
out["off_all"] = snapshot.rules_sha256(config.PromptValidatorCfg(prompt_source="dataless", model_selection="all"),
                                       _base())
try:
    config.PromptValidatorCfg(prompt_source="dataless", judge_miner_prompts=True)
    out["judge_on"] = "accepted"
except ValueError as e:
    out["judge_on"] = str(e)
print(json.dumps(out))
"""


def test_without_the_judges_rules_the_config_loads_and_the_judge_is_refused():
    env = {**os.environ, "PYTHONPATH": str(REPO), "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""}
    done = subprocess.run([sys.executable, "-c", CHECK], cwd=REPO, env=env, capture_output=True, text=True,
                          timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    out = json.loads(done.stdout.strip().splitlines()[-1])

    assert out["rules"] is False and out["judge_rules"] is None
    # Every default is empty: no rule value or text of the judge is in the public code.
    assert all(v is None or v == [] or (isinstance(v, dict) and set(v.values()) == {None})
               for v in out["miner_defaults"].values()), out["miner_defaults"]
    # The judge-off pins (as in test_prompt_validator_loss_rank.py): judge off, the
    # hash is the one with the rules present.
    assert out["off"] == "18df5bb8e0ca3d85179c4b4db7c11486d67c4f8df1e721a675067377a8cb1799"
    assert out["off_all"] == "10f482f8487214468eed252e31593d31f438aff493c793d0bfb8dba20cdf769a"
    assert "judge_miner_prompts needs the judge's rules, judge_rules.json" in out["judge_on"]
