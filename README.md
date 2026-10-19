# fraud-mlops — Chapter 2 lab starter

Tested with Python 3.11 and the versions pinned in requirements.txt (30 Sept 2026).

    python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    pytest -q                        # 9 tests should pass

Follow the lab sheet (Chapter 2, Section 9) from Step 2.

Files: src/make_data.py (synthetic data) · src/features.py (shared feature logic) ·
src/train.py · src/evaluate.py (gates) · src/register.py · src/promote.py ·
dvc.yaml (pipeline) · params.yaml (all parameters and gates) · tests/ · .github/workflows/ci.yml
