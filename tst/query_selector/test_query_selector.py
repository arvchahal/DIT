import os
import tempfile
import pandas as pd
from query_selector.query_selector import QuerySelector

def create_temp_csv(contents):
    """Helper to create a temporary CSV and return its path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", newline="")
    df = pd.DataFrame(contents)
    df.to_csv(tmp.name, index=False)
    tmp.close()
    return tmp.name

def test_query_selector_single_cycle():
    files = []
    for i in range(3):
        data = [
            {"type": "t1", "question": f"Q{i}_1", "oracle_response": "A"},
            {"type": "t1", "question": f"Q{i}_2", "oracle_response": "B"}
        ]
        files.append(create_temp_csv(data))

    selector = QuerySelector(files, cycles=1, seed=42)
    results = list(selector)

    total_rows = sum(len(pd.read_csv(f)) for f in files)
    assert len(results) == total_rows

    all_questions = [r["question"] for r in results]
    sequential = [f"Q{i}_{j}" for i in range(3) for j in [1, 2]]
    assert all_questions != sequential

def test_query_selector_multiple_cycles():
    data = [{"type": "x", "question": "Q1", "oracle_response": "A"}]
    file = create_temp_csv(data)

    selector = QuerySelector([file], cycles=3, seed=1)
    results = list(selector)

    assert len(results) == 3
    assert all(r["question"] == "Q1" for r in results)

def test_query_selector_reset_changes_order():
    data = [{"type": "x", "question": f"Q{i}", "oracle_response": "A"} for i in range(5)]
    file = create_temp_csv(data)

    selector = QuerySelector([file], cycles=1)
    first_order = [r["question"] for r in selector]
    selector.reset()
    second_order = [r["question"] for r in selector]

    assert first_order != second_order

def teardown_module(module):
    """Clean up temporary CSVs after tests."""
    for f in os.listdir(tempfile.gettempdir()):
        if f.endswith(".csv"):
            try:
                os.remove(os.path.join(tempfile.gettempdir(), f))
            except:
                pass
