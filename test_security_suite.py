"""The whole Security Verification Lab must pass."""
from core.dataset import generate_dataset, train_selector
from core.security_tests import run_all_security_tests


def test_all_security_tests_pass():
    bundle = train_selector(generate_dataset(1200, 5), 5)
    table, rate = run_all_security_tests(bundle)
    failed = table[table["Result"] != "PASS"]
    assert failed.empty, failed.to_string()
    assert rate == 100.0
    assert len(table) == 3 * 10 + 1
