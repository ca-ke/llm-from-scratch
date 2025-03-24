from src.utils import get_freq_pair


def test_most_freq_pair():
    ids = [1, 2, 2, 3, 1, 1]
    current_result = get_freq_pair(ids, mode="most")
    expected_result = (1, 2)
    assert current_result == expected_result

def test_least_freq_pair():
    ids = [1, 2, 2, 3, 1, 1]
    current_result = get_freq_pair(ids, mode="least")
    expected_result = (1, 2)
    assert current_result == expected_result