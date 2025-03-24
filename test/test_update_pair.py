from src.utils import update_pair


def test_update_pair():
    expected_result = [1,4,3]
    current_result = update_pair([1,2,2,3], (2,2), 4)
    assert current_result == expected_result
    
def test_update_non_existing_pair():
    expected_result = [1,2,3]
    current_result = update_pair([1,2,3], (2,4), 5)
    assert current_result == expected_result
    