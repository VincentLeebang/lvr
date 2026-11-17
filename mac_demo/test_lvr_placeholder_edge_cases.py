
from src.dataset.data_utils import replace_lvr_tokens
from src.constants import LVR_PLACEHOLDER

def run_test(name, test_function):
    print(f"\n{name}")
    try:
        test_function()
    except Exception as e:
        print(f"Expected/Detected error: {type(e).__name__}: {e}")


# TEST 1: One placeholder and one bbox
def test1():
    text = f"{LVR_PLACEHOLDER}\nWhat is in the image?"
    groups = [[1, 2, 3, 4, 5]]

    result = replace_lvr_tokens(text, groups, False, None)

    print(result)
    print("LVR tokens:", result.count("<|lvr|>"))
    assert result.count("<|lvr|>") == 5


# TEST 2: One placeholder and three bbox groups
def test2():
    text = f"{LVR_PLACEHOLDER} What is in the image?"
    groups = [[1, 2], [3, 4, 5], [6, 7, 8, 9]]

    replace_lvr_tokens(text, groups, False, None)


# TEST 3: Three placeholders and one bbox group
def test3():
    text = (
        f"{LVR_PLACEHOLDER} First "
        f"{LVR_PLACEHOLDER} Second "
        f"{LVR_PLACEHOLDER} Third"
    )

    groups = [[1, 2, 3, 4, 5]]

    replace_lvr_tokens(text, groups, False, None)


# TEST 4: Preserve text before placeholder
def test4():
    text = f"Important context: {LVR_PLACEHOLDER} Answer"
    groups = [[1, 2, 3]]

    result = replace_lvr_tokens(text, groups, False, None)

    print(result)
    assert "Important context:" in result
    print("Prefix preserved: True")


# TEST 5: Three placeholders and three bbox groups
def test5():
    text = (
        f"{LVR_PLACEHOLDER} First "
        f"{LVR_PLACEHOLDER} Second "
        f"{LVR_PLACEHOLDER} Third"
    )

    groups = [[1, 2], [3, 4, 5], [6, 7, 8, 9]]

    result = replace_lvr_tokens(text, groups, False, None)

    print(result)
    print("Expected LVR tokens: 9")
    print("Actual LVR tokens:", result.count("<|lvr|>"))
    print("Remaining placeholders:", result.count(LVR_PLACEHOLDER))

    assert result.count("<|lvr|>") == 9
    assert LVR_PLACEHOLDER not in result


run_test("TEST 1: One placeholder, one bbox", test1)
run_test("TEST 2: One placeholder, three bboxes", test2)
run_test("TEST 3: Three placeholders, one bbox", test3)
run_test("TEST 4: Preserve original prefix", test4)
run_test("TEST 5: Three placeholders, three bboxes", test5)

print("\nEDGE CASE TESTING COMPLETED")
