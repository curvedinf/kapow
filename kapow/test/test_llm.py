from kapow.llm import call_llm


def test_llm():
    output = call_llm("Hello, how are you?")
    print(output)

    assert len(output) > 0
    assert isinstance(output, str)