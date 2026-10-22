from backend.prompt import build_chat_prompt, extract_reply


def test_prompt_is_a_script_ending_with_the_bot_label():
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Good morrow"},
        {"role": "user", "content": "How are you?"},
    ]
    assert build_chat_prompt(messages) == (
        "YOU:\nHello\n\nBOT:\nGood morrow\n\nYOU:\nHow are you?\n\nBOT:\n"
    )


def test_blank_lines_inside_a_message_cannot_fake_the_end_of_a_speech():
    prompt = build_chat_prompt([{"role": "user", "content": "one\n\n\n\ntwo"}])
    assert prompt == "YOU:\none\ntwo\n\nBOT:\n"


def test_long_history_drops_the_oldest_text_but_keeps_the_ending():
    messages = [{"role": "user", "content": "x" * 300}] * 20
    prompt = build_chat_prompt(messages, max_chars=500)
    assert len(prompt) == 500 and prompt.endswith("BOT:\n")


def test_reply_stops_at_a_blank_line():
    assert extract_reply("Good sir, well met.\n\nYOU:\nmore") == "Good sir, well met."


def test_reply_stops_at_the_next_speaker_label():
    assert extract_reply("First line\nSecond line\nROMEO:\nwords") == "First line\nSecond line"


def test_reply_keeps_normal_lines_that_contain_colons():
    assert extract_reply("He said: go\nand left") == "He said: go\nand left"


def test_empty_and_whitespace_replies():
    assert extract_reply("") == ""
    assert extract_reply("\n\nBOT:\nlater") == ""
    assert extract_reply("   \n  ") == ""
