from setter_eyes.classify.outcome import point_outcome


def test_outcome_from_the_last_touch_and_the_winner():
    assert point_outcome("attack", "us", "us")[0] == "kill"
    assert point_outcome("attack", "us", "them")[0] == "attack_error"
    assert point_outcome("block", "them", "them")[0] == "block_point"
    assert point_outcome("block", "them", "us")[0] == "kill"       # our attack beat the block
    assert point_outcome("serve", "them", "them")[0] == "ace"
    assert point_outcome("serve", "us", "them")[0] == "service_error"
    assert point_outcome("receive", "us", "them")[0] == "reception_error"
    assert point_outcome("dig", "us", "them")[0] == "ball_handling_error"
    assert point_outcome(None, None, "us") == ("other", 0.2)
