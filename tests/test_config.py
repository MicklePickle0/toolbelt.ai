from setter_eyes import config


def test_court_dimensions():
    court = config.load("court")
    assert court["length"] == 18.0
    assert court["width"] == 9.0
    assert court["net"]["height"] == {"men": 2.43, "women": 2.24}


def test_pass_scale_grades_are_ordered_and_complete():
    scale = config.load("pass_scale")
    scores = [g["score"] for g in scale["grades"]]
    assert scores == [4, 3, 2, 1]
    # Last grade is the catch-all.
    assert scale["grades"][-1]["max_depth_error"] is None
    # Depth limits must widen as the grade drops.
    depth = [g["max_depth_error"] for g in scale["grades"][:-1]]
    assert depth == sorted(depth)


def test_pipeline_config_has_paths():
    pipeline = config.load("pipeline")
    assert pipeline["paths"]["db_path"].startswith("data/")
