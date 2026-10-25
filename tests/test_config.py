from setter_eyes import config


def test_court_dimensions():
    court = config.load("court")
    assert court["length"] == 18.0
    assert court["width"] == 9.0
    assert court["net"]["height_presets"] == {
        "boys_17_18u": 2.43, "boys_15_16u": 2.35, "boys_13_14u": 2.24, "women": 2.24}


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


def test_example_manifest_is_valid():
    from setter_eyes.labels import load_manifest

    m = load_manifest(config.CONFIG_DIR / "manifest.example.yaml")
    assert m.videos[0].lineups[1] == [1, 7, 3, 4, 5, 6]


def test_label_vocab():
    labels = config.load("labels")
    assert {"serve", "receive", "set", "attack", "block", "dig"} <= set(labels["actions"])
    assert {"kill", "attack_error", "reception_error", "ace", "service_error"} <= \
        set(labels["outcomes"])
