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


def test_pixel_settings_scale_with_width():
    section = {"gate_px": 100, "jerk_px_s3": 2000, "metres_per_px_x": 0.004, "count": 10,
               "on": True, "init_sigma_px": {"pos": 4.0, "vel": 30.0},
               "arcs": {"break_px": 6.0, "min_samples": 6}, "range_px": [10, 20]}
    s = config.scale_px(section, 1920)
    assert s["gate_px"] == 50 and s["jerk_px_s3"] == 1000
    assert s["metres_per_px_x"] == 0.008                         # inverse
    assert s["count"] == 10 and s["on"] is True
    assert s["init_sigma_px"] == {"pos": 2.0, "vel": 15.0}
    assert s["arcs"] == {"break_px": 3.0, "min_samples": 6}
    assert s["range_px"] == [5, 10]
    assert config.scale_px(section, 3840) == section


def test_sample_stride_keeps_30_samples_per_second():
    assert [config.sample_stride(f) for f in (59.94, 60, 30, 29.97, 25, 120)] == \
        [2, 2, 1, 1, 1, 4]
