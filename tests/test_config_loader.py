from prismatic_adapter.config_loader import flatten_script_config


def test_flatten_script_config_maps_sections_to_argparse_names():
    flat = flatten_script_config(
        {
            "dataset": {"libero_hdf5_root": "data/libero"},
            "libero": {"image_keys": "image_primary,image_wrist"},
            "rlds": {"tfds_name": "bridge", "split": "train"},
            "vision": {"cache_dir": "pretrained_models/vision_cache/hf"},
            "trainable": {"language_model": True, "action_head": False},
            "lora": {"use_lora": True, "rank": 32},
        }
    )

    assert flat["libero_hdf5_root"] == "data/libero"
    assert flat["libero_image_keys"] == "image_primary,image_wrist"
    assert flat["rlds_tfds_name"] == "bridge"
    assert flat["rlds_split"] == "train"
    assert flat["vision_cache_dir"] == "pretrained_models/vision_cache/hf"
    assert flat["train_language_model"] is True
    assert flat["train_action_head"] is False
    assert flat["use_lora"] is True
    assert flat["lora_rank"] == 32
