"""Example of creating multi-episode datasets."""

if __name__ == "__main__":
    from pprint import pprint

    from mcap_data_loader.datasets.mcap_dataset import (
        McapMultiEpisodeDatasets,
        McapMultiEpisodeDatasetsConfig,
        DataRearrangeConfig,
        RearrangeType,
    )

    root = "data/train_data"
    config = McapMultiEpisodeDatasetsConfig(
        common={
            "strict": False,
            "rearrange": DataRearrangeConfig(dataset=RearrangeType.SORT_STEM_DIGITAL),
        },
        configs={
            0: {
                "data_root": root + "/close_hinge_door",
                "keys": ["arm/pose/position", "arm/pose/orientation"],
            },
            1: {
                "data_root": root + "/close_hinge_door_processed",
                "keys": ["arm/pose/position_rela", "arm/pose/rotation_6d_rela"],
            },
        },
    )
    pprint(config.model_dump())
    dataset = McapMultiEpisodeDatasets(config)
    # pprint(dataset.statistics())
    for episodes in zip(*dataset):
        for samples in zip(*episodes):
            merged = {}
            for sample in samples:
                merged.update(sample)
            print(merged)
            exit()
    #          pprint([sample["arm/pose/position"] for sample in samples])
    # for idx, episodes in enumerate(dataset):
    #     for episode in episodes:
    #         pprint(
    #             f"Dataset {idx} in {episode.config.data_root}: {len(episode)} samples"
    #         )
