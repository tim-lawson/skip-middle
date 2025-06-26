from lab.train.tokenize import Config, DatasetConfig, tokenize

if __name__ == "__main__":
    for split in ["train", "validation"]:
        tokenize(
            config=Config(
                cache_dir="data/tinystories_gpt2",
                dataset=DatasetConfig(path="roneneldan/TinyStories", split=split),
            )
        )
