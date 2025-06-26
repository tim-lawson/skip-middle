import os

from huggingface_hub import hf_hub_download


def download(filename: str, cache_dir: str) -> None:
    if not os.path.exists(os.path.join(cache_dir, filename)):
        hf_hub_download(
            repo_id="kjj0/fineweb10B-gpt2",
            filename=filename,
            repo_type="dataset",
            local_dir=cache_dir,
        )


if __name__ == "__main__":
    cache_dir = "data/fineweb_10B_gpt2"
    download(f"fineweb_val_{0:06d}.bin", cache_dir)
    for i in range(1, 104):  # 10B tokens
        download(f"fineweb_train_{i:06d}.bin", cache_dir)
