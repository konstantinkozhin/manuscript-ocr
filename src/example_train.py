from manuscript.detectors.east import train

if __name__ == "__main__":
    best_model = train(
        r"C:\school_notebooks_RU\east_dataset_split\val\images",
        r"C:\school_notebooks_RU\east_dataset_split\val\annotations.json",
        r"C:\school_notebooks_RU\east_dataset_split\val\images",
        r"C:\school_notebooks_RU\east_dataset_split\val\annotations.json",
        epochs=50,
        batch_size=2,
    )
