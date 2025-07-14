from manuscript.detectors.east import train

if __name__ == "__main__":
    train_images = [
        r"C:\Users\USER\Desktop\data02065\Archives020525\train_images",
        r"C:\Users\USER\Desktop\data02065\DDI_100\train_images",
        r"C:\Users\USER\Desktop\data02065\IAM\train_images",
        r"C:\Users\USER\Desktop\data02065\ICDAR2015\train_images",
        r"C:\Users\USER\Desktop\data02065\school_notebooks_RU\train_images",
        r"C:\Users\USER\Desktop\data02065\TotalText\train_images",
    ]
    train_anns = [
        r"C:\Users\USER\Desktop\data02065\Archives020525\train.json",
        r"C:\Users\USER\Desktop\data02065\DDI_100\train.json",
        r"C:\Users\USER\Desktop\data02065\IAM\train.json",
        r"C:\Users\USER\Desktop\data02065\ICDAR2015\train.json",
        r"C:\Users\USER\Desktop\data02065\school_notebooks_RU\train.json",
        r"C:\Users\USER\Desktop\data02065\TotalText\train.json",
    ]
    val_images = [
        r"C:\Users\USER\Desktop\data02065\Archives020525\test_images",
        r"C:\Users\USER\Desktop\data02065\DDI_100\test_images",
        r"C:\Users\USER\Desktop\data02065\IAM\test_images",
        r"C:\Users\USER\Desktop\data02065\ICDAR2015\test_images",
        r"C:\Users\USER\Desktop\data02065\school_notebooks_RU\test_images",
        r"C:\Users\USER\Desktop\data02065\TotalText\test_images",
    ]
    val_anns = [
        r"C:\Users\USER\Desktop\data02065\Archives020525\test.json",
        r"C:\Users\USER\Desktop\data02065\DDI_100\test.json",
        r"C:\Users\USER\Desktop\data02065\IAM\test.json",
        r"C:\Users\USER\Desktop\data02065\ICDAR2015\test.json",
        r"C:\Users\USER\Desktop\data02065\school_notebooks_RU\test.json",
        r"C:\Users\USER\Desktop\data02065\TotalText\test.json",
    ]

    best_model = train(
        train_images=train_images,
        train_anns=train_anns,
        val_images=val_images,
        val_anns=val_anns,
        target_size=1024,
        epochs=1000,
        batch_size=4,
        use_sam=False,
        freeze_first=False,
    )
