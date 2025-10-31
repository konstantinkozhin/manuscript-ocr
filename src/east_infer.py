from manuscript.detectors import EAST

# Model initialization
model = EAST()

# Path to the image
img_path = r"example\ocr_example_image.jpg"

# Inference with visualization
result = model.predict(img_path, vis=True)
page = result["page"]
img = result["vis_image"]

# Show the result
if img is not None:
    img.show()
