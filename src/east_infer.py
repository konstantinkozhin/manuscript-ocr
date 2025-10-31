from manuscript.detectors import EASTInfer

# Model initialization
model = EASTInfer()

# Path to the image
img_path = r"example\ocr_example_image.jpg"

# Inference with visualization
result = model.predict(img_path, vis=True)
page = result["page"]
img = result["vis_image"]

# Show the result
img.show()
