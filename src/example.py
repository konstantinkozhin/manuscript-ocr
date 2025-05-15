from PIL import Image
from manuscript.detectors import EASTInfer

# инициализация
det = EASTInfer(score_thresh=0.9)

# инфер
page, image = det.infer(r"example\ocr_example_image.jpg", vis=True)
print(page)

pil_img = Image.fromarray(image)

pil_img.show()