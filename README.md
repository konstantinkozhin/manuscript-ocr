## Installation

### Для пользователей
```bash
pip install manuscript-ocr
```

### Для разработчиков
```bash
pip install -r requirements-dev.txt
```

> **Примечание**: `requirements-dev.txt` включает GPU версию PyTorch, инструменты тестирования, форматирования и сборки.

### GPU поддержка
Если вы пользователь и хотите GPU поддержку:
```bash
pip install manuscript-ocr
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --force-reinstall
```

**Проверка GPU:**
```python
import torch
print(f"CUDA доступна: {torch.cuda.is_available()}")
```

## Usage Example

```python
from PIL import Image
from manuscript.detectors import EASTInfer

# Инициализация
det = EASTInfer(score_thresh=0.9)

# Инфер с визуализацией
page, vis_image = det.infer(r"example\ocr_example_image.jpg", vis=True)

print(page)

# Покажет картинку с наложенными боксами
Image.fromarray(vis_image).show()

# Или сохранить результат на диск:
Image.fromarray(vis_image).save(r"example\ocr_example_image_infer.png")
```

### Результат

Текстовые блоки будут выведены в консоль, например:

```
Page(blocks=[Block(words=[Word(polygon=[(874.1005, 909.1005), (966.8995, 909.1005), (966.8995, 956.8995), (874.1005, 956.8995)]),
                          Word(polygon=[(849.1234, 810.5678), … ])])])
```

А визуализация сохранится в файл `example/ocr_example_image_infer.png`:

![OCR Inference Result](example/ocr_example_image_infer.png)

