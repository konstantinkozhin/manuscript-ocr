"""
Пример использования OCRPipeline для массовой обработки изображений из папки.
Включает детекцию и распознавание с детальной статистикой производительности.
"""

import os
import time
import glob
from src.manuscript.detectors._east import EASTInfer
from src.manuscript.recognizers.trba import TRBAInfer
from src.manuscript.pipeline import OCRPipeline


def main():
    # Пути к моделям
    recognizer_model_path = (
        "C:\\Users\\USER\\Desktop\\OCR_MODELS\\exp1_model_64\\best_acc_weights.pth"
    )
    config_path = "C:\\Users\\USER\\Desktop\\OCR_MODELS\\exp1_model_64\\config.json"

    # Папка с изображениями для обработки
    images_folder = "C:\\shared\\data0205\\Archives020525\\test_images"

    # Поддерживаемые форматы изображений
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif"]

    # Собираем все изображения из папки
    image_files = set()  # Используем set для избежания дублей
    for ext in image_extensions:
        image_files.update(glob.glob(os.path.join(images_folder, ext)))
        image_files.update(glob.glob(os.path.join(images_folder, ext.upper())))

    image_files = sorted(list(image_files))  # Преобразуем в отсортированный список

    if not image_files:
        print(f"Изображения не найдены в папке: {images_folder}")
        print("Поддерживаемые форматы: jpg, jpeg, png, bmp, tiff, tif")
        return

    print(f"Найдено {len(image_files)} изображений для обработки")
    print(f"Папка: {images_folder}")

    print("\nИнициализация OCR Pipeline...")

    # Инициализация детектора
    print("Загрузка детектора EAST...")
    start_time = time.time()
    detector = EASTInfer()
    detector_time = time.time() - start_time
    print(f"Детектор загружен за {detector_time:.2f} сек")

    # Инициализация распознавателя
    print("Загрузка распознавателя TRBA...")
    start_time = time.time()
    recognizer = TRBAInfer(
        model_path=recognizer_model_path,
        config_path=config_path,
    )
    recognizer_time = time.time() - start_time
    print(f"Распознаватель загружен за {recognizer_time:.2f} сек")

    # Создание пайплайна
    pipeline = OCRPipeline(detector=detector, recognizer=recognizer, min_text_size=10)

    print("\n" + "=" * 80)
    print("НАЧАЛО МАССОВОЙ ОБРАБОТКИ")
    print("=" * 80)
    print("💡 Нажмите Ctrl+C для прерывания обработки")
    print("📊 Промежуточная статистика каждые 10 изображений")

    # Статистика для всех изображений
    total_images = 0
    total_words_found = 0
    total_words_recognized = 0
    total_processing_time = 0
    failed_images = []

    # Обрабатываем каждое изображение
    for i, image_path in enumerate(image_files, 1):
        print(f"\n📷 [{i}/{len(image_files)}] {os.path.basename(image_path)}")
        print(f"    🔄 Загрузка изображения...")

        try:
            # Замер времени обработки одного изображения
            start_time = time.time()
            print(f"    🔍 Детекция текста...")

            # Сначала только детекция, чтобы понять где зависает
            detection_start = time.time()
            detection_result = pipeline.detector.infer(image_path, vis=False)
            detection_time = time.time() - detection_start
            print(f"    ✅ Детекция завершена за {detection_time:.2f} сек")

            # Подсчитываем найденные области
            total_detected = sum(len(block.words) for block in detection_result.blocks)
            print(f"    📊 Найдено {total_detected} текстовых областей")

            if total_detected > 0:
                print(f"    🔤 Распознавание текста...")
                recognition_start = time.time()
                result_page = pipeline.process(image_path, recognize_text=True)
                recognition_time = time.time() - recognition_start
                print(f"    ✅ Распознавание завершено за {recognition_time:.2f} сек")
            else:
                print(f"    ⏭️  Пропускаем распознавание (нет областей)")
                result_page = detection_result

            processing_time = time.time() - start_time

            total_processing_time += processing_time
            total_images += 1

            # Подсчитываем слова
            words_in_image = 0
            recognized_in_image = 0

            for block in result_page.blocks:
                for word in block.words:
                    words_in_image += 1
                    if word.text:
                        recognized_in_image += 1

            total_words_found += words_in_image
            total_words_recognized += recognized_in_image

            # Результаты для этого изображения
            print(f"   ⏱️  Время: {processing_time:.2f} сек")
            print(
                f"   📊 Найдено: {words_in_image} областей, распознано: {recognized_in_image} слов"
            )

            # Промежуточная статистика каждые 10 изображений
            if i % 10 == 0:
                avg_time = (
                    total_processing_time / total_images if total_images > 0 else 0
                )
                remaining = len(image_files) - i
                eta = remaining * avg_time
                print(f"\n📈 ПРОМЕЖУТОЧНАЯ СТАТИСТИКА [{i}/{len(image_files)}]:")
                print(f"   ⏱️ Среднее время на изображение: {avg_time:.2f} сек")
                print(f"   🕐 Примерное время до завершения: {eta/60:.1f} мин")
                print(f"   📊 Общий прогресс: {(i/len(image_files)*100):.1f}%")

            if recognized_in_image > 0:
                words_per_sec = recognized_in_image / processing_time
                time_per_word = processing_time / recognized_in_image * 1000
                print(
                    f"   🏃 Скорость: {words_per_sec:.1f} слов/сек, {time_per_word:.1f} мс/слово"
                )

                # Показываем распознанный текст (первые 5 слов)
                recognized_words = []
                for block in result_page.blocks:
                    for word in block.words:
                        if word.text:
                            recognized_words.append(word.text)

                preview = " ".join(recognized_words[:5])
                if len(recognized_words) > 5:
                    preview += "..."
                print(f"   📝 Текст: {preview}")
            else:
                print(f"   ❌ Текст не распознан")

        except KeyboardInterrupt:
            print(
                f"\n⚠️ Прерывание пользователем! Обработано {i-1} из {len(image_files)} изображений"
            )
            break
        except Exception as e:
            print(f"   ❌ ОШИБКА: {e}")
            print(f"   📁 Файл: {image_path}")
            failed_images.append((image_path, str(e)))

            # Если слишком много ошибок подряд, возможно проблема с моделью
            if len(failed_images) >= 5 and len(failed_images) > total_images:
                print(
                    f"⚠️ Слишком много ошибок ({len(failed_images)}). Возможна проблема с моделью!"
                )
                break

    # Итоговая статистика
    print("\n" + "=" * 80)
    print("ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 80)

    if total_images > 0:
        avg_time_per_image = total_processing_time / total_images
        print(f"📊 Обработано изображений: {total_images}")
        print(f"⏱️  Общее время обработки: {total_processing_time:.2f} сек")
        print(f"📷 Среднее время на изображение: {avg_time_per_image:.2f} сек")
        print(f"📝 Всего найдено областей: {total_words_found}")
        print(f"✅ Успешно распознано слов: {total_words_recognized}")

        if total_words_recognized > 0:
            overall_words_per_sec = total_words_recognized / total_processing_time
            overall_time_per_word = (
                total_processing_time / total_words_recognized * 1000
            )
            recognition_rate = (
                (total_words_recognized / total_words_found) * 100
                if total_words_found > 0
                else 0
            )

            print(
                f"🏃 Общая скорость распознавания: {overall_words_per_sec:.1f} слов/сек"
            )
            print(f"⚡ Среднее время на слово: {overall_time_per_word:.1f} мс")
            print(f"🎯 Процент распознавания: {recognition_rate:.1f}%")

            # Пропускная способность
            images_per_min = 60 / avg_time_per_image
            words_per_min = overall_words_per_sec * 60
            print(f"📈 Пропускная способность:")
            print(f"   - {images_per_min:.1f} изображений в минуту")
            print(f"   - {words_per_min:.0f} слов в минуту")

    if failed_images:
        print(f"\n❌ Ошибки обработки ({len(failed_images)}):")
        for img_path, error in failed_images:
            print(f"   {os.path.basename(img_path)}: {error}")

    print("\n✅ Массовая обработка завершена!")


if __name__ == "__main__":
    main()
