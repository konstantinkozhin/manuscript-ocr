Library Structure
=================

.. mermaid::

    graph LR
        manuscript[manuscript]

        manuscript --> Pipeline[Pipeline]
        manuscript --> data[data]
        manuscript --> detectors[detectors]
        manuscript --> layouts[layouts]
        manuscript --> recognizers[recognizers]
        manuscript --> correctors[correctors]
        manuscript --> utils[utils]
        manuscript --> api[api]

        Pipeline --> p1[predict]
        Pipeline --> p2[get_text]
        Pipeline --> p3[last_detection_page]
        Pipeline --> p4[last_layout_page]
        Pipeline --> p5[last_recognition_page]
        Pipeline --> p6[last_correction_page]

        detectors --> EAST[EAST]
        layouts --> SimpleSorting[SimpleSorting]
        recognizers --> TRBA[TRBA]
        correctors --> CharLM[CharLM]

        utils --> organize_page[organize_page wrapper]

Overview
--------

- ``Pipeline`` orchestrates OCR stages.
- ``detectors`` provide detection models.
- ``layouts`` provide ordering/grouping models (for example ``SimpleSorting``).
- ``recognizers`` provide text recognition models.
- ``correctors`` provide text post-processing models.
- ``utils.organize_page`` remains as a compatibility wrapper over layout logic.
