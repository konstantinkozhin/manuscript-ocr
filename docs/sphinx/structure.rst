Library Structure
=================

.. mermaid::

    graph LR
        manuscript[manuscript] --> Pipeline
        manuscript --> Stack
        manuscript --> data
        manuscript --> detectors
        manuscript --> layouts
        manuscript --> recognizers
        manuscript --> correctors
        manuscript --> models
        manuscript --> utils
        manuscript --> api

        data --> elements["Page, Element, TextRegion<br/>Table, Cell, Image"]
        detectors --> detection["YOLO, EAST, PPOCR<br/>RFDETR, Mask2Former"]
        layouts --> ordering["SimpleSorting<br/>ReadingSorting"]
        recognizers --> recognition["TRBA, PPOCRRec, TrOCR"]
        recognizers --> ensembles["MajorityVotingRecognizer<br/>MedoidRecognizer"]
        correctors --> CharLM
        models --> registry["Registry and artifact cache"]
        api --> contracts["Base classes and protocols"]

Modules
-------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Module or class
     - Responsibility
   * - :doc:`Pipeline <api/pipeline>`
     - Runs configured stages and exposes Page snapshots and execution timings.
   * - :doc:`Stack <api/pipeline>`
     - Composes processors, accumulates detector regions and optionally runs
       independent detectors in parallel.
   * - :doc:`data <api/data>`
     - Defines elements, parent-child links, validation, COCO serialization and text output.
   * - :doc:`detectors <api/detectors>`
     - Produces configured regions and optional detailed detection results.
   * - :doc:`layouts <api/layouts>`
     - Groups, reparents and orders existing regions.
   * - :doc:`recognizers <api/recognizers>`
     - Recognizes selected text regions and combines independent predictions.
   * - :doc:`correctors <api/correctors>`
     - Updates recognized text, including CharLM character substitutions.
   * - ``models``
     - Resolves remote catalogs and manages local model artifacts.
   * - :doc:`utils <api/utils>`
     - Provides image input, geometry, grouping, transformations and visualization.
   * - ``api``
     - Defines stage protocols, base classes and shared artifact-loading behavior.

Public entry points
-------------------

Import the components you need from their packages:

.. code-block:: python

    from manuscript import Pipeline, Stack
    from manuscript.data import Page, TextRegion
    from manuscript.detectors import YOLO
    from manuscript.layouts import SimpleSorting, ReadingSorting
    from manuscript.recognizers import TRBA, MajorityVotingRecognizer
    from manuscript.correctors import CharLM
    from manuscript import models
