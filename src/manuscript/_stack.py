"""Sequential composition of image and document-tree processors."""
import inspect
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable, Optional

from .api._page_helpers import accepts_predict_kwarg, filter_predict_kwargs
from .data import Page


class Stack:
    """Combine processors in one Pipeline stage or run them directly.

    Parameters
    ----------
    stages : iterable of processors
        Nonempty ordered collection implementing predict(image, ...) or
        predict(page, image=None, ...). Nested Stacks are supported.
    parallel : bool, optional
        Run image detectors concurrently in threads if True; default is False.
        Results are merged in list order. Only detector-only Stacks support
        this mode; page processors require the preceding result.

    Notes
    -----
    Each detector receives the original image; its root regions are appended
    to the common Page. Duplicates are retained. A layout processor can then
    build containment and reading order.

    Layouts, recognizers and correctors receive the preceding Page. Later
    processors can overwrite earlier text or corrections. The supplied Page
    and detector Pages are copied before accumulation.

    A page processor's first predict parameter must be named page or annotated
    as Page. Other processors receive the image. Supported predict options are
    forwarded; individual batch sizes are preserved unless overridden.

    Examples
    --------
    >>> detector = Stack([block_detector, word_detector])
    >>> page = detector.predict(image)
    >>> recognizer = Stack([recognizer_a, recognizer_b])
    >>> page = recognizer.predict(page, image=image)

    See Also
    --------
    manuscript.recognizers.MajorityVotingRecognizer : Vote by whole-string agreement.
    manuscript.recognizers.MedoidRecognizer : Select by summed edit distance.
    """

    # Pipeline normally supplies a recognizer's batch_size. None lets each
    # member use its own default instead of imposing Pipeline's fallback of 32.
    batch_size = None

    def __init__(self, stages: Iterable[Any], *, parallel: bool = False):
        self.stages = tuple(stages)
        if not self.stages:
            raise ValueError("Stack requires at least one processor")
        for stage in self.stages:
            if not callable(getattr(stage, "predict", None)):
                raise TypeError("Stack processors must implement predict")
        if not isinstance(parallel, bool):
            raise TypeError("parallel must be a boolean")
        self.parallel = parallel
        if parallel:
            self._detector_stages()

    def _detector_stages(self):
        stages = []
        for stage in self.stages:
            if isinstance(stage, Stack):
                stages.extend(stage._detector_stages())
            elif self._uses_page(stage):
                raise ValueError("Stack parallel=True supports only image detectors; "
                                 "page processors depend on the preceding result")
            else:
                stages.append(stage)
        return stages

    @staticmethod
    def _uses_page(stage):
        try:
            parameters = tuple(inspect.signature(stage.predict).parameters.values())
        except (TypeError, ValueError) as exc:
            raise TypeError("Stack requires an inspectable predict signature") from exc
        inputs = [p for p in parameters if p.kind not in (
            inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)]
        if not inputs:
            raise TypeError("Stack processors must declare an image or page input")
        first = inputs[0]
        annotation = first.annotation
        return (first.name == "page" or annotation is Page
                or annotation in ("Page", "'Page'", "manuscript.data.Page"))

    @staticmethod
    def _call(stage, value, options):
        options = filter_predict_kwargs(stage, options)
        first = next(p for p in inspect.signature(stage.predict).parameters.values()
                     if p.kind not in (inspect.Parameter.VAR_POSITIONAL,
                                       inspect.Parameter.VAR_KEYWORD))
        # The primary input is supplied separately, never also as a keyword.
        options.pop(first.name, None)
        if first.kind == inspect.Parameter.KEYWORD_ONLY:
            return stage.predict(**{first.name: value, **options})
        return stage.predict(value, **options)

    @staticmethod
    def _detection_context(page, results):
        return {
            "page": page,
            "detections": [item for result in results
                           for item in result.get("detections", [])],
            "image_size": {"height": page.image_height, "width": page.image_width},
            "metadata": {"processor": "Stack"},
            # Keep model-specific classes, masks and outputs in their own result.
            "detector_results": list(results),
        }

    def predict(
        self, page: Optional[Page] = None, image: Any = None, *,
        return_raw: bool = False, return_masks: bool = False,
        return_outputs: bool = False, detector_result=None, **kwargs: Any,
    ):
        """Accept an image, a Page with optional image, or named inputs.

        Return Page by default. Raw/mask/output flags return a dictionary with
        the final Page and detector results grouped by model. Page processors
        always run in Page mode. No prediction results are cached on the Stack.
        """
        if page is not None and not isinstance(page, Page):
            if image is not None:
                raise TypeError("Pass an image positionally or by keyword, not both")
            image, page = page, None
        current = page.clone() if page is not None else None
        raw_results = []
        context = detector_result
        want_raw = return_raw or return_masks or return_outputs
        image_array = None
        stages = self._detector_stages() if self.parallel else self.stages
        parallel_results = None
        if self.parallel:
            if image is None:
                raise ValueError("Parallel detector Stack requires an image")
            def detect(stage):
                options = dict(kwargs)
                if options.get("batch_size") is None:
                    options.pop("batch_size", None)
                    member_batch_size = getattr(stage, "batch_size", None)
                    if member_batch_size is not None:
                        options["batch_size"] = member_batch_size
                options.update(return_raw=True, return_masks=return_masks,
                               return_outputs=return_outputs)
                return self._call(stage, image, options)
            with ThreadPoolExecutor(max_workers=len(stages)) as executor:
                parallel_results = list(executor.map(detect, stages))
        for index, stage in enumerate(stages):
            options = dict(kwargs)
            if options.get("batch_size") is None:
                options.pop("batch_size", None)
                member_batch_size = getattr(stage, "batch_size", None)
                if member_batch_size is not None:
                    options["batch_size"] = member_batch_size
            if context is not None:
                options["detector_result"] = context
            if isinstance(stage, Stack):
                result = stage.predict(
                    page=current, image=image, return_raw=True,
                    return_masks=return_masks, return_outputs=return_outputs,
                    **options,
                )
                current = result["page"]
                raw_results.extend(result["detector_results"])
            elif self._uses_page(stage):
                if current is None:
                    raise ValueError(f"{type(stage).__name__} requires a Page; "
                                     "place a detector first or supply page")
                if image is not None and accepts_predict_kwarg(stage, "image"):
                    if image_array is None:
                        from .utils import read_image
                        image_array = read_image(image)
                    options["image"] = image_array
                result = self._call(stage, current, {**options, "return_raw": False})
                if not isinstance(result, Page):
                    raise TypeError(f"{type(stage).__name__}.predict must return Page")
                current = result
            else:
                if image is None:
                    raise ValueError(f"{type(stage).__name__} requires an image")
                result = parallel_results[index] if parallel_results is not None else self._call(stage, image, {
                    **options, "return_raw": True, "return_masks": return_masks,
                    "return_outputs": return_outputs,
                })
                detected = result.get("page") if isinstance(result, dict) else result
                if not isinstance(detected, Page):
                    raise TypeError(f"{type(stage).__name__}.predict must return Page "
                                    "or a dictionary containing Page")
                detected.validate_tree()
                if current is None:
                    current = detected.clone()
                else:
                    if ((current.image_width, current.image_height)
                            != (detected.image_width, detected.image_height)):
                        raise ValueError("Stack detector Pages must have the same image size")
                    additions = detected.clone()
                    current.children = (*current.children, *additions.children)
                if isinstance(result, dict):
                    raw_results.append(result)
            current.validate_tree()
            if raw_results:
                context = self._detection_context(current, raw_results)
        if want_raw:
            return self._detection_context(current, raw_results)
        return current


__all__ = ["Stack"]
