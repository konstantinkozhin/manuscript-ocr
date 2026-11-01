import numpy as np
import pytest

from manuscript.data import Page, TextRegion
from manuscript.recognizers import TRBA
from manuscript.api.base import BaseArtifactModel
from manuscript.recognizers._common.region_preparers import (
    prepare_bbox_regions, prepare_text_regions,
)


def prepare(bounds, options, rotate=lambda crop: crop):
    x1, y1, x2, y2 = bounds
    image = np.arange(60 * 80 * 3, dtype=np.uint16).reshape(60, 80, 3)
    page = Page(image_width=80, image_height=60)
    page.children = [TextRegion(type='line', polygon=[(x1,y1),(x2,y1),(x2,y2),(x1,y2)], label='kept', detection_confidence=.9)]
    before = page.to_tree_dict()
    regions = prepare_bbox_regions(page, image, min_text_size=0,
                                   rotate_region=rotate, options={**options, 'levels': ['line']})
    assert page.to_tree_dict() == before
    return image, regions


@pytest.mark.parametrize('bounds,options,slices', [
    ((10,10,50,30), {'inset_y': .1}, (slice(12,28), slice(10,50))),
    ((10,10,50,30), {'padding_x': .25, 'padding_y': .1}, (slice(8,32), slice(0,60))),
    ((-10,-5,30,15), {'padding_x': .25, 'inset_y': .1}, (slice(0,13), slice(0,40))),
    ((10.25,5.25,30.75,15.75), {'inset_x': .1, 'padding_y': .1}, (slice(4,17), slice(12,29))),
])
def test_relative_margins_preserve_pixels_and_geometry(bounds, options, slices):
    image, regions = prepare(bounds, options)
    assert len(regions) == 1
    np.testing.assert_array_equal(regions[0].image, image[slices])
    assert not np.shares_memory(regions[0].image, image)
    for name, value in options.items():
        assert regions[0].meta[name] == value


def test_rotation_receives_adjusted_crop():
    seen = []
    def rotate(crop, region):
        seen.append((crop.shape, region.type))
        return np.rot90(crop)
    image, regions = prepare((10,10,50,30), {'inset_y': .1}, rotate)
    assert seen == [((16,40,3), 'line')]
    np.testing.assert_array_equal(regions[0].image, np.rot90(image[12:28,10:50]))


def test_clipped_empty_crop_is_skipped():
    _, regions = prepare((90,10,110,30), {'inset_y': .1})
    assert regions == []


@pytest.mark.parametrize('options,slices', [
    ({}, (slice(5,15), slice(10,30))),
    ({'padding_x': 0., 'inset_y': 0.}, (slice(5,15), slice(10,30))),
    ({'pad': 2}, (slice(3,18), slice(8,33))),
])
def test_zero_margins_preserve_legacy_rounding_and_pixel_pad(options, slices):
    image, regions = prepare((10.25,5.25,30.75,15.75), options)
    np.testing.assert_array_equal(regions[0].image, image[slices])


@pytest.mark.parametrize('name', ['padding_x', 'padding_y', 'inset_x', 'inset_y'])
@pytest.mark.parametrize('value', [-.1, float('nan'), float('inf'), True, '0.1'])
def test_invalid_values_fail_before_weight_resolution(monkeypatch, name, value):
    monkeypatch.setattr(BaseArtifactModel, '_resolve_weights', lambda *args: pytest.fail('Weights resolved before validation'))
    with pytest.raises(ValueError, match=name):
        TRBA(region_preparer_options={name: value})


@pytest.mark.parametrize('options', [
    {'inset_x': .5}, {'inset_y': .6},
    {'padding_x': .1, 'inset_x': .1}, {'padding_y': .1, 'inset_y': .1},
    {'pad': 2, 'inset_y': .1}, {'pad': -1, 'padding_x': .1},
])
def test_invalid_combinations_are_rejected(options):
    with pytest.raises(ValueError):
        prepare((10,10,50,30), options)


@pytest.mark.parametrize('preset', ['polygon_mask', 'quad_warp'])
def test_other_presets_reject_relative_options_even_on_empty_pages(preset):
    with pytest.raises(ValueError, match="require region_preparer='bbox'"):
        prepare_text_regions(Page(image_width=20, image_height=20), np.zeros((20,20,3)), preset=preset,
                             min_text_size=0, rotate_region=lambda x:x, options={'inset_y': 0.})
