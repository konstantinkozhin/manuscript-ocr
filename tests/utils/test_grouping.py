import pytest
from manuscript.data import Page, TextRegion
from manuscript.utils import group_regions


def word(x):
    return TextRegion(type='word',polygon=[(x,0),(x+5,0),(x+5,5),(x,5)],text=str(x))


@pytest.mark.parametrize('geometry',['bbox','convex_hull','concave_hull',None])
def test_geometry_and_preservation(geometry):
    p=Page(image_width=100,image_height=100)
    a,b,c=word(0),word(10),word(20);p.children=(a,b,c)
    group=group_regions([c,a],type='line',geometry=geometry)
    assert p.children==(group,b) and group.children==(c,a)
    assert a.text=='0' and a.parent is group
    assert group.text is None and group.detection_confidence is None
    assert (group.polygon is None)==(geometry is None)


def test_errors_do_not_mutate():
    p=Page(image_width=100,image_height=100);a,b=word(0),word(10);p.children=(a,b)
    before=p.to_tree_dict()
    for nodes,target,mode in [([a,a],'line','bbox'),([a],'word','bbox'),([a,b],'line','invalid')]:
        with pytest.raises(ValueError): group_regions(nodes,type=target,geometry=mode)
        assert p.to_tree_dict()==before
    def fail(polygons): raise RuntimeError('fail')
    with pytest.raises(RuntimeError): group_regions([a,b],type='line',geometry=fail)
    assert p.to_tree_dict()==before
    line=TextRegion(type='line')
    with pytest.raises(ValueError): group_regions([line],type='line',geometry=None)
    with pytest.raises(ValueError): group_regions([TextRegion(type='block')],type='line',geometry=None)
    with pytest.raises(ValueError): group_regions([a,word(30)],type='line')


def test_detached_regions_group_without_fabricating_geometry():
    first, second = TextRegion(type="word", text="first"), TextRegion(type="word", text="second")
    group = group_regions([first, second], type="line", geometry=None)
    assert group.parent is None and group.polygon is None
    assert group.children == (first, second)
    assert first.parent is group and second.parent is group


def test_custom_geometry_receives_copies_and_cannot_mutate_originals():
    page = Page(image_width=100, image_height=100)
    first, second = word(0), word(10)
    page.children = [first, second]
    original = list(first.polygon)
    def geometry(polygons):
        polygons[0][0] = (99, 99)
        return [(0,0),(15,0),(15,5),(0,5)]
    group = group_regions([first, second], type="line", geometry=geometry)
    assert first.polygon == original
    assert group.polygon == [(0,0),(15,0),(15,5),(0,5)]


@pytest.mark.parametrize("concavity", [-.1, 1.1, float("nan")])
def test_invalid_concavity_keeps_source_tree_unchanged(concavity):
    page = Page(image_width=100, image_height=100)
    page.children = [word(0), word(10)]
    before = page.to_tree_dict()
    with pytest.raises(ValueError, match="concavity"):
        group_regions(page.children, type="line", geometry="concave_hull", concavity=concavity)
    assert page.to_tree_dict() == before


def test_geometry_failure_with_unpositioned_regions_is_atomic():
    page = Page(image_width=100, image_height=100)
    page.children = [TextRegion(type="word"), TextRegion(type="word")]
    before = page.to_tree_dict()
    with pytest.raises(ValueError, match="positioned"):
        group_regions(page.children, type="line")
    assert page.to_tree_dict() == before
