"""Unit tests and regression for MetaData classes
"""

import pytest
from run_brer.metadata import MetaData, MultiMetaData


def test_metadata():
    metadata = MetaData(name="test")
    metadata.set_requirements(["param1", "param2"])

    metadata.set(param1="string", param2=0.)
    assert not metadata.get_missing_keys()

    with pytest.warns(Warning):
        metadata.set_from_dictionary(data={"bad_parameter": "bad"})

    with pytest.warns(Warning):
        metadata.set("bad_parameter", "bad")

    assert metadata.get_as_dictionary() == {"param1": "string", "param2": 0., "bad_parameter": "bad"}


def test_multi_metadata(tmpdir):
    multi = MultiMetaData()

    metadata = MetaData(name="test1")
    metadata.set_requirements(["param1", "param2"])

    with pytest.raises(IndexError):
        multi.names
    with pytest.raises(IndexError):
        multi.name_to_id("random name")

    multi.add_metadata(metadata)
    assert multi.names
    assert multi.name_to_id("test1") == 0
    assert multi.id_to_name(0) == "test1"

    multi.write_to_json("{}/state.json".format(tmpdir))
    old_multi = multi
    multi = MultiMetaData()
    multi.read_from_json("{}/state.json".format(tmpdir))

    assert old_multi.get_as_single_dataset() == multi.get_as_single_dataset()
