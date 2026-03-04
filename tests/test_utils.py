from cocotbext.interface.utils import is_match, _get_compiled_pattern


def test_is_match_glob():
    assert is_match("u_axi_*", "u_axi_tdata")
    assert is_match("u_axi_*_tdata", "u_axi_0_tdata")
    assert is_match("u_axi_?", "u_axi_0")
    assert not is_match("u_axi_?", "u_axi_10")
    assert is_match("u_axi_+", "u_axi_0")
    assert is_match("u_axi_+", "u_axi_10")


def test_is_match_regex():
    assert is_match(r"/u_axi_[0-9]_tdata/", "u_axi_5_tdata")
    assert not is_match(r"/u_axi_[0-9]_tdata/", "u_axi_a_tdata")


def test_is_match_exact():
    assert is_match("tdata", "tdata")
    assert not is_match("tdata", "atdata")


def test_pattern_caching():
    p1 = _get_compiled_pattern("u_axi_*")
    p2 = _get_compiled_pattern("u_axi_*")
    assert p1 is p2
