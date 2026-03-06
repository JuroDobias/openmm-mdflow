from __future__ import annotations

import types
import sys

from openmm_mdflow.builder import _build_template_generator


class _DummyGen:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.generator = lambda *args, **kwargs: None


def test_template_generator_selection(monkeypatch):
    module = types.SimpleNamespace(
        GAFFTemplateGenerator=_DummyGen,
        SMIRNOFFTemplateGenerator=_DummyGen,
        EspalomaTemplateGenerator=_DummyGen,
    )
    monkeypatch.setitem(sys.modules, "openmmforcefields.generators", module)

    gen1 = _build_template_generator("openff", "openff-2.0.0", [], "ff.json")
    gen2 = _build_template_generator("gaff", "gaff-2.11", [], "ff.json")
    gen3 = _build_template_generator("espaloma", "espaloma-0.3.2", [], "ff.json")

    assert isinstance(gen1, _DummyGen)
    assert isinstance(gen2, _DummyGen)
    assert isinstance(gen3, _DummyGen)
