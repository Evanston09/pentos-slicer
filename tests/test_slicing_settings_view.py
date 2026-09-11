from dataclasses import replace
from types import SimpleNamespace

from models import SlicingSettings
from models.slicing_settings import filament_preset
from views.slicing_settings_view import SlicingSettingsView


class Handle:
    def __init__(self, value=None, **kwargs):
        self._value = value
        self.callback = lambda _: None
        self.disabled = kwargs.get("disabled", False)
        self.removed = False

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if value != self._value:
            self._value = value
            self.callback(SimpleNamespace(target=self))

    def on_update(self, callback):
        self.callback = callback
        return callback

    on_click = on_update

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def remove(self):
        self.removed = True


class Gui:
    def __init__(self):
        self.handles = {}

    def add_number(self, label, value=None, **kwargs):
        handle = Handle(kwargs.get("initial_value", value), **kwargs)
        self.handles[label] = handle
        return handle

    add_folder = add_number
    add_dropdown = add_number
    add_text = add_number
    add_button = add_number
    add_markdown = add_number


def test_presets_edits_reset_and_lifecycle() -> None:
    gui = Gui()
    changed = []

    def apply(settings):
        changed.append(settings)
        view.update(settings)

    view = SlicingSettingsView(SimpleNamespace(gui=gui), SlicingSettings(), apply)
    assert changed == []
    assert len(view.inputs) == 8
    assert len(view.folders) == 2
    assert "Speeds" not in gui.handles
    gui.handles["Material"].value = "PETG"
    assert changed[-1].filament == filament_preset("PETG")
    gui.handles["Nozzle (°C)"].value = 245
    assert changed[-1].filament.temperature == 245
    assert gui.handles["Material"].value == "PETG"
    gui.handles["Infill (%)"].value = 30.0
    gui.handles["Reset to Filament Preset"].callback(None)
    assert view.settings.filament == filament_preset("PETG")
    assert view.settings.print.fill_density == 30
    gui.handles["Reset Print Settings"].callback(None)
    assert view.settings.print == SlicingSettings().print
    previous = view.settings
    gui.handles["Nozzle (°C)"].value = 0
    assert view.settings == previous
    assert gui.handles["Nozzle (°C)"].value == 240
    view.set_enabled(False)
    assert all(handle.disabled for handle in view.inputs.values())
    gui.handles["Nozzle (°C)"].value = 250
    assert view.settings == previous
    view.set_enabled(True)
    restored = replace(
        previous, filament=replace(filament_preset("PLA"), temperature=215)
    )
    view.update(restored)
    assert view.settings == restored
    assert gui.handles["Material"].value == "PLA"
    view.remove()
    assert all(folder.removed for folder in view.folders)
