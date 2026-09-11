from collections.abc import Callable
from dataclasses import replace

import viser

from models import PrintSettings, SlicingSettings
from models.slicing_settings import filament_preset


class SlicingSettingsView:
    def __init__(
        self,
        client: viser.ClientHandle,
        settings: SlicingSettings,
        on_change: Callable[[SlicingSettings], None],
    ) -> None:
        self.settings = settings
        self.on_change = on_change
        self.inputs = {}
        self.folders = []
        self.syncing = False
        self.enabled = True
        self.buttons = []
        sections = (
            (
                "Filament",
                "filament",
                (
                    {
                        "name": "temperature",
                        "label": "Nozzle (°C)",
                        "value_type": int,
                        "step": 1,
                        "min": 0,
                        "max": None,
                    },
                    {
                        "name": "first_layer_temperature",
                        "label": "First Layer Nozzle (°C)",
                        "value_type": int,
                        "step": 1,
                        "min": 0,
                        "max": None,
                    },
                    {
                        "name": "bed_temperature",
                        "label": "Bed (°C)",
                        "value_type": int,
                        "step": 1,
                        "min": 0,
                        "max": None,
                    },
                    {
                        "name": "first_layer_bed_temperature",
                        "label": "First Layer Bed (°C)",
                        "value_type": int,
                        "step": 1,
                        "min": 0,
                        "max": None,
                    },
                ),
            ),
            (
                "Print Settings",
                "print",
                (
                    {
                        "name": "layer_height",
                        "label": "Layer Height (mm)",
                        "value_type": float,
                        "step": 0.01,
                        "min": 0.06,
                        "max": 0.32,
                    },
                    {
                        "name": "first_layer_height",
                        "label": "First Layer Height (mm)",
                        "value_type": float,
                        "step": 0.01,
                        "min": 0.06,
                        "max": 0.32,
                    },
                    {
                        "name": "perimeters",
                        "label": "Perimeters",
                        "value_type": int,
                        "step": 1,
                        "min": 0,
                        "max": None,
                    },
                    {
                        "name": "fill_density",
                        "label": "Infill (%)",
                        "value_type": float,
                        "step": 1.0,
                        "min": 0,
                        "max": 100,
                    },
                ),
            ),
        )
        for title, group, entries in sections:
            folder = client.gui.add_folder(title, expand_by_default=False)
            self.folders.append(folder)
            with folder:
                if group == "filament":
                    self.material = client.gui.add_dropdown(
                        "Material",
                        ("PLA", "PETG"),
                        initial_value=settings.filament.filament_type,
                    )
                    client.gui.add_markdown(
                        "Presets are starting values; adjust for your filament."
                    )
                group_settings = getattr(settings, group)
                for entry in entries:
                    name = entry["name"]
                    value_type = entry["value_type"]
                    handle = client.gui.add_number(
                        entry["label"],
                        value_type(getattr(group_settings, name)),
                        step=entry["step"],
                        min=entry["min"],
                        max=entry["max"],
                    )
                    self.inputs[group, name] = handle
                    self._bind_input(handle, group, name, value_type)
                if title == "Filament":
                    button = client.gui.add_button("Reset to Filament Preset")
                    button.on_click(
                        lambda _: self._select_material(
                            self.settings.filament.filament_type
                        )
                    )
                    self.buttons.append(button)
                elif title == "Print Settings":
                    button = client.gui.add_button("Reset Print Settings")
                    button.on_click(
                        lambda _: (
                            self.on_change(
                                replace(self.settings, print=PrintSettings())
                            )
                            if self.enabled
                            else None
                        )
                    )
                    self.buttons.append(button)
        self.material.on_update(lambda _: self._select_material(self.material.value))
        self.update(settings)

    def _select_material(self, material: str) -> None:
        if self.syncing or not self.enabled:
            return
        self.on_change(replace(self.settings, filament=filament_preset(material)))

    def _bind_input(self, handle, group: str, name: str, value_type: type) -> None:
        @handle.on_update
        def _(_) -> None:
            if self.syncing or not self.enabled:
                return
            current = getattr(self.settings, group)
            value = value_type(handle.value)
            try:
                updated = replace(current, **{name: value})
                self.on_change(replace(self.settings, **{group: updated}))
            except ValueError:
                self.update(self.settings)

    def update(self, settings: SlicingSettings) -> None:
        self.syncing = True
        try:
            self.settings = settings
            self.material.value = settings.filament.filament_type
            for (group, name), handle in self.inputs.items():
                handle.value = getattr(getattr(settings, group), name)
        finally:
            self.syncing = False

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
        for handle in (*self.inputs.values(), self.material, *self.buttons):
            handle.disabled = not enabled

    def remove(self) -> None:
        for folder in self.folders:
            folder.remove()
