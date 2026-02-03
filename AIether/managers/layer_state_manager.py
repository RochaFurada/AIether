import torch
import json
from pathlib import Path
from copy import deepcopy
from aiether.utils import logger_setup

logger = logger_setup()

class LayerStateManager:
    """
    Manages model layer states and metadata.

    Stores only metadata in memory, saving weights to disk
    in structure base/Layer{layer_id}/{state_name}/*.pt.
    Supports multiple states per layer with associated metadata.
    """

    def __init__(self, output_dir: str = None):
        """
        Args:
            output_dir: Root directory where layer data will be saved.
                        If None, uses './output' as default.
        """
        self.layer_records = {}  # {layer_id: layer_info}
        self.next_id = 0

        if output_dir is None:
            output_dir = './output'

        self.base_dir = Path(output_dir) / "LayersData"
        self.base_dir.mkdir(parents=True, exist_ok=True)

        print(f"📁 LayerStateManager initialized at: {self.base_dir}")

    # Helper methods for path construction

    def _get_layer_dir(self, layer_id: int) -> Path:
        return self.base_dir / f"Layer{layer_id}"

    def _get_state_dir(self, layer_id: int, state_name: str) -> Path:
        """
        Returns directory path for specific layer state.
        """
        return self._get_layer_dir(layer_id) / state_name

    # State persistence methods

    def load_state_from_disk(self, layer_id: int, state_name: str):
        """
        Loads saved state_dict from disk for specified state.

        Args:
            layer_id: Layer ID
            state_name: state name (e.g., 'L0', 'L1', 'T_1000')

        Returns:
            dict: Loaded state_dict with tensors, or None if doesn't exist.
        """
        load_dir = self._get_state_dir(layer_id, state_name)
        if not load_dir.exists():
            print(f"❌ Directory {load_dir} not found")
            return None

        state_dict = {}
        for pt_file in load_dir.glob("*.pt"):
            key = pt_file.stem.replace('_', '.')
            tensor = torch.load(pt_file)
            state_dict[key] = tensor

        print(f"✅ State '{state_name}' from layer {layer_id} loaded from: {load_dir}")
        return state_dict

    # Metadata management methods

    def _ensure_layer_record(self, layer_id: int):
        """
        Verifies and creates, if necessary, layer metadata record in memory.
        """
        if layer_id not in self.layer_records:
            self.layer_records[layer_id] = {
                'ID': layer_id,
                'Generator': None,
                'Generator parameters': {},
                'Generation step': None,
                'states': {}  # state_name -> metadata
            }

    def _state_saved_flag(self, layer_id: int, state_name: str) -> bool:
        """
        Checks if state files are saved on disk for specified layer and state.
        """
        state_dir = self._get_state_dir(layer_id, state_name)
        return state_dir.exists() and any(state_dir.glob("*.pt"))

    def _save_metadata(self, layer_id: int):
        """
        Persists layer metadata to JSON file including information about
        generator, parameters, generation step and saved states.
        """
        if layer_id not in self.layer_records:
            return

        record = self.layer_records[layer_id]
        layer_dir = self._get_layer_dir(layer_id)
        layer_dir.mkdir(parents=True, exist_ok=True)

        states_meta = {}
        for state_name, srec in record['states'].items():
            states_meta[state_name] = {
                'step': srec.get('step'),
                'strategy': srec.get('strategy'),
                'metrics': srec.get('metrics', {}),
                'saved': self._state_saved_flag(layer_id, state_name),
                'dir': str(self._get_state_dir(layer_id, state_name)),
            }

        metadata = {
            'ID': record['ID'],
            'Generator': record['Generator'],
            'Generator parameters': record['Generator parameters'],
            'Generation step': record['Generation step'],
            'states': states_meta,
        }

        json_path = layer_dir / "metadata.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"💾 Layer {layer_id} metadata saved to: {json_path}")

    # Public methods for layer registration and state saving

    def register_layer(
        self,
        layer_id: int,
        strategy: str = None,
        strategy_params: dict = None,
        generation_step: int = None
    ):
        """
        Registers layer metadata without saving weights.
        Allows later state saving through save_state.
        """
        self._ensure_layer_record(layer_id)

        self.layer_records[layer_id]['Generator'] = strategy
        self.layer_records[layer_id]['Generator parameters'] = (
            deepcopy(strategy_params) if strategy_params else {}
        )
        self.layer_records[layer_id]['Generation step'] = generation_step

        self._save_metadata(layer_id)
        print(f"📝 Layer {layer_id} registered with strategy '{strategy}' at step {generation_step}")

    def save_state(
        self,
        layer_id: int,
        state_name: str,
        state_dict: dict,
        step: int = None,
        strategy: str = None,
        metrics: dict = None,
    ):
        """
        Saves a new state (any name) for a layer, with metadata.

        Args:
            layer_id: Layer ID.
            state_name: state name (e.g., 'L0', 'L1', 'T_1000', 'checkpoint', etc.).
            state_dict: PyTorch state_dict.
            step: training step associated with this snapshot.
            strategy: strategy used to generate this state (optional).
            metrics: metrics dictionary (tau, kappa, loss, etc.).
        """
        self._ensure_layer_record(layer_id)

        # Save weights to disk
        state_dir = self._get_state_dir(layer_id, state_name)
        logger.info(f"💾 State '{state_name}' from layer {layer_id} saved to: {state_dir}")

        # Register metadata in memory
        self.layer_records[layer_id]['states'][state_name] = {
            'step': step,
            'strategy': strategy,
            'metrics': deepcopy(metrics) if metrics else {},
        }

        # Update metadata on disk
        self._save_metadata(layer_id)

    def update_generation_step(self, layer_id: int, step: int):
        """
        Updates layer generation step and reflects in metadata.
        """
        self._ensure_layer_record(layer_id)
        self.layer_records[layer_id]['Generation step'] = step
        self._save_metadata(layer_id)
        print(f"🔄 Layer {layer_id} generation step updated to {step}")

    # Public query methods

    def get_state(self, layer_id: int, state_name: str):
        """
        Returns state_dict for specific state loaded from disk.
        """
        return self.load_state_from_disk(layer_id, state_name)

    def get_layer_info(self, layer_id: int) -> dict:
        """
        Returns dictionary with complete metadata for specified layer.
        """
        if layer_id not in self.layer_records:
            print(f"❌ Layer {layer_id} not found")
            return None

        record = self.layer_records[layer_id]
        # Rebuild view with "saved" flags
        states_info = {}
        for state_name, srec in record['states'].items():
            states_info[state_name] = {
                'step': srec.get('step'),
                'strategy': srec.get('strategy'),
                'metrics': srec.get('metrics', {}),
                'saved': self._state_saved_flag(layer_id, state_name),
                'dir': str(self._get_state_dir(layer_id, state_name)),
            }

        info = {
            'ID': record['ID'],
            'Generator': record['Generator'],
            'Generator parameters': record['Generator parameters'],
            'Generation step': record['Generation step'],
            'states': states_info,
        }
        return info

    def get_summary(self) -> dict:
        """
        Returns summary of all registered layers.
        """
        summary = {}
        for layer_id, record in self.layer_records.items():
            states = record.get('states', {})
            summary[layer_id] = {
                'Generator': record['Generator'],
                'Generation step': record['Generation step'],
                'Num states': len(states),
                'States': list(states.keys()),
                'Directory': str(self._get_layer_dir(layer_id)),
            }
        return summary
