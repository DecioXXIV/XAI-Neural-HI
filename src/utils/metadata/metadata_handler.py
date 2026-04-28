import json
from typing import Dict, Any

from src.utils.logger import Logger

class MetadataHandler:
    def __init__(self, metadata_path: str):
        self.metadata_path = metadata_path
        self.logger = Logger()

    def save_metadata(self, metadata: Dict[str, Any]):
        try:
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
        except Exception as e:
            self.logger.exception(f"Failed to save metadata: {e}")
            raise

    def load_metadata(self) -> Dict[str, Any]:
        try:
            with open(self.metadata_path, 'r') as f:
                data = json.load(f)
                self.logger.info(f"Metadata loaded from '{self.metadata_path}'")
                return data
        except FileNotFoundError:
            self.logger.warning(f"Metadata file '{self.metadata_path}' not found. Initializing with empty metadata.")
            return {}
        except json.JSONDecodeError as e:
            self.logger.exception(f"Failed to decode metadata: {e}")
            raise
        except Exception as e:
            self.logger.exception(f"Unexpected error while loading metadata: {e}")
            raise
    
    def update_metadata(self, metadata: Dict[str, Any], key: str, value: Any):
        metadata[key] = value
        self.save_metadata(metadata)