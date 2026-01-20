from dataclasses import dataclass
from typing import List, Any, Dict
import os
import yaml
import logging


@dataclass
class AppConfig:
    magnetars_list_path_file: str
    intermediate_path_file: str
    data_path_folder: str
    rap_cam_pix: float
    rbackin_cam_pix: float
    rbackout_cam_pix: float
    spacing: float
    prf_path: str
    result_path_file: str
    channels: List[int]
    grid: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            'magnetars_list_path_file': self.magnetars_list_path_file,
            'intermediate_path_file': self.intermediate_path_file,
            'data_path_folder': self.data_path_folder,
            'rap_(cam_pix)': self.rap_cam_pix,
            'rbackin_(cam_pix)': self.rbackin_cam_pix,
            'rbackout_(cam_pix)': self.rbackout_cam_pix,
            'spacing': self.spacing,
            'prf_path': self.prf_path,
            'result_path_file': self.result_path_file,
            'channels': self.channels,
            'grid': self.grid,
        }


def _validate_and_normalize(raw: Dict[str, Any]) -> AppConfig:
    logger = logging.getLogger(__name__)
    logger.debug("Validating YAML configuration and normalizing types")
    required_keys = [
        'magnetars_list_path_file', 'intermediate_path_file', 'data_path_folder',
        'rap_(cam_pix)', 'rbackin_(cam_pix)', 'rbackout_(cam_pix)', 'spacing',
        'prf_path', 'result_path_file', 'channels', 'grid'
    ]

    missing = [k for k in required_keys if k not in raw]
    if missing:
        logger.error(f"Missing required config keys: {', '.join(missing)}")
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")

    # Types and coercion
    try:
        rap = float(raw['rap_(cam_pix)'])
        rbi = float(raw['rbackin_(cam_pix)'])
        rbo = float(raw['rbackout_(cam_pix)'])
        spacing = float(raw['spacing'])
    except Exception as exc:
        logger.error("Radius or spacing values must be numeric", exc_info=True)
        raise ValueError(f"Radius or spacing values must be numeric: {exc}")

    # channels
    channels_val = raw['channels']
    if isinstance(channels_val, str):
        # allow comma/space separated strings
        tokens = channels_val.strip().strip('[]').replace(',', ' ').split()
        channels = [int(t) for t in tokens]
    elif isinstance(channels_val, list):
        channels = [int(c) for c in channels_val]
    else:
        channels = [int(channels_val)]

    # grid
    try:
        grid = int(raw['grid'])
    except Exception as exc:
        logger.error("grid must be an integer", exc_info=True)
        raise ValueError(f"grid must be an integer: {exc}")

    # Value validations
    errors: List[str] = []
    if rap <= 0:
        errors.append('rap_(cam_pix) must be > 0')
    if rbi <= 0:
        errors.append('rbackin_(cam_pix) must be > 0')
    if rbo <= 0:
        errors.append('rbackout_(cam_pix) must be > 0')
    if not (rap < rbi < rbo):
        errors.append('radii must satisfy rap_(cam_pix) < rbackin_(cam_pix) < rbackout_(cam_pix)')
    if spacing <= 0:
        errors.append('spacing must be > 0')
    if grid < 1:
        errors.append('grid must be >= 1')
    if not channels:
        errors.append('channels must be a non-empty list')
    if any(c not in (1, 2, 3, 4) for c in channels):
        errors.append('channels must be a subset of [1,2,3,4]')

    # Paths existence checks
    for path_key in ['magnetars_list_path_file', 'data_path_folder', 'prf_path']:
        p = str(raw[path_key])
        if path_key == 'data_path_folder':
            if not os.path.isdir(p):
                errors.append(f"data_path_folder not found: {p}")
        elif path_key == 'prf_path':
            if not os.path.isdir(p):
                errors.append(f"prf_path not found: {p}")
        else:
            if not os.path.isfile(p):
                errors.append(f"magnetars_list_path_file not found: {p}")

    if errors:
        logger.error('Invalid configuration:\n- ' + '\n- '.join(errors))
        raise ValueError('Invalid configuration:\n- ' + '\n- '.join(errors))

    cfg = AppConfig(
        magnetars_list_path_file=str(raw['magnetars_list_path_file']),
        intermediate_path_file=str(raw['intermediate_path_file']),
        data_path_folder=str(raw['data_path_folder']),
        rap_cam_pix=rap,
        rbackin_cam_pix=rbi,
        rbackout_cam_pix=rbo,
        spacing=spacing,
        prf_path=str(raw['prf_path']),
        result_path_file=str(raw['result_path_file']),
        channels=channels,
        grid=grid,
    )
    logger.debug(f"Normalized configuration: {cfg.as_dict()}")
    return cfg


def load_yaml_config(filename: str = 'configs/default.yml') -> AppConfig:
    logger = logging.getLogger(__name__)
    logger.info(f"Loading YAML configuration from {filename}")
    with open(filename, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        logger.error('Top-level YAML must be a mapping of keys to values.')
        raise ValueError('Top-level YAML must be a mapping of keys to values.')
    return _validate_and_normalize(data)