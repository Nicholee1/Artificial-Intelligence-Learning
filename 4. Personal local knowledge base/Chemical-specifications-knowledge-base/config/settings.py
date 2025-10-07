#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目配置设置
"""

import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdf"
JSON_DIR = DATA_DIR / "json"
VECTOR_DB_DIR = DATA_DIR / "vector_db"

# 配置目录
CONFIG_DIR = PROJECT_ROOT / "config"
AI_CONFIG_FILE = CONFIG_DIR / "ai_config.json"

# 日志目录
LOG_DIR = PROJECT_ROOT / "logs"

# 默认设置
DEFAULT_SETTINGS = {
    # 向量数据库设置
    "vector_db": {
        "persist_directory": str(VECTOR_DB_DIR),
        "collection_name": "chemical_documents",
        "model_name": "all-MiniLM-L6-v2"
    },
    
    # PDF处理设置
    "pdf_processing": {
        "pdf_directory": str(PDF_DIR),
        "json_directory": str(JSON_DIR),
        "chunk_size": 500,
        "overlap": 50
    },
    
    # AI服务设置
    "ai_service": {
        "config_file": str(AI_CONFIG_FILE),
        "default_provider": "ollama",
        "max_tokens": 1000,
        "temperature": 0.7
    },
    
    # API设置
    "api": {
        "host": "0.0.0.0",
        "port": 5000,
        "debug": True
    },
    
    # 日志设置
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": str(LOG_DIR / "chemical_kb.log")
    }
}

class Settings:
    """配置管理类"""
    
    def __init__(self):
        self.settings = DEFAULT_SETTINGS.copy()
        self._ensure_directories()
    
    def _ensure_directories(self):
        """确保必要的目录存在"""
        directories = [
            DATA_DIR, PDF_DIR, JSON_DIR, VECTOR_DB_DIR,
            CONFIG_DIR, LOG_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default=None):
        """获取配置值"""
        keys = key.split('.')
        value = self.settings
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value):
        """设置配置值"""
        keys = key.split('.')
        setting = self.settings
        
        for k in keys[:-1]:
            if k not in setting:
                setting[k] = {}
            setting = setting[k]
        
        setting[keys[-1]] = value
    
    def update(self, new_settings: dict):
        """更新配置"""
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(self.settings, new_settings)

# 全局配置实例
settings = Settings()

# 环境变量覆盖
if os.getenv('CHEMICAL_KB_ENV') == 'production':
    settings.update({
        "api": {
            "debug": False
        },
        "logging": {
            "level": "WARNING"
        }
    })
elif os.getenv('CHEMICAL_KB_ENV') == 'development':
    settings.update({
        "api": {
            "debug": True
        },
        "logging": {
            "level": "DEBUG"
        }
    })

