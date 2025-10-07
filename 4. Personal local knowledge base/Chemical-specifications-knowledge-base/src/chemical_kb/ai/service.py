#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI模型服务模块
支持多种AI模型API集成
"""

import os
import json
import requests
import openai
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIModel(ABC):
    """AI模型基类"""
    
    @abstractmethod
    def generate_response(self, prompt: str, **kwargs) -> str:
        """生成响应"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查模型是否可用"""
        pass

class OpenAIProvider(AIModel):
    """OpenAI API提供者"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", base_url: str = None):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """初始化OpenAI客户端"""
        try:
            if self.base_url:
                self.client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
            else:
                self.client = openai.OpenAI(api_key=self.api_key)
            logger.info("OpenAI客户端初始化成功")
        except Exception as e:
            logger.error(f"OpenAI客户端初始化失败: {e}")
            self.client = None
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """生成响应"""
        if not self.is_available():
            return "OpenAI服务不可用"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": kwargs.get('system_prompt', '你是一个专业的化工技术助手。')},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=kwargs.get('max_tokens', 1000),
                temperature=kwargs.get('temperature', 0.7)
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API调用失败: {e}")
            return f"生成响应时出错: {e}"
    
    def is_available(self) -> bool:
        """检查模型是否可用"""
        return self.client is not None

class ClaudeProvider(AIModel):
    """Claude API提供者"""
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.anthropic.com/v1/messages"
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """生成响应"""
        if not self.is_available():
            return "Claude服务不可用"
        
        try:
            headers = {
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            data = {
                "model": self.model,
                "max_tokens": kwargs.get('max_tokens', 1000),
                "messages": [
                    {
                        "role": "user",
                        "content": f"{kwargs.get('system_prompt', '你是一个专业的化工技术助手。')}\n\n{prompt}"
                    }
                ]
            }
            
            response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result['content'][0]['text']
        except Exception as e:
            logger.error(f"Claude API调用失败: {e}")
            return f"生成响应时出错: {e}"
    
    def is_available(self) -> bool:
        """检查模型是否可用"""
        return bool(self.api_key)

class LocalOllamaProvider(AIModel):
    """本地Ollama模型提供者"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama2"):
        self.base_url = base_url
        self.model = model
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """生成响应"""
        if not self.is_available():
            return "本地Ollama服务不可用"
        
        try:
            url = f"{self.base_url}/api/generate"
            data = {
                "model": self.model,
                "prompt": f"{kwargs.get('system_prompt', '你是一个专业的化工技术助手。')}\n\n{prompt}",
                "stream": False,
                "options": {
                    "temperature": kwargs.get('temperature', 0.7),
                    "num_predict": kwargs.get('max_tokens', 1000)
                }
            }
            
            response = requests.post(url, json=data, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            return result.get('response', '')
        except Exception as e:
            logger.error(f"Ollama API调用失败: {e}")
            return f"生成响应时出错: {e}"
    
    def is_available(self) -> bool:
        """检查模型是否可用"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

class AIService:
    """AI服务管理器"""
    
    def __init__(self, config_file: str = "config/ai_config.json"):
        self.config_file = config_file
        self.providers = {}
        self.default_provider = None
        self.load_config()
    
    def load_config(self):
        """加载AI配置"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            else:
                # 创建默认配置
                config = self.create_default_config()
                self.save_config(config)
            
            self._initialize_providers(config)
            logger.info("AI配置加载成功")
        except Exception as e:
            logger.error(f"加载AI配置失败: {e}")
            self._initialize_providers({})
    
    def create_default_config(self) -> Dict[str, Any]:
        """创建默认配置"""
        return {
            "default_provider": "openai",
            "providers": {
                "openai": {
                    "type": "openai",
                    "api_key": "",
                    "model": "gpt-3.5-turbo",
                    "base_url": None,
                    "enabled": False
                },
                "claude": {
                    "type": "claude",
                    "api_key": "",
                    "model": "claude-3-sonnet-20240229",
                    "enabled": False
                },
                "ollama": {
                    "type": "ollama",
                    "base_url": "http://localhost:11434",
                    "model": "llama2",
                    "enabled": False
                }
            }
        }
    
    def save_config(self, config: Dict[str, Any]):
        """保存配置"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
    
    def _initialize_providers(self, config: Dict[str, Any]):
        """初始化AI提供者"""
        self.providers = {}
        
        for name, provider_config in config.get('providers', {}).items():
            if not provider_config.get('enabled', False):
                continue
                
            try:
                if provider_config['type'] == 'openai':
                    provider = OpenAIProvider(
                        api_key=provider_config['api_key'],
                        model=provider_config['model'],
                        base_url=provider_config.get('base_url')
                    )
                elif provider_config['type'] == 'claude':
                    provider = ClaudeProvider(
                        api_key=provider_config['api_key'],
                        model=provider_config['model']
                    )
                elif provider_config['type'] == 'ollama':
                    provider = LocalOllamaProvider(
                        base_url=provider_config['base_url'],
                        model=provider_config['model']
                    )
                else:
                    logger.warning(f"未知的提供者类型: {provider_config['type']}")
                    continue
                
                if provider.is_available():
                    self.providers[name] = provider
                    logger.info(f"AI提供者 {name} 初始化成功")
                else:
                    logger.warning(f"AI提供者 {name} 不可用")
                    
            except Exception as e:
                logger.error(f"初始化AI提供者 {name} 失败: {e}")
        
        # 设置默认提供者
        self.default_provider = config.get('default_provider')
        if self.default_provider not in self.providers:
            self.default_provider = list(self.providers.keys())[0] if self.providers else None
    
    def generate_response(self, prompt: str, provider: str = None, **kwargs) -> str:
        """生成AI响应"""
        provider_name = provider or self.default_provider
        
        if not provider_name or provider_name not in self.providers:
            return "没有可用的AI提供者"
        
        return self.providers[provider_name].generate_response(prompt, **kwargs)
    
    def get_available_providers(self) -> List[str]:
        """获取可用的提供者列表"""
        return list(self.providers.keys())
    
    def test_provider(self, provider_name: str) -> bool:
        """测试提供者是否可用"""
        if provider_name not in self.providers:
            return False
        
        try:
            response = self.providers[provider_name].generate_response("测试", max_tokens=10)
            return "出错" not in response
        except:
            return False


def main():
    """测试AI服务"""
    print("🤖 AI服务测试")
    print("=" * 30)
    
    # 初始化AI服务
    ai_service = AIService()
    
    # 显示可用提供者
    providers = ai_service.get_available_providers()
    print(f"可用提供者: {providers}")
    
    if not providers:
        print("❌ 没有可用的AI提供者")
        print("请配置 ai_config.json 文件")
        return
    
    # 测试生成
    test_prompt = "请介绍一下化工管道设计的基本要求"
    print(f"\n测试提示: {test_prompt}")
    
    for provider in providers:
        print(f"\n使用提供者: {provider}")
        response = ai_service.generate_response(test_prompt, provider=provider)
        print(f"响应: {response[:200]}...")

if __name__ == "__main__":
    main()
