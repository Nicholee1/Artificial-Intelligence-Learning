#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理工具
用于管理AI模型配置和系统设置
"""

import os
import json
import argparse
from typing import Dict, Any, List
from ..ai.service import AIService

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: str = "config/ai_config.json"):
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """加载配置"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return self.create_default_config()
        except Exception as e:
            print(f"❌ 加载配置失败: {e}")
            return self.create_default_config()
    
    def save_config(self, config: Dict[str, Any] = None):
        """保存配置"""
        if config is None:
            config = self.config
        
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            print(f"✅ 配置已保存到 {self.config_file}")
        except Exception as e:
            print(f"❌ 保存配置失败: {e}")
    
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
                    "model": "llama3.1:8b",
                    "enabled": True
                },
                "qwen": {
                    "type": "openai",
                    "api_key": "",
                    "model": "qwen-turbo",
                    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "enabled": False
                },
                "deepseek": {
                    "type": "openai",
                    "api_key": "",
                    "model": "deepseek-chat",
                    "base_url": "https://api.deepseek.com/v1",
                    "enabled": False
                }
            }
        }
    
    def list_providers(self):
        """列出所有提供者"""
        print("\n🤖 AI模型提供者配置:")
        print("=" * 50)
        
        for name, config in self.config['providers'].items():
            status = "✅ 启用" if config.get('enabled', False) else "❌ 禁用"
            default_mark = " (默认)" if name == self.config.get('default_provider') else ""
            
            print(f"\n{name}{default_mark}: {status}")
            print(f"  类型: {config.get('type', 'unknown')}")
            print(f"  模型: {config.get('model', 'N/A')}")
            
            if config.get('api_key'):
                masked_key = config['api_key'][:8] + "..." + config['api_key'][-4:]
                print(f"  API密钥: {masked_key}")
            else:
                print(f"  API密钥: 未设置")
            
            if config.get('base_url'):
                print(f"  Base URL: {config['base_url']}")
    
    def set_provider_config(self, provider_name: str, **kwargs):
        """设置提供者配置"""
        if provider_name not in self.config['providers']:
            print(f"❌ 提供者 '{provider_name}' 不存在")
            return False
        
        provider_config = self.config['providers'][provider_name]
        
        for key, value in kwargs.items():
            if key in provider_config:
                provider_config[key] = value
                print(f"✅ 设置 {provider_name}.{key} = {value}")
            else:
                print(f"⚠️  未知配置项: {key}")
        
        return True
    
    def enable_provider(self, provider_name: str):
        """启用提供者"""
        if provider_name not in self.config['providers']:
            print(f"❌ 提供者 '{provider_name}' 不存在")
            return False
        
        self.config['providers'][provider_name]['enabled'] = True
        print(f"✅ 已启用提供者: {provider_name}")
        return True
    
    def disable_provider(self, provider_name: str):
        """禁用提供者"""
        if provider_name not in self.config['providers']:
            print(f"❌ 提供者 '{provider_name}' 不存在")
            return False
        
        self.config['providers'][provider_name]['enabled'] = False
        print(f"✅ 已禁用提供者: {provider_name}")
        return True
    
    def set_default_provider(self, provider_name: str):
        """设置默认提供者"""
        if provider_name not in self.config['providers']:
            print(f"❌ 提供者 '{provider_name}' 不存在")
            return False
        
        if not self.config['providers'][provider_name].get('enabled', False):
            print(f"⚠️  提供者 '{provider_name}' 未启用")
            return False
        
        self.config['default_provider'] = provider_name
        print(f"✅ 已设置默认提供者: {provider_name}")
        return True
    
    def test_provider(self, provider_name: str):
        """测试提供者"""
        if provider_name not in self.config['providers']:
            print(f"❌ 提供者 '{provider_name}' 不存在")
            return False
        
        provider_config = self.config['providers'][provider_name]
        
        if not provider_config.get('enabled', False):
            print(f"❌ 提供者 '{provider_name}' 未启用")
            return False
        
        try:
            # 创建临时AI服务进行测试
            temp_config = {
                "default_provider": provider_name,
                "providers": {provider_name: provider_config}
            }
            
            # 保存临时配置
            temp_file = f"temp_config_{provider_name}.json"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(temp_config, f, ensure_ascii=False, indent=2)
            
            # 测试
            ai_service = AIService(temp_file)
            if provider_name in ai_service.get_available_providers():
                if ai_service.test_provider(provider_name):
                    print(f"✅ 提供者 '{provider_name}' 测试成功")
                    result = True
                else:
                    print(f"❌ 提供者 '{provider_name}' 测试失败")
                    result = False
            else:
                print(f"❌ 提供者 '{provider_name}' 不可用")
                result = False
            
            # 清理临时文件
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            return result
            
        except Exception as e:
            print(f"❌ 测试提供者 '{provider_name}' 时出错: {e}")
            return False
    
    def interactive_setup(self):
        """交互式设置"""
        print("\n🔧 AI模型配置向导")
        print("=" * 30)
        
        # 选择提供者
        print("\n可用的AI模型提供者:")
        providers = list(self.config['providers'].keys())
        for i, provider in enumerate(providers, 1):
            print(f"  {i}. {provider}")
        
        while True:
            try:
                choice = input(f"\n请选择提供者 (1-{len(providers)}): ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(providers):
                    provider_name = providers[int(choice) - 1]
                    break
                else:
                    print("❌ 无效选择")
            except KeyboardInterrupt:
                print("\n👋 取消设置")
                return
        
        print(f"\n配置提供者: {provider_name}")
        
        # 配置API密钥
        if provider_name in ['openai', 'claude', 'qwen', 'deepseek']:
            api_key = input("请输入API密钥: ").strip()
            if api_key:
                self.set_provider_config(provider_name, api_key=api_key)
        
        # 配置模型
        current_model = self.config['providers'][provider_name].get('model', '')
        model = input(f"请输入模型名称 (当前: {current_model}): ").strip()
        if model:
            self.set_provider_config(provider_name, model=model)
        
        # 配置Base URL (如果需要)
        if provider_name in ['qwen', 'deepseek']:
            current_url = self.config['providers'][provider_name].get('base_url', '')
            base_url = input(f"请输入Base URL (当前: {current_url}): ").strip()
            if base_url:
                self.set_provider_config(provider_name, base_url=base_url)
        
        # 启用提供者
        enable = input(f"是否启用 {provider_name}? (y/n): ").strip().lower()
        if enable in ['y', 'yes']:
            self.enable_provider(provider_name)
            
            # 设置为默认提供者
            set_default = input(f"是否设置为默认提供者? (y/n): ").strip().lower()
            if set_default in ['y', 'yes']:
                self.set_default_provider(provider_name)
        
        # 测试提供者
        if self.config['providers'][provider_name].get('enabled', False):
            test = input(f"是否测试 {provider_name}? (y/n): ").strip().lower()
            if test in ['y', 'yes']:
                self.test_provider(provider_name)
        
        # 保存配置
        self.save_config()
        print(f"\n✅ 配置完成!")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='AI模型配置管理工具')
    parser.add_argument('--list', '-l', action='store_true', help='列出所有提供者')
    parser.add_argument('--test', '-t', type=str, help='测试指定提供者')
    parser.add_argument('--enable', '-e', type=str, help='启用指定提供者')
    parser.add_argument('--disable', '-d', type=str, help='禁用指定提供者')
    parser.add_argument('--default', type=str, help='设置默认提供者')
    parser.add_argument('--setup', '-s', action='store_true', help='交互式设置')
    parser.add_argument('--set', nargs=3, metavar=('PROVIDER', 'KEY', 'VALUE'), 
                       help='设置提供者配置 (提供者 键 值)')
    
    args = parser.parse_args()
    
    config_manager = ConfigManager()
    
    if args.list:
        config_manager.list_providers()
    elif args.test:
        config_manager.test_provider(args.test)
    elif args.enable:
        config_manager.enable_provider(args.enable)
        config_manager.save_config()
    elif args.disable:
        config_manager.disable_provider(args.disable)
        config_manager.save_config()
    elif args.default:
        config_manager.set_default_provider(args.default)
        config_manager.save_config()
    elif args.set:
        provider, key, value = args.set
        config_manager.set_provider_config(provider, **{key: value})
        config_manager.save_config()
    elif args.setup:
        config_manager.interactive_setup()
    else:
        print("AI模型配置管理工具")
        print("使用 --help 查看帮助")

if __name__ == "__main__":
    main()
