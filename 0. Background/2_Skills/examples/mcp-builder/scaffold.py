"""
MCP Builder Skill scaffold — 使用本地 Ollama 运行 MCP Builder 技能。

解析 SKILL.md（YAML frontmatter + Markdown 流程说明），将用户请求与技能描述
一并交给 Ollama 模型，由模型按技能指南给出建议或步骤。
"""

import argparse
import os
import re
from pathlib import Path

try:
    import ollama
except ImportError:
    raise ImportError("请先安装 ollama Python 客户端: pip install ollama")

# 默认使用的 Ollama 模型，可通过环境变量 OLLAMA_MODEL 或 --model 覆盖
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")


def _skill_dir() -> Path:
    """返回 scaffold 所在目录（即 mcp-builder skill 根目录）。"""
    return Path(__file__).resolve().parent


def parse_skill_md(skill_md_path: str | Path | None = None) -> dict:
    """
    解析 SKILL.md：提取 YAML frontmatter（name, description, license）及正文内容。

    Args:
        skill_md_path: SKILL.md 路径，默认使用与 scaffold 同目录下的 SKILL.md。

    Returns:
        包含 name, description, license, content 的字典。
    """
    path = Path(skill_md_path) if skill_md_path else _skill_dir() / "SKILL.md"
    if not path.is_file():
        raise FileNotFoundError(f"未找到技能文件: {path}")

    text = path.read_text(encoding="utf-8")

    # 解析 YAML frontmatter (--- ... ---)
    front_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    config = {}
    if front_match:
        front = front_match.group(1)
        for line in front.strip().split("\n"):
            m = re.match(r"^(\w+):\s*(.*)$", line.strip())
            if m:
                config[m.group(1).strip()] = m.group(2).strip()
        content = text[front_match.end() :].strip()
    else:
        content = text.strip()

    config["content"] = content
    return config


def run_skill_with_ollama(
    skill_config: dict,
    user_input: str,
    model: str | None = None,
) -> str:
    """
    使用本地 Ollama 模型按 MCP Builder 技能指南响应用户请求。

    Args:
        skill_config: parse_skill_md() 返回的配置（含 name, description, content）。
        user_input: 用户的问题或需求（例如：帮我生成一个 MCP Server，用于调用计算器工具）。
        model: Ollama 模型名，默认使用 DEFAULT_MODEL（或环境变量 OLLAMA_MODEL）。

    Returns:
        模型生成的回复文本。
    """
    model = model or DEFAULT_MODEL
    name = skill_config.get("name", "mcp-builder")
    description = skill_config.get("description", "")
    content = skill_config.get("content", "")

    # 将技能流程与用户输入一起交给模型
    prompt = f"""你正在使用「{name}」技能。该技能的说明如下：

## 技能描述
{description}

## 技能流程与参考（请按此指南回答）
{content[:12000]}

---

用户请求：
{user_input}

请严格按照上述 MCP Builder 技能指南，针对用户请求给出具体、可执行的建议或步骤（例如如何设计工具、项目结构、调用方式等）。若需生成代码，请直接写出可用代码片段。"""

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.get("message", {}).get("content", "")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="使用本地 Ollama 运行 MCP Builder 技能（解析 SKILL.md 并调用模型）"
    )
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        help=f"Ollama 模型名（默认: {DEFAULT_MODEL}，或环境变量 OLLAMA_MODEL）",
    )
    parser.add_argument(
        "--input", "-i",
        default="帮我生成一个 MCP Server，用于调用计算器工具",
        help="用户请求内容",
    )
    parser.add_argument(
        "--skill-file",
        default=None,
        help="SKILL.md 路径（默认使用与 scaffold 同目录的 SKILL.md）",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="仅解析并打印技能配置，不调用 Ollama",
    )
    args = parser.parse_args()

    skill_path = args.skill_file or (_skill_dir() / "SKILL.md")
    skill_config = parse_skill_md(skill_path)

    print("解析到的技能配置：")
    for k in ("name", "description", "license"):
        if k in skill_config:
            print(f"  {k}: {skill_config[k]}")
    print()

    if args.no_run:
        print("（已使用 --no-run，跳过 Ollama 调用）")
        return

    print(f"使用模型: {args.model}")
    print(f"用户请求: {args.input}")
    print("-" * 40)
    result = run_skill_with_ollama(skill_config, args.input, model=args.model)
    print("技能执行结果：")
    print(result)


if __name__ == "__main__":
    main()
