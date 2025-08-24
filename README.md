# Here is a repository includes AI learing roadmap:

- **LangChain**：LangChain 像是一个专门给 AI 开发者用的 “多功能工具箱” 兼 “施工图纸”。它有各种各样的小工具模块，比如能帮 AI 记住之前对话内容的记忆模块，还有可以连接搜索引擎、数据库这些外部工具的连接模块。同时它还能帮开发者设计 AI 完成任务的流程，比如先做什么、再做什么，如果遇到问题怎么选择下一步动作等。借助 LangChain，开发者就能更轻松地把 AI 模型和各种功能组合起来，开发出更复杂、更智能的 AI 应用程序。
- **MCP（Model Context Protocol）**：MCP 就像是一种 AI 与外部工具沟通的 “通用语言标准”。比如说 AI 要去查天气、访问数据库或者操作文件的时候，不同的天气应用、数据库软件、文件系统它们跟 AI 对话的方式可能都不一样，这就会让 AI 觉得很麻烦。MCP 就是来解决这个问题的，它制定了一套统一的规则，让 AI 只要按照这个规则来，就能够以一种标准的方式去和各种外部工具 “聊天”，获取自己需要的数据或者功能，从而更好地完成各种任务，像个 “翻译官” 一样统一 AI 与外部工具交流的方式。
- **Agent2Agent（A2A）**：Agent2Agent 可以想象成是 AI 小助手们的 “合作协议”。现在有好多不同的 AI 小助手，每个都有自己的专长，有的擅长找资料，有的擅长写文案。Agent2Agent 就是让这些不同的 AI 小助手之间能够相互交流、一起合作。有了它，这些小助手们能知道彼此能做什么，然后互相帮忙，一起去处理一些复杂的事情，而且它们在合作的时候也不用把自己内部最核心的秘密和特殊方法都公开，在保障隐私的同时一起把活干好。
- **Dify**：Dify 是一个很方便的 AI 应用搭建工厂。以前要是想做个 AI 应用，比如一个智能客服或者文档分析工具，对于不懂很多技术的人来说特别难。但是 Dify 给大家准备了好多现成的 “小零件” 和 “组装方法”，不管你是不是程序员，都能通过简单的操作，像拼积木那样，把不同的 AI 功能模块组合起来，快速做出一个能用的 AI 应用。而且你还能通过它很容易地管理这个应用，比如看看有多少人用、花了多少钱等等。

## 1.Simple demo
The basic demo using Ollama in HTTP request type/ OpenAI type/ langchain type.

For this project, you can know how to call an llm with ollama in your local env.

## 2. Langchain practice
Here is an open-source project, [langchain-crash-course](https://github.com/bhancockio/langchain-crash-course)

And here is the corresponding video in Youtube [Youtube-link](http://youtube.com/watch?v=yF9kGESAi3M)

In this session, you will know how to use langchain in basic chat model/prompt tamplate/langchain chains/Retrieve-Augmented Generation(RAG)/AI Agent and Tools with 21 practice demos.

In my own project, there are some records and analysis for every demo, and model has been changed from openAI to llama3 locally maintained by Ollama.
