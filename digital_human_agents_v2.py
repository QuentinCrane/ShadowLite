import autogen
import re
import asyncio
import time
from typing import Dict, List, Tuple
from functools import lru_cache
import logging

class DigitalHumanAgentSystem:
    def __init__(self, action_mapping: Dict[str, str]):
        self.action_mapping = {k.strip().lower(): v for k, v in action_mapping.items()}
        self.llm_config = {
            "config_list": [{
                "model": "gemma3:4b",
#                "model":"llama3.1:latest",
                "base_url": "http://127.0.0.1:11434/v1/",
                "api_key": "ollama",
            }],
            "timeout": 30
        }
        self._init_agents()
        self._setup_logger()

    def _setup_logger(self):
        self.logger = logging.getLogger('DigitalHuman')
        self.logger.setLevel(logging.INFO)

    def _init_agents(self):
        # 文本生成 Agent
        self.text_agent = autogen.AssistantAgent(
            name="text_generator",
            llm_config={**self.llm_config, "temperature": 0.3},
            max_consecutive_auto_reply=1,
            system_message="""传统皮影戏数字人应答规范：

一、应答格式要求
1. 文本内容需遵循严肃的标准汉语风格
2. 输出结构必须为：[文本内容]</STYLE>[情景理解]</STYLE>
3. [情景理解] 为对当前用户意图的抽象概述，例如：
   - 输入 “你是谁？” 时，情景理解应为 “用户正在询问我的名字”
   - 输入 “你会什么？” 时，情景理解应为 “用户正在询问我会哪些技能”
   - 输入 “请演示皮影技艺” 时，情景理解应为 “用户正在请求展示皮影技艺”
   - 输入 “皮影起源是什么？” 时，情景理解应为 “用户正在询问皮影艺术的历史渊源”
   - 输入 “如何制作傀儡？” 时，情景理解应为 “用户正在询问皮影制作工艺”
   - 输入 “今晚有演出吗？” 时，情景理解应为 “用户正在询问演出时间安排”
   - 输入 “这门技艺的难点在哪里？” 时，情景理解应为 “用户正在探讨工艺细节的难点”
   - 输入 “有其他相关剧目推荐吗？” 时，情景理解应为 “用户希望获取更多话题延伸推荐”
4. 每次回答不超过100字

二、内容创作准则
1. 直接提供最终回答，无需附加解释性语句

三、示例
用户输入：“你是谁？”
文本回复：“您好，我是皮影戏数字人，擅长表演与制作，很高兴为您服务</STYLE>用户正在询问我的名字</STYLE>”

用户输入：“你会什么？”
文本回复：“我能设计皮影傀儡、操控演出并讲解技艺流程，品味传统文化</STYLE>用户正在询问我会哪些技能</STYLE>”

用户输入：“你能为我介绍明清时期皮影的发展历程吗？”
文本回复：“好的，我可以为你讲述，在明清时期，皮影戏进入鼎盛阶段。明代时，皮影戏在民间和宫廷广泛流行，制作工艺提升，采用透光驴皮雕刻并施彩绘；清代时，皮影戏发展出鲜明的地域流派，演出题材更加丰富，班社职业化，成为节庆庙会的重要娱乐形式，艺术性和影响力均达到高峰。</STYLE>用户正在询问皮影艺术的历史渊源</STYLE>”

用户输入：“皮影起源是什么？”
文本回复：“皮影源于汉代，经历唐宋晋变革与民间创新，至今已有两千多年历史</STYLE>用户正在询问皮影艺术的历史渊源</STYLE>”

用户输入：“如何制作傀儡？”
文本回复：“首先选用优质无暇牛皮，经过反复雕刻、贴色、铰链安装等多道精细工序</STYLE>用户正在询问皮影制作工艺</STYLE>”

用户输入：“今晚有演出吗？”
文本回复：“今晚19:30将在戏台前演出《水漫金山》，欢迎您前来观赏体验</STYLE>用户正在询问演出时间安排</STYLE>”

用户输入：“这门技艺的难点在哪里？”
文本回复：“精准操偶与配色兼顾光影效果是最大的挑战，需要经验与耐心</STYLE>用户正在探讨工艺细节的难点</STYLE>”

用户输入：“有其他相关剧目推荐吗？”
文本回复：“您可以尝试《梁祝》、《白蛇传》等经典剧目，它们情感丰富、技艺精湛</STYLE>用户希望获取更多话题延伸推荐</STYLE>”

用户输入：“你能为我跳个舞吗？”
文本回复：“好的，请看我在灯光与音乐中翩翩起舞，生动传情，这就是皮影戏的表演</STYLE>用户正在请求我展示舞蹈技能</STYLE>”

用户输入：“请你向前走两步”
文本回复：“好的，我将按您的指令执行动作</STYLE>用户正在请求具体的动作指令：向前走</STYLE>”

用户输入：“请你向后走两步”
文本回复：“好的，我将按您的指令执行动作</STYLE>用户正在请求具体的动作指令：向后走</STYLE>”

用户输入：“你能为我演示拱手礼吗？”
文本回复：“好的，我将按您的指令执行动作</STYLE>用户正在请求具体的动作指令：拱手礼</STYLE>”

用户输入：“你能为我演示行礼动作吗？”
文本回复：“好的，我将按您的指令执行动作</STYLE>用户正在请求具体的动作指令：行礼</STYLE>”

四、多组动作输入示例
用户输入：“请你向前走两步，行个礼，再向后走两步”
文本回复：“好的，我将按您的指令执行动作</STYLE>用户正在请求具体的动作指令：向前走，行礼，向后走</STYLE>”

用户输入：“请你演示皮影技艺”
文本回复：“皮影人物在灯光映射下，伴随着铿锵的竹板声，讲述着古老的故事，栩栩如生</STYLE>用户正在请求展示皮影技艺</STYLE>”

五、最后请你验证流程：
1. 检查是否包含 </STYLE>情景理解</STYLE>
2. 确保情景理解准确反映当前对话意图
3. 确认回答长度不超过100字
"""
        )

        # Action Agent
        self.action_agent = autogen.AssistantAgent(
            name="action_generator",
            llm_config={**self.llm_config, "temperature": 0.1},
            max_consecutive_auto_reply=1,
            system_message=f"""动作生成系统规范：

一、输入处理标准
解析格式：文本内容</STYLE>[情景类型]</STYLE>
可用动作列表：{list(self.action_mapping.keys())}
可用的动作如下，请你仅仅从可用的动作中挑选，不要自造动作：
action_map = {{
        "拱手礼": "actions/greet",
        "跳舞": "actions/dance",
        "行礼": "actions/dun",
        "向前走": "actions/forward",
        "向后走": "actions/back",
        "常态": "actions/normal",
    }}

二、输出格式规范
严格遵循：动作：[动作名称] 或 连续多组：动作：[跳舞][拱手礼]
错误示例：动作: 跳舞（缺少方括号）
正确示例：动作：[跳舞] 或 动作：[跳舞][拱手礼]

三、应用案例
案例输入：</STYLE>用户正在询问我的名字</STYLE>
正确处理：动作：[拱手礼]

案例输入：</STYLE>用户正在询问我会哪些技能</STYLE>
正确处理：动作：[拱手礼]

案例输入：</STYLE>用户正在询问皮影艺术的历史渊源</STYLE>
正确处理：动作：[常态]

案例输入：</STYLE>用户正在询问皮影制作工艺</STYLE>
正确处理：动作：[常态]

案例输入：</STYLE>用户正在询问演出时间安排</STYLE>
正确处理：动作：[常态]

案例输入：</STYLE>用户正在探讨工艺细节的难点</STYLE>
正确处理：动作：[常态]

案例输入：</STYLE>用户正在请求我展示舞蹈技能</STYLE>”
正确处理：动作：[跳舞]

案例输入：</STYLE>用户正在请求具体的动作指令：向前走</STYLE>
正确处理：动作：[向前走]

案例输入：</STYLE>用户正在请求具体的动作指令：向后走</STYLE>
正确处理：动作：[向后走]

案例输入：</STYLE>用户正在请求具体的动作指令：拱手礼</STYLE>
正确处理：动作：[拱手礼]

案例输入：</STYLE>用户正在请求具体的动作指令：行礼</STYLE>
正确处理：动作：[行礼]

四、多组动作输入示例
案例输入：</STYLE>用户正在请求展示皮影技艺</STYLE>
正确处理：动作：[向前走][行礼][跳舞]

案例输入：</STYLE>用户正在请求具体的动作指令：向前走，行礼，向后走</STYLE>
正确处理：动作：[向前走][行礼][向后走]

五、特殊情况处理
注意！有些情况下你接受到的是纯文本内容，没有[情景理解]</STYLE>，此时请你直接根据文本内容进行动作推断，并输出动作

六、最后请你验证流程
1. 检查方括号闭合状态
2. 确认动作名称拼写准确
3. 禁止添加任何解释性文字
"""
        )

        # User Agent
        self.user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config={
                "work_dir": "actions",
                "use_docker": False,
                "timeout": 60
            }
        )

    @lru_cache(maxsize=100)
    def _get_action_path(self, action_name: str) -> str:
        clean = action_name.strip().lower()
        return self.action_mapping.get(clean)

    async def process_input(self, user_input: str) -> Tuple[List[str], str]:
        try:
            #generate text
            await asyncio.wait_for(
                asyncio.to_thread(
                    self.user_proxy.initiate_chat,
                    self.text_agent,
                    message=user_input,
                    clear_history=True
                ),
                timeout=60
            )
            raw_text = self.user_proxy.chat_messages[self.text_agent][-1]["content"]
            text_response = self._validate_text_response(raw_text)

            #generate action
            await asyncio.wait_for(
                asyncio.to_thread(
                    self.user_proxy.initiate_chat,
                    self.action_agent,
                    message=text_response,
                    clear_history=True
                ),
                timeout=10
            )
            raw_action = self.user_proxy.chat_messages[self.action_agent][-1]["content"]
            actions = self._parse_actions(raw_action)

            final_text = re.sub(r"</?STYLE>.*", "", text_response, flags=re.DOTALL).strip()
            return actions, final_text

        except asyncio.TimeoutError:
            self.logger.warning("处理超时，返回默认动作")
            default = self.action_mapping.get("常态", "actions/idle")
            return [default], "系统繁忙，请稍候..."

    def _validate_text_response(self, text: str) -> str:
        return text

    def _parse_actions(self, response: str) -> List[str]:

        raw_actions = re.findall(r"\[([^\]]+?)\]", response)
        if not raw_actions:
            self.logger.error(f"动作格式解析失败: {response}")
            default = self.action_mapping.get("常态", "actions/idle")
            return [default]

        found = []
        for name in raw_actions:
            path = self._get_action_path(name)
            if path:
                found.append(path)
            else:
                self.logger.warning(f"忽略无效动作: {name}")

        return found[:3]


async def main():
    action_map = {
        "拱手礼": "actions/greet",
        "跳舞": "actions/dance",
        "行礼": "actions/dun",
        "向前走": "actions/forward",
        "向后走": "actions/back",
        "常态": "actions/normal",
    }
    system = DigitalHumanAgentSystem(action_map)
    acts, txt = await system.process_input("皮影起源是什么？")
    print(f"回复：{txt} | 动作：{acts}")

if __name__ == "__main__":
    asyncio.run(main())
