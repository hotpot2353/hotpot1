from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (
    search_travel_guide,
    get_weather,
    search_poi,
    get_route_info,
    search_nearby,
    get_user_travel_prefs,
    fill_context_for_report
)
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch


class TravelAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[
                search_travel_guide,
                get_weather,
                search_poi,
                get_route_info,
                search_nearby,
                get_user_travel_prefs,
                fill_context_for_report
            ],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )

    def execute_stream(self, messages: list):
        input_dict = {
            "messages": messages
        }

        # context 用于存储运行时上下文信息，report 默认为 False
        prev_content = ""
        for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                content = latest_message.content.strip().replace("~~", "").replace("~", "\\~")
                # 只输出增量部分
                if content.startswith(prev_content):
                    delta = content[len(prev_content):]
                    prev_content = content
                    if delta:
                        yield delta
                else:
                    prev_content = content
                    yield content


if __name__ == '__main__':
    agent = TravelAgent()

    for chunk in agent.execute_stream("帮我生成北京3日游的详细攻略报告"):
        print(chunk, end="", flush=True)