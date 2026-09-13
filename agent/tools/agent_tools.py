import os
import requests
from utils.logger_handler import logger
from langchain_core.tools import tool
from rag.rag_service import RagSummarizeService
from utils.config_handler import agent_conf, get_env_var
from utils.path_tool import get_abs_path

rag = RagSummarizeService()

# 用户偏好数据缓存
user_travel_prefs = {}


# ==================== 工具1：搜索旅游攻略（RAG） ====================
@tool(description="从知识库检索旅游攻略，回答景点介绍、游玩路线、注意事项等问题")
def search_travel_guide(query: str) -> str:
    """搜索旅游攻略知识库"""
    logger.info(f"[search_travel_guide] query: {query}")
    return rag.rag_summarize(query)


# ==================== 工具2：获取天气（高德API） ====================
@tool(description="获取指定城市的实时天气和未来3天预报，包含温度、湿度、降雨概率")
def get_weather(city: str) -> str:
    """调用高德天气API获取实时天气"""
    api_key = get_env_var("AMAP_API_KEY")

    # 获取城市编码
    geo_url = f"https://restapi.amap.com/v3/geocode/geo?address={city}&key={api_key}"
    geo_res = requests.get(geo_url).json()

    if geo_res.get("status") != "1" or not geo_res.get("geocodes"):
        return f"未找到城市 {city} 的天气信息"

    adcode = geo_res["geocodes"][0]["adcode"]

    # 获取天气信息
    weather_url = f"https://restapi.amap.com/v3/weather/weatherInfo?city={adcode}&key={api_key}&extensions=all"
    weather_res = requests.get(weather_url).json()

    if weather_res.get("status") != "1":
        return f"获取 {city} 天气失败"

    forecasts = weather_res["forecasts"][0]["casts"][:3]

    result = f"【{city}未来3天天气预报】\n"
    for day in forecasts:
        date = day["date"]
        day_temp = day["daytemp"]
        night_temp = day["nighttemp"]
        day_weather = day["dayweather"]
        night_weather = day["nightweather"]
        result += f"{date}: {day_weather}转{night_weather}，{night_temp}~{day_temp}℃\n"

    logger.info(f"[get_weather] {result}")
    return result


# ==================== 工具3：搜索POI（高德API） ====================
@tool(description="搜索指定城市的景点、餐厅、酒店等地点信息，返回名称、地址、评分")
def search_poi(city: str, keyword: str) -> str:
    """调用高德POI搜索API"""
    api_key = get_env_var("AMAP_API_KEY")

    url = f"https://restapi.amap.com/v3/place/text?keywords={keyword}&city={city}&key={api_key}&offset=5"
    response = requests.get(url).json()

    if response.get("status") != "1" or not response.get("pois"):
        return f"未找到 {city} 的 {keyword} 相关信息"

    result = f"【{city} {keyword}推荐】\n"
    for i, poi in enumerate(response["pois"][:5], 1):
        name = poi["name"]
        address = poi.get("address", "暂无地址")
        rating = poi.get("biz_ext", {}).get("rating", "暂无评分")
        result += f"{i}. {name}\n   地址: {address}\n   评分: {rating}\n\n"

    logger.info(f"[search_poi] 找到{len(response['pois'])}个结果")
    return result


# ==================== 工具4：路线规划（高德API） ====================
@tool(description="获取两个地点之间的交通路线信息，包含距离、时间、出行方式建议")
def get_route_info(origin: str, destination: str, city: str) -> str:
    """调用高德路线规划API"""
    api_key = get_env_var("AMAP_API_KEY")

    # 获取起点坐标
    origin_url = f"https://restapi.amap.com/v3/geocode/geo?address={origin}&city={city}&key={api_key}"
    origin_res = requests.get(origin_url).json()

    # 获取终点坐标
    dest_url = f"https://restapi.amap.com/v3/geocode/geo?address={destination}&city={city}&key={api_key}"
    dest_res = requests.get(dest_url).json()

    if origin_res.get("status") != "1" or dest_res.get("status") != "1":
        return f"无法获取 {origin} 或 {destination} 的位置信息"

    origin_loc = origin_res["geocodes"][0]["location"]
    dest_loc = dest_res["geocodes"][0]["location"]

    # 驾车路线规划
    route_url = f"https://restapi.amap.com/v3/direction/driving?origin={origin_loc}&destination={dest_loc}&key={api_key}"
    route_res = requests.get(route_url).json()

    if route_res.get("status") != "1":
        return f"路线规划失败"

    route = route_res["route"]["paths"][0]
    distance = int(route["distance"]) / 1000
    duration = int(route["duration"]) / 60

    result = f"【{origin} → {destination} 路线信息】\n"
    result += f"距离: {distance:.1f}公里\n"
    result += f"预计时间: {duration:.0f}分钟\n"
    result += f"出行建议: 打车/公交/地铁"

    logger.info(f"[get_route_info] {distance:.1f}km, {duration:.0f}min")
    return result


# ==================== 工具5：周边搜索（高德API） ====================
@tool(description="搜索指定位置周边的餐厅、酒店、停车场等地点，用于规划行程时查找附近设施")
def search_nearby(location: str, keyword: str, city: str) -> str:
    """调用高德周边搜索API"""
    api_key = get_env_var("AMAP_API_KEY")

    # 获取中心位置坐标
    geo_url = f"https://restapi.amap.com/v3/geocode/geo?address={location}&city={city}&key={api_key}"
    geo_res = requests.get(geo_url).json()

    if geo_res.get("status") != "1" or not geo_res.get("geocodes"):
        return f"未找到位置 {location}"

    center_loc = geo_res["geocodes"][0]["location"]

    # 周边搜索
    nearby_url = f"https://restapi.amap.com/v3/place/around?location={center_loc}&keywords={keyword}&key={api_key}&radius=1000&offset=5"
    nearby_res = requests.get(nearby_url).json()

    if nearby_res.get("status") != "1" or not nearby_res.get("pois"):
        return f"在 {location} 附近未找到 {keyword}"

    result = f"【{location}周边{keyword}（1公里内）】\n"
    for i, poi in enumerate(nearby_res["pois"][:5], 1):
        name = poi["name"]
        address = poi.get("address", "暂无地址")
        distance = poi.get("distance", "未知")
        rating = poi.get("biz_ext", {}).get("rating", "暂无评分")
        result += f"{i}. {name}\n   地址: {address}\n   距离: {distance}米\n   评分: {rating}\n\n"

    logger.info(f"[search_nearby] 找到{len(nearby_res['pois'])}个结果")
    return result


# ==================== 工具6：获取用户旅行偏好 ====================
@tool(description="获取用户的历史旅行偏好记录，包含去过的目的地、预算范围、旅行风格等，用于个性化推荐")
def get_user_travel_prefs(user_id: str) -> str:
    """
    获取用户旅行偏好
    :param user_id: 用户ID，如"1001"
    :return: 用户旅行偏好信息
    """
    load_travel_prefs()

    if user_id not in user_travel_prefs:
        logger.warning(f"[get_user_travel_prefs] 未找到用户{user_id}的偏好记录")
        return f"未找到用户{user_id}的历史记录"

    prefs = user_travel_prefs[user_id]

    result = f"【用户{user_id}旅行偏好】\n"
    result += f"历史目的地: {', '.join(prefs['destinations'])}\n"
    result += f"平均预算: {prefs['avg_budget']}元\n"
    result += f"偏好风格: {prefs['style']}\n"
    result += f"平均时长: {prefs['avg_duration']}天\n"
    result += f"备注: {prefs['notes']}"

    logger.info(f"[get_user_travel_prefs] {result}")
    return result


def load_travel_prefs():
    """加载用户旅行偏好数据"""
    if user_travel_prefs:
        return

    try:
        prefs_path = get_abs_path(agent_conf["travel_prefs_path"])

        if not os.path.exists(prefs_path):
            logger.warning(f"[load_travel_prefs] 偏好文件不存在: {prefs_path}")
            return

        with open(prefs_path, "r", encoding="gbk") as f:
            lines = f.readlines()

        # 临时存储每个用户的所有记录
        temp_data = {}

        for line in lines[1:]:  # 跳过表头
            parts = line.strip().split(",")
            if len(parts) >= 7:
                user_id = parts[0].replace('"', '')
                destination = parts[1].replace('"', '')
                month = parts[2].replace('"', '')
                budget = int(parts[3].replace('"', ''))
                style = parts[4].replace('"', '')
                duration = int(parts[5].replace('"', '').replace('天', ''))
                notes = parts[6].replace('"', '')

                if user_id not in temp_data:
                    temp_data[user_id] = []

                temp_data[user_id].append({
                    "destination": destination,
                    "month": month,
                    "budget": budget,
                    "style": style,
                    "duration": duration,
                    "notes": notes
                })

        # 汇总用户偏好
        for user_id, records in temp_data.items():
            destinations = [r["destination"] for r in records]
            avg_budget = sum(r["budget"] for r in records) // len(records)
            styles = [r["style"] for r in records]
            avg_duration = sum(r["duration"] for r in records) // len(records)
            notes = records[-1]["notes"]  # 取最新备注

            # 统计最常见的风格
            style_count = {}
            for s in styles:
                style_count[s] = style_count.get(s, 0) + 1
            main_style = max(style_count, key=style_count.get)

            user_travel_prefs[user_id] = {
                "destinations": destinations,
                "avg_budget": avg_budget,
                "style": main_style,
                "avg_duration": avg_duration,
                "notes": notes
            }

        logger.info(f"[load_travel_prefs] 成功加载{len(user_travel_prefs)}个用户的偏好数据")

    except Exception as e:
        logger.error(f"[load_travel_prefs] 加载失败: {str(e)}")

# ==================== 工具7：触发报告生成上下文 ====================
@tool(description="无入参，调用后触发中间件为报告生成场景动态注入上下文，为后续提示词切换提供支撑")
def fill_context_for_report():
    """触发报告生成上下文注入"""
    logger.info("[fill_context_for_report] 已触发报告生成上下文注入")
    return "fill_context_for_report已调用"


if __name__ == "__main__":
    test_cases = [
        (search_travel_guide, {"query": "北京有哪些好玩的景点"}),
        (get_weather, {"city": "北京"}),
        (search_poi, {"city": "北京", "keyword": "烤鸭"}),
        (get_route_info, {"origin": "天安门", "destination": "八达岭长城", "city": "北京"}),
        (search_nearby, {"location": "天安门", "keyword": "餐厅", "city": "北京"}),
        (get_user_travel_prefs, {"user_id": "1001"}),
        (fill_context_for_report, {}),
    ]

    for tool_obj, args in test_cases:
        print(f"\n{'='*20} 测试工具: {tool_obj.name} {'='*20}")
        try:
            result = tool_obj.invoke(args)
            print(result)
            print(f"[OK] {tool_obj.name} 调用成功")
        except Exception as e:
            print(f"[FAIL] {tool_obj.name} 调用失败: {e}")