import json
import os
from openai import OpenAI
from wrappers.constants import CLIENT_API_KEY, CLIENT_BASE_URL, MODEL_EXTRACT


class ExperiencePool:
    def __init__(
        self,
        pool_file="./log/experience_pool.json",
        update_path="./log/task_logs_new",
        file_name="action_log.json",
    ):
        self.pool_file = pool_file
        self.update_path = update_path
        self.file_name = file_name

        self.experiences = self._load_pool()
        self.map = self._load_map()

    def _load_pool(self):
        """加载经验池"""
        if os.path.exists(self.pool_file):
            with open(self.pool_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("experiences", [])
        return []

    def _load_map(self):
        """加载映射"""
        path_map = {exp["query"]: exp["log_path"] for exp in self.experiences}
        return path_map

    def save_pool(self):
        """保存经验池"""
        with open(self.pool_file, "w", encoding="utf-8") as f:
            json.dump(
                {"experiences": self.experiences}, f, indent=2, ensure_ascii=False
            )

    @property
    def all_queries(self):
        """获取所有query列表"""
        return [exp["query"] for exp in self.experiences]

    def match_query(self, query):
        client = OpenAI(api_key=CLIENT_API_KEY, base_url=CLIENT_BASE_URL)
        model_name = MODEL_EXTRACT
        history_prompt = "以下是你之前执行过的任务：\n" + "\n".join(
            [f"- {q}" for q in self.all_queries]
        )

        system_prompt = f"""你是一个经验池匹配专家，负责判断新任务描述是否与已有经验池中的任务语义相似。请严格遵循以下规则：
        1. 目前已有经验池任务列表{history_prompt}
        2. 仅当新指令与经验池中的指令在以下两个维度完全一致时才返回匹配项：
        - 平台名称（如bilibili/百度/QQ）必须字符级一致
        - 操作对象（如老番茄/账号名称）必须完全匹配
        3. 忽略其他修饰词差异（如"打开"/"访问"等）
        4. 匹配query时，一定要严格注意数字的区别
        5. 如果两个维度都匹配那么返回经验池中所匹配的那个query(仅仅返回query即可,不要有多余的其他内容)
        6. 若没有相匹配的query即返回"no_match"
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        response = client.chat.completions.create(model=model_name, messages=messages)

        query_map = response.choices[0].message.content
        if query_map in self.all_queries:
            return query_map
        else:
            return "no_match"

    def add_experience(self, query, log_path):
        """添加新经验"""
        # 检查是否已存在相同query
        if query in self.all_queries:
            return False

        self.experiences.append({"query": query, "log_path": log_path})
        return True

    # 读取日志更新经验池
    def update_query(self):
        for file in os.listdir(self.update_path):
            file_path = os.path.join(self.update_path, file)
            file_path_log = os.path.join(file_path, self.file_name)
            print(f"Reading: {file_path_log}")
            with open(file_path_log, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data["metadata"]["experience_flag"] == True:
                self.add_experience(data["metadata"]["query"], file_path)
                print("Adding experience...")
        self.save_pool()
        return True


if __name__ == "__main__":
    pool = ExperiencePool()
    pool.update_query()
