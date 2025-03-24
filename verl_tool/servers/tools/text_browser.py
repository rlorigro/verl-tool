from .base import BaseTool, register_tool
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


@register_tool
class TextBrowserTool(BaseTool):
    tool_type = "text_browser"

    def get_usage_inst(self):
        return ("Usage instructions for TextBrowser. This code is based on mini_webarena, using playwright to get "
                "accessibility tree for LLMs agent easier access. The code is modified from AutoWebGLM. To get start, run `pip install -e .` under the mini_webarena repo.")

    def conduct_action(self, trajectory_id, action, extra_field):
        # extra_fields: {question: str or None, gt: str or None, url: str or None}
        print(trajectory_id, action, extra_field)
        from mini_webarena.env_worker import WikiQAEnv
        import copy
        env_state = self.load_env(trajectory_id)
        if env_state.get("trajectory_id") is not None: # New Environment, need start
            question = extra_field['question'] if extra_field['question'] is not None else "placeholder"
            gt = extra_field['gt'] if extra_field['gt'] is not None else "placeholder"
            url = extra_field['url']
            env = WikiQAEnv(question, gt, url = url)
            env_state = copy.deepcopy(env.get_state())
            self.save_env(trajectory_id, env_state)
            if action is None:
                observation = env.render()
                env.close()
                return observation, False, True
            env.close()
            del env
        env = WikiQAEnv(env_state["question"], env_state["gt"], url=env_state["url"])
        env.load_state(env_state)
        observation, _, done, _ = env.step(action)
        if done:
            self.delete_env(trajectory_id)
        else:
            env_state = copy.deepcopy(env.get_state())
            self.save_env(trajectory_id, env_state)
        env.close()
        return observation, done, True

    def get_observations(self, trajectory_ids, actions, extra_fields):
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
        # with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(
                tqdm(executor.map(self.conduct_action, trajectory_ids, actions, extra_fields),
                     total=len(trajectory_ids), desc=f"Getting observations using tool {self.tool_type}"))

        observations, dones, valids = zip(*results)
        return observations, dones, valids
