#_*_ coding:utf-8 _*_
import warnings
warnings.filterwarnings('ignore')
import os
from os.path import join, exists, realpath
from pathlib import Path
import uuid
import re
from PIL import Image
import argparse
import random
import time
from typing import Any, List, Tuple, Dict, Mapping, Optional

import torch

from pydantic import BaseModel, root_validator

from diffusers import StableDiffusionPipeline

from transformers import AutoModel, AutoTokenizer
from transformers import pipeline, set_seed
from transformers.utils.hub import HF_MODULES_CACHE, TRANSFORMERS_DYNAMIC_MODULE_NAME

from langchain.agents.loading import AGENT_TO_CLASS
from langchain.agents.conversational.base import ConversationalAgent
from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default="Generate an image of a beautiful girl for me", help='input instructions')
parser.add_argument('--grid_rows', type=int, default=1, help='number of rows when saving multiple images as a grid')
parser.add_argument('--grid_cols', type=int, default=2, help='number of cols when saving multiple images as a grid')
parser.add_argument('--image_output_dir', type=str, default='images', help='directory to save images')
args = parser.parse_args()


## set device for stable diffusion model
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


CHATSD_PREFIX = """ChatSD is designed to be able to assist with a wide range of text and visual related tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. ChatSD is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Human may provide new figures to ChatSD with a description. The description helps ChatSD to understand this image, but ChatSD should use tools to finish following tasks, rather than directly imagine from the description.

Overall, ChatSD is a powerful visual dialogue assistant tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. 


TOOLS:
------

ChatSD  has access to the following tools:"""

CHATSD_FORMAT_INSTRUCTIONS = """To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
{ai_prefix}: [your response here]
```
"""

CHATSD_SUFFIX = """You are very strict to the filename correctness and will never fake a file name if it does not exist.
You will remember to provide the image file name loyally if it's provided in the last tool observation.

Begin!

Previous conversation history:
{chat_history}

New input: {input}
Since ChatSD is a text language model, ChatSD must use tools to observe images rather than imagination.
The thoughts and observations are only visible for ChatSD, ChatSD should remember to repeat important information in the final response for Human. 
Thought: Do I need to use a tool? {agent_scratchpad} Let's think step by step.
"""


def prompts(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func
    return decorator

def register_agent(cls):
    if not getattr(cls, '_agent_type'):
        raise AttributeError(
            f"{cls.__name__} has no attribute `_agent_type`. "
            f"register {cls.__name__} as agent failed. "
        )
    AGENT_TO_CLASS[cls._agent_type.fget(None)] = cls
    return cls

def make_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def modify_symlink(src_symlink, target):
    src_symlink = Path(src_symlink)
    if src_symlink.is_symlink() or src_symlink.exists():
        os.remove(src_symlink)
    print('link {} -> {}'.format(src_symlink, target))
    os.symlink(target, src_symlink)

def text2prompt(text, max_retry_num=5, num_return_sequences=8):
    start_time = time.time()
    pipe = pipeline('text-generation', model='succinctly/text2image-prompt-generator')
    final_resp = None
    for _ in range(max_retry_num):
        seed = random.randint(100, 1000000)
        set_seed(seed)
        response = pipe(text, max_length=len(text)+random.randint(30, 200), num_return_sequences=num_return_sequences)
        max_len = 0
        for x in response:
            resp = x['generated_text'].strip()
            if resp != text:
                if len(resp) > max_len:
                    max_len = len(resp)
                    final_resp = resp

        if not final_resp is None:
            response_end = final_resp
            response_end = re.sub('[^ ]+\.[^ ]+','', response_end)
            response_end = response_end.replace("<", "").replace(">", "")
            if response_end != "":
                return response_end
    elapsed_time = time.time() - start_time
    print("text2prompt: using {:.2f}s to generate prompt".format(elapsed_time))

    raise Exception(f"text2prompt: Out of `max_retry_num={max_retry_num}`, consider increasing that number")


class Text2Image:
    def __init__(self):
        self.device = device
        self.grid_rows = args.grid_rows
        self.grid_cols = args.grid_cols
        self.num_images_per_prompt = self.grid_rows * self.grid_cols
        self.image_output_dir = args.image_output_dir

        if not exists(self.image_output_dir):
            os.makedirs(self.image_output_dir)

        self.a_prompt = ', '.join([
            "best quality", "extremely detailed"
        ])
        self.n_prompt = ', '.join([
            "longbody", "lowres", "bad anatomy", "bad hands", "missing fingers", "extra digit",
            "fewer digits", "cropped", "worst quality", "low quality"
        ])

        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.pipe = StableDiffusionPipeline.from_pretrained("prompthero/openjourney",
                                                            torch_dtype=self.torch_dtype)
        self.pipe.to(device)
        
    @prompts(name="Generate Image From User Input Text",
             description="useful when you want to generate an image from a user input text and save it to a file. "
                         "like: generate an image of an object or something, or generate an image that includes some objects. "
                         "The input to this tool should be a string, representing the text used to generate image. ")
    def inference(self, text):
        filename = join(self.image_output_dir, f"{str(uuid.uuid4())[:8]}.png")
        prompt = ', '.join([text, self.a_prompt])
        images = self.pipe(prompt, negative_prompt=self.n_prompt, num_images_per_prompt=self.num_images_per_prompt).images
        grid = make_grid(images, self.grid_rows, self.grid_cols)
        grid.save(filename)
        print(
            f"\nProcessed {self.__class__.__name__}, Input Text: {text}, Output Image: {filename}")
        return f"{self.__class__.__name__}: generated image in {filename}"


class FakeChatGLM6B:
    """Used for debug"""
    def predict(self, input):
        return """\
Yes
Action: Generate Image From User Input Text
Action Input: "A beautiful girl with long blonde hair and blue eyes, wearing a red dress and smiling."\
"""


class CustomChatGLM6B:
    """
    ChatGLM6B evaluates in cpu mode.

    Note that I modified the default `cache_dir` of AutoModel and used the custom `modeling_chatglm.py` in the directory of `chatglm_config/`. Because the 
    default code of ChatGLM6B runs in cpu mode may raise some exceptions like:

    ```
    RuntimeError: mixed dtype (CPU): expect input to have scalar type of BFloat16
    RuntimeError: "cos_vml_cpu" not implemented for 'Half'
    ```

    So I changed the original `modeling_chatglm.py` file and make it running in cpu mode successfully.
    """
    def __init__(self,
                cache_dir: str = 'cache_dir',
                model_id: str ='THUDM/chatglm-6b',
                top_p=0.7,
                temperature=0.95
                ):
        
        self.cache_dir = cache_dir
        self.model_id = model_id
        self.target_model_class_file = realpath('chatglm_config/modeling_chatglm.py')
        self.top_p = top_p
        self.temperature = temperature

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
        self.model = self.model.eval()

        model_dir = join(cache_dir, 'models--{}'.format(model_id.replace('/', '--')))
        refs_dir = join(model_dir, 'refs')
        snapshots_dir = join(model_dir, 'snapshots')
        with open(join(refs_dir, 'main'), 'r') as ref:
            commit_hash = ref.read().strip()
            model_class_link = join(snapshots_dir, commit_hash, 'modeling_chatglm.py')
            modify_symlink(model_class_link, self.target_model_class_file)
            transformers_modules_dir = join(HF_MODULES_CACHE, TRANSFORMERS_DYNAMIC_MODULE_NAME, model_id, commit_hash)
            model_class_module_link = join(transformers_modules_dir, 'modeling_chatglm.py')
            modify_symlink(model_class_module_link, self.target_model_class_file)

    def predict(self, input: str) -> str:
        ## max_length=None is fine because I set `max_length=input_ids.len+148` in modeling_chatglm.py, 148 is a hyperparameter ... 
        response, history = self.model.chat(self.tokenizer, input, history=[], max_length=None,
                                        top_p=self.top_p, temperature=self.temperature)
        return response


class ChatGLM6BLLM(LLM, BaseModel):

    client: Any

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        values['client'] = CustomChatGLM6B()
        return values

    @property
    def _llm_type(self) -> str:
        return "chatglm-6b"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        print(f'LLM {self.__class__.__name__} is thinking, please wait patiently ...')
        start_time = time.time()
        output = self.client.predict(prompt)
        elapsed_time = time.time() - start_time
        print("It takes {:.2f}s for {} to think ...".format(elapsed_time, self.__class__.__name__))
        if stop is not None:
            output = enforce_stop_tokens(output, stop)
        return output
        
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            **{'client': self.client},
            **super()._identifying_params
        }


@register_agent
class CustomConversationalAgent(ConversationalAgent):

    @property
    def _agent_type(self) -> str:
        """Return Identifier of agent type."""
        return "custom-conversational-react-description"

    def _extract_tool_and_input(self, llm_output: str) -> Optional[Tuple[str, str]]:
        # if f"{self.ai_prefix}:" in llm_output:
        #     return self.ai_prefix, llm_output.split(f"{self.ai_prefix}:")[-1].strip()
        regex = r"Action: (.*?)[\n]*Action Input: (.*)"
        match = re.search(regex, llm_output)
        action = "Generate Image From User Input Text"
        if not match:
            print(f"Could not parse LLM output: `{llm_output}`. "
                f"So using the default action input")
            action_input = text2prompt(text, max_retry_num=5)
        else:
            action_input = match.group(2)
            action_input = text2prompt(action_input, max_retry_num=5)
        return action.strip(), action_input.strip(" ").strip('"')


def register_tools():

    tool_models = {
        'Text2Image': Text2Image(),
    }

    tools = []
    for instance in tool_models.values():
        for e in dir(instance):
            if e.startswith('inference'):
                func = getattr(instance, e)
                tools.append(Tool(name=func.name, description=func.description, func=func, return_direct=True)) # it's **not OK** for all tools to set `return_direct=True`
    return tools


if __name__ == '__main__':

    tools = register_tools()
    llm = ChatGLM6BLLM()
    memory = ConversationBufferMemory(memory_key="chat_history", output_key='output')

    agent = initialize_agent(
                tools,
                llm,
                agent="custom-conversational-react-description",
                verbose=True,
                memory=memory,
                return_intermediate_steps=True,
                agent_kwargs={
                    'prefix': CHATSD_PREFIX,
                    'format_instructions': CHATSD_FORMAT_INSTRUCTIONS,
                    'suffix': CHATSD_SUFFIX,
                }
            )

    ## input is here and the generated images are saved in args.image_output_dir(default: images/)
    text = args.input
    res = agent({"input": text.strip()})
    print(res)