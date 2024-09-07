""" Main application file for the Streamlit app. """

import streamlit as st
from utils.conversation import Conversation
from utils.utils import get_help, clear_chat, about_app, singleton, AVAILABLE_FUNCTIONS
from tools.tools import Tools
from tools.text_response import TextResponse
from vllm.lora.request import LoRARequest



from typing import Any, Dict, Union
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from vllm import LLM, SamplingParams


@singleton
class GenAIModel:
    def __init__(self, system_message: str = "I'm an AI assistant here to help you with any questions you have."):
        """
        Initializes a new instance of the ModelManager.
        """
        self.model_name = "t3ai-org/pt-model" #mrbesher/t3ai-cogitators-fc-v2
        self.lora_adapters = "mrbesher/t3ai-cogitators-fc-v2" 
        self.model = None
        self.initialize_model()
        self.system_message = system_message

    def initialize_model(self):
        """
        Initializes the Model  based on the configuration.
        """
        self.model = AutoModelForCausalLM.from_pretrained(self.lora_adapters, load_in_4bit=True, device_map= "auto")
        # model = PeftModel.from_pretrained(model, lora_path)
        # self.model = LLM(self.model_name)
        # self.model = LLM(self.model_name, enable_lora=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.lora_adapters)
        # self.sampling_params = SamplingParams(
        #                                 temperature=0.2,
        #                                 max_tokens=1024,
        #                                 stop=["<eot>"]
        #                             )


    def switch_model(self):
        """
        Switches between  models.
        :param use_local: Determines whether to switch to a local model or an OpenAI model.
        """
        self.initialize_model()

    def generate_response(self, data: str, model_settings: Dict[str, Any], response_model: Union[None, Any]=TextResponse):
        """
        Generates a response from the GPT model based on the provided data and model settings.
        :param data: The input data for the GPT model.
        :param model_settings: Settings for the model such as 'model', 'temperature', etc.
        :param response_model: The Pydantic model for structuring the response, if any.
        :return: The response from the GPT model.
        """

        try:
            messages=[
                        {"role": "system", "content": self.system_message,
                        "tools": [{'name': 'calculate_age', 'description': 'Calculate the age based on birth date', 'parameters': {'type': 'object', 'properties': {'birth_date': {'type': 'string', 'format': 'date', 'description': 'The birth date of the person'}}, 'required': ['birth_date']}}]
                        }, {"role": "user", "content": data}
                    ],
            # formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # print(f"formatted prompt is: {formatted_prompt}")
            formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            print(f"formatted_prompt: {formatted_prompt}")
            formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
            formatted_prompt.to("cuda")
            response = self.model.generate(formatted_prompt, max_new_tokens = 1024, eos_token_id = 128004) #self.sampling_params ,lora_request=LoRARequest("func_call", 1,self.lora_adapters))
            docoded_response = self.tokenizer.decode(response[0][len(formatted_prompt):], skip_special_tokens = True)
            print(f"Full response is: {docoded_response}")
            # response = response[0].outputs[0].text
            return docoded_response
        except Exception as e:
            # response = TextResponse(response="An error occurred while generating the response.")
            print(f"An error occurred while generating the response: {e}")
            return None
             



generic_system_message = """Sen aşağıdaki fonksiyonlara erişimi olan yardımcı bir asistansın. Kullanıcı sorusuna yardımcı olabilmek için bir veya daha fazla fonksiyon çağırabilirsin. Fonksiyon parametreleri ile ilgili varsayımlarda bulunma. Her fonksiyon çağrısı fonksiyon ismi ve parametreleri ile olmalıdır. İşte, kullanabileceğin fonksiyonlar:"""
models_map = {
    "google/gemma-2-9b": {
        'use_local': True,
        'avatar': "🦙",
        'system_message': generic_system_message.format(model_name="google/gemma-2-9b")
    },
}

functions_list = list(AVAILABLE_FUNCTIONS.keys())
with st.sidebar:
    selected_functions = st.multiselect(
        "Select Functions:",
        options=functions_list,
        default=[functions_list[0]],  # Default to the first model
    )
    st.write(f"Selected Functions: {', '.join(selected_functions)}")

# Update session state to handle multiple selected models
# if 'selected_models' not in st.session_state or st.session_state.selected_models != selected_models:
#     st.session_state.selected_models = selected_models

# Initialize the Conversation
if "conversation" not in st.session_state:
    st.session_state.conversation = Conversation()

for message in st.session_state.conversation.messages:    
    if message["role"]=="user":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    elif message["role"]=="tool":
        with st.chat_message(message["role"], avatar='🛠️'):
            st.markdown(message['content'])
    else:
        with st.chat_message(message["role"], avatar=models_map[message["role"]]['avatar']):
            st.markdown(message["content"])
    
# Chat input
prompt = st.chat_input("Type something...", key="prompt")

if prompt:

    if prompt[0] == '/':
            commands_map = {
                "help": get_help,
                "clear": clear_chat,
                "about": about_app,
            }
            command = prompt[1:]
            if command in commands_map:
                command_response = commands_map[command]()
                if command_response:
                    with st.chat_message("tool", avatar='🛠️'):
                        st.write(command_response)
            else:
                with st.chat_message("tool", avatar='🛠️'):
                    st.write(f"Command '{command}' not recognized.")
  
    else:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.conversation.add_message("user", prompt)            
        model = GenAIModel(system_message=generic_system_message)
        model_name = "google/gemma-2-9b"
        model_settings = {
            "model": model_name,
            "temperature": 0.2,  # Customize as needed
        }
        
        function_response = ""
        full_prompt = st.session_state.conversation.format_for_gpt()

        function_response = model.generate_response(full_prompt, model_settings)
        with st.chat_message(model_name, avatar=models_map[model_name]['avatar']):
            st.markdown(function_response)
        st.session_state.conversation.add_message(model_name, function_response)
        # try:
        #     response_dict = json.loads(response)

        
        # functions_obj = model.generate_response(full_prompt, model_settings, response_model=Tools)

        # if functions_obj:
        #     function_response, tool = functions_obj.process()

        #     if function_response:
        #         if tool == "text":
        #             with st.chat_message(model_name, avatar=models_map[model_name]['avatar']):
        #                 st.markdown(function_response)
        #             st.session_state.conversation.add_message(model_name, function_response)
        #         else:            
        #             with st.chat_message("tool", avatar='🛠️'):
        #                 st.markdown(function_response)
        #             st.session_state.conversation.add_message("tool", function_response)
        #             break
        # else:
        #     response = model.generate_response(full_prompt, model_settings)
        #     if response:
        #         with st.chat_message(model_name, avatar=models_map[model_name]['avatar']):
        #             st.markdown(response)
        #         st.session_state.conversation.add_message(model_name, response)
        #     else:
        #         with st.chat_message(model_name, avatar=models_map[model_name]['avatar']):
        #             st.write("I'm sorry, I don't have a response to that.")