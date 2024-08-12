import html
import os
import time

import google.generativeai as genai
import gradio as gr
import torch
import transformers

# from dotenv import load_dotenv
from modules import (
    devices,
    generation_parameters_copypaste,
    script_callbacks,
    scripts,
    shared,
    ui,
)
from modules.ui_components import FormRow

# load_dotenv()  # take environment variables from .env.

# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=GOOGLE_API_KEY)
# model = genai.GenerativeModel("gemini-1.5-flash")

default_prompt = """You are an expert in creating detailed prompts using natural language to generate high quality images. The image generation model uses two different encoders to process the language:

CLIP-optimized:
- Use concise, keyword-rich descriptions
- Separate concepts with commas
- Prioritize visual attributes and style descriptors
- Include artistic references or specific visual techniques when relevant

T5-optimized:
- Focus on clear, descriptive language
- Use complete sentences or phrases
- Emphasize overall scene composition and context
- Include specific details about objects, colors, and spatial relationships

-----------------------------------------------------------------------

So use keywords and terms at the beginning of the new prompt to optimize for CLIP. Here, for example, the style and setting can be well directed.
After the keywords to describe the image on a general level, optimize the prompt for T5. Here you can now use natural language to describe details of the image.

Below I give you my request, in any language and roughly describing what I want for a picture. Then write ONLY the optimized prompt in ENGLISH LANGUAGE!!!! Really only generate the one prompt and nothing more!!! Not even in bullet points or split for CLIP and T5, but ONE prompt that I can use directly to generate an image.

My request:
{{input}}
"""


def generate_batch(model, input_text):
    try:
        response = model.generate_content(input_text, stream=False)
        return [response.text]
    except Exception as e:
        raise ValueError(f"Error generating text: {e}")


def generate(prompt):
    print(f"The prompt is: {prompt}")

    api_key = shared.opts.Google_API_KEY
    prompt_template = shared.opts.gemini_prompt
    model_name = shared.opts.promptgen_model

    if api_key == "" or api_key is None:
        raise ValueError("Google API key is required for text generation")

    if "{{input}}" not in prompt_template:
        raise ValueError("Prompt template has to include {{input}}")

    prompt = prompt_template.replace("{{input}}", prompt)
    print(f"Using prompt: {prompt}")

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        raise ValueError(f"Error initializing model: {e}")

    # shared.state.textinfo = "Loading model..."
    # shared.state.job_count = batch_count
    # shared.state.textinfo = ""

    markup = "<table><tbody>"

    index = 0
    for i in range(1):
        texts = generate_batch(model, prompt)
        print(texts)
        shared.state.nextjob()
        for generated_text in texts:
            index += 1
            markup += f"""
                <tr>
                <td>
                <div class="prompt gr-box gr-text-input">
                    <p id='promptgen_res_{index}'>{html.escape(generated_text)}</p>
                </div>
                </td>
                <td class="sendto">
                    <a class='gr-button gr-button-lg gr-button-secondary' onclick="promptgen_send_to_txt2img(gradioApp().getElementById('promptgen_res_{index}').textContent)">to txt2img</a>
                    <a class='gr-button gr-button-lg gr-button-secondary' onclick="promptgen_send_to_img2img(gradioApp().getElementById('promptgen_res_{index}').textContent)">to img2img</a>
                </td>
                </tr>
            """

    markup += "</tbody></table>"

    return markup, ""


def find_prompts(fields):
    field_prompt = [x for x in fields if x[1] == "Prompt"][0]
    field_negative_prompt = [x for x in fields if x[1] == "Negative prompt"][0]
    return [field_prompt[0], field_negative_prompt[0]]


def send_prompts(text):
    params = generation_parameters_copypaste.parse_generation_parameters(text)
    negative_prompt = params.get("Negative prompt", "")
    return params.get("Prompt", ""), negative_prompt or gr.update()


def add_tab():

    with gr.Blocks(analytics_enabled=False) as tab:
        with gr.Row():
            with gr.Column(scale=80):
                prompt = gr.Textbox(
                    label="Prompt",
                    elem_id="promptgen_prompt",
                    show_label=False,
                    lines=2,
                    placeholder="Beginning of the prompt (press Ctrl+Enter or Alt+Enter to generate)",
                )
            with gr.Column(scale=10):
                submit = gr.Button(
                    "Generate", elem_id="promptgen_generate", variant="primary"
                )

            with gr.Row(elem_id="promptgen_main"):
                with gr.Column(variant="compact"):
                    selected_text = gr.TextArea(
                        elem_id="promptgen_selected_text", visible=False
                    )
                    send_to_txt2img = gr.Button(
                        elem_id="promptgen_send_to_txt2img", visible=False
                    )
                    send_to_img2img = gr.Button(
                        elem_id="promptgen_send_to_img2img", visible=False
                    )

            #         with FormRow():
            #             sampling_mode = gr.Radio(
            #                 label="Sampling mode",
            #                 elem_id="promptgen_sampling_mode",
            #                 value="Top K",
            #                 choices=["Top K", "Top P"],
            #             )
            #             top_k = gr.Slider(
            #                 label="Top K",
            #                 elem_id="promptgen_top_k",
            #                 value=12,
            #                 minimum=1,
            #                 maximum=50,
            #                 step=1,
            #             )
            #             top_p = gr.Slider(
            #                 label="Top P",
            #                 elem_id="promptgen_top_p",
            #                 value=0.15,
            #                 minimum=0,
            #                 maximum=1,
            #                 step=0.001,
            #             )

            #         with gr.Row():
            #             num_beams = gr.Slider(
            #                 label="Number of beams",
            #                 elem_id="promptgen_num_beams",
            #                 value=1,
            #                 minimum=1,
            #                 maximum=8,
            #                 step=1,
            #             )
            #             temperature = gr.Slider(
            #                 label="Temperature",
            #                 elem_id="promptgen_temperature",
            #                 value=1,
            #                 minimum=0,
            #                 maximum=4,
            #                 step=0.01,
            #             )
            #             repetition_penalty = gr.Slider(
            #                 label="Repetition penalty",
            #                 elem_id="promptgen_repetition_penalty",
            #                 value=1,
            #                 minimum=1,
            #                 maximum=4,
            #                 step=0.01,
            #             )

            #         with FormRow():
            #             length_penalty = gr.Slider(
            #                 label="Length preference",
            #                 elem_id="promptgen_length_preference",
            #                 value=1,
            #                 minimum=-10,
            #                 maximum=10,
            #                 step=0.1,
            #             )
            #             min_length = gr.Slider(
            #                 label="Min length",
            #                 elem_id="promptgen_min_length",
            #                 value=20,
            #                 minimum=1,
            #                 maximum=400,
            #                 step=1,
            #             )
            #             max_length = gr.Slider(
            #                 label="Max length",
            #                 elem_id="promptgen_max_length",
            #                 value=150,
            #                 minimum=1,
            #                 maximum=400,
            #                 step=1,
            #             )

            #         with FormRow():
            #             batch_count = gr.Slider(
            #                 label="Batch count",
            #                 elem_id="promptgen_batch_count",
            #                 value=1,
            #                 minimum=1,
            #                 maximum=100,
            #                 step=1,
            #             )
            #             batch_size = gr.Slider(
            #                 label="Batch size",
            #                 elem_id="promptgen_batch_size",
            #                 value=10,
            #                 minimum=1,
            #                 maximum=100,
            #                 step=1,
            #             )

            with gr.Column():
                with gr.Group(elem_id="promptgen_results_column"):
                    res = gr.HTML()
                    res_info = gr.HTML()

        submit.click(
            fn=ui.wrap_gradio_gpu_call(generate, extra_outputs=[""]),
            _js="submit_promptgen",
            inputs=[
                # model_selection,
                # model_selection,
                # batch_count,
                # batch_size,
                prompt,
                # min_length,
                # max_length,
                # num_beams,
                # temperature,
                # repetition_penalty,
                # length_penalty,
                # sampling_mode,
                # top_k,
                # top_p,
            ],
            outputs=[res, res_info],
        )

        send_to_txt2img.click(
            fn=send_prompts,
            inputs=[selected_text],
            outputs=find_prompts(ui.txt2img_paste_fields),
        )

        send_to_img2img.click(
            fn=send_prompts,
            inputs=[selected_text],
            outputs=find_prompts(ui.img2img_paste_fields),
        )

    return [(tab, "Promptgen", "promptgen")]


def on_ui_settings():
    section = ("promptgen", "Promptgen")

    shared.opts.add_option(
        "Google_API_KEY",
        shared.OptionInfo(
            "",
            "Google API key (required for text generation)",
            gr.Textbox,
            {"type": "password"},
            section=section,
        ),
    )

    shared.opts.add_option(
        "promptgen_model",
        shared.OptionInfo(
            "gemini-1.5-flash",
            "Model to use for text generation",
            gr.Radio,
            {"choices": ["gemini-1.5-flash", "gemini-1.5-pro"]},
            section=section,
        ),
    )

    shared.opts.add_option(
        "gemini_prompt",
        shared.OptionInfo(
            default_prompt,
            "Prompt to use for text generation (has to include {{input}})",
            gr.Textbox,
            {"lines": 20},
            section=section,
        ),
    )


def on_unload():
    pass


script_callbacks.on_ui_tabs(add_tab)
script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_script_unloaded(on_unload)
