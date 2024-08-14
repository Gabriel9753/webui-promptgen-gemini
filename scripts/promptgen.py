import html
import os
import time

import google.generativeai as genai
import gradio as gr
import torch
import transformers
from loguru import logger

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

current_dir = os.path.dirname(os.path.realpath(__file__))
log_file = os.path.join(current_dir, "promptgen.log")
print(f"logger to: {log_file}")
logger.add(log_file, rotation="10 MB", level="INFO")

if os.path.exists(log_file):
    print("Log file exists")
else:
    print("Log file does not exist")
    with open(log_file, "w") as f:
        f.write("")

logger.info("logger started")

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

Just optimize the prompt for T5 but keep CLIP in mind. Here you can now use natural language to describe general and detailed information about the image.

Below I give you my request, in any language and roughly describing what I want for a picture. Then write ONLY the optimized prompt in ENGLISH LANGUAGE!!!! Really only generate the one prompt (natural language) and nothing more!!!
Not even in bullet points or split for CLIP and T5, but ONE prompt that I can use directly to generate an image.

My request:
{{input}}
"""

upload_image_prompt = "\n\nAlso, I uploaded an image that can be used as a reference for the prompt. Use the image to generate the prompt more accurately."

style_prompt = "\n   Additionally, I want the following style for the image: {{style}}"
style_dict = {
    "Photorealistic": "Photorealistic - An image that looks as close to a real photograph as possible. High detail, realistic lighting, shadows, textures, and colors.",
    "Minimalist": "Minimalist -  Simple and clean with minimal details and use of color. Focuses on basic shapes and primary colors to convey the idea.",
    "Abstract": "Abstract -  Non-representational art that uses shapes, colors, and forms to achieve its effect, without necessarily depicting real-world objects.",
    "Surreal": "Surreal -  A style that blends reality with dream-like elements, creating bizarre, unexpected, and fantastical scenarios.",
    "Cartoon": "Cartoon -  Features exaggerated characteristics, bold outlines, and vibrant colors. Often conveys a light-hearted or whimsical feel.",
    "Vintage/Retro": "Vintage/Retro -  Uses a color palette and aesthetic reminiscent of a past era, often with a sepia tone, grainy textures, and classic design elements.",
    "Steampunk": "Steampunk -  A fusion of Victorian-era aesthetics with modern or futuristic elements, often incorporating gears, cogs, and steam-powered machinery.",
    "Pop Art": "Pop Art -  A style characterized by bold colors, sharp lines, and iconic imagery. Often draws inspiration from commercial art, media, and popular culture.",
    "Cyberpunk": "Cyberpunk -  A futuristic style that combines high-tech elements with a gritty, urban atmosphere. Often features neon lights, advanced technology, and dystopian themes.",
    "Fantasy": "Fantasy -  A style that brings to life magical or mythical elements, often featuring mystical creatures, enchanted landscapes, and otherworldly scenarios.",
}

color_brightness_prompt = "\n   Additionally, I want the following color and brightness for the image: {{color_brightness}}"
color_brightness_dict = {
    "Vibrant": "Vibrant - A style that emphasizes bold, bright, and vivid colors to create a lively and energetic feel.",
    "Muted": "Muted - A style featuring colors that are less saturated, resulting in a softer and more subdued look.",
    "Monochrome": "Monochrome - A style that uses variations of a single color or shades of black and white, creating a cohesive and minimalist aesthetic.",
    "High Contrast": "High Contrast - A style that emphasizes strong differences between light and dark areas, enhancing visual impact and depth.",
    "Pastel": "Pastel - A style characterized by soft, light colors, often used to create a gentle and soothing atmosphere.",
    "Dark": "Dark - A style that features predominantly dark tones and colors, creating a moody and dramatic effect.",
    "Sepia": "Sepia - A style that gives images a warm, brownish tone, reminiscent of old photographs.",
    "Neon": "Neon - A style that incorporates bright, glowing colors typically associated with neon lights, creating a futuristic or retro-futuristic vibe.",
    "Earthy": "Earthy - A style that uses natural, warm colors such as browns, greens, and ochres, often evoking a rustic or organic feel.",
    "High Key": "High Key - A style that utilizes predominantly light tones and minimal shadows, creating an airy, bright, and optimistic appearance.",
    "Low Key": "Low Key - A style that features predominantly dark tones, with a focus on shadows and subtle highlights, creating a dramatic and mysterious atmosphere.",
}

camera_shot_prompt = "\n   Additionally, I want the following color shot style for the image: {{color_shot}}"
camera_shot_dict = {
    "Macro": "Macro - A close-up shot that captures fine details of small subjects, often resulting in a highly detailed and intimate view.",
    "Wide Angle": "Wide Angle - A shot that captures a broad view of a scene, often used to emphasize the scale or setting of the subject.",
    "Telephoto": "Telephoto - A shot taken with a long lens that compresses the perspective, making distant subjects appear closer together.",
    "Panoramic": "Panoramic - A shot that captures a wide, horizontal view of a landscape or scene, often achieved by stitching together multiple images.",
    "Aerial": "Aerial - A shot taken from above, typically using a drone or aircraft, to provide a bird’s-eye view of the scene.",
    "Tilt-Shift": "Tilt-Shift - A shot that uses selective focus to create a miniature effect, making real-world scenes appear like small-scale models.",
    "Long Exposure": "Long Exposure - A shot that uses a slow shutter speed to capture motion blur, often resulting in smooth water or light trails.",
    "Time-Lapse": "Time-Lapse - A sequence of shots taken over a long period and played back at high speed, showing changes over time in a compressed format.",
    "Fish-eye": "Fish-eye - A shot that uses an ultra-wide lens to create a circular, distorted view, often resulting in a unique and exaggerated perspective.",
    "Portrait": "Portrait - A shot that focuses on a person or group, often using a shallow depth of field to emphasize the subject against a blurred background.",
    "Bokeh": "Bokeh - A shot that features a blurred background with out-of-focus points of light, creating an aesthetic and artistic effect.",
    "Night": "Night - A shot taken in low light conditions, often using techniques to enhance visibility and detail in dark environments.",
}


def generate_batch(model, input_text, uploaded_image=None):
    logger.info(f"Generating text for input: {input_text}")
    try:
        if uploaded_image:
            response = model.generate_content(
                [input_text, uploaded_image], stream=False
            )
        else:
            response = model.generate_content(input_text, stream=False)
        return [response.text.strip().replace("\n", "")]
    except Exception as e:
        logger.error(f"Error generating text: {e}")
        raise ValueError(f"Error generating text: {e}")


def generate(
    prompt,
    uploaded_image,
    temperature,
    style_setting,
    color_brightness_setting,
    camera_shot_setting,
):
    logger.info("Generating text")

    api_key = shared.opts.Google_API_KEY
    # prompt_template = shared.opts.gemini_prompt
    model_name = shared.opts.promptgen_model
    prompt_template = default_prompt

    logger.info(f"Using model: {model_name}")

    if api_key == "" or api_key is None:
        raise ValueError("Google API key is required for text generation")

    # if "{{input}}" not in prompt_template:
    #     raise ValueError("Prompt template has to include {{input}}")

    prompt = prompt_template.replace("{{input}}", prompt)
    # print(f"Using prompt: {prompt}")

    if uploaded_image:
        prompt += upload_image_prompt

    if style_setting != "default":
        prompt += style_prompt.replace("{{style}}", style_dict[style_setting])

    if color_brightness_setting != "default":
        prompt += color_brightness_prompt.replace(
            "{{color_brightness}}", color_brightness_dict[color_brightness_setting]
        )

    if camera_shot_setting != "default":
        prompt += camera_shot_prompt.replace(
            "{{color_shot}}", camera_shot_dict[camera_shot_setting]
        )

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        raise ValueError(f"Error initializing model: {e}")

    markup = "<table><tbody>"

    index = 0
    for i in range(1):
        texts = generate_batch(model, prompt, uploaded_image)
        logger.info(f"Generated text: {texts}")
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
                    lines=10,
                    placeholder="Beginning of the prompt (press Ctrl+Enter or Alt+Enter to generate)",
                )
            with gr.Column(scale=20):
                uploaded_image = gr.Image(
                    elem_id="promptgen_image",
                    label="Upload image",
                    show_label=True,
                    type="pil",
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

                with FormRow():
                    style_setting = gr.Dropdown(
                        label="style mode",
                        elem_id="promptgen_style_mode",
                        value="default",
                        choices=["default"] + list(style_dict.keys()),
                        info="Choose a style mode for the image",
                    )

                    color_brightness_setting = gr.Dropdown(
                        label="color and brightness mode",
                        elem_id="promptgen_color_brightness_mode",
                        value="default",
                        choices=["default"] + list(color_brightness_dict.keys()),
                        info="Choose a color and brightness mode for the image",
                    )

                    camera_shot_setting = gr.Dropdown(
                        label="camera shot mode",
                        elem_id="promptgen_camera_shot_mode",
                        value="default",
                        choices=["default"] + list(camera_shot_dict.keys()),
                        info="Choose a camera shot mode for the image",
                    )

                with gr.Row():
                    temperature = gr.Slider(
                        label="Temperature",
                        elem_id="promptgen_temperature",
                        value=0.2,
                        minimum=0,
                        maximum=1,
                        step=0.01,
                    )

                # with FormRow():
                #     min_length = gr.Slider(
                #         label="Min length",
                #         elem_id="promptgen_min_length",
                #         value=20,
                #         minimum=1,
                #         maximum=400,
                #         step=1,
                #     )
                #     max_length = gr.Slider(
                #         label="Max length",
                #         elem_id="promptgen_max_length",
                #         value=150,
                #         minimum=1,
                #         maximum=400,
                #         step=1,
                #     )

                # with FormRow():
                #     batch_count = gr.Slider(
                #         label="Batch count",
                #         elem_id="promptgen_batch_count",
                #         value=1,
                #         minimum=1,
                #         maximum=100,
                #         step=1,
                #     )

            with gr.Column():
                with gr.Group(elem_id="promptgen_results_column"):
                    res = gr.HTML()
                    res_info = gr.HTML()

        submit.click(
            fn=ui.wrap_gradio_gpu_call(generate, extra_outputs=[""]),
            _js="submit_promptgen",
            inputs=[
                prompt,
                uploaded_image,
                temperature,
                style_setting,
                color_brightness_setting,
                camera_shot_setting,
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

    # shared.opts.add_option(
    #     "gemini_prompt",
    #     shared.OptionInfo(
    #         default_prompt,
    #         "Prompt to use for text generation (has to include {{input}})",
    #         gr.Textbox,
    #         {"lines": 20},
    #         section=section,
    #     ),
    # )


def on_unload():
    pass


script_callbacks.on_ui_tabs(add_tab)
script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_script_unloaded(on_unload)
