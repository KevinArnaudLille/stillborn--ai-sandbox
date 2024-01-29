import torch
from transformers import pipeline, AutoProcessor, BarkModel
import scipy
import uuid
import os
import tempfile
from IPython.display import Audio
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    sentimentClassifier = pipeline("sentiment-analysis")
    res1 = sentimentClassifier(["I would like a cookie, I love sugar!", "I didn't like the movie, it was so bad.", "J'adore les chiens"])
    print(res1)


    # ==============================
    # labelClassifier = pipeline("zero-shot-classification")
    # res2 = labelClassifier(
    #     ["Picture a vertical line that runs through the center of your body, and the goal is to keep your weight balanced on both sides of that line.",
    #      "Wouldn’t it be nice if the crux on your project had a huge, sticky, perfectly formed foothold in just the right place?"],
    #     candidate_labels=["soccer", "climbing", "politics", "medicine"]
    # )
    # print(res2)


    # ==============================
    # en_fr_translator = pipeline("translation_fr_to_en")
    # res3 = en_fr_translator("Le père, Jean-Michel, également percuté lors de l’accident et grièvement blessé, est toujours hospitalisé, mais il « va de mieux en mieux », a précisé Sébastien Durand, après l’avoir vu la veille. Refusant « que le drame soit récupéré », les organisateurs avaient tenu à ce que les élus « viennent en civil », « sans écharpe », et que la presse « se tienne à l’écart du cortège ». « Nous voulons rester dans le respect et la dignité », ont-ils souligné.")
    # print(res3)


    # ==============================
    # processor = AutoProcessor.from_pretrained("suno/bark")
    # model = BarkModel.from_pretrained("suno/bark")
    #
    # voice_preset = "v2/fr_speaker_6"
    #
    # inputs = processor("J'espère que t'as bien mangé Hadrien! [laughter]", voice_preset=voice_preset)
    #
    # audio_array = model.generate(**inputs)
    # audio_array = audio_array.cpu().numpy().squeeze()
    #
    # sample_rate = model.generation_config.sample_rate
    # scipy.io.wavfile.write("generated_assets/bark_out.wav", rate=sample_rate, data=audio_array)


    # ==============================
    # model_id = "stabilityai/stable-diffusion-2"
    # scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    # pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
    # pipe = pipe.to("cuda")
    #
    # prompt = "a dog and a cat fighting in the street with red lights"
    # image = pipe(prompt).images[0]
    # image.save("generated_assets/test.png")
