import gradio as gr
from fastai.vision.all import *
import skimage


def name_label_func(fname):
    return 'rem' in fname.lower()


learn = load_learner('model.pkl')

title = "Is this Rem?"
description = "Is your image include Rem in it?"
article = "<p style='text-align: center'><a href='https://github.com/HelloYeew/is-this-rem target='_blank'>GitHub Source Code</a></p>"
examples = ['girl.webp', 'another-girl.jpg']


def predict(img):
    img = PILImage.create(img)
    pred, pred_idx, probs = learn.predict(img)
    labels = ['This is not Rem!', 'This is Rem!']
    return {labels[i]: float(probs[i]) for i in range(len(labels))}


interface = gr.Interface(fn=predict, inputs="image", outputs="label", title=title, description=description, article=article, examples=examples)
interface.launch()
