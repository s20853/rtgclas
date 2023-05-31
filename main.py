import streamlit as st
import pickle
import pathlib
from pathlib import Path

from fastai.vision.all import *
from fastai.vision.widgets import *


class HookBwd():
    def __init__(self, m):
        self.hook = m.register_backward_hook(self.hook_func)
    def hook_func(self, m, gi, go): self.stored = go[0].detach().clone()
    def __enter__(self, *args): return self
    def __exit__(self, *args): self.hook.remove()

class Hook():
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_func)
    def hook_func(self, m, i, o): self.stored = o.detach().clone()
    def __enter__(self, *args): return self
    def __exit__(self, *args): self.hook.remove()

class Predict:
    def __init__(self, resnet_model: str, vgg_model: str):
        self.learn_interface_resnet = load_learner(Path() / resnet_model)
        self.learn_interface_vgg = load_learner(Path() / vgg_model)

        st.set_page_config(page_title='RTGClas', page_icon='🩻', layout="centered", initial_sidebar_state="auto",
                           menu_items=None)

        st.header("Klasyfikacja zdjęć rentgenowskich")
        left, right = st.columns(2)
        with left:
            self.img = self.get_image_from_upload()

        if self.img is not None:
            with right:
                st.image(self.img)
            self.get_prediction()

    @staticmethod
    def get_image_from_upload():

        uploaded_file = st.file_uploader("Wybierz zdjęcie RTG do klasyfikacji", type=['png', 'jpg', 'jpeg'])
        if uploaded_file is not None:
            return PILImage.create(uploaded_file)

    def get_prediction(self):
        if st.button("Wykonaj klasyfikację"):
            st.divider()
            pred, pred_idx, probs = self.learn_interface_resnet.predict(self.img)
            prob = probs[pred_idx].item() * 100

            predictions_col, cam_col = st.columns(2)
            with predictions_col:
                st.subheader("Model ResNet")
                left, right = st.columns(2)
                with left:
                    st.metric("Klasa", pred)
                with right:
                    st.metric("Prawdopodobieństwo", "{0:.0f}".format(prob))

                pred_vgg, pred_idx_vgg, probs_vgg = self.learn_interface_vgg.predict(self.img)
                prob_vgg = probs_vgg[pred_idx_vgg].item() * 100
                st.subheader("Model VGG")
                left, right = st.columns(2)
                with left:
                    st.metric("Klasa", pred_vgg)
                with right:
                    st.metric("Prawdopodobieństwo", "{0:.0f}".format(prob_vgg))

            with cam_col:
                self.show_cam()

    def show_cam(self):
        st.subheader("Mapa aktywacji (sieć ResNet)")
        data = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            get_items=get_image_files,
            item_tfms=Resize(224),
            batch_tfms=[*aug_transforms(size=224), Normalize.from_stats(*imagenet_stats)],
        )
        dls = data.dataloaders(Path(), bs=18)
        x, = first(dls.test_dl([self.img]))
        with Hook(self.learn_interface_resnet.model.cpu()[0]) as hook:
            with torch.no_grad(): output = self.learn_interface_resnet.model.eval()(x.cpu())
            act = hook.stored
        a1 = F.softmax(output, dim=-1)
        a2 = self.learn_interface_resnet.dls.vocab[output.argmax().item()]
        cls = 1
        with HookBwd(self.learn_interface_resnet.model[0]) as hookg:
            with Hook(self.learn_interface_resnet.model[0]) as hook:
                output = self.learn_interface_resnet.model.eval()(x.cpu())
                act = hook.stored
            output[0, cls].backward()
            grad = hookg.stored
        w = grad[0].mean(dim=[1, 2], keepdim=True)
        cam_map = (w * act[0]).sum(0)
        x_dec = TensorImage(dls.train.decode((x,))[0][0])
        a = self.learn_interface_resnet.dls.vocab[output.argmax().item()]
        _, ax = plt.subplots()
        x_dec.show(ctx=ax)
        ax.imshow(cam_map.detach().cpu(), alpha=0.7, extent=(0, 224, 224, 0),
                  interpolation='bilinear', cmap='magma')
        fig = plt.gcf()
        st.pyplot(fig)


def main():
    resnet_model = 'resnet152_nvb_v2_5_cam.pckl'
    vgg_model = 'vgg19_nvb_v2_5.pckl'
    x = Predict(resnet_model, vgg_model)


if __name__ == '__main__':
    main()