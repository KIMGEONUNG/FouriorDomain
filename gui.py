#!/usr/bin/env python
import gradio as gr
from glob import glob
import numpy as np


class GUI(object):
    """docstring for GUI."""

    def __init__(self):
        super(GUI, self).__init__()
        self.ft = None

        self._init_gui_components()

    def _init_gui_components(self, height=300):

        css = r"img { image-rendering: pixelated; }"
        with gr.Blocks(css=css) as self.demo:
            with gr.Box(), gr.Row():
                view_gt = gr.Image(label="GT",
                                   interactive=False).style(height=height)
                view_gray = gr.Image(label="Gray",
                                     interactive=False).style(height=height)
                view_fourier = gr.Image(label="Fourier",
                                        interactive=False).style(height=height)
            with gr.Box(), gr.Row():
                mask = gr.ImageMask(label="Mask").style(height=height)
                with gr.Column():
                    view_recon = gr.Image(
                        label="Reon", interactive=False).style(height=height)
                    btn_recon = gr.Button("Recon")

            gr.Examples(sorted(glob("assets/*")), inputs=view_gt)

            view_gt.change(self._togray, inputs=view_gt, outputs=view_gray)
            view_gray.change(self._tofourier,
                             inputs=view_gray,
                             outputs=view_fourier)
            view_fourier.change(lambda x: x, inputs=view_fourier, outputs=mask)
            btn_recon.click(self._fromfourier, inputs=mask, outputs=view_recon)

    def launch(self):
        self.demo.launch()

    def _fromfourier(self, mask):
        if self.ft is None or mask is None:
            return None

        mask = 1 - mask["mask"][..., 0] / 255

        x = np.fft.ifft2(np.fft.ifftshift(self.ft * mask))
        x = x.clip(0, 255)
        x = x.astype('uint8')
        return x

    def _tofourier(self, x):
        x = x[..., 0]
        x = np.fft.fftshift(np.fft.fft2(x))
        self.ft = x
        x = np.log(abs(x))
        x = x / x.max()
        x = (x * 255).astype('uint8')
        x = self._repeat3(x)
        return x

    def _repeat3(self, x):
        if len(x.shape) == 2:
            x = x[..., None]
        x = np.tile(x, (1, 1, 3))
        return x

    def _togray(self, x):
        if x is None:
            return None
        g = x.astype('float') / 255
        g = 0.299 * g[:, :, 0] + 0.587 * g[:, :, 1] + 0.114 * g[:, :, 2]
        g = self._repeat3(g)
        return g


if __name__ == "__main__":
    gui = GUI()
    gui.launch()
