#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 10:23:09 2021

@author: lingjia
"""

from abc import ABC
from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import hsv_to_rgb
from matplotlib.colors import rgb_to_hsv
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from decode.generic import emitter

class Renderer(ABC):
    def __init__(self, plot_axis: tuple, xextent: tuple, yextent: tuple, zextent: tuple,
                 px_size: float, abs_clip: float, rel_clip: float, contrast: float):
        """Renderer. Takes emitters and outputs a rendered image."""
        super().__init__()

        self.xextent = xextent
        self.yextent = yextent
        self.zextent = zextent

        self.px_size = px_size
        self.plot_axis = plot_axis

        self.abs_clip = abs_clip
        self.rel_clip = rel_clip

        self.contrast = contrast

        assert (
                self.abs_clip is None or self.rel_clip is None
        ), "Define either an absolute or a relative value for clipping, but not both"

    def forward(self, em: emitter.EmitterSet) -> torch.Tensor:
        """
        Forward emitterset through rendering and output rendered data.
        Args:
            em: emitter set
        """
        raise NotImplementedError

    def render(self, em: emitter.EmitterSet, ax=None):
        """
        Render emitters
        Args:
            em: emitter set
            ax: plot axis
        """
        raise NotImplementedError


class Renderer2D_v2(Renderer):
    def __init__(self, px_size, sigma_blur, plot_axis=(0, 1, 2),
                 xextent=None, yextent=None, zextent=None, colextent=None,
                 abs_clip=None, rel_clip=None, contrast=1):
        """
        2D histogram renderer with constant gaussian blur.
        Args:
            px_size: pixel size of the output image in nm
            sigma_blur: sigma of the gaussian blur applied in nm
            plot_axis: determines which dimensions get plotted. 0,1,2 = x,y,z. (0,1) is x over y.
            xextent: extent in x in nm
            yextent: extent in y in nm
            zextent: extent in z in nm.
            cextent: extent of the color variable. Values outside of this range get clipped.
            abs_clip: absolute clipping value of the histogram in counts
            rel_clip: clipping value relative to the maximum count. i.e. rel_clip = 0.8 clips at 0.8*hist.max()
            contrast: scaling factor to increase contrast
        """
        super().__init__(
            plot_axis=plot_axis,
            xextent=xextent,
            yextent=yextent,
            zextent=zextent,
            px_size=px_size,
            abs_clip=abs_clip,
            rel_clip=rel_clip,
            contrast=contrast,
        )

        self.sigma_blur = sigma_blur
        self.colextent = colextent

        self.jet_hue = self._get_jet_cmap()

    def render(self, em: torch.Tensor, col_vec=None, ax=None):
        """
        Forward emitterset through rendering and output rendered data.
        Args:
            em: emitter set
            col_vec: torch tensor (1 dim) with the same length as em
            ax: plot axis
        """
        hist = self.forward(em, col_vec).numpy()

        if ax is None:
            ax = plt.gca()

        if col_vec is not None:

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size=0.25, pad=-0.25)
            colb = mpl.colorbar.ColorbarBase(
                cax, cmap=plt.get_cmap("jet"), values=np.linspace(0, 1.0, 101),
                norm=mpl.colors.Normalize(0.0, 1.0)
            )
            colb.outline.set_visible(False)

            cax.text(
                0.12, 0.04, f"{self.colextent[0]}", rotation=90, color="white", fontsize=15,
                transform=cax.transAxes
            )
            cax.text(
                0.12, 0.88, f"{self.colextent[1]}", rotation=90, color="white", fontsize=15,
                transform=cax.transAxes
            )
            cax.axis("off")

            ax.imshow(np.transpose(hist, [1, 0, 2]))

        else:

            # because imshow use different ordering
            ax.imshow(np.transpose(hist), cmap="gray")

        return ax,np.transpose(hist, [1, 0, 2])

    def forward(self, em: torch.Tensor, col_vec=None) -> torch.Tensor:
        """
        Forward emitterset through rendering and output rendered data.
        Args:
            em: emitter set
            col_vec: torch tensor (1 dim) with the same length as em
        """

        xyz_extent = self.get_extent(em)
        ind_mask = (
                  (em[:, 0] >= xyz_extent[0][0])
                * (em[:, 0] <= xyz_extent[0][1])
                * (em[:, 1] >= xyz_extent[1][0])
                * (em[:, 1] <= xyz_extent[1][1])
                * (em[:, 2] >= xyz_extent[2][0])
                * (em[:, 2] <= xyz_extent[2][1]))

        em_sub = em[ind_mask]

        if col_vec is not None:

            col_vec = col_vec[ind_mask]
            self.colextent = (
            col_vec.min(), col_vec.max()) if self.colextent is None else self.colextent
            int_hist, col_hist = self._hist2d(
                em_sub, col_vec, xyz_extent[self.plot_axis[0]], xyz_extent[self.plot_axis[1]],
                self.colextent
            )

            with np.errstate(divide="ignore", invalid="ignore"):
                c_avg = col_hist / int_hist

            if self.rel_clip is not None:
                int_hist = np.clip(int_hist * self.contrast, 0.0, int_hist.max() * self.rel_clip)
                val = int_hist / int_hist.max()
            elif self.abs_clip is not None:
                int_hist = np.clip(int_hist, 0.0, self.abs_clip)
                val = int_hist / self.abs_clip
            else:
                val = int_hist / int_hist.max()

            val *= self.contrast

            c_avg[np.isnan(c_avg)] = 0
            sat = np.ones(int_hist.shape)
            hue = np.interp(c_avg, np.linspace(0, 1, 256), self.jet_hue)
            # hue = np.ones(int_hist.shape)
            # aa = hue<0.3
            # val = aa.astype(int).astype(float)
            # val = np.ones(int_hist.shape)

            HSV = np.concatenate((hue[:, :, None], sat[:, :, None], val[:, :, None]), -1)
            RGB = hsv_to_rgb(HSV)

            if self.sigma_blur:
                RGB = np.array(
                    [
                        gaussian_filter(
                            RGB[:, :, i],
                            sigma=[self.sigma_blur / self.px_size, self.sigma_blur / self.px_size]
                        )
                        for i in range(3)
                    ]
                ).transpose(1, 2, 0)

            RGB = np.clip(RGB, 0, 1)
            return torch.from_numpy(RGB)

        else:

            hist = self._hist2d(em_sub, None, xyz_extent[self.plot_axis[0]],
                                xyz_extent[self.plot_axis[1]])

            if self.rel_clip is not None:
                hist = np.clip(hist, 0.0, hist.max() * self.rel_clip)
            if self.abs_clip is not None:
                hist = np.clip(hist, 0.0, self.abs_clip)

            if self.sigma_blur is not None:
                hist = gaussian_filter(hist, sigma=[self.sigma_blur / self.px_size,
                                                    self.sigma_blur / self.px_size])

            hist = np.clip(hist, 0, hist.max() / self.contrast)
            return torch.from_numpy(hist)

    def get_extent(self, em) -> Tuple[tuple, tuple, tuple]:

        xextent = (
        em[:, 0].min(), em[:, 0].max()) if self.xextent is None else self.xextent
        yextent = (
        em[:, 1].min(), em[:, 1].max()) if self.yextent is None else self.yextent
        zextent = (
        em[:, 2].min(), em[:, 2].max()) if self.zextent is None else self.zextent

        return xextent, yextent, zextent

    def _hist2d(self, em: torch.Tensor, col_vec, x_hist_ext, y_hist_ext, c_range=None):

        xy = em[:, self.plot_axis].numpy()

        hist_bins_x = np.arange(x_hist_ext[0], x_hist_ext[1] + self.px_size, self.px_size)
        hist_bins_y = np.arange(y_hist_ext[0], y_hist_ext[1] + self.px_size, self.px_size)

        int_hist, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=(hist_bins_x, hist_bins_y))

        if col_vec is not None:

            c_pos = np.clip(col_vec, c_range[0], c_range[1])
            c_weight = (c_pos - c_range[0]) / (c_range[1] - c_range[0])

            col_hist, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=(hist_bins_x, hist_bins_y),
                                            weights=c_weight)

            return int_hist, col_hist

        else:

            return int_hist

    @staticmethod
    def _get_jet_cmap():
        lin_hue = np.linspace(0, 1, 256)
        cmap = plt.get_cmap("jet", lut=256)
        cmap = cmap(lin_hue)
        cmap_hsv = rgb_to_hsv(cmap[:, :3])
        jet_hue = cmap_hsv[:, 0]
        _, b = np.unique(jet_hue, return_index=True)
        jet_hue = [jet_hue[index] for index in sorted(b)]
        jet_hue = np.interp(np.linspace(0, len(jet_hue), 256), np.arange(len(jet_hue)), jet_hue)
        return jet_hue
