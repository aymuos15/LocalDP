# Torch only Simple UNet

This is a repo which allows one to understand UNet in the easiest and simplest form using only torch (as that is the way I personally like it!)

## Motivation

Most of the current work of my Ph.D. is on medical image segmentation. In that, UNet is a fairly common baseline. I initially struggled to search for a repo which helped me quickly understand the architecture with a fairly simple data loader pipeline. I then came across https://github.com/usuyama/pytorch-unet.

Inspired by that, I've created (what I believe) to be even simpler and quite easy to enter.

## To Run
1. Clone The repo
2. `cd` into the directory
3. `python main.py`

## To-Do's

- [x]  Make it torch only.
- [ ]  Adapt it to multi-class segmentation.
- [ ]  Add other losses.
- [ ]  Add a real world dataset. (Maybe: Medical Segmentation Decathlon (MSD))
- [ ]  Add more variants. Example Res-U-Net.

## Contribution:

Code -- Me: 80%, LLMs: 20%

Documentation -- Me: 1%, LLMs 99%.
