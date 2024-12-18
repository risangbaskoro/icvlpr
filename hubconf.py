"""
Load Model:
    model = torch.hub.load('risangbaskoro/icvlpr', 'lprnet')

Decoder API:
    decoder = torch.hub.load('risangbaskoro/icvlpr', 'decoder', decoder='greedy')
    decoder = torch.hub.load('risangbaskoro/icvlpr', 'decoder', decoder='beam', beam_width=5)


Converter API:
    converter = torch.hub.load('risangbaskoro/icvlpr', 'converter')
"""

import torch
from model import LPRNet, SpatialTransformerLayer, LocNet


dependencies = ["torch"]


def lprnet(pretrained: bool = True):
    locnet = LocNet()
    stn = SpatialTransformerLayer(localization=locnet, align_corners=True)

    model = LPRNet(stn=stn)

    if pretrained:
        url = "https://data.risangbaskoro.com/icvlp/models/epoch_811.pth"
        model.load_state_dict(
            torch.hub.load_state_dict_from_url(
                url, map_location="cpu", progress=True)
        )

    return model


def dataset(*args, **kwargs):
    from dataset import ICVLPDataset

    return ICVLPDataset(*args, **kwargs)


def decoder(decoder: str = "greedy", beam_width: int = 5):
    assert decoder in [
        "greedy",
        "beam",
    ], f"Decoder must either 'greedy' or 'beam'. Got {decoder}"

    if decoder == "greedy":
        from decoder import GreedyCTCDecoder

        ret = GreedyCTCDecoder()

    elif decoder == "beam":
        from decoder import BeamCTCDecoder

        ret = BeamCTCDecoder(beam_width=beam_width)

    return ret


def converter():
    from utils import Converter

    return Converter()


def rln(blank=0):
    from metrics import LetterNumberRecognitionRate

    return LetterNumberRecognitionRate(blank=blank)
