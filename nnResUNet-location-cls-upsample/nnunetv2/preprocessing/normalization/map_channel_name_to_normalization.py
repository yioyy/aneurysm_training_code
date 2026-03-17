from typing import Type

from nnunetv2.preprocessing.normalization.default_normalization_schemes import CTNormalization, NoNormalization, \
    ZScoreNormalization, RescaleTo01Normalization, RGBTo01Normalization, ImageNormalization, ADCNormalization, ZScoreBrainNormalization, ZScoreImageNormalization, MRI_custom_normalize_1, CTANormalization, Max995_Min005_Normalization

channel_name_to_normalization_mapping = {
    'CT': CTNormalization,
    'noNorm': NoNormalization,
    'zscore': ZScoreNormalization,
    'rescale_0_1': RescaleTo01Normalization,
    'rgb_to_0_1': RGBTo01Normalization,
    'ADC': ADCNormalization,
    'DWI1000': ZScoreBrainNormalization,
    'T2FLAIR':  ZScoreBrainNormalization,
    'SynthSEG33': NoNormalization,
    'SynthSegDWI': NoNormalization,
    'MRA_BRAIN': NoNormalization,
    'MRI_custom_normalize_1': MRI_custom_normalize_1,
    'CTA': CTANormalization,
    'MultiSeries': Max995_Min005_Normalization,
    'T1post': NoNormalization,
    'T2': NoNormalization
}


def get_normalization_scheme(channel_name: str) -> Type[ImageNormalization]:
    """
    If we find the channel_name in channel_name_to_normalization_mapping return the corresponding normalization. If it is
    not found, use the default (ZScoreNormalization)
    """
    norm_scheme = channel_name_to_normalization_mapping.get(channel_name)
    if norm_scheme is None:
        norm_scheme = ZScoreNormalization
    # print('Using %s for image normalization' % norm_scheme.__name__)
    return norm_scheme
