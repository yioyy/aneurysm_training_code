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


_WARNED_CHANNEL_NAMES = set()


def get_normalization_scheme(channel_name: str) -> Type[ImageNormalization]:
    """
    If we find the channel_name in channel_name_to_normalization_mapping return the corresponding normalization. If it is
    not found, use the default (ZScoreNormalization)

    ⚠ 查不到就靜默套 ZScore 是個危險的預設 —— dataset.json 把 channel 名稱寫成描述性的
    字串（例如 'MRA' 而非 'MRA_BRAIN' 或 'noNorm'）時，整個資料集會被額外做一次 z-score，
    而且不會有任何錯誤訊息。實際發生過：Dataset082 寫 'MRA' → ZScore，
    Dataset080/083 寫 'noNorm' → NoNormalization，同一批影像因此前處理不同、模型不可比。
    故此處在退回預設時發出警告，並提示最接近的合法名稱。
    """
    norm_scheme = channel_name_to_normalization_mapping.get(channel_name)
    if norm_scheme is None:
        norm_scheme = ZScoreNormalization
        if channel_name not in _WARNED_CHANNEL_NAMES:
            _WARNED_CHANNEL_NAMES.add(channel_name)
            import difflib
            import warnings
            close = difflib.get_close_matches(
                str(channel_name), list(channel_name_to_normalization_mapping), n=3, cutoff=0.5)
            warnings.warn(
                f"[nnUNet] dataset.json 的 channel_name {channel_name!r} 不在對照表中，"
                f"已退回預設的 ZScoreNormalization。若本意是不要正規化，請改寫 'noNorm'。"
                + (f" 最接近的合法名稱：{close}。" if close else "")
                + f" 合法名稱：{sorted(channel_name_to_normalization_mapping)}",
                stacklevel=2)
    # print('Using %s for image normalization' % norm_scheme.__name__)
    return norm_scheme
