"""onnx_cv_graph パッケージ. 全操作クラスをここで公開する."""

import importlib as _importlib

# 数字始まりのサブパッケージは通常の import 構文が使えないため importlib を使用
_elementwise = _importlib.import_module(".01_elementwise", __name__)
_color_space = _importlib.import_module(".02_color_space", __name__)
_conv_filter = _importlib.import_module(".03_conv_filter", __name__)
_morphology = _importlib.import_module(".04_morphology", __name__)
_geometric = _importlib.import_module(".05_geometric", __name__)
_normalization = _importlib.import_module(".06_normalization", __name__)
_blend = _importlib.import_module(".07_blend", __name__)
_threshold = _importlib.import_module(".08_threshold", __name__)
_feature = _importlib.import_module(".09_feature", __name__)
_ml_preprocess = _importlib.import_module(".10_ml_preprocess", __name__)

# 公開クラスの re-export
AutoLevelsOp = _elementwise.AutoLevelsOp
BinarizeOp = _elementwise.BinarizeOp
BrightnessOp = _elementwise.BrightnessOp
ContrastOp = _elementwise.ContrastOp
ExposureOp = _elementwise.ExposureOp
GammaOp = _elementwise.GammaOp
GrayscaleOp = _elementwise.GrayscaleOp
InvertOp = _elementwise.InvertOp
LevelsOp = _elementwise.LevelsOp
PosterizeOp = _elementwise.PosterizeOp
SaturationOp = _elementwise.SaturationOp
SolarizeOp = _elementwise.SolarizeOp

ChannelExtractOp = _color_space.ChannelExtractOp
ColorSuppressOp = _color_space.ColorSuppressOp
ColorTemperatureOp = _color_space.ColorTemperatureOp
ColormapOp = _color_space.ColormapOp
HsvExtractOp = _color_space.HsvExtractOp
HsvRangeOp = _color_space.HsvRangeOp
Rgb2BgrOp = _color_space.Rgb2BgrOp
SepiaOp = _color_space.SepiaOp
WbGainOp = _color_space.WbGainOp
WbGrayWorldOp = _color_space.WbGrayWorldOp
WbWhitePatchOp = _color_space.WbWhitePatchOp

BgNormalizeOp = _conv_filter.BgNormalizeOp
KuwaharaOp = _conv_filter.KuwaharaOp
XDoGOp = _conv_filter.XDoGOp
BlurOp = _conv_filter.BlurOp
DogOp = _conv_filter.DogOp
EdgeMagnitudeOp = _conv_filter.EdgeMagnitudeOp
EmbossOp = _conv_filter.EmbossOp
GaussianBlurOp = _conv_filter.GaussianBlurOp
LaplacianOp = _conv_filter.LaplacianOp
LogFilterOp = _conv_filter.LogFilterOp
PrewittOp = _conv_filter.PrewittOp
ScharrOp = _conv_filter.ScharrOp
SharpenOp = _conv_filter.SharpenOp
SobelOp = _conv_filter.SobelOp
UnsharpMaskOp = _conv_filter.UnsharpMaskOp

BlackHatOp = _morphology.BlackHatOp
ClosingOp = _morphology.ClosingOp
DilateOp = _morphology.DilateOp
ErodeOp = _morphology.ErodeOp
GradientOp = _morphology.GradientOp
HitMissOp = _morphology.HitMissOp
OpeningOp = _morphology.OpeningOp
TopHatOp = _morphology.TopHatOp

AffineOp = _geometric.AffineOp
CenterCropOp = _geometric.CenterCropOp
CropOp = _geometric.CropOp
HFlipOp = _geometric.HFlipOp
HVFlipOp = _geometric.HVFlipOp
PaddingReflectOp = _geometric.PaddingReflectOp
PaddingColorOp = _geometric.PaddingColorOp
PerspectiveOp = _geometric.PerspectiveOp
PyrDownOp = _geometric.PyrDownOp
PyrUpOp = _geometric.PyrUpOp
VFlipOp = _geometric.VFlipOp
ResizeOp = _geometric.ResizeOp
ResizeToOp = _geometric.ResizeToOp
Rotate90Op = _geometric.Rotate90Op
Rotate180Op = _geometric.Rotate180Op
Rotate270Op = _geometric.Rotate270Op
Rotate3dOp = _geometric.Rotate3dOp
Rotate3dPadColorOp = _geometric.Rotate3dPadColorOp
RotateArbitraryOp = _geometric.RotateArbitraryOp

L1NormOp = _normalization.L1NormOp
L1NormChOp = _normalization.L1NormChOp
L2NormOp = _normalization.L2NormOp
L2NormChOp = _normalization.L2NormChOp
LcnOp = _normalization.LcnOp
MinMaxNormOp = _normalization.MinMaxNormOp

AlphaBlendOp = _blend.AlphaBlendOp
MaskCompositeOp = _blend.MaskCompositeOp
OverlayOp = _blend.OverlayOp
WeightedAddOp = _blend.WeightedAddOp

AdaptiveThreshGaussianOp = _threshold.AdaptiveThreshGaussianOp
AdaptiveThreshMeanOp = _threshold.AdaptiveThreshMeanOp
InrangeOp = _threshold.InrangeOp
InvBinarizeOp = _threshold.InvBinarizeOp
SauvolaOp = _threshold.SauvolaOp
ThreshTruncOp = _threshold.ThreshTruncOp
ThreshTozeroOp = _threshold.ThreshTozeroOp
ThreshTozeroInvOp = _threshold.ThreshTozeroInvOp

HarrisCornerOp = _feature.HarrisCornerOp
LineExtractOp = _feature.LineExtractOp
ShiTomasiOp = _feature.ShiTomasiOp

BatchSqueezeNchwOp = _ml_preprocess.BatchSqueezeNchwOp
BatchSqueezeNhwcOp = _ml_preprocess.BatchSqueezeNhwcOp
BatchUnsqueezeNchwOp = _ml_preprocess.BatchUnsqueezeNchwOp
BatchUnsqueezeNhwcOp = _ml_preprocess.BatchUnsqueezeNhwcOp
ChannelMeanSubOp = _ml_preprocess.ChannelMeanSubOp
ChwToHwcOp = _ml_preprocess.ChwToHwcOp
FloatToUint8Op = _ml_preprocess.FloatToUint8Op
HwcToChwOp = _ml_preprocess.HwcToChwOp
ImageNetNormOp = _ml_preprocess.ImageNetNormOp
LetterboxOp = _ml_preprocess.LetterboxOp
NormalizeNeg1Pos1Op = _ml_preprocess.NormalizeNeg1Pos1Op
PixelMeanSubOp = _ml_preprocess.PixelMeanSubOp
ScaleFrom255Op = _ml_preprocess.ScaleFrom255Op
ScaleTo255Op = _ml_preprocess.ScaleTo255Op
Uint8ToFloatOp = _ml_preprocess.Uint8ToFloatOp

__all__ = [
    "AutoLevelsOp",
    "BinarizeOp", "BrightnessOp", "ContrastOp", "ExposureOp", "GammaOp",
    "GrayscaleOp", "InvertOp", "LevelsOp", "PosterizeOp", "SaturationOp", "SolarizeOp",
    "ChannelExtractOp", "ColorSuppressOp", "ColorTemperatureOp", "ColormapOp",
    "HsvExtractOp", "HsvRangeOp", "Rgb2BgrOp", "SepiaOp",
    "WbGainOp", "WbGrayWorldOp", "WbWhitePatchOp",
    "BgNormalizeOp", "BlurOp", "DogOp", "EdgeMagnitudeOp", "EmbossOp", "GaussianBlurOp",
    "KuwaharaOp", "XDoGOp",
    "LaplacianOp", "LogFilterOp", "PrewittOp", "ScharrOp",
    "SharpenOp", "SobelOp", "UnsharpMaskOp",
    "BlackHatOp", "ClosingOp", "DilateOp", "ErodeOp",
    "GradientOp", "HitMissOp", "OpeningOp", "TopHatOp",
    "AffineOp", "CenterCropOp", "CropOp", "HFlipOp", "HVFlipOp", "PaddingReflectOp", "PaddingColorOp",
    "PerspectiveOp", "PyrDownOp", "PyrUpOp", "VFlipOp",
    "ResizeOp", "ResizeToOp",
    "Rotate3dOp", "Rotate3dPadColorOp", "Rotate90Op", "Rotate180Op", "Rotate270Op", "RotateArbitraryOp",
    "L1NormOp", "L1NormChOp", "L2NormOp", "L2NormChOp", "LcnOp", "MinMaxNormOp",
    "AlphaBlendOp", "MaskCompositeOp", "OverlayOp", "WeightedAddOp",
    "AdaptiveThreshGaussianOp", "AdaptiveThreshMeanOp", "InrangeOp",
    "InvBinarizeOp", "SauvolaOp", "ThreshTruncOp", "ThreshTozeroOp", "ThreshTozeroInvOp",
    "HarrisCornerOp", "LineExtractOp", "ShiTomasiOp",
    "BatchSqueezeNchwOp", "BatchSqueezeNhwcOp",
    "BatchUnsqueezeNchwOp", "BatchUnsqueezeNhwcOp",
    "ChannelMeanSubOp", "ChwToHwcOp", "FloatToUint8Op", "HwcToChwOp",
    "ImageNetNormOp", "LetterboxOp", "NormalizeNeg1Pos1Op",
    "PixelMeanSubOp", "ScaleFrom255Op", "ScaleTo255Op", "Uint8ToFloatOp",
    "CATEGORIES",
]

# カテゴリ定義: UI の折りたたみグループに使用
# export_all.py が models_meta.json にも出力する
CATEGORIES = [
    {
        "id": "01_elementwise",
        "label_ja": "ピクセル単位演算",
        "label_en": "Element-wise",
        "ops": [GrayscaleOp, BinarizeOp, BrightnessOp, ContrastOp, GammaOp,
                LevelsOp, InvertOp, SolarizeOp, PosterizeOp, SaturationOp, ExposureOp,
                AutoLevelsOp],
    },
    {
        "id": "02_color_space",
        "label_ja": "色空間変換",
        "label_en": "Color Space",
        "ops": [SepiaOp, Rgb2BgrOp, ColorTemperatureOp, ColormapOp, ChannelExtractOp,
                HsvExtractOp, HsvRangeOp, ColorSuppressOp,
                WbGainOp, WbGrayWorldOp, WbWhitePatchOp],
    },
    {
        "id": "03_conv_filter",
        "label_ja": "畳み込みフィルタ",
        "label_en": "Conv Filters",
        "ops": [BlurOp, GaussianBlurOp, SharpenOp, EmbossOp,
                SobelOp, ScharrOp, LaplacianOp, PrewittOp,
                UnsharpMaskOp, EdgeMagnitudeOp, LogFilterOp, BgNormalizeOp, DogOp,
                KuwaharaOp, XDoGOp],
    },
    {
        "id": "04_morphology",
        "label_ja": "モルフォロジー演算",
        "label_en": "Morphology",
        "ops": [DilateOp, ErodeOp, OpeningOp, ClosingOp,
                GradientOp, TopHatOp, BlackHatOp, HitMissOp],
    },
    {
        "id": "05_geometric",
        "label_ja": "幾何変換",
        "label_en": "Geometric",
        "ops": [ResizeOp, ResizeToOp, HFlipOp, VFlipOp, HVFlipOp,
                Rotate90Op, Rotate180Op, Rotate270Op, RotateArbitraryOp,
                CropOp, CenterCropOp, PaddingReflectOp, PaddingColorOp, PyrDownOp, PyrUpOp,
                AffineOp, PerspectiveOp, Rotate3dOp, Rotate3dPadColorOp],
    },
    {
        "id": "06_normalization",
        "label_ja": "正規化・統計",
        "label_en": "Normalization",
        "ops": [MinMaxNormOp, L2NormOp, L1NormOp, L2NormChOp, L1NormChOp, LcnOp],
    },
    {
        "id": "08_threshold",
        "label_ja": "閾値処理",
        "label_en": "Threshold",
        "ops": [InvBinarizeOp, ThreshTruncOp, ThreshTozeroOp, ThreshTozeroInvOp,
                AdaptiveThreshMeanOp, AdaptiveThreshGaussianOp, InrangeOp, SauvolaOp],
    },
    {
        "id": "09_feature",
        "label_ja": "特徴量・コーナー検出",
        "label_en": "Feature Detection",
        "ops": [HarrisCornerOp, ShiTomasiOp, LineExtractOp],
    },
    {
        "id": "10_ml_preprocess",
        "label_ja": "ML 前処理",
        "label_en": "ML Preprocessing",
        "ops": [ScaleTo255Op, ScaleFrom255Op, NormalizeNeg1Pos1Op,
                ImageNetNormOp, ChannelMeanSubOp, PixelMeanSubOp,
                HwcToChwOp, ChwToHwcOp,
                BatchUnsqueezeNchwOp, BatchUnsqueezeNhwcOp,
                BatchSqueezeNchwOp, BatchSqueezeNhwcOp,
                FloatToUint8Op, Uint8ToFloatOp, LetterboxOp],
        "hidden": True,
    },
    {
        "id": "07_blend",
        "label_ja": "ブレンド・合成",
        "label_en": "Blend / Composite",
        "ops": [WeightedAddOp, AlphaBlendOp, MaskCompositeOp, OverlayOp],
        "hidden": True,
    },
]
