from hw_asr.model.deep_speech_2 import DeepSpeech2
from hw_asr.model.baseline_model import BaselineModel
from hw_asr.collate_fn.collate import collate_fn
from hw_asr.datasets import LibrispeechDataset
from hw_asr.utils.parse_config import ConfigParser
from hw_asr.augmentations.wave_augmentations import PitchShift, RoomImpulseResponse, GaussianNoise

model = DeepSpeech2(128, 10)
baseline = BaselineModel(128, 10)
config_parser = ConfigParser.get_test_configs()
ds = LibrispeechDataset(
    "dev-clean", text_encoder=config_parser.get_text_encoder(),
    config_parser=config_parser
)

batch_size = 3
batch = collate_fn([ds[i] for i in range(batch_size)])

out1 = baseline(**batch)
out2 = model(**batch)

pass