from pytest import fixture

from readable_implementations.tokenizers.trainer import TrainerBPE, create_text_char_dicts_from_files, create_line_reader

# TODO: add tempfile

@fixture()
def real_file_path():
    path = "../notebooks/data/bel_news_2020_10K/bel_news_2020_10K-sentences.txt"
    return path


def test_create_line_reader(real_file_path):
    reader = create_line_reader(real_file_path)

def test_create_freq_dict_from_files(real_file_path):
    line_reader = create_line_reader(real_file_path)
    text_vocab = create_text_char_dicts_from_files(line_reader)
    assert len(text_vocab) > 0

class TestTrainerBPE:
    def test_happy_path(self, real_file_path):
        trainer = TrainerBPE(
            files_path=real_file_path,
            vocab_size=10000,
            special_tokens_list=[]
        )
        trainer.fit()