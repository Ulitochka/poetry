from gensim.models import KeyedVectors
from tqdm import tqdm

from linguistic_style_transfer.linguistic_style_transfer_model.utils import data_processor

data_path = '/home/mdomrachev/work_rep/content_vs_style_problem/linguistic_style_transfer/dataset_forming/data_set_rew/'
w2v_vectors = KeyedVectors.load_word2vec_format("/home/mdomrachev/Data/cc.ru.300.vec", binary=False)

for data in [(data_path,'train.txt'), (data_path, 'test.txt'), (data_path, 'valid.txt')]:
    [word_index, x, _, _, _] = data_processor.get_text_sequences(
        data[0] + data[1],
        40000,
        None)

    vocabulary_index_sorted = sorted([w for w in word_index])
    print(len(vocabulary_index_sorted))

    vectors = dict()
    cover_voc = 0
    for t in vocabulary_index_sorted:
        try:
            vectors[t] = w2v_vectors[t]
            cover_voc += 1
        except KeyError:
            pass
    print('voc size: %s; cover_voc: %s' % (len(vocabulary_index_sorted), cover_voc), '\n')

    with open("/home/mdomrachev/work_rep/content_vs_style_problem/linguistic_style_transfer/w2v_models/%s_ft_300.txt" % (data[1].split('.')[0],), "wt") as outfile:
        for word, vec in tqdm(vectors.items()):
            vec = " ".join([str(i) for i in vec])
            wordvec = " ".join([word, vec])
            outfile.write(wordvec + "\n")