import argparse

def get_sentence_encodings(sents, output_dir, model_name = 'universal-sentence-encoder'):
    if model_name == 'universal-sentence-encoder':
        import tensorflow as tf
        import tensorflow_hub as hub
        model = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
    elif model_name == 'fasttext':
        import sister
        model = sister.MeanEmbedding(lang="en")

#     count = len(sents)
#     for i in range(0,count, args.batch_size):
#         path_ = f'{output_dir}/sentence_encoding_batch_{int(i/self.batch_size)}.pkl'
#         print('extracting embeddings ', path_)
#         sentence_encoding_batch = [(' '.join(s), model([' '.join(s)])) for s in sents[i:i+self.batch_size]]
#         utils.dump_picklefile(sentence_encoding_batch, path_)

    embeddings = [(s, model([s])) for s in sents]
    return embeddings


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default = 'output', help = 'output directory')
    parser.add_argument('--batch-size', type = int, default = 2048, help = 'batch size')
    parser.add_argument('--input', default = 'input', help = 'input sentences as pickle file')
    args = parser.parse_args()
    
    sents = ['this is a test sentence!', 'this is another test sentence!']
    
    embeddings = get_sentence_encodings(sents, 'output')

    