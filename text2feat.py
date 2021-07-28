import argparse
import utilities as utils

def load_model(model_name='universal-sentence-encoder'):
    if model_name == 'universal-sentence-encoder':
        import tensorflow as tf
        import tensorflow_hub as hub
        model = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
    elif model_name == 'fasttext':
        import sister
        model = sister.MeanEmbedding(lang="en")
    return model
        
def get_sentence_encodings(sents, output_dir, model_name='universal-sentence-encoder', split_output=False):
    '''
    Input Parameters:
        sents: a list of sentences 
        output_dir: directory to write output embeddings 
        model_name: choose between universal sentence encoder and fasttext
        split_output: if True, splits the output into smaller chunks. Useful when dealing with a large number of sentences
    Output:
        write embedings to output_dir. If split_output is False, also returns a list of embeddings.
    '''
    model = load_model('universal-sentence-encoder')

    embeddings = []
    if split_output:
        count = len(sents)
        chunk_size = 10000
        for i in range(0,count, chunk_size):
            sentence_encoding_batch = model(sents[i:i+chunk_size])
            # embeddings.extend(sentence_encoding_batch)
            
            # write output
            path_ = f'{output_dir}/sentence_encoding_batch_{int(i/chunk_size)}.pkl'
            utils.dump_picklefile(sentence_encoding_batch, path_)
    else:
        embeddings = model(sents)
        
        # write output
        path_ = f'{output_dir}/sentence_encoding_all.pkl'
        utils.dump_picklefile(embeddings, path_)
    
    return embeddings

def get_msrvtt_captions(annotations_path):
    import json
    with open(annotations_path) as f:
            anno = json.load(f)

    sents = []
    for item in anno['sentences']:
            sents.append(item['caption'])
    return sents

if __name__ == '__main__':
    # simple example
    # sents = ['this is a test sentence!', 'this is another test sentence!']
    # embeddings = get_sentence_encodings(sents, 'output')
    # print(embeddings.shape) # (2, 512)

    # extract embeddings for msrvtt captions    
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default = 'output', help = 'output directory')
    parser.add_argument('--input-path', 
                        default = '../datasets/MSRVTT/train_val_annotation/train_val_videodatainfo.json', 
                        help = 'dataset annotations file contatining sentences. example: msrvtt')
    args = parser.parse_args()
    
    # load captions from msrvtt
    captions = get_msrvtt_captions(args.input_path)
    
    # create output dir
    utils.create_dir_if_not_exist(args.output_dir)
    
    # obtain embeddings and store the output dir
    embeddings = get_sentence_encodings(captions, 
                                        args.output_dir, 
                                        model_name='universal-sentence-encoder', 
                                        split_output=False)
