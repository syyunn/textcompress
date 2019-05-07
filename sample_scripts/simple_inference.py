import torch

import src.datasets.data as data
import src.utils.conf as conf
import src.utils.devices as devices
import src.runners.inference as inference


def run_inference(model_path,
                  data_path,
                  beam_k=None,
                  output_path=None):
    loaded_model = torch.load(
        model_path
    )

    config = loaded_model["config"]
    model = loaded_model["model"]
    env = conf.EnvConfiguration.from_env()
    device = devices.device_from_env(env)

    word_embeddings, dictionary = data.resolve_embeddings_and_dictionary(
        data_vocab_path=env.data_dict[config.dataset_name]["vocab_path"],
        max_vocab=config.max_vocab,
        vector_cache_path=env.vector_cache_path,
        vector_file_name=config.vector_file_name,
        device=device,
        num_oov=config.num_oov,
        verbose=False
    )
    word_embeddings.learned_embeddings = model.learned_embeddings
    corpus = data.resolve_corpus(
        data_path=data_path,
        max_sentence_length=200,
    )
    inf = inference.Inference(
        model=model,
        dictionary=dictionary,
        word_embeddings=word_embeddings,
        device=device,
        config=config,
        beam_k=beam_k,
    )

    if output_path is None:
        def write(string):
            print("output_path is none! just print string")
            print(string)
            pass
    else:
        f = open(output_path, "w")

        def write(string):
            f.write(string + "\n")

    for translations, all_logprobs, sent_batch in \
            inf.corpus_inference(corpus, lambda _: _//2 + 1, batch_size=16):
        oov_dicts = dictionary.get_oov_dicts(sent_batch)

        batch_count = 0
        for line in dictionary.ids2sentences(
                translations, oov_dicts, oov_fallback=True):
            write("[input] : " + sent_batch[batch_count])
            batch_count += 1
            if batch_count == 15:
                batch_count = 0
            write("[infrc] : " + line)


if __name__ == "__main__":
    model_path = "/home/zachary/hdd/usc_dae/saves/pc_4_lstm_nli_2g_2019.05.04.21.08.55.251_080000.p"
    data_path = "/home/zachary/hdd/nlp/sumdata/train/valid.article.filter.txt"
    output_path = "/home/zachary/hdd/usc_dae/infrc/result.txt"
    import argparse
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--model-path',
                        default=model_path)
    parser.add_argument('--data-path',
                        default=data_path)
    parser.add_argument('--beam-k',
                        default=None,
                        type=int)
    parser.add_argument('--output-path',
                        default=output_path)
    args = parser.parse_args()

    run_inference(
        model_path=args.model_path,
        data_path=args.data_path,
        beam_k=args.beam_k,
        output_path=args.output_path,
    )

"""
python sample_scripts/simple_inference.py \
  --model-path {model_path} \
  --data-path {data_path}
"""
