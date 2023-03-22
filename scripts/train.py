import os
from argparse import ArgumentParser


def eval_model(model_name_list, ckpt_name_list, device):
    train_format = "CUDA_VISIBLE_DEVICES={1} python train.py " \
                      "--output_dir {0} " \
                      "--model_name_or_path  {2} " \
                      "--save_steps 20000 " \
                      "--dataset_name Tevatron/msmarco-passage " \
                      "--fp16 " \
                      "--per_device_train_batch_size 64 " \
                      "--train_n_passages 8 " \
                      "--learning_rate 5e-6 " \
                      "--q_max_len 32 " \
                      "--p_max_len 128 " \
                      "--num_train_epochs 3 " \
                      "--logging_steps 500 " \
                      "--overwrite_output_dir"
    encode_format_1 = "CUDA_VISIBLE_DEVICES={1} python encode.py " \
                      "--output_dir temp " \
                      "--model_name_or_path {0} " \
                      "--fp16 " \
                      "--per_device_eval_batch_size 1024 " \
                      "--p_max_len 128 " \
                      "--dataset_name Tevatron/msmarco-passage-corpus " \
                      "--encoded_save_path {0}/corpus_emb.pkl "
    encode_format_2 = "CUDA_VISIBLE_DEVICES={1} python encode.py " \
                      "--output_dir temp " \
                      "--model_name_or_path {0} " \
                      "--fp16 " \
                      "--per_device_eval_batch_size 1024 " \
                      "--q_max_len 32 " \
                      "--dataset_name Tevatron/msmarco-passage/dev " \
                      "--encoded_save_path {0}/test_emb.pkl " \
                      "--encode_is_qry "
    retrieve_format = "python retrieve.py " \
                      "--query_reps {0}/test_emb.pkl " \
                      "--passage_reps {0}/corpus_emb.pkl " \
                      "--depth 100 " \
                      "--batch_size -1 " \
                      "--save_text " \
                      "--save_ranking_to {0}/test_rank.csv"
    convert_format = "python tevatron/utils/format/convert_result_to_marco.py " \
                     "--input {0}/rank.txt " \
                     "--output {0}/rank.txt.marco "
    eval_format = "python -m pyserini.eval.msmarco_passage_eval " \
                  "msmarco-passage-dev-subset " \
                  "{0}/rank.txt.marco"
    for model_name, ckpt_name in zip(model_name_list, ckpt_name_list):
        os.system(train_format.format(ckpt_name, device, model_name))
        os.system(encode_format_1.format(ckpt_name, device))
        os.system(encode_format_2.format(ckpt_name, device))
        os.system(retrieve_format.format(ckpt_name))
        os.system(convert_format.format(ckpt_name))
    for ckpt_name in ckpt_name_list:
        os.system(eval_format.format(ckpt_name))


def eval_main(args):
    model_name_list = args.model_name.split(' ')
    ckpt_name_list = args.ckpt_name.split(' ')
    device = args.device

    eval_model(model_name_list, ckpt_name_list, device)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--ckpt_name', type=str)
    parser.add_argument('--device', type=int)
    main_args = parser.parse_args()

    eval_main(main_args)
