import argparse

def get_train_test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=5)
    parser.add_argument('--num-epochs-pretrain', type=int, default=5)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--num-visuals', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-dir', type=str, default='save/')
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--run-name', type=str, default='multitask_distilbert')
    parser.add_argument('--model-type', type=str, default='distilbert')
    parser.add_argument('--use_cache', action='store_true') # only use_cache if the argument is present
    parser.add_argument('--do-train', action='store_true')
    parser.add_argument('--do-eval', action='store_true')
    parser.add_argument('--sub-file', type=str, default='')
    parser.add_argument('--visualize-predictions', action='store_true')
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--dim', type=int, default=768)
    parser.add_argument('--num_experts', type=int, default=16)
    parser.add_argument('--hidden_dim', type=int, default=768 * 4)
    parser.add_argument('--back_translate', action='store_true')
    parser.add_argument('--eda', action='store_true')
    parser.add_argument("--num_aug_pretrain", default=4, required=False, type=int, help="number of augmented sentences per original sentence for pretraining")
    parser.add_argument("--num_aug", default=4, required=False, type=int, help="number of augmented sentences per original sentence")
    parser.add_argument("--alpha_sr", default=0.3, required=False, type=float, help="percent of words in each sentence to be replaced by synonyms")
    parser.add_argument("--alpha_ri", default=0, required=False, type=float, help="percent of words in each sentence to be inserted")
    parser.add_argument("--alpha_rs", default=0, required=False, type=float, help="percent of words in each sentence to be swapped")
    parser.add_argument("--alpha_rd", default=0, required=False, type=float, help="percent of words in each sentence to be deleted")
    parser.add_argument("--train_with_ood", default="simple_mix", type=str, help="Method to mix in-domain and OOD data. Options: [no_ood, simple_mix]. For wandb recording purpose")
    parser.add_argument('--freeze_basemodel', action='store_true', help="Freeze the base model weights when training MoE")
    parser.add_argument('--freeze_expert', action='store_true', help="Freeze the experts weights after pretraining")

    args = parser.parse_args()
    return args


DATASET_CONFIG = {
    "train": [
        "datasets/indomain_train/nat_questions",
        "datasets/indomain_train/newsqa",
        "datasets/indomain_train/squad",
        "datasets/oodomain_train/duorc",
        "datasets/oodomain_train/race",
        "datasets/oodomain_train/relation_extraction",
    ],
    "finetune": [
        "datasets/oodomain_train/duorc",
        "datasets/oodomain_train/race",
        "datasets/oodomain_train/relation_extraction",
    ],
    "id_val": [
        "datasets/indomain_val/nat_questions",
        "datasets/indomain_val/newsqa",
        "datasets/indomain_val/squad",
    ],
    "ood_val": [
        "datasets/oodomain_val/duorc",
        "datasets/oodomain_val/race",
        "datasets/oodomain_val/relation_extraction",
    ],
    "test": [
        "datasets/oodomain_test/duorc",
        "datasets/oodomain_test/race",
        "datasets/oodomain_test/relation_extraction",
    ]
}
