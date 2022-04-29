import numpy as np
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig


def load_model(model, ngpus, load_path):
    config = AutoConfig.from_pretrained(model, num_labels=1)
    # this assert is to silence Pylance (which infers that .from_pretrained is
    # of type Optional)
    assert AutoModelForSequenceClassification.from_pretrained is not None
    model = AutoModelForSequenceClassification.from_pretrained(model, config=config)

    # Use strict=False since the provided models were originally trained with an earlier version of Huggingface
    model.load_state_dict(torch.load(load_path), strict=False)
    if ngpus > 0:
        model = model.cuda()
        model = torch.nn.DataParallel(model, device_ids=[i for i in range(ngpus)])
    return model


def get_ids_mask(sentences, tokenizer, max_length, truncate_prefix):
    tokenized = [tokenizer.tokenize(s) for s in sentences]
    if truncate_prefix:
        # we truncate by removing the prefix so that total length is <=
        # max_length -1, then add a "SEP" token
        assert max_length > 1
        tokenized = [t[- (max_length - 1):] + ["SEP"] for t in tokenized]
    else:
        # we truncate by removing the suffix instead
        tokenized = [t[: (max_length - 1)] + ["SEP"] for t in tokenized]
    # assert that all tokenized sequences have length <= max_length
    assert all(len(t) <= max_length for t in tokenized)

    ids = [tokenizer.convert_tokens_to_ids(t) for t in tokenized]
    ids = np.array([np.pad(i, (0, max_length - len(i)), mode="constant") for i in ids])
    amasks = []
    for seq in ids:
        seq_mask = [float(i > 0) for i in seq]
        amasks.append(seq_mask)
    return ids, amasks


def load_process_sentences(model, sentences, max_length=512, *, truncate_prefix):
    sentences = ["[CLS] " + s for s in sentences]
    # TODO(sam): cache all of this stuff (i.e. pass in the tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(model)
    ids, amasks = get_ids_mask(sentences, tokenizer, max_length, truncate_prefix)
    inputs = torch.tensor(ids)
    masks = torch.tensor(amasks)
    return inputs, masks


def main(args):
    load_path = "{}_{}.pt".format(args.data, args.model)
    model = load_model(args.model, args.ngpus, load_path)
    model.eval()

    while True:
        sentence = input("Input: ")
        input_ids, input_mask = load_process_sentences(
            args.model, [sentence], args.max_length,
            truncate_prefix=False,
        )
        with torch.no_grad():
            output = model(input_ids, attention_mask=input_mask)[0]
        if args.data == "cm":  # commonsense morality
            prob = torch.sigmoid(output).squeeze(dim=1).detach().cpu().numpy()[0]
            if output >= 0:
                print("Morally wrong, with confidence: {:.3f}".format(prob))
            else:
                print("Morally right, with confidence: {:.3f}".format(1 - prob))
        else:  # utilitarianism
            assert args.data == "util", args.data
            print("Utility Value: {:.3f}".format(output.item()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type=str, default="cm")
    parser.add_argument("--model", "-m", type=str, default="roberta-large")
    parser.add_argument("--ngpus", "-n", type=int, default=1)
    parser.add_argument("--max_length", "-t", type=int, default=64)
    main(parser.parse_args())