#!/usr/bin/env python3

import argparse
import glob
import json
import os
import subprocess
import tarfile
from threading import Lock

from g2p_en import G2p
import tqdm

def remove_space(s: str):
    return s.replace(" ", "_")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir',
                        required=True,
                        type=str)
    parser.add_argument('--dict_dir',
                        required=True,
                        type=str)
    parser.add_argument('--input_dir',
                        required=True,
                        type=str)
    parser.add_argument('--stage',
                        required=True,
                        type=int)

    args = parser.parse_args()

    data_dir = args.data_dir
    dict_dir = args.dict_dir
    input_dir = args.input_dir
    stage = args.stage

    if stage <= 0:
        print("Create wav.scp file")
        os.makedirs(data_dir, exist_ok=True)
        tar_file_names = glob.glob(os.path.join(input_dir, "tars/*.tar"))
        tar_file_names = tar_file_names[:1]
        lock = Lock()
        with open(os.path.join(data_dir, "wav.scp"), "w") as wav_scp_fh:
            def write_data(tar_file_name):
                with tarfile.open(tar_file_name) as tar_fh:
                    members = tar_fh.getmembers()
                    lines = [f"{remove_space(member.name)} dd if=input.binary of=output.binary skip={member.offset_data} count={member.size} iflag=skip_bytes,count_bytes|\n"
                             for member in members]
                    lines = "".join(lines)
                # May need to acquire the lock in a particular order...
                with lock:
                    wav_scp_fh.write(lines)
                
            list(tqdm(executor.map(write_data, tar_file_names), total=len(tar_file_names)))
            # for tar_file_name in tqdm.tqdm(tar_file_names):
            #     with tarfile.open(tar_file_name) as tar_fh:
            #         members = tar_fh.getmembers()
            #         for member in members:
            #             # if " " in member.name:
            #             #     print(f"Kaldi does not support keys with a space in them! Got {member.name}")
            #             name = remove_space(member.name)
            #             wav_scp_fh.write(f"{name} dd if=input.binary of=output.binary skip={member.offset_data} count={member.size} iflag=skip_bytes,count_bytes|\n")

    if stage <= 1:
        print("Create text file")
        manifest_name,  = glob.glob(os.path.join(input_dir, "manifest/*.json"))

        with open(os.path.join(data_dir, "text"), "w") as text_fh, \
             open(os.path.join(data_dir, "utt2spk"), "w") as utt2spk_fh, \
             open(manifest_name, "r") as manifest_fh:
            for manifest_line in tqdm.tqdm(manifest_fh):
                manifest_line = manifest_line.rstrip()
                manifest_obj = json.loads(manifest_line)
                key = remove_space(manifest_obj['audio_filepath'])
                text_fh.write(f"{key} {manifest_obj['text']}\n")
                utt2spk_fh.write(f"{key} {key}\n")
        with open(os.path.join(data_dir, "utt2spk"), "r") as utt2spk_fh, \
             open(os.path.join(data_dir, "spk2utt"), "w") as spk2utt_fh:
            subprocess.check_call(["utils/utt2spk_to_spk2utt.pl"], stdin=utt2spk_fh, stdout=spk2utt_fh)

    if stage <= 2:
        os.makedirs(dict_dir, exist_ok=True)
        print("Create dictionary directory")
        word_to_phonemes = {}
        all_phonemes = set()
        g2p = G2p()
        with open(os.path.join(data_dir, "text"), "r") as text_fh:
            for line in tqdm.tqdm(text_fh.readlines()):
                line = line.rstrip()
                _, words = line.split(" ", 1)
                for word in words.split():
                    if word in word_to_phonemes:
                        continue
                    phonemes = g2p(word)
                    word_to_phonemes[word] = phonemes
                    for phoneme in phonemes:
                        all_phonemes.add(phoneme)

        with open(os.path.join(dict_dir, "words.txt"), "w") as dict_fh:
            for word in word_to_phonemes.keys():
                dict_fh.write(f"{word}\n")

        with open(os.path.join(dict_dir, "silence_phones.txt"), "w") as silence_phones_fh:
            silence_phones_fh.write("SIL\nSPN\n")
        with open(os.path.join(dict_dir, "optional_silence.txt"), "w") as optional_silence_fh:
            optional_silence_fh.write("SIL\n")
        with open(os.path.join(dict_dir, "nonsilence_phones.txt"), "w") as nonsilence_phones_fh:
            for phoneme in sorted(phonemes):
                nonsilence_phones_fh.write(f"{phoneme}\n")
        # with open(os.path.join(dict_dir, "extra_questions.txt"), "w") as extra_questions_fh:
        #     assert False
        #     # extra_questions_fh.write()

        with open(os.path.join(dict_dir, "lexicon.txt"), "w") as dict_fh:
            dict_fh.write('!SIL SIL\n<spoken_noise> SPN\n<unk> SPN\n')
            for word, phonemes in word_to_phonemes.items():
                dict_fh.write(f"{word} {' '.join(phonemes)}\n")

    if stage <= 3:
        subprocess.check_call(["utils/validate_data_dir.sh", "--no-feats", data_dir])
        
if __name__ == '__main__':
    main()
