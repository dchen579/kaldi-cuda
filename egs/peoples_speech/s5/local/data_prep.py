#!/usr/bin/env python3

import argparse
from concurrent.futures import ThreadPoolExecutor
import contextlib
import glob
import json
import locale
import os
import subprocess
import sys
import tarfile
from threading import Lock

from g2p_en import G2p
from smart_open import open
import tqdm

def remove_space(s: str):
    return s.replace(" ", "_")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir',
                        required=True,
                        type=str)
    parser.add_argument('--words_txt',
                        required=True,
                        type=str)
    parser.add_argument('--dict_dir',
                        required=True,
                        type=str)
    parser.add_argument('--input_dir',
                        required=True,
                        type=str)
    parser.add_argument('--unk_string',
                        required=True,
                        type=str)
    parser.add_argument('--stage',
                        required=True,
                        type=int)
    parser.add_argument('--nj',
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
        tar_file_names = glob.glob(os.path.join(input_dir, "repartitioned_dataset_tars_jul_28_wav_new_join_no_space/*.tar"))
        # tar_file_names = tar_file_names[:1]
        lock = Lock()
        all_lines = []
        with open(os.path.join(data_dir, "wav.scp"), "w") as wav_scp_fh:
            def write_data(tar_file_name):
                with open(tar_file_name, mode="rb") as blob_fh, \
                     tarfile.open(fileobj=blob_fh, mode="r") as tar_fh:
                    members = tar_fh.getmembers()
                    lines = [f"{remove_space(member.name)} dd if={tar_file_name} of=/dev/stdout skip={member.offset_data} count={member.size} iflag=skip_bytes,count_bytes |\n"
                             for member in members]
                    # lines = [f"{remove_space(member.name)} dd if=input.binary of=output.binary skip={member.offset_data} count={member.size} iflag=skip_bytes,count_bytes < tar_file_name | sox -t flac - -t wav -|\n"
                    #          for member in members]
                with lock:
                    all_lines.extend(lines)
            executor = ThreadPoolExecutor(args.nj)
            list(tqdm.tqdm(executor.map(write_data, tar_file_names), total=len(tar_file_names)))
            all_lines.sort()
            for line in all_lines:
                wav_scp_fh.write(line)

            # for tar_file_name in tqdm.tqdm(tar_file_names):
            #     with tarfile.open(tar_file_name) as tar_fh:
            #         members = tar_fh.getmembers()
            #         for member in members:
            #             # if " " in member.name:
            #             #     print(f"Kaldi does not support keys with a space in them! Got {member.name}")
            #             name = remove_space(member.name)
            #             wav_scp_fh.write(f"{name} dd if=input.binary of=output.binary skip={member.offset_data} count={member.size} iflag=skip_bytes,count_bytes|\n")
        del all_lines

    if stage <= 1:
        print("Create text file")
        manifest_name,  = glob.glob(os.path.join(input_dir, "dataset_manifest_nemo_jul_28_wav_filtered_single_new_join_no_space/*.json"))

        with open(manifest_name, "r") as manifest_fh:
            key_to_transcript = []
            for manifest_line in tqdm.tqdm(manifest_fh):
                manifest_line = manifest_line.rstrip()
                manifest_obj = json.loads(manifest_line)
                key = remove_space(manifest_obj['audio_filepath'])
                key_to_transcript.append((key, manifest_obj['text']))

        # key_to_transcript = [(k, v) for (k, v) in key_to_transcript.items()]
        key_to_transcript.sort(key=lambda x: locale.strxfrm(x[0]))

        with open(os.path.join(data_dir, "text"), "w") as text_fh, \
             open(os.path.join(data_dir, "utt2spk"), "w") as utt2spk_fh:
            for key, transcript in key_to_transcript:
                text_fh.write(f"{key} {transcript}\n")
                utt2spk_fh.write(f"{key} {key}\n")

        # Conserve memory. This list is multiple gigabytes in size.
        del key_to_transcript

        with open(os.path.join(data_dir, "utt2spk"), "r") as utt2spk_fh, \
             open(os.path.join(data_dir, "spk2utt"), "w") as spk2utt_fh:
            subprocess.check_call(["utils/utt2spk_to_spk2utt.pl"], stdin=utt2spk_fh, stdout=spk2utt_fh)

    if stage <= 2:
        os.makedirs(dict_dir, exist_ok=True)
        print("Create dictionary directory")
        word_to_phonemes = {}
        all_phonemes = set()
        g2p = G2p()

        with open(args.words_txt) as words_fh:
            for line in tqdm.tqdm(words_fh.readlines()):
                word = line.rstrip()
                if word in word_to_phonemes:
                    continue
                phonemes = g2p(word)
                for phoneme in phonemes:
                    if phoneme == " ":
                        print(f"Offending word {word}: {phonemes}.\n in line: {line}", file=sys.stderr)
                phonemes = [p for p in phonemes if p != " "]
                word_to_phonemes[word] = phonemes
                for phoneme in phonemes:
                    all_phonemes.add(phoneme)

        with open(os.path.join(data_dir, "text"), "r") as text_fh:
            for line in tqdm.tqdm(text_fh.readlines()):
                line = line.rstrip()
                _, words = line.split(" ", 1)
                for word in words.split():
                    if word in word_to_phonemes:
                        continue
                    phonemes = g2p(word)
                    for phoneme in phonemes:
                        if phoneme == " ":
                            print(f"Offending word {word}: {phonemes}.\n in line: {line}", file=sys.stderr)
                    phonemes = [p for p in phonemes if p != " "]
                    word_to_phonemes[word] = phonemes
                    for phoneme in phonemes:
                        all_phonemes.add(phoneme)

        # word_to_phonemes = [(k, v) for (k, v) in word_to_phonemes.items()]
        # word_to_phonemes.sort(key=lambda x: locale.strxfrm(x[0]))
        with open(os.path.join(dict_dir, "words.txt"), "w") as dict_fh:
            for word, _ in word_to_phonemes.items():
                dict_fh.write(f"{word}\n")

        with open(os.path.join(dict_dir, "silence_phones.txt"), "w") as silence_phones_fh:
            silence_phones_fh.write("SIL\nSPN\n")
        with open(os.path.join(dict_dir, "optional_silence.txt"), "w") as optional_silence_fh:
            optional_silence_fh.write("SIL\n")
        with open(os.path.join(dict_dir, "nonsilence_phones.txt"), "w") as nonsilence_phones_fh:
            for phoneme in sorted(all_phonemes):
                nonsilence_phones_fh.write(f"{phoneme}\n")
        with open(os.path.join(dict_dir, "extra_questions.txt"), "w") as extra_questions_fh:
            extra_questions_fh.write("SIL SPN\n")
            # extra_questions_fh.write("SIL_B SPN_B\n")
            # extra_questions_fh.write("SIL_E SPN_E\n")
            # extra_questions_fh.write("SIL_I SPN_I\n")
            # extra_questions_fh.write("SIL_S SPN_S\n")

        with open(os.path.join(dict_dir, "lexicon.txt"), "w") as dict_fh:
            dict_fh.write(f'!SIL SIL\n<spoken_noise> SPN\n{args.unk_string} SPN\n')
            for word, phonemes in word_to_phonemes.items():
                dict_fh.write(f"{word} {' '.join(phonemes)}\n")

    if stage <= 3:
        # validate_data_dir.sh will complain if lexiconp.txt doesn't
        # match lexicon.txt. However, if you are modifying the inputs
        # to local/data_prep.py after having run
        # utils/prepare_lang.sh, lexiconp.txt will have been created
        # already by utils/prepare_lang.sh on the old lexicon.txt
        # created by the previous run of data_prep.py
        with contextlib.suppress(FileNotFoundError):
            os.remove(os.path.join(dict_dir, "lexiconp.txt"))
        subprocess.check_call(["utils/validate_data_dir.sh", "--no-feats", data_dir])
        
if __name__ == '__main__':
    main()
