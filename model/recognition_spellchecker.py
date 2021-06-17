import csv
import re
from pathlib import Path
from typing import Sequence, Dict

import copy

import enchant

from braille_utils.postprocess import Line


class InDelSymbols:
    ins = "▲"
    delet = "◂"
    match = "⊙"
    dummy = "$"


class OneLineString:  # TODO duplication with brl_data_tools
    """
    Basic form of string. All texts for alignment/region finding are first converted into this, then to specific forms.
    """

    def __init__(self, raw_text: str, convert_ampersands: bool = False):
        no_tabs_text = re.sub("[\t\r\n]", " ", raw_text)

        # we'll denote capital sign as `^`
        # TODO maybe convert only the first match after space?
        caps_converted_text = re.sub("[A-Z]", lambda ch: StringForAlignment.caps_sign + ch.group(0).lower(),
                                     no_tabs_text)
        single_quotes_converted_text = re.sub(r"([^a-z])‘(.*)’([^a-z])", r"\1“\2”\3", caps_converted_text)
        apostrope_converted_text = re.sub("[‘’]", "'", single_quotes_converted_text)
        no_line_numbers_text = re.sub(r"\[([0-9]|[a-z])*" + StringForAlignment.number_sign_regex, " ",
                                      apostrope_converted_text)
        dash_converted_text = re.sub("—", "--", no_line_numbers_text)
        # in Braille, dots (…) are three separate dots (...)
        dots_exploded_text = re.sub("…", "...", dash_converted_text)
        # in Jane Eyre ampersand is always preceded by stress (') TODO do not do it in general case or create parameter
        amp_converted_text = re.sub("&", "'&", dots_exploded_text)
        whitespace_reduced_text = re.sub(" +", " ", amp_converted_text if convert_ampersands else dots_exploded_text)

        special_letters_map = {
            "è": "e",
            "é": "e",
            "ê": "e",
            "ë": "e",
            "ï": "i",
            "ä": "a",
            "â": "a",
            "à": "a",
            "ç": "c",
            "ô": "o",
            "œ": "ae",
            "æ": "ae",
        }
        text_no_special = "".join(
            [special_letters_map[ch] if ch in special_letters_map else ch for ch in whitespace_reduced_text])
        self.text = text_no_special


class StringForAlignment:
    """
    A form of text for alignment. All numbers converted to number sign + letters
    """
    number_sign = "]"
    number_sign_regex = r"\]"
    caps_sign = "^"
    caps_sign_regex = r"\^"
    num_to_letters = {1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 6: "f", 7: "g", 8: "h", 9: "i", 0: "j"}

    @classmethod
    def to_letters(cls, number_str: str) -> str:
        return cls.number_sign + "".join(cls.num_to_letters[int(ch)] for ch in number_str)

    @classmethod
    def return_digits_to_text(cls, text: str, remove_number_sign: bool) -> str:
        letters_to_num = {value: key for key, value in cls.num_to_letters.items()}

        def substitute_with_num(letters_str: str):
            result = "".join(str(letters_to_num[ch]) for ch in letters_str[1:])
            return result if remove_number_sign else StringForAlignment.number_sign + result

        return re.sub(cls.number_sign_regex + "[a-j]*", lambda match: substitute_with_num(match.group(0)), text)

    def __init__(self, one_line_str: OneLineString):
        self.text = re.sub(r"\d+", lambda match: StringForAlignment.to_letters(match.group(0)), one_line_str.text). \
            replace("«", "“").replace("»", "”")


class RecognitionSpellchecker:

    @staticmethod
    def _read_csv_stats(stats_file_name: Path) -> Dict[str, Dict[str, int]]:
        result = {}
        replacements = {
            "unknown": "■",
            "del": InDelSymbols.delet,
            "ins": InDelSymbols.ins
        }
        with open(str(stats_file_name)) as csvfile:
            csvreader = csv.reader(csvfile)
            header = next(csvreader)[1:]
            for row in csvreader:
                k = row[0].strip()
                k = replacements[k] if k in replacements else k
                result[k] = {replacements[key] if key in replacements else key: int(value) for key, value
                             in zip(header, row[1:])}
        return result

    def __init__(self, common_errs_json_fname: Path, subst_mtx_csv_fname: Path):
        self.substitutions = self._read_csv_stats(subst_mtx_csv_fname)
        self.dic = enchant.Dict("en_US")

    def _correct(self, text: str) -> str:
        words = text.split()
        replacements = []
        for word in words:
            word_alpha = "".join([ch for ch in word if ch.isalpha()])
            if not len(word_alpha) or self.dic.check(word_alpha.capitalize()):
                replacements.append(word)
                continue

            feasible_subs = {replacement: d
                             for replacement, d in self.substitutions.items() if replacement in word_alpha}
            feasible_subs = [(replacement, orig, count)
                             for replacement, d in feasible_subs.items() for orig, count in d.items()]
            feasible_subs.sort(key=lambda tup: tup[2], reverse=True)
            max_check = 20
            for replacement, orig, _ in feasible_subs[:max_check]:
                new_word = word.replace(replacement, orig, 1)  # TODO try to replace 2nd occurence, etc
                new_word_alpha = "".join([ch for ch in new_word if ch.isalpha()])
                if self.dic.check(new_word_alpha.capitalize()):
                    word = new_word
                    break  # TODO split into functions to eliminate nested `for`
            replacements.append(word)

        result = " ".join(replacements)
        return re.sub(StringForAlignment.caps_sign_regex + "([a-z])", lambda match: " " + match.group(1).upper(),
                      result)

    def improve(self, lines: Sequence[Line]) -> Sequence[Line]:
        lines = copy.deepcopy(lines)

        for line in lines:
            text_initial = "".join([" " * ch.spaces_before + ch.char for ch in line.chars])
            text_initial = StringForAlignment(OneLineString(text_initial))
            text_initial = StringForAlignment.return_digits_to_text(text=text_initial.text, remove_number_sign=False)
            text_corrected = self._correct(text=text_initial)
            if text_initial == text_corrected or InDelSymbols.ins in text_corrected:  # TODO find what's wrong with ins
                continue

            indices_to_remove = []
            i_label = 0
            replacements = {
                StringForAlignment.number_sign: "##",
                "“": "«",
                "”": "»",
            }
            for q_symbol, ref_symbol in zip(text_initial, text_corrected):
                if q_symbol in (" ", InDelSymbols.delet, StringForAlignment.caps_sign):
                    continue
                if ref_symbol in (InDelSymbols.ins, " "):
                    indices_to_remove.append(i_label)
                new_label = replacements[ref_symbol] if ref_symbol in replacements.keys() else ref_symbol
                line.chars[i_label].char = new_label
                i_label += 1
            for i_lbl in reversed(indices_to_remove):
                line.chars.pop(i_lbl)

        return lines
