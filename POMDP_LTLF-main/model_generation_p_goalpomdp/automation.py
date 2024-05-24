import itertools as it
import os
import re
import signal
from subprocess import PIPE, Popen, TimeoutExpired, check_output
from sympy import And, Not, Or, simplify, symbols
from ltlf2dfa.base import MonaProgram
from itertools import chain, combinations
from ltlf2dfa.parser.ltlf import LTLfParser


def powerset(iterable):  # same
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def get_value(text, regex, value_type=float):  # same
    """Dump a value from a file based on a regex passed in."""
    pattern = re.compile(regex, re.MULTILINE)
    results = pattern.search(text)
    if results:
        return value_type(results.group(1))
    else:
        print("Could not find the value {}, in the text provided".format(regex))
        return value_type(0.0)


def ter2symb(ap, ternary):  # same
    """Translate ternary output to symbolic."""
    expr = And()
    i = 0
    for value in ternary:
        if value == "1":
            expr = And(expr, ap[i] if isinstance(ap, tuple) else ap)
        elif value == "0":
            assert value == "0"
            expr = And(expr, Not(ap[i] if isinstance(ap, tuple) else ap))
        else:
            assert value == "X", "[ERROR]: the guard is not X"
        i += 1
    return expr


class DFA():  # same

    def __init__(self, formula_str, file_str):  # same

        self.Q = []
        self.q0 = None
        self.n_qs = -1
        self.acc = []
        self.T = {}

        __file__ = file_str
        PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
        parser = LTLfParser()
        formula = parser(formula_str)
        p = MonaProgram(formula)
        mona_p_string = p.mona_program()

        file = open("{}/automa.mona".format(PACKAGE_DIR), "w+")
        file.write(mona_p_string)
        file.close()

        command = "/Users/dongmingshen/Downloads/mona-1.4/Front/mona -q -u -w {}/automa.mona".format(PACKAGE_DIR)
        process = Popen(args=command,
                        stdout=PIPE,
                        stderr=PIPE,
                        preexec_fn=os.setsid,
                        shell=True,
                        encoding="utf-8",
                        )

        output, error = process.communicate(timeout=30)
        mona_output = str(output).strip()
        # print(mona_output)

        free_variables = get_value(mona_output, r".*DFA for formula with free variables:[\s]*(.*?)\n.*", str)
        free_variables = symbols(" ".join(x.strip().lower() for x in free_variables.split() if len(x.strip()) > 0))

        char_map = dict(enumerate(map(str, free_variables)))
        char_ind = {c: i for i, c in enumerate(free_variables)}
        ap_list = [tuple(ap) for ap in powerset(sorted(char_map.values()))]

        self.n_qs = get_value(mona_output, '.*Automaton has[\s]*(\d+)[\s]states.*', int) - 1
        self.Q = tuple(str(i) for i in range(1, self.n_qs + 1))
        self.q0 = str(1)

        accepting_states = get_value(mona_output, r".*Accepting states:[\s]*(.*?)\n.*", str)
        self.acc = [str(x.strip()) for x in accepting_states.split() if len(x.strip()) > 0]

        for line in mona_output.splitlines():
            if line.startswith("State "):

                orig_state = get_value(line, r".*State[\s]*(\d+):\s.*", str)

                if orig_state == '0':
                    continue
                else:
                    guard = get_value(line, r".*:[\s](.*?)[\s]->.*", str)
                    guard = ter2symb(free_variables, guard)
                    dest_state = get_value(line, r".*state[\s]*(\d+)[\s]*.*", str)
                    c = list(map(lambda x: x.strip(), str(guard).split('&')))

                    if c[0] == 'True':
                        label_acc, label_rej = set(()), set(())
                    else:
                        label_acc = set(l for l in c if not l.startswith('~'))
                        label_rej = set([l[1:] for l in c if l.startswith('~')])

                    for ap in ap_list:
                        if not (label_acc - set(ap)) and (label_rej - set(ap)) == label_rej:
                            self.T.update({(orig_state, ap): dest_state})