import itertools as it
import os
import pickle
import re
import signal
from copy import deepcopy
from subprocess import PIPE, Popen, TimeoutExpired, check_output
from sympy import And, Not, Or, simplify, symbols
from ltlf2dfa.base import MonaProgram
from itertools import chain, combinations
from ltlf2dfa.parser.ltlf import LTLfParser


def powerset(iterable):
    """ print the powerset of an iterable """
    # example: powerset([1,2]) gives (), (1,), (2,), (1,2)
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def get_value(text, regex, value_type=float):
    """Dump a value from a file based on a regex passed in."""
    pattern = re.compile(regex, re.MULTILINE)
    results = pattern.search(text)
    if results:
        return value_type(results.group(1))
    else:
        print("Could not find the value {}, in the text provided".format(regex))
        return value_type(0.0)


def ter2symb(ap, ternary):
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


class DFA():

    def __init__(self, formula_str, file_str):
        """ DFA automation object
        :param formula_str: ltlf formula
        :param file_str: /Users/shendongming/PycharmProjects/pythonProject/venv/lib/python3.9/site-packages
        """
        self.Q = []  # tuple with n_qs (num_states - 1) index in str format (from '1' to 'n_qs')
        self.q0 = None   # q0 = '1' (start?)
        self.n_qs = -1  # (states_num - 1)? => Q
        self.acc = []  # accepting states as list
        self.T = {}  # put into T final format of Transitions

        __file__ = file_str  # file_str: site-package holding the "ltlf2dfa"
        PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))  # site-package directory absolute path
        parser = LTLfParser()  # from ltlf2dfa.parser.ltlf import LTLfParser
        formula = parser(formula_str)  # parse str into LTLf format
        p = MonaProgram(formula)  # p is a class holding multiple informations about the LTLf format
        """
        p.formula => LTLf formula
        p.vars => variables in the formula
        p.HEADER => not sure what it does
        p.mona_program() => a long list, expanding the LTLf formula?
        ...
        """
        mona_p_string = p.mona_program()  # a long list, expanding the LTLf formula?
        file = open("{}/automa.mona".format(PACKAGE_DIR), "w+")  # automa.mona TextIOWrapper (create new file to write)
        file.write(mona_p_string)
        file.close()  # write the mona_str into the file & close file

        # paste mona executable location before the "-q -u -w {}/automa.mona"
        command = "/Users/shendongming/Desktop/AIDyS/AIDyS_pomdp/pomdp_ltlf/model_generation_decpomdp/mona-1.4/Front/mona -q -u -w {}/automa.mona".format(PACKAGE_DIR)
        process = Popen(args=command,  # command: mona -q -u -w automa.mona
                        stdout=PIPE,
                        stderr=PIPE,
                        preexec_fn=os.setsid,  # can only use in MAC (not working in Windows)
                        shell=True,
                        encoding="utf-8",
                        )  # Popen?

        output, error = process.communicate(timeout=30)  # error message of the process (if any, otherwise '')
        print(error)
        assert error == ''  # added assertion so if error in process, will stop the program
        mona_output = str(output).strip()  # DFA format of the LTLf formula
        """
        DFA for formula with free variables: ...
        Initial state: ...
        Accepting states: ... 
        Rejecting states: ...
        Automaton has ... states and ... BDD-nodes
        Transitions: ...
        A counter-example of least length (0) is: ...
        A satisfying example of least length (1) is:
        """

        free_variables = get_value(mona_output, r".*DFA for formula with free variables:[\s]*(.*?)\n.*", str)
        free_variables = symbols(" ".join(x.strip().lower() for x in free_variables.split() if len(x.strip()) > 0))
        # free_variables => tuple of free variables in lower case from the DFA

        char_map = dict(enumerate(map(str, free_variables)))  # map: index (0,1,2,...) to free_variables
        char_ind = {c: i for i, c in enumerate(free_variables)}  # reverse map: free_variables to index
        ap_list = [tuple(ap) for ap in powerset(sorted(char_map.values()))]  # powerset of the free_variables tuple

        """n_qs => the number of automaton states"""
        self.n_qs = get_value(mona_output, '.*Automaton has[\s]*(\d+)[\s]states.*', int) - 1  # states_num - 1???
        self.Q = tuple(str(i) for i in range(1, self.n_qs + 1))  # tuple with n_qs index in str format ('1' to 'n_qs')
        self.q0 = str(1)  # q0 = '1' (start?)

        accepting_states = get_value(mona_output, r".*Accepting states:[\s]*(.*?)\n.*", str)  # accepting states as str
        self.acc = [str(x.strip()) for x in accepting_states.split() if len(x.strip()) > 0]  # accepting states as list

        for line in mona_output.splitlines():
            if line.startswith("State "):  # mona_output line that starts with State => in "Transitions: ..."
                """ format of each such line:
                "State S: ? -> state F"
                S => orig_state
                ? => guard (get symbol using ter2symb)
                F => dest_state
                """
                # print(line)
                orig_state = get_value(line, r".*State[\s]*(\d+):\s.*", str)  # original state (start state?)
                # print(orig_state)
                if orig_state == '0':  # don't care about initial state
                    continue
                else:  # not initial state
                    guard = get_value(line, r".*:[\s](.*?)[\s]->.*", str)
                    guard = ter2symb(free_variables, guard)  # logic symbol
                    # print(guard)
                    dest_state = get_value(line, r".*state[\s]*(\d+)[\s]*.*", str)
                    # print(dest_state)
                    c = list(map(lambda x: x.strip(), str(guard).split('&')))  # logic, parse & differently
                    # print(c)
                    """
                    label_acc => labels that can be accepted by the logic
                    label_rej => labels that should be rejected by the logic
                    """
                    if c[0] == 'True':
                        label_acc, label_rej = set(()), set(())
                    else:
                        label_acc = set(l for l in c if not l.startswith('~'))
                        label_rej = set([l[1:] for l in c if l.startswith('~')])
                    # print(label_acc, label_rej)

                    for ap in ap_list:  # ap_list => powerset (tuples) of the free_variables tuple
                        if not (label_acc - set(ap)) and (label_rej - set(ap)) == label_rej:
                            self.T.update({(orig_state, ap): dest_state})  # put into T final format of Transitions?
                            # formate of T (delta) input: {(orig_state, ap): dest_state, ...}


def main():
    specifications = 'F(a) & G(!w)'
    A = DFA(specifications, '/Users/shendongming/PycharmProjects/pythonProject/venv/lib/python3.9/site-packages')
    output_path = "DFA/reachAavoidW.txt"
    f_out = open(output_path, "w")
    f_out.writelines(specifications + "\n")
    f_out.writelines("A.n_qs: \n")
    f_out.writelines(str(A.n_qs) + "\n")
    f_out.writelines("A.Q: \n")
    f_out.writelines(str(A.Q) + "\n")
    f_out.writelines("A.q0: \n")
    f_out.writelines(str(A.q0) + "\n")
    f_out.writelines("A.acc: \n")
    f_out.writelines(str(A.acc) + "\n")
    f_out.writelines("A.T: \n")
    f_out.writelines(str(A.T) + "\n")
    f_out.close()
    print("complete")
    exit(0)


if __name__ == '__main__':
    main()
